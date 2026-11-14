"""Both levels learned, in alternating rounds.

In each round the bid trains for `rounds.episodes_per_level` episodes, then the dispatch trains
for as many. Each trains against a partner:

| `co_train` | the bid trains against | the dispatch trains against | policy kind |
|---|---|---|---|
| False | the best rule's dispatch | the best rule's bid | `bid-and-dispatch` |
| True | the current learned dispatch | the current learned bid | `hierarchical` |

After every round the two learned levels run together on the validation episodes, paired against
the best rule pair. Round 0 is the starting pair. Training stops by the rule in
`schedule.Rounds`. The round with the best validation gain is kept and run once on the test episodes.

**Starting pair.** Any saved bid and dispatch. Where this step needs a wider network, the start
is widened with zero weights (`flexbid.policies.nets`), so round 0 acts exactly as the start:

* `message="fresh"` or `"fixed"`: the bid gets a fifth output, the message, starting at 0;
* `dispatch_extra="commitments"`: the dispatch gets 4 extra inputs;
* a bid lead the start was not trained with: the bid gets the extra inputs.

A bid that needs no widening is reloaded with its optimizer state. A widened bid, and the
dispatch, start with a fresh optimizer.

Writes into `out`: `bid.zip` and `dispatch.pt` (the chosen round), `last_bid.zip`,
`last_dispatch.pt`, `checkpoints/`, `monitor.jsonl`, `log.txt` and `result.json`.
"""
from __future__ import annotations

import shutil
import time

import numpy as np

from .evaluate import evaluate_pair, paired
from .learners import (RunLog, make_venv, new_dispatch_model, save_bid, save_dispatch,
                       steps_per_episode)
from .schedule import assert_not_test, episodes_per_env, training_seed


def stop_reason(curve, rnd, rounds):
    """Why training stops after round `rnd`, or None to go on. `curve` holds one validation
    point per round, round 0 first."""
    if rnd >= rounds.max_rounds:
        return f"{rounds.max_rounds} rounds done"
    if rnd < rounds.min_rounds:
        return None
    best, last_up = -np.inf, 0
    for q in curve:
        if q["gain"] > best + rounds.tol:
            last_up = q["round"]
        best = max(best, q["gain"])
    if rnd - last_up >= rounds.patience:
        return f"no gain above {rounds.tol} $/day in {rnd - last_up} rounds"
    return None


def _bid_model(start, venv, lr, seed):
    """The start bid on this environment: reloaded if its shape fits, else widened."""
    from stable_baselines3 import PPO
    from flexbid.policies.nets import widen_bid
    old = PPO.load(start, device="cpu")
    same = (old.observation_space.shape == venv.observation_space.shape
            and old.action_space.shape == venv.action_space.shape)
    if same:
        return PPO.load(start, env=venv, device="cpu"), "reloaded"
    return widen_bid(old, venv, lr, seed), "widened"


def train_rounds(cfg, best, bid_start, dispatch_start, rounds, schedule, out, *, step, seed=0,
                 co_train=True, message=None, dispatch_extra=None, pool=None):
    """Train both levels in rounds. See the module docstring.

    | argument | meaning |
    |---|---|
    | best | the rule search's best pair, {"bid": name, "dispatch": name} |
    | bid_start, dispatch_start | saved bid (.zip) and dispatch (.pt) to start from |
    | rounds | a `schedule.Rounds`: episodes per level per round and the stopping rule |
    | step | the name of this step, which also selects its block of training episodes |
    | message | None, "fresh" or "fixed" |
    | dispatch_extra | None, or "commitments" |
    """
    import torch as th
    from flexbid.policies.features import DISPATCH_EXTRA
    from flexbid.policies.nets import widen_dispatch
    s = schedule
    if message is not None and not co_train:
        raise ValueError("a message needs the two levels trained together")
    log = RunLog(out)
    ck = log.dir / "checkpoints"
    r_bid, r_dis = best["bid"], best["dispatch"]
    n_extra = DISPATCH_EXTRA[dispatch_extra]
    per_env = episodes_per_env(rounds.episodes_per_level, s.n_envs)
    assert_not_test(s.validation_seeds, "the validation episodes", s.test_seeds)

    def B(path):
        return {"path": str(path), "message": message}

    def D(path):
        return {"path": str(path), "extra": dispatch_extra}

    def venv_for(level, partner, rnd):
        first = [training_seed(seed, step, level == "dispatch", rnd, i) for i in range(s.n_envs)]
        return make_venv(level, cfg, partner, first, message if level == "bid" else dispatch_extra)

    # ---- round 0: the starting pair, widened where needed
    t0 = time.time()
    d_cur = str(ck / "round0_dispatch.pt")
    th.save(widen_dispatch(th.load(dispatch_start, map_location="cpu"), n_extra).state_dict(), d_cur)
    venv_b = venv_for("bid", r_dis if not co_train else D(d_cur), 1)
    model_b, how = _bid_model(str(bid_start), venv_b, s.lr_bid, seed)
    b_cur = save_bid(model_b, ck / "round0_bid.zip")
    venv_d = venv_for("dispatch", r_bid if not co_train else B(b_cur), 1)
    model_d = new_dispatch_model(venv_d, cfg, s.lr_dispatch, seed, n_extra)
    model_d.policy.net.load_state_dict(th.load(d_cur, map_location="cpu"))
    log(f"{step}: {'trained together' if co_train else 'each against the best rule'}; message={message}, "
        f"dispatch extra={dispatch_extra}; bid {how} from {bid_start}; dispatch from {dispatch_start}; "
        f"{rounds.episodes_per_level} episodes per level per round (about {per_env} per environment); "
        f"rounds {rounds.min_rounds}-{rounds.max_rounds}, patience {rounds.patience}, tol {rounds.tol} $/day")

    R_val = evaluate_pair(cfg, r_bid, r_dis, s.validation_seeds, pool)

    def point(rnd, b, d):
        M = evaluate_pair(cfg, B(b), D(d), s.validation_seeds, pool)
        g, se = paired(M, R_val)
        return dict(round=rnd, episodes=2 * rounds.episodes_per_level * rnd, mean=M["mean"], se=M["se"],
                    rule_pair=R_val["mean"], gain=g, gain_se=se, message=M["message"], minutes=(time.time() - t0) / 60)

    curve = [point(0, b_cur, d_cur)]
    log.point(curve[0], first=True)
    log(f"round 0 (the starting pair): gain over the rule pair {curve[0]['gain']:+.1f} ± {curve[0]['gain_se']:.1f} $/day (validation)")

    # ---- the rounds
    rnd, reason = 0, stop_reason(curve, 0, rounds)
    while reason is None:
        rnd += 1
        if rnd > 1:
            venv_b.close()
            venv_b = venv_for("bid", D(d_cur) if co_train else r_dis, rnd)
            model_b.set_env(venv_b)
        model_b.learn(total_timesteps=rounds.episodes_per_level * steps_per_episode("bid", cfg), reset_num_timesteps=False)
        b_cur = save_bid(model_b, ck / f"round{rnd}_bid.zip")
        venv_d.close()
        venv_d = venv_for("dispatch", B(b_cur) if co_train else r_bid, rnd)
        model_d.set_env(venv_d)
        model_d.learn(total_timesteps=rounds.episodes_per_level * steps_per_episode("dispatch", cfg), reset_num_timesteps=False)
        d_cur = save_dispatch(model_d, ck / f"round{rnd}_dispatch.pt")
        pt = point(rnd, b_cur, d_cur)
        curve.append(pt)
        log.point(pt)
        top = max(curve, key=lambda q: q["gain"])
        log(f"round {rnd}: gain over the rule pair {pt['gain']:+.1f} ± {pt['gain_se']:.1f} $/day (best: round {top['round']}, "
            f"{top['gain']:+.1f}); message mean {pt['message']['mean']:+.4f}, sd {pt['message']['sd']:.4f}; {pt['minutes']:.0f} min")
        reason = stop_reason(curve, rnd, rounds)
    venv_b.close()
    venv_d.close()
    log(f"stop after round {rnd}: {reason}")

    # ---- choose on the validation episodes, then the test episodes once
    top = max(curve, key=lambda q: q["gain"])
    br = top["round"]
    shutil.copy(ck / f"round{br}_bid.zip", log.dir / "bid.zip")
    shutil.copy(ck / f"round{br}_dispatch.pt", log.dir / "dispatch.pt")
    shutil.copy(b_cur, log.dir / "last_bid.zip")
    shutil.copy(d_cur, log.dir / "last_dispatch.pt")
    log(f"chosen round {br}, validation gain {top['gain']:+.1f} ± {top['gain_se']:.1f} $/day")
    fb, fd = B(log.dir / "bid.zip"), D(log.dir / "dispatch.pt")
    R_test = evaluate_pair(cfg, r_bid, r_dis, s.test_seeds, pool)
    T = evaluate_pair(cfg, fb, fd, s.test_seeds, pool)
    g, se = paired(T, R_test)
    res = dict(step=step, seed=seed, best_rule=best, bid=fb, dispatch=fd, co_train=co_train, message=message,
               dispatch_extra=dispatch_extra, start=dict(bid=str(bid_start), dispatch=str(dispatch_start)),
               rounds=rnd, chosen_round=br, stop_reason=reason, test=T, rule_pair_test=R_test, gain=g, gain_se=se,
               curve=curve)
    if message is not None:
        T0 = evaluate_pair(cfg, dict(fb, message="off"), fd, s.test_seeds, pool)
        res.update(test_message_off=T0, message_on_minus_off=paired(T, T0))
        log(f"message on minus off: {res['message_on_minus_off'][0]:+.1f} ± {res['message_on_minus_off'][1]:.1f} $/day (test episodes)")
    res["minutes"] = (time.time() - t0) / 60
    log(f"TEST {step}: {T['mean']:+.1f} $/day, gain over the rule pair {g:+.1f} ± {se:.1f} (paired, {len(s.test_seeds)} episodes; round {br})")
    return log.result(res)
