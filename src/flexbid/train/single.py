"""One level learned, the other level the best rule's: `learned-bid` and `learned-dispatch`.

1. **Copy.** Record the best rule pair and fit the learned level to it (`copy_rule`).
2. **Check.** Run the copy, acting greedily, against the rule pair on the validation episodes. Stop
   if it is off by more than `schedule.copy_tol` of the rule pair's money.
3. **Improve.** PPO, with a checkpoint every `schedule.monitor_every` episodes. Each checkpoint
   is saved to its own file and evaluated, paired against the rule pair, on the validation episodes.
4. **Choose.** Keep the checkpoint with the best validation gain. The copy, at 0 episodes, is a
   candidate. Then run it once on the test episodes.

Writes into `out`: `bid.zip` or `dispatch.pt` (the chosen checkpoint), `last.*`, `checkpoints/`,
`monitor.jsonl`, `log.txt` and `result.json`.
"""
from __future__ import annotations

import shutil
import time

from .copy_rule import copy_into, record
from .evaluate import evaluate_pair, paired
from .learners import (RunLog, make_venv, new_bid_model, new_dispatch_model, save_bid, save_dispatch,
                       steps_per_episode)
from .schedule import assert_not_test, episodes_per_env, training_seed


def train_single(level, cfg, best, schedule, out, seed=0, pool=None):
    """Train one level. `level` is "bid" or "dispatch"; `best` is the rule search's best pair,
    {"bid": name, "dispatch": name}. Returns the result dict, also written to out/result.json."""
    if level not in ("bid", "dispatch"):
        raise ValueError(level)
    s = schedule
    step = f"learned-{level}"
    log = RunLog(out)
    r_bid, r_dis = best["bid"], best["dispatch"]
    ext = ".zip" if level == "bid" else ".pt"
    save = save_bid if level == "bid" else save_dispatch

    def pair(path):
        return ({"path": path}, r_dis) if level == "bid" else (r_bid, {"path": path})

    assert_not_test(s.validation_seeds, "the validation episodes", s.test_seeds)
    n_env = episodes_per_env(s.single_episodes, s.n_envs)
    first = [training_seed(seed, step, 0, 0, i) for i in range(s.n_envs)]
    partner = r_dis if level == "bid" else r_bid
    venv = make_venv(level, cfg, partner, first)
    model = (new_bid_model(venv, cfg, s.lr_bid, seed) if level == "bid"
             else new_dispatch_model(venv, cfg, s.lr_dispatch, seed))
    log(f"{step}: best rule pair ({r_bid}, {r_dis}); {s.single_episodes} episodes, about {n_env} per environment; "
        f"{s.n_envs} environments; step size {s.lr_bid if level == 'bid' else s.lr_dispatch:g}")

    # 1-2. copy the rule, and check the copy
    t0 = time.time()
    obs, tgt, ret = record(cfg, r_bid, r_dis, level, s.copy_episodes, training_seed(seed, "copy", level == "dispatch", 0, 0))
    clog = copy_into(model, level, obs, tgt, ret, grad_steps=s.copy_grad_steps)
    log(f"copy: {len(obs)} decisions from {s.copy_episodes} episodes; action error {clog[0]['action_mse']:.4g} -> "
        f"{clog[-1]['action_mse']:.4g}, value error {clog[0]['value_mse']:.4g} -> {clog[-1]['value_mse']:.4g}")
    R_val = evaluate_pair(cfg, r_bid, r_dis, s.validation_seeds, pool)
    ck = log.dir / "checkpoints"
    path = save(model, ck / f"ep0{ext}")
    C = evaluate_pair(cfg, *pair(path), s.validation_seeds, pool)
    g, se = paired(C, R_val)
    log(f"copy check, validation episodes: rule pair {R_val['mean']:+.1f}, copy {C['mean']:+.1f}, gap {g:+.1f} ± {se:.1f} $/day")
    curve = [dict(episodes=0, mean=C["mean"], se=C["se"], rule_pair=R_val["mean"], gain=g, gain_se=se)]
    log.point(curve[0], first=True)
    if abs(g) > s.copy_tol * max(abs(R_val["mean"]), 1.0):
        venv.close()
        raise RuntimeError(f"the copy is off by more than {s.copy_tol:.0%} of the rule pair's money; nothing trained")

    # 3. improve
    done = 0
    while done < s.single_episodes:
        chunk = min(s.monitor_every, s.single_episodes - done)
        model.learn(total_timesteps=chunk * steps_per_episode(level, cfg), reset_num_timesteps=False)
        done += chunk
        path = save(model, ck / f"ep{done}{ext}")
        M = evaluate_pair(cfg, *pair(path), s.validation_seeds, pool)
        g, se = paired(M, R_val)
        pt = dict(episodes=done, mean=M["mean"], se=M["se"], rule_pair=R_val["mean"], gain=g, gain_se=se,
                  minutes=(time.time() - t0) / 60)
        log.point(pt)
        curve.append(pt)
        log(f"{done:>6} episodes: gain over the rule pair {g:+.1f} ± {se:.1f} $/day (validation), {pt['minutes']:.0f} min")
    venv.close()

    # 4. choose on the validation episodes, then the test episodes once
    best_pt = max(curve, key=lambda q: q["gain"])
    final = log.dir / f"{level}{ext}"
    shutil.copy(ck / f"ep{best_pt['episodes']}{ext}", final)
    shutil.copy(ck / f"ep{s.single_episodes}{ext}", log.dir / f"last{ext}")
    log(f"chosen checkpoint: {best_pt['episodes']} episodes, validation gain {best_pt['gain']:+.1f} ± {best_pt['gain_se']:.1f} $/day")
    R_test = evaluate_pair(cfg, r_bid, r_dis, s.test_seeds, pool)
    T = evaluate_pair(cfg, *pair(str(final)), s.test_seeds, pool)
    g, se = paired(T, R_test)
    log(f"TEST {step}: {T['mean']:+.1f} $/day, gain over the rule pair {g:+.1f} ± {se:.1f} (paired, {len(s.test_seeds)} episodes)")
    return log.result(dict(step=step, seed=seed, best_rule=best, bid=T["bid"], dispatch=T["dispatch"],
                           chosen_episodes=best_pt["episodes"], test=T, rule_pair_test=R_test, gain=g, gain_se=se,
                           curve=curve, copy_log=clog, minutes=(time.time() - t0) / 60))
