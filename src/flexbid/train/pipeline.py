"""`train()`: every policy kind, as one call that runs the steps it needs.

Each step starts from the output of the step before. A step whose `result.json` already exists
in `out` is not run again, so a stopped run continues where it stopped, and the kinds share
their common steps.

| step | folder in `out` | trains | needs |
|---|---|---|---|
| A. best rule | `best-rule.json` | nothing: every rule pair on the tuning episodes | — |
| B. one level | `learned-bid/`, `learned-dispatch/` | one level, the other the best rule's | A |
| C′. levels apart | `bid-and-dispatch/` | both, each against the best rule's other level | both B |
| C. hierarchical | `hierarchical/` | both, each against the current other level | both B |
| D. continuation | `hierarchical-<option>/` | both, from C, with a message or longer | C |

| kind | steps | the policy reported |
|---|---|---|
| "best-rule" | A | the best rule pair |
| "learned-bid", "learned-dispatch" | A, B | that step's |
| "bid-and-dispatch" | A, B, C′ | C′'s |
| "hierarchical" | A, B, C, and D when a message or commitment inputs are asked for, or when the schedule continues every pair (`long`) | the last step's |

"bid-and-dispatch" is the upper and lower level trained separately, and "hierarchical" the
two trained jointly. The MPC baseline needs no training: evaluate it with
`evaluate_pair(cfg, "mpc:6", best_dispatch_rule, seeds)`.

**Message options of "hierarchical"** (step D):

| option | what |
|---|---|
| message=None | no message |
| message="fresh" | the bid learns a message that acts in the interval it is chosen |
| message="fixed" | the bid learns a message sent with its bid, which acts when that bid is delivered |
| message="computed" | the "fresh" pair, with its message replaced by the computed message; its step size is tuned on validation episodes |
| dispatch_sees="commitments" | no message; the dispatch network reads the last requirement and the fixed commitments |
"""
from __future__ import annotations

import json
import pathlib

from .evaluate import evaluate_pair, make_pool, paired
from .schedule import get_schedule

KINDS = ("best-rule", "learned-bid", "learned-dispatch", "bid-and-dispatch", "hierarchical")
MESSAGES = (None, "fresh", "fixed", "computed")


def _done(folder):
    f = pathlib.Path(folder) / "result.json"
    return json.loads(f.read_text()) if f.exists() else None


def train(kind, cfg, out, *, seed=0, schedule="standard", message=None, dispatch_sees=None, start=None, **changes):
    """Train one policy kind and return its result dict.

    | argument | meaning |
    |---|---|
    | kind | one of `KINDS` |
    | cfg | the scenario, from `flexbid.make_config` |
    | out | folder for this run; one folder per scenario and seed |
    | seed | seed of the networks and of the training episodes |
    | schedule | "standard", "long", "smoke", or a `TrainSchedule` |
    | message, dispatch_sees | options of "hierarchical", see the module docstring |
    | start | a folder holding `bid.zip` and `dispatch.pt` to start step D from, instead of step C. Use it to continue a pair trained in another scenario, for example a pair trained without a bid lead |
    | changes | any `TrainSchedule` field, e.g. `n_envs=8` |
    """
    from .evaluate import _limit_threads
    _limit_threads()
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; use one of {KINDS}")
    if message not in MESSAGES:
        raise ValueError(f"unknown message {message!r}; use one of {MESSAGES}")
    if dispatch_sees not in (None, "commitments"):
        raise ValueError("dispatch_sees is None or 'commitments'")
    if (message or dispatch_sees or start) and kind != "hierarchical":
        raise ValueError("message, dispatch_sees and start are options of kind 'hierarchical'")
    if message and dispatch_sees:
        raise ValueError("choose a message or commitment inputs, not both")
    s = get_schedule(schedule, **changes)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenario.json").write_text(json.dumps(dict(fingerprint=cfg.physics_fingerprint(),
                                                       n_chargers=cfg.n_chargers,
                                                       market_lead_intervals=cfg.market_lead_intervals,
                                                       station_limit_method=cfg.station_limit_method,
                                                       plan_stay_fraction=cfg.plan_stay_fraction,
                                                       schedule=s.name), indent=1))
    pool = make_pool(s.workers)
    try:
        return _run(kind, cfg, out, seed, s, message, dispatch_sees, start, pool)
    finally:
        if pool is not None:
            pool.close()


def _run(kind, cfg, out, seed, s, message, dispatch_sees, start, pool):
    from .rounds import train_rounds
    from .rule_search import rule_search
    from .single import train_single

    # A. best rule pair
    f = out / "best-rule.json"
    if f.exists():
        A = json.loads(f.read_text())
    else:
        A = rule_search(cfg, s.tune_seeds, s.bid_rules, s.dispatch_rules, pool, s.test_seeds)
        f.write_text(json.dumps(A, indent=1))
    best = A["best"]
    print(f"best rule pair: {best['bid']} | {best['dispatch']}, {best['mean']:+.1f} $/day on the tuning episodes", flush=True)
    if kind == "best-rule":
        return A

    # B. one level learned
    def single(level):
        return _done(out / f"learned-{level}") or train_single(level, cfg, best, s, out / f"learned-{level}", seed, pool)

    if kind in ("learned-bid", "learned-dispatch"):
        return single(kind.split("-")[1])
    if start is None:
        single("bid")
        single("dispatch")

    # C' and C
    def pair(name, co_train):
        return _done(out / name) or train_rounds(
            cfg, best, out / "learned-bid" / "bid.zip", out / "learned-dispatch" / "dispatch.pt", s.rounds, s,
            out / name, step=name, seed=seed, co_train=co_train, pool=pool)

    if kind == "bid-and-dispatch":
        return pair("bid-and-dispatch", False)
    C = pair("hierarchical", True) if start is None else None
    trained = "fresh" if message == "computed" else message
    if not (trained or dispatch_sees or s.continue_without_message or start):
        return C

    # D. continuation
    src = pathlib.Path(start) if start is not None else out / "hierarchical"
    name = "hierarchical-" + (trained or ("commitments" if dispatch_sees else "long"))
    step = {"fresh": "continue-fresh", "fixed": "continue-fixed"}.get(trained, "continue-commitments" if dispatch_sees else "continue")
    D = _done(out / name) or train_rounds(cfg, best, src / "bid.zip", src / "dispatch.pt", s.continuation, s, out / name,
                                          step=step, seed=seed, co_train=True, message=trained,
                                          dispatch_extra=dispatch_sees, pool=pool)
    if message != "computed":
        return D
    return _done(out / "hierarchical-computed") or _computed(cfg, D, s, out / "hierarchical-computed", pool)


def tune_computed_eta(cfg, bid, dispatch, seeds, etas, pool=None):
    """The computed message's step size: the one with the best mean money on the episodes `seeds`, for a
    trained pair (bid and dispatch descriptions). Returns (eta, means by eta, money with the
    message off)."""
    means = {}
    for eta in etas:
        means[eta] = evaluate_pair(cfg, dict(bid, message="computed", eta=eta), dispatch, seeds, pool)["mean"]
    off = evaluate_pair(cfg, dict(bid, message="off"), dispatch, seeds, pool)["mean"]
    return max(means, key=means.get), means, off


def _computed(cfg, D, s, folder, pool):
    """The "fresh" pair of step D with the computed message in place of its own."""
    from .learners import RunLog
    from .schedule import assert_not_test
    assert_not_test(s.computed_eta_seeds, "the seeds that tune the computed message", s.test_seeds)
    log = RunLog(folder)
    if int(cfg.market_lead_intervals) < 2:
        log("note: under a bid lead below 2 intervals no later commitment is fixed, so the computed message is 0")
    eta, means, off = tune_computed_eta(cfg, D["bid"], D["dispatch"], s.computed_eta_seeds, s.computed_etas, pool)
    log(f"step size tuned on episodes {s.computed_eta_seeds[0]}-{s.computed_eta_seeds[-1]}: {eta:g} "
        f"({means[eta] - off:+.1f} $/day over the message switched off)")
    bid = dict(D["bid"], message="computed", eta=eta)
    T = evaluate_pair(cfg, bid, D["dispatch"], s.test_seeds, pool)
    g, se = paired(T, D["rule_pair_test"])
    gl, sel = paired(D["test"], T)
    log(f"TEST computed message: {T['mean']:+.1f} $/day, gain over the rule pair {g:+.1f} ± {se:.1f}; "
        f"learned minus computed message {gl:+.1f} ± {sel:.1f} $/day")
    return log.result(dict(step="hierarchical-computed", eta=eta, eta_tuning=dict(means={str(k): v for k, v in means.items()}, off=off),
                           bid=bid, dispatch=D["dispatch"], test=T, rule_pair_test=D["rule_pair_test"], gain=g, gain_se=se,
                           learned_minus_computed=(gl, sel)))
