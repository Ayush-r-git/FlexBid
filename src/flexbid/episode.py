"""Run one episode, or evaluate a policy pair on a list of days.

A policy pair is a bid policy and a dispatch policy. Each is a rule name from
`flexbid.policies.rules`, "mpc:<N>" for the MPC bid, a description of a trained policy (see
`flexbid.policies.learned`), or a callable:

    bid(sim)      -> (anchor, q_share, h_share, beta_mult)     once per market interval
    dispatch(sim) -> action, one number per charger in [-1, 1] once per dispatch step

Inside a callable, read the fleet through ``sim.cfleet`` (the controller's view) and the
market through ``sim.clearing`` and ``sim.market_obs()``. Never read ``sim.fleet``: it holds
the hidden truth.
"""
from __future__ import annotations

import numpy as np

from .sim.simulator import Simulator

#: money parts reported per episode
PARTS = ("revenue_usd", "imbalance_usd", "degradation_usd", "unmet_usd", "unmet_kwh")


def _bid_policy(bid):
    from .policies.learned import make_bid
    return make_bid(bid)


def _dispatch_policy(dispatch):
    from .policies.learned import make_dispatch
    return make_dispatch(dispatch)


def run_episode(cfg, seed: int, bid="fixed:8", dispatch="qp") -> dict:
    """Run the episode with random seed ``seed`` and return its money per day.

    An episode lasts ``cfg.episode_length`` market intervals (3 days for every fleet). The same
    seed always gives the same episode.

    | key | meaning | units |
    |---|---|---|
    | `seed` | the episode's seed | — |
    | `revenue` | payment for delivered flexibility | $/day |
    | `imbalance` | charge for missing commitments | $/day |
    | `wear` | battery wear | $/day |
    | `undelivered` | charge for energy departing vehicles did not receive | $/day |
    | `settlement` | settlement of the vehicles still plugged in at the end | $/day |
    | `total` | revenue - imbalance - wear - undelivered + settlement | $/day |
    | `undelivered_kwh` | energy departing vehicles did not receive | kWh/day |
    | `mean_price` | mean cleared price | price units |
    """
    up, lo = _bid_policy(bid), _dispatch_policy(dispatch)
    sim = Simulator(cfg)
    sim.reset(int(seed))
    parts = dict.fromkeys(PARTS, 0.0)
    while not sim.truncated:
        if sim.phase == "market":
            a, q, h, beta = up(sim)
            sim.market_step(float(a), q_frac=float(q), h_frac=float(h), beta_mult=float(beta))
        o = sim.dispatch_step(lo(sim))
        for k in PARTS:
            parts[k] += float(o.components[k])
    n_days = cfg.episode_length * cfg.market_interval_min / 1440.0     # episode length, days
    settlement = sim.settle()
    money = parts["revenue_usd"] - parts["imbalance_usd"] - parts["degradation_usd"] - parts["unmet_usd"]
    return dict(
        seed=int(seed),
        revenue=parts["revenue_usd"] / n_days,
        imbalance=parts["imbalance_usd"] / n_days,
        wear=parts["degradation_usd"] / n_days,
        undelivered=parts["unmet_usd"] / n_days,
        settlement=settlement / n_days,
        total=(money + settlement) / n_days,
        undelivered_kwh=parts["unmet_kwh"] / n_days,
        mean_price=float(np.mean(sim.cleared_prices)) if sim.cleared_prices else 0.0,
    )


def evaluate(cfg, seeds, bid="fixed:8", dispatch="qp") -> dict:
    """Run one episode per seed in ``seeds``. Returns the mean of every money part, in $/day,
    the standard error of the total over episodes, and each episode's result under ``per_episode``."""
    rows = [run_episode(cfg, s, bid, dispatch) for s in seeds]
    keys = [k for k in rows[0] if k != "seed"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    tot = np.array([r["total"] for r in rows])
    out["total_se"] = float(tot.std(ddof=1) / np.sqrt(len(tot))) if len(tot) > 1 else 0.0
    out["per_episode"] = rows
    return out
