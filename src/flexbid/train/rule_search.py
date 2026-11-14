"""The best rule pair: every bid rule with every dispatch rule, on the tuning episodes.

The pair with the highest mean money is the baseline every learned policy starts from and is
compared with.
"""
from __future__ import annotations

from flexbid.policies.rules import BID_RULES, BID_RULES_CAUTIOUS, DISPATCH_RULES

import numpy as np

from .evaluate import PARTS, _episode
from .schedule import TEST_SEEDS, assert_not_test


def default_bid_rules(cfg):
    """Every bid rule; under a bid lead also the cautious ones, which offer only part of the room."""
    return BID_RULES + (BID_RULES_CAUTIOUS if int(cfg.market_lead_intervals) > 0 else ())


def rule_search(cfg, seeds, bids=None, dispatches=None, pool=None, test_seeds=TEST_SEEDS):
    """Try every pair on the episodes `seeds`. Returns {"bid", "dispatch", "mean"} of the best pair under "best",
    and every pair's result under "pairs", keyed "bid|dispatch"."""
    assert_not_test(seeds, "the rule search seeds", test_seeds)
    bids = tuple(bids) if bids else default_bid_rules(cfg)
    dispatches = tuple(dispatches) if dispatches else DISPATCH_RULES
    jobs = [(cfg, int(s), b, d) for b in bids for d in dispatches for s in seeds]
    rows = pool.map(_episode, jobs) if pool is not None else [_episode(j) for j in jobs]
    pairs = {}
    for (_, _, b, d), r in zip(jobs, rows):
        pairs.setdefault(f"{b}|{d}", []).append(r)
    for k, rs in pairs.items():
        tot = np.array([r["total"] for r in rs])
        pairs[k] = dict(mean=float(tot.mean()),
                        se=float(tot.std(ddof=1) / np.sqrt(len(tot))) if len(tot) > 1 else 0.0,
                        per_episode={str(r["seed"]): r["total"] for r in rs},
                        parts={q: float(np.mean([r[q] for r in rs])) for q in PARTS})
    best = max(pairs, key=lambda k: pairs[k]["mean"])
    b, d = best.split("|")
    return dict(best=dict(bid=b, dispatch=d, mean=pairs[best]["mean"]), seeds=[int(x) for x in seeds], pairs=pairs)
