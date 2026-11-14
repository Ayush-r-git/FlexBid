"""Evaluate a policy pair on a list of days, in parallel, and compare two pairs day by day.

A pair is a bid description and a dispatch description (see `flexbid.policies.learned`). Every
comparison in the trainer is paired: both pairs run on the same episode seeds, and the standard
error is that of the per-episode difference.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

PARTS = ("revenue", "imbalance", "wear", "undelivered", "settlement", "undelivered_kwh")


def _limit_threads():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _episode(job):
    """One episode in a worker: money per day, and the message of every interval."""
    _limit_threads()
    from flexbid.episode import run_episode
    from flexbid.policies.learned import make_bid, make_dispatch
    cfg, seed, bid, dispatch = job
    b, msgs = make_bid(bid), []

    def bid_and_record(sim):
        out = b(sim)
        msgs.append(float(sim.message))
        return out
    r = run_episode(cfg, seed, bid_and_record, make_dispatch(dispatch))
    r["message"] = msgs
    return r


def make_pool(workers):
    """A process pool for `evaluate_pair`, or None to run in this process."""
    return mp.get_context("spawn").Pool(int(workers)) if int(workers) > 1 else None


def evaluate_pair(cfg, bid, dispatch, seeds, pool=None):
    """Money of a policy pair on the episodes `seeds`.

    | key | meaning | units |
    |---|---|---|
    | mean, se | mean and standard error of the total over episodes | $/day |
    | per_episode | the total of every episode, by seed | $/day |
    | parts | mean of each money part (see `flexbid.run_episode`) | $/day, undelivered_kwh in kWh/day |
    | message | mean, standard deviation and share above 0 of the message over all intervals | score units, — |
    """
    jobs = [(cfg, int(s), bid, dispatch) for s in seeds]
    rows = pool.map(_episode, jobs) if pool is not None else [_episode(j) for j in jobs]
    tot = np.array([r["total"] for r in rows])
    th = np.concatenate([np.asarray(r["message"], float) for r in rows])
    return dict(bid=bid, dispatch=dispatch, seeds=[int(s) for s in seeds],
                mean=float(tot.mean()), se=float(tot.std(ddof=1) / np.sqrt(len(tot))) if len(tot) > 1 else 0.0,
                per_episode={str(r["seed"]): r["total"] for r in rows},
                parts={k: float(np.mean([r[k] for r in rows])) for k in PARTS},
                message=dict(mean=float(th.mean()), sd=float(th.std()), share_pos=float((th > 0).mean())))


def paired(a, b):
    """Mean and standard error of the per-day difference a − b over the days both ran, $/day."""
    d = np.array([a["per_episode"][k] - b["per_episode"][k] for k in b["per_episode"]])
    return float(d.mean()), (float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0)
