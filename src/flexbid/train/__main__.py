"""Command line for `flexbid.train.train`.

    python -m flexbid.train --kind hierarchical --fleet hub-6h --market lead --lead-min 60 \\
        --message fresh --schedule long --seed 0 --out runs/hub-6h-lead60/seed0
"""
import argparse

from flexbid.scenarios import FLEETS, MARKETS, make_config

from .pipeline import KINDS, train
from .schedule import SCHEDULES


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m flexbid.train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--fleet", default="hub-6h", choices=list(FLEETS))
    ap.add_argument("--market", default="base", choices=list(MARKETS))
    ap.add_argument("--lead-min", type=int, default=0, help="bid lead, minutes, a multiple of 15")
    ap.add_argument("--n-chargers", type=int, default=None)
    ap.add_argument("--message", default=None, choices=["fresh", "fixed", "computed"])
    ap.add_argument("--dispatch-sees", default=None, choices=["commitments"])
    ap.add_argument("--start", default=None, help="folder with bid.zip and dispatch.pt to continue from")
    ap.add_argument("--schedule", default="standard", choices=list(SCHEDULES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=None, help="environments per learner")
    ap.add_argument("--workers", type=int, default=None, help="processes for evaluation")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = make_config(a.fleet, a.market, market_lead_min=a.lead_min, n_chargers=a.n_chargers)
    changes = {k: v for k, v in (("n_envs", a.n_envs), ("workers", a.workers)) if v is not None}
    train(a.kind, cfg, a.out, seed=a.seed, schedule=a.schedule, message=a.message,
          dispatch_sees=a.dispatch_sees, start=a.start, **changes)


if __name__ == "__main__":
    main()
