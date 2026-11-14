"""A short training run of the hierarchical policy with a message, from start to finish.

    python examples/train_short.py [--out runs/example]

It runs every training step on the 15-charger depot with a 30-minute bid lead:

1. the best rule pair on the tuning episodes;
2. the learned bid and the learned dispatch, each starting as a copy of the rule pair;
3. the two levels trained together in rounds;
4. the pair continued with a message from the bid to the dispatch.

It uses the `smoke` schedule, a few episodes per step, so it finishes in a few minutes. Its
policies are not trained enough to mean anything. For a real run use `schedule="standard"` or
`schedule="long"`, which take hours to days on a multi-core machine.

Needs the training packages: pip install ".[train]".
"""
import argparse

import flexbid
from flexbid.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/example", help="output folder")
    a = ap.parse_args()

    cfg = flexbid.make_config("depot-15", market="lead", market_lead_min=30)
    result = train("hierarchical", cfg, a.out, seed=0, schedule="smoke", message="fresh")

    print(f"\ntest money of the trained pair: {result['test']['mean']:+.1f} $/day, "
          f"{result['gain']:+.1f} ± {result['gain_se']:.1f} $/day against the best rule pair")
    print(f"trained policies: {result['bid']['path']} and {result['dispatch']['path']}")

    # a trained pair runs like any other policy
    r = flexbid.run_episode(cfg, seed=200, bid=result["bid"], dispatch=result["dispatch"])
    print(f"one episode with the trained pair, seed 200: {r['total']:+.1f} $/day")


if __name__ == "__main__":      # needed: training starts worker processes
    main()
