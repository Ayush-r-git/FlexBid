"""Run a rule-based policy pair and print its money.

    python examples/run_rule_pair.py

The bid rule `fixed:8` offers all of the fleet's room at the highest anchor. The dispatch rule
`qp-depth:3e-3` solves the dispatch QP with a credit for charging vehicles pulled below their
session peak. The pair runs on the 6-hour hub, first without a bid lead and then with a
60-minute bid lead. Takes about half a minute.
"""
import flexbid

SEEDS = range(200, 205)        # five episodes of 3 days each
PARTS = ("revenue", "imbalance", "wear", "undelivered", "settlement", "total")


def main():
    scenarios = {
        "no bid lead": flexbid.make_config("hub-6h", market="base"),
        "60-minute bid lead": flexbid.make_config("hub-6h", market="lead", market_lead_min=60),
    }
    print(f"{'scenario':20s}" + "".join(f"{p:>13s}" for p in PARTS) + "   (all in $/day)")
    for name, cfg in scenarios.items():
        r = flexbid.evaluate(cfg, SEEDS, bid="fixed:8", dispatch="qp-depth:3e-3")
        print(f"{name:20s}" + "".join(f"{r[p]:13.1f}" for p in PARTS)
              + f"   ± {r['total_se']:.1f} standard error of the total")


if __name__ == "__main__":
    main()
