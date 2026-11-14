"""The package reproduces the money of every reference episode in reference_totals.json.

The totals were computed by the original research code. On the same platform the match is
exact up to the order of floating-point additions. A different CPU or NumPy version can move
the last digits, so the tolerance is 1e-6 $/day.
"""
import json
import pathlib

import pytest

import flexbid

REF = json.loads((pathlib.Path(__file__).with_name("reference_totals.json")).read_text())["episodes"]


@pytest.mark.parametrize("e", REF, ids=lambda e: f"{e['fleet']}-{e['market']}-{e['market_lead_min']}-{e['bid']}-{e['dispatch']}")
def test_reference_total(e):
    cfg = flexbid.make_config(e["fleet"], e["market"], market_lead_min=e["market_lead_min"])
    r = flexbid.run_episode(cfg, e["seed"], e["bid"], e["dispatch"])
    assert r["total"] == pytest.approx(e["total_usd_per_day"], abs=1e-6)
