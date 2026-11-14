"""Behaviour of the simulator and the scenario API."""
import numpy as np
import pytest

import flexbid
from flexbid.policies.rules import bid_rule, dispatch_rule


def _short(**kw):
    return flexbid.make_config("hub-6h", episode_length=24, **kw)


def test_scenarios_arrival_rates():
    """75 % occupancy: arrivals per day = 0.75 x chargers x 24 / mean stay."""
    expected = {"hub-6h": 150.0, "hub-12h": 75.0, "hub-mixed": 100.0, "depot-15": 45.0}
    for fleet, rate in expected.items():
        assert flexbid.make_config(fleet).arr_rate_per_day == pytest.approx(rate)


def test_market_presets():
    base = flexbid.make_config("hub-6h")
    lead = flexbid.make_config("hub-6h", "lead", market_lead_min=60)
    assert (base.station_limit_method, base.plan_stay_fraction, base.market_lead_intervals) == ("projection", 1.0, 0)
    assert (lead.station_limit_method, lead.plan_stay_fraction, lead.market_lead_intervals) == ("exact", 0.8, 4)


def test_bad_arguments():
    with pytest.raises(ValueError):
        flexbid.make_config("hub-6h", market_lead_min=20)
    with pytest.raises(KeyError):
        flexbid.make_config("no-such-fleet")
    with pytest.raises(ValueError):
        flexbid.make_config("hub-6h", plan_stay_fraction=1.5)


def test_fleet_size_scales_arrivals():
    cfg = flexbid.make_config("hub-12h", n_chargers=100)
    assert cfg.n_chargers == 100 and cfg.arr_rate_per_day == pytest.approx(150.0)


def test_same_seed_same_episode():
    cfg = _short()
    a = flexbid.run_episode(cfg, 5, "fixed:8", "qp")
    b = flexbid.run_episode(cfg, 5, "fixed:8", "qp")
    assert a == b


def test_money_adds_up():
    r = flexbid.run_episode(_short(), 3, "fixed:4", "qp")
    parts = r["revenue"] - r["imbalance"] - r["wear"] - r["undelivered"] + r["settlement"]
    assert r["total"] == pytest.approx(parts, abs=1e-9)


def test_call_order_is_enforced():
    sim = flexbid.Simulator(_short())
    sim.reset(0)
    with pytest.raises(RuntimeError):
        sim.dispatch_step(np.zeros(sim.cfg.n_chargers))
    sim.market_step(0.0)
    with pytest.raises(RuntimeError):
        sim.market_step(0.0)


@pytest.mark.parametrize("lead", [1, 2, 4])
def test_market_lead(lead):
    """Under a lead of L intervals the first L intervals carry no commitment, and the bid
    placed at interval t is delivered at t + L."""
    cfg = _short(market_lead_intervals=lead)
    sim = flexbid.Simulator(cfg)
    sim.reset(7)
    rule = dispatch_rule("qp")
    for t in range(cfg.episode_length):
        cl = sim.market_step(*bid_rule("fixed:8", sim)[:1], q_frac=1.0, h_frac=1.0)
        if t < lead:
            assert cl.price == 0.0 and cl.commitment_kw == 0.0
        if t + lead < cfg.episode_length:
            assert (t + lead) in sim.pending
        for _ in range(cfg.K):
            sim.dispatch_step(rule(sim))
    assert len(sim.cleared_prices) == cfg.episode_length - lead


def test_controller_view_hides_the_truth():
    sim = flexbid.Simulator(_short())
    sim.reset(1)
    view = sim.cfleet
    for hidden in ("temp_k", "r_mult", "cool_mult", "v_pol", "pol_tau_mult", "true_type"):
        assert getattr(view, hidden) is None
    assert np.all(view.departure_steps[view.connected] == 1)
