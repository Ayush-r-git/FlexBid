"""The learned policies and the trainer. Skipped without the `train` extra.

These tests check that the code runs and that widening keeps a policy unchanged. They train
nothing useful: every budget is a few short episodes.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

import flexbid
from flexbid.train import train
from flexbid.train.envs import BidEnv, DispatchEnv

SMALL = dict(n_chargers=8, episode_length=12)


def test_envs_pass_gymnasium_check():
    from gymnasium.utils.env_checker import check_env
    cfg = flexbid.make_config("hub-6h", "lead", market_lead_min=30, **SMALL)
    check_env(BidEnv(cfg, "qp", first_seed=1, message="fixed"), skip_render_check=True)
    check_env(DispatchEnv(cfg, "fixed:8", first_seed=1, extra="commitments"), skip_render_check=True)


def test_widened_bid_acts_like_its_source(tmp_path):
    from flexbid.policies.features import N_BID_OBS
    from flexbid.policies.nets import widen_bid
    from flexbid.train.learners import new_bid_model
    cfg0 = flexbid.make_config("hub-6h", "lead", **SMALL)
    cfg4 = flexbid.make_config("hub-6h", "lead", market_lead_min=60, **SMALL)
    old = new_bid_model(BidEnv(cfg0, "qp", 1), cfg0, 1e-4, 0)
    new = widen_bid(old, BidEnv(cfg4, "qp", 1, message="fresh"), 1e-4, 0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        o = rng.normal(size=N_BID_OBS + 4).astype(np.float32)
        a_old = old.predict(o[:N_BID_OBS], deterministic=True)[0]
        a_new = new.predict(o, deterministic=True)[0]
        assert np.allclose(a_new[:4], a_old, atol=1e-6)
        assert a_new[4] == 0.0                      # the message starts at 0


def test_widened_dispatch_gives_the_same_scores():
    from flexbid.policies.nets import DeepSetNet, widen_dispatch
    net = DeepSetNet()
    wide = widen_dispatch(net.state_dict(), 4)
    cfg = flexbid.make_config("hub-6h", **SMALL)
    x = np.random.default_rng(1).normal(size=cfg.n_chargers * 14 + 9).astype(np.float32)
    x[cfg.n_chargers * 13: cfg.n_chargers * 14] = 1.0          # every charger occupied
    extra = np.array([0.3, -0.2, 0.5, 0.1], np.float32)
    assert np.allclose(net.mean_np(x), wide.mean_np(np.concatenate([x, extra])), atol=1e-6)


def test_every_kind_runs(tmp_path):
    cfg0 = flexbid.make_config("hub-6h", "lead", **SMALL)
    cfg2 = flexbid.make_config("hub-6h", "lead", market_lead_min=30, **SMALL)
    a, b = tmp_path / "lead0", tmp_path / "lead30"
    for kind in ("best-rule", "learned-bid", "learned-dispatch", "bid-and-dispatch", "hierarchical"):
        train(kind, cfg0, a, schedule="smoke")
    r = train("hierarchical", cfg0, a, schedule="smoke", message="fresh")
    assert r["curve"][0]["message"]["sd"] == 0.0               # round 0: the widened pair sends no message
    start = a / "hierarchical"
    for kw in (dict(message="fixed"), dict(message="computed"), dict(dispatch_sees="commitments")):
        r = train("hierarchical", cfg2, b, schedule="smoke", start=start, **kw)
        assert np.isfinite(r["test"]["mean"])
    starts = {f.name: __import__("json").loads((f / "result.json").read_text())["curve"][0]["mean"]
              for f in b.iterdir() if f.is_dir() and f.name != "hierarchical-computed"}
    assert len(set(round(v, 9) for v in starts.values())) == 1  # widening changes no starting policy


def test_learned_policies_run_in_run_episode(tmp_path):
    from flexbid.train.learners import new_bid_model, new_dispatch_model, save_bid, save_dispatch
    cfg = flexbid.make_config("depot-15", **SMALL)
    bid = save_bid(new_bid_model(BidEnv(cfg, "qp", 1), cfg, 1e-4, 0), tmp_path / "bid.zip")
    dis = save_dispatch(new_dispatch_model(DispatchEnv(cfg, "fixed:8", 1), cfg, 3e-5, 0), tmp_path / "dispatch.pt")
    r = flexbid.run_episode(cfg, 3, bid={"path": bid}, dispatch={"path": dis})
    assert np.isfinite(r["total"])
