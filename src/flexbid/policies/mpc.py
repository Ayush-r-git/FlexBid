"""Model predictive control (MPC) for the bid, a baseline for the learned bid.

Each market interval the MPC scores the nine fixed bids of the rule menu (`rules.MENU`) on a
copy of the simulator and keeps the best. The copy knows only what a controller can know:

| process | the copy uses |
|---|---|
| requirement d_t | a forecast from the last requirement seen (`DemandForecast`) |
| rivals' anchors | their long-run mean |
| vehicles plugged in | each leaves at its stated departure |
| future arrivals | none |
| hidden battery state | the estimated temperature and polarization, with nominal parameters |

So the copy's battery, charger and market equations are exact, but its future is a forecast.

**Score of a bid.** With lookahead N, a bid is scored by its money over N intervals: the bid
itself, then N − 1 intervals in which the copy picks the best bid for one interval only. The
energy left at the end is valued at the water value p_wv:

    score = Σ (money over N intervals) + V(end) − V(now)

| symbol | meaning | units |
|---|---|---|
| V | `settlement.terminal_usd` on stated departures: stored energy valued at p_wv, energy owed charged at the undelivered-energy price | $ |
| p_wv | water value: the mean cleared price of the one-interval MPC on the fit episodes | $/kWh |

The copy always dispatches with the plain QP (`qp`), whatever dispatch actually runs. The
water value and the forecast are fitted once per configuration, on the fit episodes only.

**The demand forecast.** AR(1) around a time-of-day mean:

    d̂(t+j) = m(t+j) + φ^(j+1) (d(t−1) − m(t−1)),     j = 0, 1, ...

| symbol | meaning | units |
|---|---|---|
| d̂(t+j) | forecast of the requirement j intervals ahead | kW |
| m(s) | mean requirement in the 15-minute time-of-day slot of interval s | kW |
| φ | AR(1) coefficient of d − m, per interval | — |
| d(t−1) | the last requirement seen | kW |
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import tempfile

import numpy as np

from flexbid.sim.settlement import terminal_usd
from flexbid.sim.simulator import Simulator

from .departures import FIT_SEEDS
from .rules import MENU, QPController

PARTS = ("revenue_usd", "imbalance_usd", "degradation_usd", "unmet_usd", "economic_usd")


def _step(sim, dispatch, bid):
    """One market interval with the menu bid (anchor, q_frac, h_frac). Returns the money parts, $."""
    a, qf, hf = bid
    sim.market_step(a, q_frac=qf, h_frac=hf)
    out = dict.fromkeys(PARTS, 0.0)
    for _ in range(sim.cfg.K):
        o = sim.dispatch_step(dispatch(sim))
        for p in PARTS:
            out[p] += o.components[p]
        if o.truncated:
            break
    return out


def greedy_index(sim, dispatch, p_wv=None, believed=True):
    """The menu bid with the best money in this interval alone, tried on copies.

    With `p_wv` ($/kWh) the score adds the change in V, so the policy does not sell now the
    energy it must buy back later.
    """
    rs = []
    v0 = terminal_usd(sim, p_wv, believed) if p_wv is not None else 0.0
    for m in MENU:
        c = copy.deepcopy(sim)
        r = _step(c, dispatch, m)["economic_usd"]
        if p_wv is not None:
            r += terminal_usd(c, p_wv, believed) - v0
        rs.append(r)
    return int(np.argmax(rs))


def water_value(cfg, seeds=FIT_SEEDS):
    """p_wv: the mean cleared price of the plain one-interval greedy bid over the fit episodes, $/kWh."""
    pr = []
    for d in seeds:
        sim = Simulator(cfg, estimator=False)
        sim.reset(d)
        qp = QPController()
        for _ in range(cfg.episode_length):
            k = greedy_index(sim, qp)
            _step(sim, qp, MENU[k])
            pr.append(sim.clearing.price)
            if sim.truncated:
                break
    return float(np.mean(pr)) * cfg.price_unit


def _cache_dir():
    return pathlib.Path(os.environ.get("FLEXBID_CACHE", pathlib.Path.home() / ".cache" / "flexbid"))


def water_value_cached(cfg, seeds=FIT_SEEDS):
    """`water_value`, cached on disk per configuration and code version. It takes about a minute.

    The cache folder is `$FLEXBID_CACHE`, or `~/.cache/flexbid`.
    """
    here = pathlib.Path(__file__).resolve()
    src = sorted((here.parents[1] / "sim").glob("*.py")) + [here, here.parent / "rules.py"]
    code = hashlib.sha256(b"".join(q.read_bytes() for q in src)).hexdigest()[:8]
    d = _cache_dir()
    f = d / f"water_value_{cfg.physics_fingerprint()}_{code}_{seeds[0]}-{seeds[-1]}_{len(seeds)}.json"
    if f.exists():
        return float(json.loads(f.read_text())["water_value_usd_per_kwh"])
    p = water_value(cfg, seeds)
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as h:
        json.dump(dict(water_value_usd_per_kwh=p, fingerprint=cfg.physics_fingerprint(), code=code,
                       seeds=list(seeds)), h)
    os.replace(tmp, f)        # atomic, so parallel workers never read half a file
    return p


def _slot(sim, t):
    """15-minute time-of-day slot of interval t, 0 .. 95."""
    minute = sim.start_minute + t * sim.cfg.market_interval_min
    return int((minute % 1440) // 15)


class DemandForecast:
    """AR(1) around a time-of-day mean. See the module docstring."""

    def __init__(self, m, phi):
        self.m, self.phi = np.asarray(m, float), float(phi)

    @classmethod
    def fit(cls, cfg, seeds=FIT_SEEDS):
        rows = []
        for d in seeds:
            s = Simulator(cfg, estimator=False)
            s.reset(d)
            rows.append([(_slot(s, t), float(s.d_process[t])) for t in range(cfg.episode_length)])
        acc, cnt = np.zeros(96), np.zeros(96)
        for r in rows:
            for b, d in r:
                acc[b] += d
                cnt[b] += 1
        # slots no fit episode visits fall back to the overall mean
        m = np.where(cnt > 0, acc / np.maximum(cnt, 1), acc.sum() / max(cnt.sum(), 1))
        num = den = 0.0
        for r in rows:
            e = np.array([d - m[b] for b, d in r])
            num += float(e[1:] @ e[:-1])
            den += float(e[:-1] @ e[:-1])
        return cls(m, num / den if den > 0 else 0.0)

    def predict(self, sim):
        """Forecast for intervals t .. T−1 of this episode, kW. At t = 0 it is the mean alone."""
        t, T = sim.t, sim.cfg.episode_length
        j = np.arange(T - t)
        m_fut = np.array([self.m[_slot(sim, t + k)] for k in j])
        if t == 0:
            return m_fut
        d_last = float(sim.d_process[t - 1])
        return m_fut + self.phi ** (j + 1) * (d_last - self.m[_slot(sim, t - 1)])


def planning_copy(sim, forecast):
    """A copy of `sim` whose future is only what a controller can know. Call it before the bid."""
    c = copy.deepcopy(sim)
    t = c.t
    c.estimator = None                     # the estimator learns from the real meter only
    f = c.fleet                            # estimated temperature and polarization, nominal parameters
    f.temp_k = f.temp_hat_k.copy()
    f.r_mult = np.ones_like(f.r_mult)
    f.cool_mult = np.ones_like(f.cool_mult)
    f.v_pol = f.v_pol_hat.copy()
    f.pol_tau_mult = np.ones_like(f.pol_tau_mult)
    c.d_process = np.array(c.d_process, dtype=float)
    c.d_process[t:] = forecast.predict(sim)
    if c.competitors is not None:          # rivals at their long-run mean anchor
        mu = float(c.cfg.comp_exo_mean)
        pool = c.competitors
        pool.path = np.array(pool.path, dtype=float)
        pool.path[pool._t:] = mu
        pool.p0 = np.full_like(pool.p0, mu)
    conn = f.connected                     # vehicles leave at their stated departure
    f.departure_steps[conn] = f.est_departure_steps[conn]
    if c.schedule is not None:             # no future arrivals
        c.schedule.cursor = len(c.schedule.arrival_step)
    return c


def mpc_index(sim, dispatch, forecast, p_wv, lookahead):
    """Index into `MENU` of the bid with the best score over `lookahead` intervals."""
    base = planning_copy(sim, forecast)
    if lookahead == 1:
        return greedy_index(base, dispatch, p_wv=p_wv, believed=True)
    v0 = terminal_usd(base, p_wv, believed=True)
    best, best_v = 0, -np.inf
    for i, m in enumerate(MENU):
        c = copy.deepcopy(base)
        v = _step(c, dispatch, m)["economic_usd"]
        for _ in range(lookahead - 1):
            if c.truncated:
                break
            k = greedy_index(c, dispatch, p_wv=p_wv, believed=True)
            v += _step(c, dispatch, MENU[k])["economic_usd"]
        v += terminal_usd(c, p_wv, believed=True) - v0
        if v > best_v:
            best, best_v = i, v
    return best


_INPUTS: dict = {}


class MPCBid:
    """The MPC as a bid policy: `MPCBid(lookahead)(sim)` returns (anchor, q_frac, h_frac, 1).

    Lookaheads of 1 and 6 intervals are the usual choices. The first call fits the water value and the
    forecast on the fit episodes, about a minute; later calls reuse them.
    """

    def __init__(self, lookahead=6):
        if int(lookahead) < 1:
            raise ValueError("lookahead must be at least 1 interval")
        self.lookahead = int(lookahead)

    def __call__(self, sim):
        key = sim.cfg.physics_fingerprint()
        if key not in _INPUTS:
            _INPUTS[key] = (water_value_cached(sim.cfg), DemandForecast.fit(sim.cfg), QPController())
        p_wv, fc, qp = _INPUTS[key]
        a, q, h = MENU[mpc_index(sim, qp, fc, p_wv, self.lookahead)]
        return float(a), float(q), float(h), 1.0
