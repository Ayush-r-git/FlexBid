"""The rule-based policies of both levels.

Bid rules. A bid is four numbers (anchor, q_share, h_share, beta_mult); see
`Simulator.market_step`. The menu holds nine fixed bids:

| index | anchor | share of room to reduce charging | share of room to increase charging |
|---|---|---|---|
| 0 | +1 | 0.02 | 1 |
| 1 | -1 | 1 | 0.02 |
| 2-8 | -1, -0.6, -0.3, 0, 0.3, 0.6, 1 | 1 | 1 |

Dispatch rules. Every dispatch rule solves the dispatch QP (`physics.qp_allocation_power`),
some with extra per-vehicle credits computed from what the controller observes:

| name | what it adds to the plain QP |
|---|---|
| `qp` | nothing: the least-wear split |
| `qp-rate` | wear priced at the reference temperature and each vehicle's full-power rate |
| `qp-band:<lambda>[:<hours>[:<side>]]` | a credit for the flexibility a kWh in this pack adds over the next hours |
| `qp-depth:<mu>` | a credit for charging vehicles pulled below their session peak |
| `qp-mono:<mu>` | a credit for keeping each vehicle moving in its net direction so far |
| `qp-few` | the QP, then its discharge moved onto as few vehicles as possible |
| `baseline` | no QP: every vehicle follows its baseline plan |
"""
from __future__ import annotations

import numpy as np

GRID = (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)
ABSORB = (1.0, 0.02, 1.0)   # anchor, share of room to reduce offered, share of room to increase
SHED = (-1.0, 1.0, 0.02)
MENU = [ABSORB, SHED] + [(a, 1.0, 1.0) for a in GRID]


def band_of(sim):
    """The fleet's deliverable room over one market interval, (Q, H) in kW: the most it can
    reduce charging and the most it can increase it."""
    from flexbid.sim.physics import baseline_charge, feasible_flex_range
    cb = baseline_charge(sim.cfleet, sim.cfg)
    lo, hi = feasible_flex_range(sim.cfleet, cb, sim.cfg)
    return hi, -lo


BID_RULES = ("balanced",) + tuple(f"fixed:{i}" for i in range(9))
#: cautious bids that offer only a share of the room in both directions; for use under a bid lead
BID_RULES_CAUTIOUS = tuple(f"share:{i}:{sh}" for sh in ("0.25", "0.5") for i in range(7))
DISPATCH_RULES = ("qp", "qp-rate", "qp-band:1:0.25", "qp-depth:3e-3", "qp-mono:1e-3", "qp-few")


def bid_rule(name, sim):
    """(anchor, q_share, h_share, beta_mult) of a bid rule. No forecast, no model.

    | name | the bid |
    |---|---|
    | `fixed:<i>` | menu entry i, see the module table |
    | `balanced` | anchor 0, offering the same kW in both directions: the smaller of the two rooms |
    | `share:<i>:<s>` | anchor GRID[i], offering the share s of the room in both directions |
    | `none` | offer nothing; there is then no commitment, which gives a no-market reference |
    """
    if name == "none":
        return 0.0, 0.0, 0.0, 1.0
    if name == "balanced":
        Q, H = band_of(sim)
        Q, H = max(Q, 1e-9), max(H, 1e-9)
        r = min(Q, H)
        return 0.0, r / Q, r / H, 1.0
    if name.startswith("fixed:"):
        a, q, h = MENU[int(name.split(":")[1])]
        return float(a), float(q), float(h), 1.0
    if name.startswith("share:"):
        _, i, sh = name.split(":")
        return float(GRID[int(i)]), float(sh), float(sh), 1.0
    raise ValueError(f"unknown bid rule {name!r}")


def _commitment(sim):
    return sim.clearing.commitment_kw if sim.clearing is not None else 0.0


class QPController:
    """`qp`: meet the commitment at the least battery wear."""
    name = "qp"

    def __call__(self, sim):
        from flexbid.sim.physics import baseline_charge, qp_allocation_power, power_to_action
        cfg = sim.cfg
        c_base = baseline_charge(sim.cfleet, cfg)
        return power_to_action(qp_allocation_power(sim.cfleet, c_base, _commitment(sim), cfg), c_base, cfg)


class ZeroController:
    """`baseline`: every vehicle follows its baseline plan (deviation 0)."""
    name = "baseline"

    def __call__(self, sim):
        return np.zeros(sim.cfleet.n, dtype=np.float64)


class RateAwareQP:
    """`qp-rate`: each vehicle's wear coefficients are scaled by the wear factor g(T_ref, C_k), with
    T_ref = `cfg.wear_t_ref_c` and C_k the vehicle's full-power rate (rating over pack size, 1/h).
    Fast-charging small packs then cost more to move."""
    name = "qp-rate"

    def __call__(self, sim):
        from flexbid.sim.physics import baseline_charge, qp_allocation_power, power_to_action, wear_factor_g
        cfg, f = sim.cfg, sim.cfleet
        cb = baseline_charge(f, cfg)
        t_ref = np.full(f.n, 273.15 + cfg.wear_t_ref_c)
        c_ref = f.max_charge_kw / np.maximum(f.battery_kwh, 1e-9)
        scale = wear_factor_g(cfg, t_ref, c_ref)
        return power_to_action(qp_allocation_power(f, cb, _commitment(sim), cfg, wear_scale=scale), cb, cfg)


class BandTrackingQP:
    """`qp-band:<lambda>[:<hours>[:<side>]]`: a credit for the flexibility a vehicle can offer later.

    For each connected vehicle, the derivative of its flexibility width over the next `hours`
    with respect to its state of charge, priced at lambda $ per kW and turned into a credit
    on its power:

        credit_k = lambda * (d width_k / d soc) * eta_c * dt / E_k          $/kW per step

    | symbol | meaning | units |
    |---|---|---|
    | width_k | P_max - P_min of `per_car_power_limits` over `hours` (`side` "sum"), or only the reduce side ("q") or the increase side ("h") | kW |
    | E_k | pack size | kWh |
    | eta_c, dt | charge efficiency, dispatch step | —, h |

    The derivative is taken numerically on the controller's view, so no hidden state is read.
    """

    def __init__(self, lam, hours=2.0, side="sum"):
        self.lam = float(lam); self.hours = float(hours); self.side = side
        self.name = f"qp-band:{lam:g}:{hours:g}:{side}"

    def marginal_band(self, f, cfg):
        import copy as _copy
        from flexbid.sim.physics import per_car_power_limits
        d = 0.02

        def width(soc):
            g = _copy.copy(f); g.soc = soc
            lo, hi = per_car_power_limits(g, cfg, self.hours)
            return {"sum": hi - lo, "q": -lo, "h": hi}[self.side]
        s = f.soc
        w_up = width(np.clip(s + d, cfg.soc_min, cfg.soc_max))
        w_dn = width(np.clip(s - d, cfg.soc_min, cfg.soc_max))
        ds = np.clip(s + d, cfg.soc_min, cfg.soc_max) - np.clip(s - d, cfg.soc_min, cfg.soc_max)
        m = np.where(ds > 1e-9, (w_up - w_dn) / np.maximum(ds, 1e-9), 0.0)   # kW of width per unit soc
        return np.where(f.connected, m, 0.0)

    def __call__(self, sim):
        from flexbid.sim.physics import baseline_charge, qp_allocation_power, power_to_action
        cfg, f = sim.cfg, sim.cfleet
        cb = baseline_charge(f, cfg)
        m = self.marginal_band(f, cfg)
        credit = self.lam * m * cfg.charge_efficiency * cfg.dt_hours / np.maximum(f.battery_kwh, 1e-9)
        return power_to_action(qp_allocation_power(f, cb, _commitment(sim), cfg, credit_extra=credit), cb, cfg)


class MonotoneQP:
    """`qp-mono:<mu>`: a credit mu * sign(net pack energy since arrival), $/kW per step, so vehicles
    that have net charged keep charging and vehicles that have net discharged keep discharging.
    Fewer reversals mean fewer closed wear cycles. A dead band of 0.5 kWh applies."""

    def __init__(self, mu):
        self.mu = float(mu); self.name = f"qp-mono:{mu:g}"

    def __call__(self, sim):
        from flexbid.sim.physics import baseline_charge, qp_allocation_power, power_to_action
        cfg, f = sim.cfg, sim.cfleet
        cb = baseline_charge(f, cfg)
        net = np.where(f.connected, f.pack_hat_kwh, 0.0)
        credit = self.mu * np.sign(np.where(np.abs(net) > 0.5, net, 0.0))
        return power_to_action(qp_allocation_power(f, cb, _commitment(sim), cfg, credit_extra=credit), cb, cfg)


class DepthAwareQP:
    """`qp-depth:<mu>`: a credit mu * d_k, $/kW per step, where d_k is how far vehicle k is below its
    session peak, as a share of its pack (from the meter). It favours charging vehicles that are
    in a dip and disfavours discharging them further."""

    def __init__(self, mu):
        self.mu = float(mu); self.name = f"qp-depth:{mu:g}"

    def __call__(self, sim):
        from flexbid.sim.physics import baseline_charge, qp_allocation_power, power_to_action
        cfg, f = sim.cfg, sim.cfleet
        cb = baseline_charge(f, cfg)
        d = np.where(f.connected, (f.pack_peak_kwh - f.pack_hat_kwh) / np.maximum(f.battery_kwh, 1e-9), 0.0)
        credit = self.mu * np.maximum(d, 0.0)
        return power_to_action(qp_allocation_power(f, cb, _commitment(sim), cfg, credit_extra=credit), cb, cfg)


class ConsolidatedQP:
    """`qp-few`: the plain QP, then the discharge it asks for is moved onto as few vehicles as
    possible: idle vehicles with the most discharge room first, each filled to its limit before the
    next. Charging is left as the QP set it and the total is unchanged. If the idle vehicles lack
    the room, the QP's own split is kept."""
    name = "qp-few"

    def __call__(self, sim):
        from flexbid.sim.physics import baseline_charge, qp_allocation_power, power_to_action, per_car_power_limits
        cfg, f = sim.cfg, sim.cfleet
        cb = baseline_charge(f, cfg)
        q = _commitment(sim)
        P = qp_allocation_power(f, cb, q, cfg)
        D = float(np.sum(np.maximum(-P, 0.0)))
        if D > 1e-9:
            lo, _ = per_car_power_limits(f, cfg, cfg.dt_hours)
            P = np.maximum(P, 0.0)
            room = np.where(f.connected & (P <= 1e-9), np.maximum(-lo, 0.0), 0.0)
            for k in np.lexsort((-room,)):
                if D <= 1e-9 or room[k] <= 1e-9:
                    break
                take = min(room[k], D); P[k] = -take; D -= take
            if D > 1e-9:
                P = qp_allocation_power(f, cb, q, cfg)
        return power_to_action(P, cb, cfg)


def dispatch_rule(name):
    """The dispatch rule called `name` (see the module table)."""
    if name == "qp":
        return QPController()
    if name == "qp-rate":
        return RateAwareQP()
    if name == "qp-few":
        return ConsolidatedQP()
    if name == "baseline":
        return ZeroController()
    if name.startswith("qp-depth:"):
        return DepthAwareQP(float(name.split(":")[1]))
    if name.startswith("qp-mono:"):
        return MonotoneQP(float(name.split(":")[1]))
    if name.startswith("qp-band:"):
        parts = name.split(":")
        return BandTrackingQP(float(parts[1]), float(parts[2]) if len(parts) > 2 else 2.0,
                              parts[3] if len(parts) > 3 else "sum")
    raise ValueError(f"unknown dispatch rule {name!r}")
