"""Settlement of the vehicles still plugged in when an episode ends.

    V_end = - c_u * sum_k max(0, e_k - e_max_k) - p_ref * sum_k (e_k - l_k) / eta_c

| symbol | meaning | units |
|---|---|---|
| e_k | energy vehicle k is still owed | kWh |
| e_max_k | most energy vehicle k can still take before it leaves: true departure for the settlement, stated departure for a controller's own estimate | kWh |
| c_u | `cfg.undelivered_energy_price` | $/kWh |
| l_k | energy vehicle k's baseline plan still has to deliver | kWh |
| p_ref | a reference price for energy delivered ahead of or behind the plan | $/kWh |
| eta_c | charge efficiency | — |

Energy that can no longer reach a vehicle before it leaves is charged as undelivered. Energy
delivered ahead of the plan is credited, and energy behind it is charged, at p_ref.
"""
from __future__ import annotations

import numpy as np


def terminal_usd(sim, p_ref_usd: float, believed: bool = False) -> float:
    """V_end, $. ``believed=False``: the settlement itself, on the true departures.
    ``believed=True``: a controller's estimate, on its view and the stated departures."""
    f, cfg = (sim.cfleet if believed else sim.fleet), sim.cfg
    c = f.connected
    if not c.any():
        return 0.0
    from .fleet import unmet_kwh_per_car
    need = unmet_kwh_per_car(f, cfg)
    dep = f.est_departure_steps if believed else f.departure_steps
    t_h = np.maximum(dep, 0) * cfg.dt_hours
    e_max = f.max_charge_kw * cfg.charge_efficiency * t_h
    v = -cfg.undelivered_energy_price * float(np.sum(np.maximum(need - e_max, 0.0)[c]))
    v -= p_ref_usd * float(np.sum((need - f.plan_left_kwh)[c])) / cfg.charge_efficiency
    return v
