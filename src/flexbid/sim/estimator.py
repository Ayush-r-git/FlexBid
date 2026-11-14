"""The controller's estimate of each vehicle's hidden cell temperature and polarization.

A DC charger does not see a vehicle's cell temperature. It sees the charge and discharge limits
the vehicle reports, its state of charge and pack size, and the metered power. This estimator
works from those alone, assuming nominal resistance and cooling.

1. It runs the nominal thermal and polarization models forward on the metered power.
2. Where the reported charge limit is set by the voltage law (below the rating and the
   plating limit), it inverts that law for the temperature:

       P_c = E (V_max - OCV(z) - V_pol) / (R(T) Q_cell)
       T   = 1 / (1/298.15 + ln(R / r25) * Rg / Ea)

| symbol | meaning | units |
|---|---|---|
| P_c | the charge limit the vehicle reports | kW |
| E | pack size | kWh |
| z | state of charge | — |
| OCV, V_pol | open-circuit voltage, and the estimated polarization voltage | V |
| R(T), r25 | cell resistance at T, and at 25 °C (`cfg.r25_mohm`) | ohm |
| Q_cell | cell capacity, `cfg.cell_capacity_ah` | Ah |
| Ea, Rg | activation energy of the resistance (`cfg.r_ea_kj`), gas constant | kJ/mol, J/(mol K) |

A vehicle with a higher resistance than nominal therefore reads as hotter than it is; that is
the estimator's model error. The estimates are written to `fleet.temp_hat_k` and
`fleet.v_pol_hat`. The estimator never reads a hidden field.
"""
from __future__ import annotations

import numpy as np


class TemperatureEstimator:
    """See the module docstring. The simulator calls ``admit`` after new arrivals and
    ``observe`` after every dispatch step."""

    def __init__(self, cfg):
        self.cfg = cfg

    def admit(self, f, cfg):
        """New vehicles start at the mean arrival temperature."""
        new = f.connected & (f.elapsed_steps == 0)
        if new.any():
            f.temp_hat_k[new] = 273.15 + cfg.t_arrival_mean_c

    def observe(self, f, result, cfg):
        from .physics import ocv_cell_v, cell_resistance_ohm, plating_c_rate, GAS_R_J_PER_MOL_K
        c = f.connected
        # 1. the nominal thermal model, run forward on the metered power
        pack_kw = cfg.charge_efficiency * result.charge_kw + result.discharge_kw / cfg.discharge_efficiency
        i_cell = pack_kw / np.maximum(f.battery_kwh, 1e-9) * cfg.cell_capacity_ah
        q_w = i_cell ** 2 * cell_resistance_ohm(cfg, f.temp_hat_k, 1.0)
        h = (cfg.cell_capacity_ah ** 2) * (cfg.r25_mohm * 1e-3) / cfg.th_rise_1c_k
        tau_s = cfg.th_tau_min * 60.0
        t_amb = 273.15 + cfg.t_amb_c
        t_ss = t_amb + q_w / h
        pred = t_ss + (f.temp_hat_k - t_ss) * np.exp(-cfg.dt_hours * 3600.0 / tau_s)
        # 1b. the nominal polarization model, run forward on the metered current
        pk = cfg.charge_efficiency * result.charge_kw - result.discharge_kw / cfg.discharge_efficiency
        ic = pk / np.maximum(f.battery_kwh, 1e-9) * cfg.cell_capacity_ah
        v_ss = ic * cfg.pol_r_mohm * 1e-3
        a = np.exp(-cfg.dt_hours * 3600.0 / (cfg.pol_tau_min * 60.0))
        f.v_pol_hat = np.where(c, v_ss + (f.v_pol_hat - v_ss) * a, f.v_pol_hat)
        vph = f.v_pol_hat
        # 2. where the reported charge limit is set by the voltage law, invert it
        ocv = ocv_cell_v(np.clip(f.soc, 0.0, 1.0)) + vph
        rating = f.max_charge_kw
        plating_kw = plating_c_rate(cfg, pred) * f.battery_kwh
        binds = (c & (f.limit_c_kw > 0.05) & (f.limit_c_kw < 0.98 * np.minimum(rating, plating_kw))
                 & (ocv < cfg.cell_v_max - 1e-3))
        r_need = f.battery_kwh * (cfg.cell_v_max - ocv) / (cfg.cell_capacity_ah * np.maximum(f.limit_c_kw, 1e-6))
        inv = 1.0 / (1.0 / 298.15 + np.log(np.maximum(r_need, 1e-6) / (cfg.r25_mohm * 1e-3))
                     * GAS_R_J_PER_MOL_K / (cfg.r_ea_kj * 1e3))
        inv = np.clip(inv, 273.15 - 20.0, 273.15 + 80.0)
        est = np.where(binds, inv, pred)
        f.temp_hat_k = np.where(c, est, f.temp_hat_k)
