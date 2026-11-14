"""Battery, charger and station physics, the baseline plan, feasibility, and battery wear.

Every function works on the fleet arrays with NumPy. Functions that a controller uses read
the controller's view of the fleet (``build_view``); the true fleet is used only by the
physics itself and by settlement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SimConfig
from .fleet import FleetState

_DATA_DIR = Path(__file__).resolve().parent / "data"
_OCV_CACHE: dict = {}


# Source of the open-circuit voltage curve in data/ocv_soc.npz:
#
#   EV-EcoSim battery data, NMC cell W8, file OCV/SOC_corr_W8_1, downsampled from
#   36,848 measured points to 201 points.
#   E. Balogun, E. Buechler, S. Bhela, S. Onori and R. Rajagopal, "EV-EcoSim: A grid-aware
#   co-simulation platform for the design and optimization of electric vehicle charging
#   infrastructure", IEEE Transactions on Smart Grid, 2023.
#   Code and data: https://github.com/ebalogun01/EV-EcoSim
#
# The curve is used in three places: the cell voltage limits on charge and discharge power
# (`v2_limits_kw`), the wear model (`degradation_cost_usd`, `marginal_cycle_cost_usd_per_kwh`)
# and the wear coefficients of the dispatch QP (`wear_quadratic_coeffs`).


def _ocv_table() -> tuple[np.ndarray, np.ndarray]:
    """Measured open-circuit voltage curve of an NMC cell, 201 points (data/ocv_soc.npz).

    Source: EV-EcoSim battery data, cell W8. See the comment above this function.
    """
    if "table" not in _OCV_CACHE:
        raw = np.load(_DATA_DIR / "ocv_soc.npz", allow_pickle=False)
        _OCV_CACHE["table"] = (raw["soc"].copy(), raw["ocv_v"].copy())
    return _OCV_CACHE["table"]


def ocv_cell_v(soc: np.ndarray) -> np.ndarray:
    """Cell open-circuit voltage, V, at the given state of charge."""
    grid, volts = _ocv_table()
    return np.interp(soc, grid, volts)


# Cycle-fade coefficients of the semi-empirical NMC aging model of Schmalstieg et al. (2014):
#
#     beta(v_rms, dDOD) = BETA_QUAD * (v_rms - v_offset)^2 + BETA_CONST + BETA_DDOD * dDOD
#
# | symbol | meaning | units |
# |---|---|---|
# | v_rms | root-mean-square cell voltage over the step | V |
# | v_offset | `cfg.schmalstieg_v_offset` | V |
# | dDOD | half the depth of discharge of the step | — |
SCHMALSTIEG_BETA_QUAD = 7.348e-3      # 1/V^2
SCHMALSTIEG_BETA_CONST = 7.600e-4     # dimensionless
SCHMALSTIEG_BETA_DDOD = 4.081e-3      # per unit half depth of discharge


def degradation_cost_usd(soc_before: np.ndarray, soc_after: np.ndarray, charge_kw: np.ndarray,
                         discharge_kw: np.ndarray, fleet: FleetState,
                         cfg: SimConfig) -> tuple[float, float, float]:
    """Battery wear of one dispatch step, $. Returns (total, cycle, calendar).

    The aging model of Schmalstieg et al. (2014) with the measured voltage curve. Both fade
    terms are concave in accumulated stress, so each step is priced at the slope around a
    reference point: `cfg.q_ref_ah` accumulated throughput and `cfg.t_ref_days` age.

        cycle fade    = beta(v_rms, dDOD) / (2 sqrt(q_ref)) * dQ * g(T, C)
        calendar fade = 0.75 * alpha(v_mean, T) / t_ref^0.25 * dt_days     (connected vehicles)
        alpha(V, T)   = (7.543 V - 23.75) * 1e6 * exp(-6976 / T)
        cost          = fade * battery_capital_usd_per_kwh * pack size

    | symbol | meaning | units |
    |---|---|---|
    | dQ | cell charge throughput this step, from the power the pack sees | Ah |
    | g(T, C) | wear factor for cell temperature T and rate C, see `wear_factor_g` | — |
    | T | the true cell temperature | K |
    | v_mean | mean cell voltage over the step | V |
    | dt_days | step length | days |

    Charging and holding at high state of charge both cost more, through the voltage.
    """
    v0 = ocv_cell_v(soc_before)
    v1 = ocv_cell_v(soc_after)
    v_rms = np.sqrt(0.5 * (v0**2 + v1**2))
    v_mean = 0.5 * (v0 + v1)
    d_dod = 0.5 * np.abs(soc_after - soc_before)
    # cell throughput uses the DC power the pack sees: eta_c * charge in, discharge / eta_d out
    pack_kw = cfg.charge_efficiency * charge_kw + discharge_kw / cfg.discharge_efficiency
    c_rate = pack_kw / fleet.battery_kwh  # 1/h
    dq_ah = c_rate * cfg.cell_capacity_ah * cfg.dt_hours
    beta = (SCHMALSTIEG_BETA_QUAD * (v_rms - cfg.schmalstieg_v_offset) ** 2
            + SCHMALSTIEG_BETA_CONST + SCHMALSTIEG_BETA_DDOD * d_dod)
    temp_k = fleet.temp_k
    fade_cycle = beta / (2.0 * np.sqrt(cfg.q_ref_ah)) * dq_ah * wear_factor_g(cfg, temp_k, c_rate)
    alpha = np.maximum((7.543 * v_mean - 23.75) * 1e6 * np.exp(-6976.0 / temp_k), 0.0)
    fade_cal = (0.75 * alpha / cfg.t_ref_days**0.25 * (cfg.dt_hours / 24.0)) * fleet.connected
    usd = cfg.battery_capital_usd_per_kwh * fleet.battery_kwh
    cycle_usd = float(np.sum(fade_cycle * usd))
    cal_usd = float(np.sum(fade_cal * usd))
    return cycle_usd + cal_usd, cycle_usd, cal_usd


def marginal_cycle_cost_usd_per_kwh(soc: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Marginal cycling cost at the given state of charge, $ per kWh of throughput, at the
    reference temperature and with dDOD = 0. Pack size cancels, so it depends on the state of
    charge only, through the cell voltage."""
    v = ocv_cell_v(soc)
    beta = SCHMALSTIEG_BETA_QUAD * (v - cfg.schmalstieg_v_offset) ** 2 + SCHMALSTIEG_BETA_CONST
    return beta * cfg.cell_capacity_ah * cfg.battery_capital_usd_per_kwh / (2.0 * np.sqrt(cfg.q_ref_ah))


def unmet_liability_usd(fleet: FleetState, cfg: SimConfig, mask: np.ndarray) -> float:
    """Minus the money owed if every masked vehicle left now, $. The potential of the reward
    shaping: it moves when the deadline cost is felt, and leaves the best policy unchanged
    (Ng, Harada and Russell, 1999)."""
    from .fleet import unmet_kwh_per_car
    return -cfg.undelivered_energy_price * float(np.sum(unmet_kwh_per_car(fleet, cfg) * mask))


def baseline_charge(fleet: FleetState, cfg: SimConfig) -> np.ndarray:
    """The baseline: each vehicle's power under its plan fixed at arrival, kW (0 when empty).

    The plan charges at its flat rate until its energy is delivered; the last step is
    pro-rated. It never reads the state of charge, so charging ahead of the plan does not
    lower the baseline later. Flexibility is measured against this baseline.
    """
    tail = fleet.plan_left_kwh / (cfg.charge_efficiency * cfg.dt_hours)
    c_base = np.maximum(np.minimum(fleet.plan_rate_kw, tail), 0.0)
    return np.where(fleet.connected, c_base, 0.0)


@dataclass
class DispatchResult:
    """What one dispatch step did.

    | field | meaning | units |
    |---|---|---|
    | `power_kw` | net power per charger, positive when charging | kW |
    | `charge_kw`, `discharge_kw` | its positive and negative parts | kW |
    | `baseline_kw` | the baseline plan per charger | kW |
    | `realized_flex_kw` | flexibility delivered: sum of baseline minus sum of power | kW |
    | `soc_violation` | energy the request asked for beyond the state-of-charge bounds | — |
    | `deviation_kw` | sum of the absolute deviations from the baseline | kW |
    | `site_unmet_kw` | power the station buses could not pass | kW |
    | `setpoint_kw` | the power the charger asked each vehicle for, before the vehicle's own limits | kW |
    """

    power_kw: np.ndarray
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    baseline_kw: np.ndarray
    realized_flex_kw: float
    soc_violation: float
    deviation_kw: float
    site_unmet_kw: float = 0.0
    setpoint_kw: np.ndarray | None = None


# ---------------------------------------------------------------------------
# The battery: voltage limits, heat and thermal derating.
# ---------------------------------------------------------------------------
GAS_R_J_PER_MOL_K = 8.314


def cell_resistance_ohm(cfg: SimConfig, temp_k, r_mult=1.0) -> np.ndarray:
    """R(T) = r25 * rho_k * exp(Ea_R / Rg * (1/T - 1/298.15)), ohm per cell."""
    return (cfg.r25_mohm * 1e-3) * r_mult * np.exp(
        cfg.r_ea_kj * 1e3 / GAS_R_J_PER_MOL_K * (1.0 / temp_k - 1.0 / 298.15))


def thermal_derate(cfg: SimConfig, temp_k) -> np.ndarray:
    """BMS power factor: 1 up to derate_t_start_c, derate_floor at derate_t_full_c."""
    t_c = temp_k - 273.15
    f = 1.0 - (1.0 - cfg.derate_floor) * (t_c - cfg.derate_t_start_c) / (
        cfg.derate_t_full_c - cfg.derate_t_start_c)
    return np.clip(f, cfg.derate_floor, 1.0)


def plating_c_rate(cfg: SimConfig, temp_k) -> np.ndarray:
    """Most charge C-rate before lithium plating, linear in T (2 C at 25 C, 1 C at 0 C)."""
    t_c = temp_k - 273.15
    return np.maximum(cfg.plating_c_at_0 + (cfg.plating_c_at_25 - cfg.plating_c_at_0) * t_c / 25.0, 0.1)


def v2_limits_kw(fleet: FleetState, cfg: SimConfig, soc=None, temp_k=None, r_mult=None, v_pol=None):
    """(charge limit, discharge limit) each vehicle accepts now, kW, grid side.

    The cell current is limited by the voltage headroom over the cell resistance R(T), and the
    charge rate also by the lithium-plating limit. Both are scaled by the thermal derating.
    Called without overrides this is the true physics, from the hidden temperature, resistance
    and polarization; passing `temp_k`, `r_mult` or `v_pol` gives an estimate instead.
    """
    z = fleet.soc if soc is None else soc
    T = fleet.temp_k if temp_k is None else temp_k
    rho = fleet.r_mult if r_mult is None else r_mult
    R = cell_resistance_ohm(cfg, T, rho)
    ocv = ocv_cell_v(np.clip(z, 0.0, 1.0))
    # the polarization voltage takes part of the headroom
    vp = fleet.v_pol if v_pol is None else v_pol
    i_c = np.maximum(cfg.cell_v_max - ocv - vp, 0.0) / R      # A per cell
    i_d = np.maximum(ocv - cfg.cell_v_min + vp, 0.0) / R
    c_c = np.minimum(i_c / cfg.cell_capacity_ah, plating_c_rate(cfg, T))   # 1/h
    c_d = i_d / cfg.cell_capacity_ah
    f = thermal_derate(cfg, T)
    cbar = fleet.max_charge_kw
    p_c = np.minimum(cbar, c_c * fleet.battery_kwh) * f
    p_d = np.minimum(cbar * cfg.discharge_rating_frac, c_d * fleet.battery_kwh) * f
    p_c = np.where(z >= cfg.soc_max, 0.0, p_c)
    p_d = np.where(z <= cfg.soc_min, 0.0, p_d)
    return p_c, p_d


def refresh_reported_limits(fleet: FleetState, cfg: SimConfig, idx=None) -> None:
    """Update the limits each vehicle reports to the charger (its present charge and discharge limits)."""
    p_c, p_d = v2_limits_kw(fleet, cfg)
    if idx is None:
        fleet.limit_c_kw, fleet.limit_d_kw = p_c, p_d
    else:
        fleet.limit_c_kw[idx], fleet.limit_d_kw[idx] = p_c[idx], p_d[idx]


def thermal_step(fleet: FleetState, cfg: SimConfig, charge_kw, discharge_kw) -> None:
    """Advance each connected vehicle's cell temperature over one dispatch step.

        C_th dT/dt = I^2 R(T) - h_k (T - T_amb)

    solved exactly over the step with R held at the step's start. h_k is set so
    that a 1 C charge at nominal resistance rises th_rise_1c_k in steady state;
    C_th = h tau makes the time constant th_tau_min. A vehicle with better cooling
    (cool_mult > 1) has a larger h_k and a SHORTER time constant.
    """
    conn = fleet.connected
    pack_kw = cfg.charge_efficiency * charge_kw + discharge_kw / cfg.discharge_efficiency
    i_cell = pack_kw / np.maximum(fleet.battery_kwh, 1e-9) * cfg.cell_capacity_ah     # A
    q_w = i_cell ** 2 * cell_resistance_ohm(cfg, fleet.temp_k, fleet.r_mult)          # W per cell
    h_nom = (cfg.cell_capacity_ah ** 2) * (cfg.r25_mohm * 1e-3) / cfg.th_rise_1c_k      # W/K
    c_th = h_nom * cfg.th_tau_min * 60.0                                              # J/K
    h_k = h_nom * fleet.cool_mult
    tau_s = c_th / h_k
    t_amb = 273.15 + cfg.t_amb_c
    t_ss = t_amb + q_w / h_k
    dt_s = cfg.dt_hours * 3600.0
    t_new = t_ss + (fleet.temp_k - t_ss) * np.exp(-dt_s / tau_s)
    fleet.temp_k = np.where(conn, t_new, fleet.temp_k)
    refresh_reported_limits(fleet, cfg)


# ---------------------------------------------------------------------------
# The station: a DC bus shared by `cfg.station_size` chargers, a converter loss,
# and one polarization state per vehicle.
# ---------------------------------------------------------------------------

def station_index(n: int, cfg: SimConfig) -> np.ndarray:
    """Station of each stall: stalls 0..size-1 are station 0, and so on."""
    return np.arange(n) // max(int(cfg.station_size), 1)


def converter_eta(fleet: FleetState, cfg: SimConfig, power_kw) -> np.ndarray:
    """Converter efficiency at grid-side power |P|, on top of the fixed charge and discharge
    efficiencies: eta_conv = 1 - l |P| / rating, with l = `cfg.conv_loss_frac_at_rated`."""
    r = np.maximum(fleet.max_charge_kw, 1e-9)
    return np.clip(1.0 - cfg.conv_loss_frac_at_rated * np.abs(power_kw) / r, 0.5, 1.0)


def project_to_cap(P, lo, hi, L, connected):
    """Enforce |sum P| <= L by moving vehicles toward the binding side in proportion to their room.
    Returns (P, power that could not be moved, kW).

    Two stages: first each vehicle moves only as far as zero (a charging vehicle is cut back, never
    pushed into discharge); only if the cap still binds does the second stage use each vehicle's
    full range."""
    P = np.asarray(P, dtype=np.float64).copy()
    total = float(np.sum(P[connected])) if connected.any() else 0.0
    if abs(total) <= L + 1e-9:
        return P, 0.0
    if total > L:
        excess, sgn = total - L, -1.0
        rooms = (np.maximum(P - np.maximum(lo, 0.0), 0.0), np.maximum(P - lo, 0.0))
    else:
        excess, sgn = -L - total, +1.0
        rooms = (np.maximum(np.minimum(hi, 0.0) - P, 0.0), np.maximum(hi - P, 0.0))
    for room in rooms:
        room = np.where(connected, room, 0.0)
        tr = float(np.sum(room))
        if tr <= 1e-12:
            continue
        move = min(excess, tr)
        P = P + sgn * move * (room / tr)
        excess -= move
        if excess <= 1e-9:
            return P, 0.0
    return P, max(excess, 0.0)


def project_to_station_caps(P, lo, hi, cfg: SimConfig, connected):
    """Apply the DC-bus cap `cfg.station_bus_kw` to every station. Returns (P, unmet kW)."""
    P = np.asarray(P, dtype=np.float64).copy()
    st = station_index(len(P), cfg)
    unmet = 0.0
    for s in np.unique(st):
        m = (st == s) & connected
        if not m.any():
            continue
        Ps, u = project_to_cap(P[m], lo[m], hi[m], float(cfg.station_bus_kw), np.ones(int(m.sum()), bool))
        P[m] = Ps; unmet += u
    return P, unmet


def polarization_step(fleet: FleetState, cfg: SimConfig, charge_kw, discharge_kw) -> None:
    """Advance each connected vehicle's polarization voltage over one step:

        dV_pol/dt = (I R_pol - V_pol) / tau_k

    | symbol | meaning | units |
    |---|---|---|
    | V_pol | polarization voltage per cell; after charging it lowers the charge headroom | V |
    | I | signed cell current, positive when charging, held over the step | A |
    | R_pol | `cfg.pol_r_mohm` | mohm |
    | tau_k | relaxation time, `cfg.pol_tau_min` times a hidden per-vehicle factor | min |
    """
    conn = fleet.connected
    pack_kw = cfg.charge_efficiency * charge_kw - discharge_kw / cfg.discharge_efficiency
    i_cell = pack_kw / np.maximum(fleet.battery_kwh, 1e-9) * cfg.cell_capacity_ah
    v_ss = i_cell * cfg.pol_r_mohm * 1e-3
    tau_s = cfg.pol_tau_min * 60.0 * fleet.pol_tau_mult
    a = np.exp(-cfg.dt_hours * 3600.0 / np.maximum(tau_s, 1.0))
    fleet.v_pol = np.where(conn, v_ss + (fleet.v_pol - v_ss) * a, fleet.v_pol)


def wear_factor_g(cfg: SimConfig, temp_k, c_rate) -> np.ndarray:
    """g(T, C): the cycle-wear factor for cell temperature T (K) and rate C (1/h), equal to 1
    at (`cfg.wear_t_ref_c`, `cfg.wear_c_ref`).

    g = A(T) * exp((d T + e) C) / exp((d T_ref + e) C_ref). A(T) is V-shaped with its minimum at
    `cfg.wear_t_min_c` (Waldmann et al., 2014): an Arrhenius rise above it and a steeper one
    below it, from lithium plating. The rate term is the NMC form of Wang et al. (2014), with
    d = `cfg.wang_d` and e = `cfg.wang_e`.
    """
    T = np.asarray(temp_k, dtype=float)
    t_ref = 273.15 + cfg.wear_t_ref_c
    t_min = 273.15 + cfg.wear_t_min_c
    ea_h = cfg.wear_ea_hot_kj * 1e3 / GAS_R_J_PER_MOL_K
    ea_c = cfg.wear_ea_cold_kj * 1e3 / GAS_R_J_PER_MOL_K
    a_hot = np.exp(ea_h * (1.0 / t_ref - 1.0 / T))
    a_min = np.exp(ea_h * (1.0 / t_ref - 1.0 / t_min))
    a_cold = a_min * np.exp(ea_c * (1.0 / T - 1.0 / t_min))
    A = np.where(T >= t_min, a_hot, a_cold)
    rate = np.exp((cfg.wang_d * T + cfg.wang_e) * c_rate) / np.exp(
        (cfg.wang_d * t_ref + cfg.wang_e) * cfg.wear_c_ref)
    return A * rate


def taper_limit_kw(fleet: FleetState, cfg: SimConfig, soc=None) -> np.ndarray:
    """The true most power each vehicle accepts when charging at state of charge `soc`, kW."""
    z = fleet.soc if soc is None else soc
    return v2_limits_kw(fleet, cfg, soc=z)[0]


def build_view(fleet: FleetState, cfg: SimConfig) -> FleetState:
    """The fleet as a controller sees it. Every controller-side function takes this view.

    A DC charger reports each vehicle's state of charge, pack size and present charge and
    discharge limits, and knows the energy the driver asked for. It does not see cell
    temperature, resistance, cooling, polarization, the driver's type, or the true departure:
    those fields are removed, and the departure countdown reads 1 while the vehicle is connected.

    | view field | from |
    |---|---|
    | target_soc | state of charge plus the energy still owed, over pack size |
    | charge_cap_kw, discharge_cap_kw | the limits the vehicle reports |
    | departure_steps | 1 while connected |
    """
    import dataclasses
    conn = fleet.connected
    cap = np.maximum(fleet.battery_kwh, 1e-6)
    tgt = np.minimum(fleet.soc + np.maximum(fleet.request_kwh - fleet.pack_hat_kwh, 0.0) / cap,
                     cfg.soc_max - 0.01)
    tgt = np.where(fleet.stopped, np.minimum(tgt, fleet.soc), tgt)
    return dataclasses.replace(
        fleet,
        target_soc=np.where(conn, tgt, fleet.target_soc),
        charge_cap_kw=np.where(conn, fleet.limit_c_kw, 0.0),
        discharge_cap_kw=np.where(conn, fleet.limit_d_kw, 0.0),
        departure_steps=conn.astype(fleet.departure_steps.dtype),
        temp_k=None, r_mult=None, cool_mult=None, taper_knee=None,
        v_pol=None, pol_tau_mult=None, true_type=None, hetero_rng=None,
    )


def apply_dispatch(
    fleet: FleetState,
    action: np.ndarray,
    c_base: np.ndarray,
    cfg: SimConfig,
    ctrl_fleet: FleetState | None = None,
) -> DispatchResult:
    """Apply one dispatch action and advance the fleet in place.

    `action` in [-1, 1] per charger scales to a deviation of up to `cfg.max_deviation_kw`
    around the baseline. The requested power is then limited, in this order, by the charger
    rating; the deadline floor computed from the controller's view `ctrl_fleet`; the vehicle's true
    voltage and thermal limits; and the station bus. Physics comes last, so it always wins.
    The power actually delivered is backed out of the energy that moved, so revenue, wear and
    settlement are paid only on energy that physically moved.
    """
    connected = fleet.connected
    delta_kw = np.clip(action, -1.0, 1.0) * cfg.max_deviation_kw * connected
    power_req = np.clip(c_base + delta_kw, -fleet.max_charge_kw, fleet.max_charge_kw)
    power_req = np.where(connected, power_req, 0.0)

    # The deadline floor is the controller's safety rule, so it comes from the controller's
    # view; the physical room comes from the true fleet.
    lo_deadline, hi_phys = per_car_power_limits(fleet, cfg, cfg.dt_hours)
    setpoint = power_req.copy()
    if ctrl_fleet is not None and ctrl_fleet is not fleet:
        lo_deadline = per_car_power_limits(ctrl_fleet, cfg, cfg.dt_hours)[0]
        # what the charger asks for: the command raised to the view's floor
        setpoint = np.where(connected, np.maximum(power_req, lo_deadline), 0.0)
        lo_deadline = np.minimum(lo_deadline, hi_phys)
    power_req = np.where(connected, np.clip(power_req, lo_deadline, hi_phys), 0.0)
    if ctrl_fleet is None or ctrl_fleet is fleet:
        setpoint = power_req.copy()

    site_unmet_kw = 0.0
    # the cell voltage limits cap both directions, at the true temperature and resistance
    p_c, p_d = v2_limits_kw(fleet, cfg)
    power_req = np.where(connected, np.clip(power_req, -p_d, p_c), 0.0)
    # The station bus is hardware, so it wins over the deadline floor: the projection may use
    # each vehicle's full physical range, and a deadline it squeezes out is settled at departure.
    power_req, st_unmet = project_to_station_caps(power_req, -p_d, p_c, cfg, connected)
    site_unmet_kw += st_unmet

    charge_req = np.maximum(power_req, 0.0)
    discharge_req = np.maximum(-power_req, 0.0)

    # state of charge, with the converter loss on top of the fixed efficiencies;
    # the violation is measured before clipping
    eta_c = cfg.charge_efficiency
    eta_cv = converter_eta(fleet, cfg, power_req)
    d_soc = (
        eta_c * eta_cv * charge_req * cfg.dt_hours / fleet.battery_kwh
        - discharge_req * cfg.dt_hours / (cfg.discharge_efficiency * eta_cv * fleet.battery_kwh)
    )
    soc_before = fleet.soc
    raw = soc_before + d_soc
    viol = np.maximum(cfg.soc_min - raw, 0.0) + np.maximum(raw - cfg.soc_max, 0.0)
    soc_after = np.clip(raw, cfg.soc_min, cfg.soc_max)

    # Back the realized power out of the energy that actually moved, inverting the converter
    # loss: charging E = eta dt P (1 - k P), discharging E_out = x dt / (eta_d (1 - k x)),
    # with k = l / rating. Where no state-of-charge bound was hit, this returns power_req.
    d_energy_kwh = (soc_after - soc_before) * fleet.battery_kwh
    k = cfg.conv_loss_frac_at_rated / np.maximum(fleet.max_charge_kw, 1e-9)
    e_in = np.maximum(d_energy_kwh, 0.0) / (eta_c * cfg.dt_hours)
    disc = np.maximum(1.0 - 4.0 * k * e_in, 0.0)
    p_ch = np.where(k > 1e-12, (1.0 - np.sqrt(disc)) / (2.0 * np.maximum(k, 1e-12)), e_in)
    e_out = np.maximum(-d_energy_kwh, 0.0) * cfg.discharge_efficiency
    p_dis = e_out / (cfg.dt_hours + e_out * k)
    power = np.where(d_energy_kwh >= 0.0, p_ch, -p_dis)
    power = np.where(connected, power, 0.0)
    fleet.soc = soc_after

    charge = np.maximum(power, 0.0)
    discharge = np.maximum(-power, 0.0)
    # the step's current builds the polarization voltage and heats the pack;
    # the thermal step then refreshes the limits each vehicle reports
    polarization_step(fleet, cfg, charge, discharge)
    thermal_step(fleet, cfg, charge, discharge)

    return DispatchResult(
        power_kw=power,
        charge_kw=charge,
        discharge_kw=discharge,
        baseline_kw=c_base,
        realized_flex_kw=float(np.sum(c_base) - np.sum(power)),
        soc_violation=float(np.sum(viol)),
        deviation_kw=float(np.sum(np.abs(power - c_base))),
        site_unmet_kw=float(site_unmet_kw),
        setpoint_kw=setpoint,
    )


def planning_steps(fleet: FleetState, cfg: SimConfig) -> np.ndarray:
    """The horizon a controller plans against, per vehicle, in dispatch steps (>= 1): the driver's
    stated remaining stay, or a controller's own estimate when ``fleet.planning_hook`` is set."""
    if fleet.planning_hook is not None:
        return np.maximum(np.asarray(fleet.planning_hook(fleet, cfg), dtype=np.float64), 1.0)
    return np.maximum(fleet.est_departure_steps.astype(np.float64), 1.0)


def deadline_min_power(fleet: FleetState, cfg: SimConfig, hours: float) -> np.ndarray:
    """The deadline floor: the least power each vehicle must hold over ``hours`` so that its
    target stays reachable by its planned departure, kW. 0 for empty chargers. A negative
    value means the vehicle may give energy back and still meet its target.

    Holding power a for H hours moves f(a) kWh into the pack, with f(a) = eta_c a H when
    charging and f(a) = a H / eta_d when discharging. Whatever is still owed afterwards must fit
    in the remaining steps at the vehicle's rate:

        e_k - f(a) <= eta_c * rate_k * (tau_k - H / dt) * dt

    | symbol | meaning | units |
    |---|---|---|
    | e_k | energy still needed to reach the target | kWh |
    | tau_k | planned steps left, `planning_steps` | steps |
    | rate_k | charger rating, or the lower charge limit the vehicle reports | kW |
    | H | `hours` | h |
    | dt | dispatch step, `cfg.dt_hours` | h |
    | eta_c, eta_d | charge and discharge efficiency | — |
    """
    steps_in_horizon = hours / cfg.dt_hours
    tau = planning_steps(fleet, cfg)
    steps_after = np.maximum(tau - steps_in_horizon, 0.0)
    e_need_kwh = np.maximum(fleet.target_soc - fleet.soc, 0.0) * fleet.battery_kwh
    # energy that can still reach the pack after this horizon, kWh; the charge limit only
    # tightens as the vehicle fills, so today's reported limit bounds every later step's rate
    rate_later = fleet.max_charge_kw if fleet.charge_cap_kw is None \
        else np.minimum(fleet.max_charge_kw, fleet.charge_cap_kw)
    payable_later_kwh = (
        cfg.charge_efficiency * rate_later * steps_after * cfg.dt_hours
    )
    # R > 0: energy that MUST reach the pack over this horizon, kWh.
    # R < 0: energy that may be taken out of it without losing the deadline.
    R = e_need_kwh - payable_later_kwh
    a_req = np.where(
        R >= 0.0,
        R / (cfg.charge_efficiency * hours),
        R * cfg.discharge_efficiency / hours,
    )
    return np.where(fleet.connected, a_req, 0.0)


def per_car_power_limits(fleet: FleetState, cfg: SimConfig, hours: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-vehicle net power limits [P_min, P_max] that can be held for ``hours``, kW.

    Bounded by the charger rating, by the energy the pack can give or take over ``hours``
    within [soc_min, soc_max], by the limits the vehicle reports (on a controller's view), and
    from below by the deadline floor ``deadline_min_power``. A vehicle whose target is already out
    of reach has P_min = P_max, so it offers no flexibility.
    """
    cbar = fleet.max_charge_kw
    conn = fleet.connected
    dischargeable_kw = (
        (fleet.soc - cfg.soc_min) * fleet.battery_kwh * cfg.discharge_efficiency / hours
    )
    chargeable_kw = (
        (cfg.soc_max - fleet.soc) * fleet.battery_kwh / (cfg.charge_efficiency * hours)
    )
    p_min = np.maximum(-cbar, -np.maximum(dischargeable_kw, 0.0))
    p_max = np.minimum(cbar, np.maximum(chargeable_kw, 0.0))
    if fleet.charge_cap_kw is not None:
        p_max = np.minimum(p_max, fleet.charge_cap_kw)
    if fleet.discharge_cap_kw is not None:
        p_min = np.maximum(p_min, -fleet.discharge_cap_kw)
    a_req = deadline_min_power(fleet, cfg, hours)
    p_min = np.minimum(p_max, np.maximum(p_min, a_req))

    return np.where(conn, p_min, 0.0), np.where(conn, p_max, 0.0)


def feasible_flex_range(fleet: FleetState, c_base: np.ndarray, cfg: SimConfig) -> tuple[float, float]:
    """The flexibility the fleet can deliver over a whole market interval, (q_min, q_max) in kW.

    Flexibility is the sum of the baseline minus the sum of the power, so the range follows
    from the per-vehicle power limits, then from each station's bus cap. q_max is the most the
    fleet can reduce charging; -q_min is the most it can increase it.
    """
    interval_hours = cfg.market_interval_min / 60.0
    p_min, p_max = per_car_power_limits(fleet, cfg, interval_hours)
    base = float(np.sum(c_base))
    # each station's DC bus caps its chargers' net power
    B = float(cfg.station_bus_kw)
    st = station_index(len(p_min), cfg)
    lo_s = np.bincount(st, weights=p_min); hi_s = np.bincount(st, weights=p_max)
    hi_c = np.where(lo_s > B, lo_s, np.minimum(hi_s, B))
    lo_c = np.where(hi_s < -B, hi_s, np.maximum(lo_s, -B))
    tot_lo, tot_hi = float(np.sum(lo_c)), float(np.sum(hi_c))

    return base - tot_hi, base - tot_lo


def urgency_stats(fleet: FleetState, cfg: SimConfig) -> tuple[float, float]:
    """(mean, 90th percentile) of urgency over connected vehicles. Urgency is the power needed to
    reach the target by the stated departure, over the charger rating; above 1 the target is
    already out of reach."""
    connected = fleet.connected
    if not np.any(connected):
        return 0.0, 0.0
    steps_left = np.maximum(fleet.est_departure_steps.astype(np.float64), 1.0)
    soc_gap = np.maximum(fleet.target_soc - fleet.soc, 0.0)
    required_kw = (soc_gap * fleet.battery_kwh) / (
        cfg.charge_efficiency * steps_left * cfg.dt_hours + 1e-9
    )
    u = (required_kw / fleet.max_charge_kw)[connected]
    return float(np.mean(u)), float(np.percentile(u, 90))


def power_to_action(
    power_kw: np.ndarray, c_base: np.ndarray, cfg: SimConfig
) -> np.ndarray:
    """Turn net powers (kW) into the dispatch action: the deviation from the baseline as a share
    of `cfg.max_deviation_kw`, in [-1, 1]."""
    return np.clip((power_kw - c_base) / cfg.max_deviation_kw, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Per-vehicle cost scores and the fleet's flexibility cost curve
# ---------------------------------------------------------------------------
# shares of the deliverable range at which the cost curve is read for the bid's observation
CURVE_FRACTIONS = (0.25, 0.50, 0.75)


def _merit_scores(fleet: FleetState, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Each vehicle's marginal cost of deviating, $/kWh, in both directions.

    up (charge less): wear + w * urgency, since taking energy from an urgent vehicle risks its target.
    down (charge more): wear - w * urgency, since charging an urgent vehicle helps it.
    w = `cfg.merit_urgency_weight_usd_per_kwh`.
    """
    aging = marginal_cycle_cost_usd_per_kwh(fleet.soc, cfg)
    steps = np.maximum(fleet.est_departure_steps.astype(np.float64), 1.0)
    gap = np.maximum(fleet.target_soc - fleet.soc, 0.0)
    req_kw = gap * fleet.battery_kwh / (
        cfg.charge_efficiency * steps * cfg.dt_hours + 1e-9
    )
    urg = req_kw / fleet.max_charge_kw
    w = cfg.merit_urgency_weight_usd_per_kwh
    return aging + w * urg, aging - w * urg


def wear_quadratic_coeffs(fleet: FleetState, cfg: SimConfig):
    """Cycle wear of each vehicle as a function of its power P for one step, $:

        cost_k(P) = a_k^+ P + b_k^+ P^2     for P >= 0
        cost_k(P) = a_k^- |P| + b_k^- P^2   for P < 0

    Returns (a_plus, a_minus, b_plus, b_minus), in $/kW and $/kW^2 per dispatch step.

    From `degradation_cost_usd` at the current state of charge: the pack size cancels in the
    linear coefficient a, and the quadratic coefficient b, which comes from the depth term,
    scales as 1 / pack size. So a small pack pays more for a deep swing. The throughput is
    measured on the pack side, so a differs by direction.
    """
    v = ocv_cell_v(fleet.soc)
    K = (
        cfg.cell_capacity_ah
        * cfg.dt_hours
        * cfg.battery_capital_usd_per_kwh
        / (2.0 * np.sqrt(cfg.q_ref_ah))
    )
    a = K * (
        SCHMALSTIEG_BETA_QUAD * (v - cfg.schmalstieg_v_offset) ** 2
        + SCHMALSTIEG_BETA_CONST
    )
    E = np.maximum(fleet.battery_kwh, 1e-9)
    base = K * SCHMALSTIEG_BETA_DDOD * 0.5 * cfg.dt_hours / E
    b_plus = base * cfg.charge_efficiency
    b_minus = base / cfg.discharge_efficiency
    a_plus = a * cfg.charge_efficiency
    a_minus = a / cfg.discharge_efficiency
    return a_plus, a_minus, b_plus, b_minus


def qp_allocation_power(
    fleet: FleetState, c_base: np.ndarray, q_target_kw: float, cfg: SimConfig,
    *, urgency_weight: float = 0.0, tol: float = 1e-10, max_iter: int = 200,
    wear_scale=None, a_extra=None, credit_extra=None, a_dis_extra=None, _fixed=None,
) -> np.ndarray:
    """The dispatch QP: the cheapest split of the commitment across vehicles.

        min_P   sum_k [ cost_k(P_k) - credit_k * P_k ]
        s.t.    sum_k P_k = sum_k c_base_k - q_target
                lo_k <= P_k <= hi_k
                |sum of P over each station| <= station_bus_kw

    | symbol | meaning | units |
    |---|---|---|
    | P_k | power of vehicle k | kW |
    | cost_k | its wear, `wear_quadratic_coeffs` (scaled by `wear_scale`, plus `a_extra`) | $ per step |
    | credit_k | `urgency_weight` * urgency + `credit_extra` | $/kW per step |
    | c_base_k | baseline plan | kW |
    | q_target | the commitment | kW |
    | lo_k, hi_k | `per_car_power_limits` over one dispatch step | kW |

    The objective is strictly convex and separable with one coupling equality, so the exact
    solution is a search on its multiplier lambda. Each vehicle's power is

        P_k(lambda) = clip(soft(credit_k + lambda) / (2 b_k), lo_k, hi_k)

    with soft(c) = c - a^+ above a^+, c + a^- below -a^-, and 0 between. P_k rises with
    lambda, so bisection converges.

    The station caps are handled by `cfg.station_limit_method`: "exact" puts them inside the
    program (each binding station gets its own multiplier); "projection" solves without them,
    projects each over-cap station onto its cap, fixes it, and re-solves the rest.
    """
    exact_caps = cfg.station_limit_method == "exact" and _fixed is None
    if cfg.station_limit_method == "projection" and _fixed is None:
        return _qp_station_capped(fleet, c_base, q_target_kw, cfg, urgency_weight=urgency_weight,
                                  tol=tol, max_iter=max_iter, wear_scale=wear_scale,
                                  a_extra=a_extra, credit_extra=credit_extra, a_dis_extra=a_dis_extra)
    conn = fleet.connected
    fixed_sum = 0.0
    if _fixed is not None:
        fm, fv = _fixed
        fixed_sum = float(np.sum(np.where(fm & conn, fv, 0.0)))
        conn = conn & ~fm
    lo, hi = per_car_power_limits(fleet, cfg, cfg.dt_hours)
    if not np.any(conn):
        return np.where(conn, c_base, 0.0).astype(np.float64)

    a_plus, a_minus, b_plus, b_minus = wear_quadratic_coeffs(fleet, cfg)
    # `wear_scale` multiplies each vehicle's wear coefficients; `a_extra` adds a linear price on
    # |P_k|, $/kW per step; `a_dis_extra` adds one on discharge only.
    if wear_scale is not None:
        w = np.asarray(wear_scale, dtype=np.float64)
        a_plus, a_minus, b_plus, b_minus = a_plus * w, a_minus * w, b_plus * w, b_minus * w
    if a_extra is not None:
        x = np.maximum(np.asarray(a_extra, dtype=np.float64), 0.0)
        a_plus, a_minus = a_plus + x, a_minus + x
    if a_dis_extra is not None:
        a_minus = a_minus + np.maximum(np.asarray(a_dis_extra, dtype=np.float64), 0.0)

    # A floor on b keeps the program strictly convex when wear is priced at zero
    # (battery_capital_usd_per_kwh = 0); otherwise every power would sit at a corner.
    B_FLOOR = 1e-9
    b_plus = np.maximum(b_plus, B_FLOOR)
    b_minus = np.maximum(b_minus, B_FLOOR)

    steps = np.maximum(fleet.est_departure_steps.astype(np.float64), 1.0)
    gap = np.maximum(fleet.target_soc - fleet.soc, 0.0)
    req = gap * fleet.battery_kwh / (
        cfg.charge_efficiency * steps * cfg.dt_hours + 1e-9
    )
    u = req / np.maximum(fleet.max_charge_kw, 1e-9)
    credit = float(urgency_weight) * u
    if credit_extra is not None:
        credit = credit + np.asarray(credit_extra, dtype=np.float64)

    T = float(np.sum(c_base)) - float(q_target_kw) - fixed_sum

    def P_of(lam: float) -> np.ndarray:
        # soft threshold with a separate price on each side of zero
        c = credit + lam
        Pp = np.where(c > a_plus, (c - a_plus) / (2.0 * b_plus), 0.0)
        Pm = np.where(c < -a_minus, (c + a_minus) / (2.0 * b_minus), 0.0)
        P = np.where(c > a_plus, Pp, np.where(c < -a_minus, Pm, 0.0))
        return np.where(conn, np.clip(P, lo, hi), 0.0)

    if exact_caps:
        # Station caps inside the program. Vehicle k on station s sees the price
        # credit_k + lambda - mu_s, with mu_s the cap's multiplier. For a given lambda, station s
        # delivers clip(S_s(lambda), -B, B), which rises with lambda; so the outer search on
        # lambda is unchanged, and each binding station then gets an inner search for its price.
        B = float(cfg.station_bus_kw)
        st = station_index(fleet.n, cfg)
        n_st = int(st.max()) + 1

        def S_of(lam: float) -> np.ndarray:
            return np.bincount(st, weights=P_of(lam), minlength=n_st)

        def total(lam: float) -> float:
            return float(np.sum(np.clip(S_of(lam), -B, B)))
    else:
        def total(lam: float) -> float:
            return float(np.sum(P_of(lam)))

    # bracket: at lam very negative every vehicle sits at lo, at lam very
    # positive every vehicle sits at hi.
    lam_lo, lam_hi = -1e4, 1e4
    if exact_caps:
        lam_min, lam_max = lam_lo, lam_hi
    if total(lam_lo) >= T:
        lam_star = lam_lo            # T below even the all-lo total
    elif total(lam_hi) <= T:
        lam_star = lam_hi            # T above even the all-hi total
    else:
        lam_star = None
    if lam_star is not None:
        return _exact_cap_finish(P_of, S_of, st, B, lam_star, lam_min, lam_max, tol, max_iter) \
            if exact_caps else P_of(lam_star)
    # stop once the remaining window cannot move the total by more than tol
    lam_eps = tol / max(float(np.sum(1.0 / (2.0 * b_plus[conn]))) +
                        float(np.sum(1.0 / (2.0 * b_minus[conn]))), 1e-12)
    for _ in range(max_iter):
        lam = 0.5 * (lam_lo + lam_hi)
        tot = total(lam)
        if abs(tot - T) < tol or (lam_hi - lam_lo) < lam_eps:
            break
        if tot < T:
            lam_lo = lam
        else:
            lam_hi = lam
    lam_star = 0.5 * (lam_lo + lam_hi)
    if exact_caps:
        return _exact_cap_finish(P_of, S_of, st, B, lam_star, lam_min, lam_max, tol, max_iter)
    return P_of(lam_star)


def _exact_cap_finish(P_of, S_of, st, B, lam_star, lam_min, lam_max, tol, max_iter):
    """Given the fleet multiplier lam_star, give each over-cap station its own multiplier
    lam_star - mu_s so that its total sits exactly on its cap. Other stations keep lam_star."""
    P = P_of(lam_star)
    S = np.bincount(st, weights=P, minlength=int(st.max()) + 1)
    for s in np.flatnonzero(np.abs(S) > B + tol):
        target = B if S[s] > 0 else -B
        # S_s is non-decreasing in its price: over +B search below lam_star,
        # under -B search above it
        lo_l, hi_l = (lam_min, lam_star) if S[s] > 0 else (lam_star, lam_max)
        m = st == s
        for _ in range(max_iter):
            mid = 0.5 * (lo_l + hi_l)
            tot = float(np.sum(P_of(mid)[m]))
            if abs(tot - target) < tol or (hi_l - lo_l) < 1e-15:
                break
            if tot < target:
                lo_l = mid
            else:
                hi_l = mid
        P[m] = P_of(0.5 * (lo_l + hi_l))[m]
    return P


def _qp_station_capped(fleet, c_base, q_target_kw, cfg, **kw):
    """The "projection" station-cap method: solve the QP without caps, project each over-cap station
    onto its cap (vehicles move in proportion to their room), fix it, and re-solve the free vehicles.
    At most one pass per station. A heuristic, not the exact optimum."""
    n = fleet.n
    conn = fleet.connected
    fm = np.zeros(n, dtype=bool); fv = np.zeros(n)
    st = station_index(n, cfg)
    B = float(cfg.station_bus_kw)
    lo, hi = per_car_power_limits(fleet, cfg, cfg.dt_hours)
    P = None
    for _ in range(int(st.max()) + 2):
        P = qp_allocation_power(fleet, c_base, q_target_kw, cfg, _fixed=(fm, fv), **kw)
        P = np.where(fm, fv, P)
        tot = np.bincount(st, weights=np.where(conn, P, 0.0), minlength=int(st.max()) + 1)
        bad = [s for s in range(len(tot)) if abs(tot[s]) > B + 1e-6 and not fm[(st == s) & conn].all()]
        if not bad:
            return P
        for s in bad:
            m = (st == s) & conn
            Ps, _ = project_to_cap(P[m], lo[m], hi[m], B, np.ones(int(m.sum()), bool))
            fv[m] = Ps; fm[m] = True
    return P


def flex_cost_curve(
    fleet: FleetState, c_base: np.ndarray, cfg: SimConfig
) -> np.ndarray:
    """The fleet's marginal cost of flexibility, read at 25, 50 and 75 % of the deliverable
    range in each direction. Returns 6 numbers in price units:
    [up 25 %, up 50 %, up 75 %, down 25 %, down 50 %, down 75 %]."""
    conn = fleet.connected
    out = np.zeros(2 * len(CURVE_FRACTIONS))
    if not np.any(conn):
        return out

    score_up, score_down = _merit_scores(fleet, cfg)
    # room over one market interval, the time a commitment must be held
    lo, hi = per_car_power_limits(fleet, cfg, cfg.market_interval_min / 60.0)
    P = np.clip(np.where(conn, c_base, 0.0).astype(np.float64), lo, hi)
    for d, (score, room) in enumerate(
        (
            (score_up, np.where(conn, np.maximum(P - lo, 0.0), 0.0)),
            (score_down, np.where(conn, np.maximum(hi - P, 0.0), 0.0)),
        )
    ):
        order = np.argsort(score)
        cap = np.cumsum(room[order])
        total = cap[-1]
        if total <= 1e-9:
            continue
        sorted_scores = score[order]
        for i, f in enumerate(CURVE_FRACTIONS):
            idx = int(np.searchsorted(cap, f * total))
            out[d * len(CURVE_FRACTIONS) + i] = sorted_scores[
                min(idx, len(sorted_scores) - 1)
            ]
    return out / cfg.price_unit
