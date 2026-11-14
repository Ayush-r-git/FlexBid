"""Fleet state, arrivals and departures.

Each charger (stall) holds at most one vehicle. A vehicle is connected while its true countdown
``departure_steps`` is above zero. The driver also states a departure time, held in
``est_departure_steps``; controllers see only the stated time. A vehicle leaves at its true time,
and the energy it did not receive is settled then.

Every array in ``FleetState`` has one entry per charger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import SimConfig

#: Arrival state of charge, a two-component beta mixture:
#: with probability ``w`` Beta(a1, b1), otherwise Beta(a2, b2).
ARRIVAL_SOC_MIXTURE = dict(
    w=0.98196109293167,
    a1=2.021036097853689, b1=2.5128131241073444,
    a2=1.4602659824358968, b2=0.2857590316497959,
)


@dataclass
class DepartureRecord:
    """One vehicle leaving."""

    ev_index: int            # charger index
    soc_at_departure: float  # true state of charge, 0-1
    target_soc: float        # target state of charge, 0-1
    unmet_kwh: float         # energy the driver asked for and did not receive, kWh
    early: bool              # True if the vehicle left before its stated time
    est_error_steps: int     # stated minus true remaining stay at departure, dispatch steps


@dataclass
class FleetState:
    """The fleet, one entry per charger. Fields marked hidden are never shown to a controller.

    | field | meaning | units | seen by controllers |
    |---|---|---|---|
    | `soc` | state of charge | 0-1 | yes |
    | `target_soc` | state of charge the driver must reach | 0-1 | via the view |
    | `departure_steps` | true dispatch steps until the vehicle leaves; > 0 while connected | steps | no |
    | `est_departure_steps` | stated dispatch steps until the vehicle leaves | steps | yes |
    | `battery_kwh` | pack size | kWh | yes |
    | `max_charge_kw` | charger rating | kW | yes |
    | `plan_rate_kw`, `plan_left_kwh` | the baseline plan, fixed at arrival: its rate and the energy it still has to deliver | kW, kWh | yes |
    | `stated_steps`, `elapsed_steps` | stated stay at arrival, and steps since arrival | steps | yes |
    | `true_type` | the driver's hidden type, index into `cfg.dwell_types` | — | no |
    | `request_kwh` | energy the driver asked for at arrival | kWh | yes |
    | `delivered_kwh` | net energy through the meter since arrival | kWh | yes |
    | `stopped`, `stopped_dis` | the vehicle no longer accepts charge / discharge | — | yes |
    | `pack_hat_kwh`, `pack_peak_kwh` | pack energy moved since arrival, from the meter and nominal efficiencies, and its peak | kWh | yes |
    | `temp_k`, `r_mult`, `cool_mult` | cell temperature, resistance factor, cooling factor | K, —, — | no |
    | `limit_c_kw`, `limit_d_kw` | charge and discharge limits the vehicle reports | kW | yes |
    | `temp_hat_k` | the controller's estimate of cell temperature | K | yes |
    | `v_pol`, `pol_tau_mult` | polarization voltage per cell, and its relaxation-time factor | V, — | no |
    | `v_pol_hat` | the controller's estimate of the polarization voltage | V | yes |
    | `charge_cap_kw`, `discharge_cap_kw` | the power caps a controller sees; set on the view only | kW | yes |
    | `planning_hook` | optional callable(fleet, cfg) giving a controller's own planning horizon per vehicle | steps | — |
    """

    soc: np.ndarray
    target_soc: np.ndarray
    departure_steps: np.ndarray
    est_departure_steps: np.ndarray
    battery_kwh: np.ndarray
    max_charge_kw: np.ndarray
    plan_rate_kw: Optional[np.ndarray] = None
    plan_left_kwh: Optional[np.ndarray] = None
    hetero_rng: object = None
    stated_steps: Optional[np.ndarray] = None
    elapsed_steps: Optional[np.ndarray] = None
    planning_hook: object = None
    true_type: Optional[np.ndarray] = None
    taper_knee: Optional[np.ndarray] = None
    request_kwh: Optional[np.ndarray] = None
    delivered_kwh: Optional[np.ndarray] = None
    stopped: Optional[np.ndarray] = None
    stopped_dis: Optional[np.ndarray] = None
    pack_hat_kwh: Optional[np.ndarray] = None
    charge_cap_kw: Optional[np.ndarray] = None
    temp_k: Optional[np.ndarray] = None
    r_mult: Optional[np.ndarray] = None
    cool_mult: Optional[np.ndarray] = None
    limit_c_kw: Optional[np.ndarray] = None
    limit_d_kw: Optional[np.ndarray] = None
    temp_hat_k: Optional[np.ndarray] = None
    discharge_cap_kw: Optional[np.ndarray] = None
    v_pol: Optional[np.ndarray] = None
    pol_tau_mult: Optional[np.ndarray] = None
    v_pol_hat: Optional[np.ndarray] = None
    pack_peak_kwh: Optional[np.ndarray] = None

    @property
    def connected(self) -> np.ndarray:
        return self.departure_steps > 0

    @property
    def overstay_steps(self) -> np.ndarray:
        """Steps a connected vehicle has stayed past its stated time, >= 0."""
        return np.where(self.connected, np.maximum(self.elapsed_steps - self.stated_steps, 0), 0)

    @property
    def n(self) -> int:
        return int(self.soc.shape[0])


@dataclass
class SessionSchedule:
    """All sessions of one episode, sorted by arrival step."""

    arrival_step: np.ndarray      # dispatch step of arrival; 0 = present at the start
    dwell_act_steps: np.ndarray   # true stay, dispatch steps
    dwell_est_steps: np.ndarray   # stated stay, dispatch steps
    energy_kwh: np.ndarray        # energy requested, kWh
    capacity_kwh: np.ndarray      # pack size, kWh
    soc_arrival: np.ndarray       # state of charge at arrival, 0-1
    cursor: int = 0               # next session to admit
    n_dropped: int = 0            # sessions that found every charger busy
    driver_type: Optional[np.ndarray] = None   # hidden driver type per session


# ---------------------------------------------------------------- arrival state of charge
def arrival_soc(rng: np.random.Generator, size: int, cfg: SimConfig | None = None) -> np.ndarray:
    """Draw the state of charge at arrival.

    With ``cfg.soc_arrival_mean`` None, the draw is the beta mixture ``ARRIVAL_SOC_MIXTURE``.
    Otherwise it is a beta distribution on [``cfg.soc_min``, 1] with mean
    ``cfg.soc_arrival_mean`` and standard deviation ``cfg.soc_arrival_sd``. The mixture is
    drawn first in both cases, so the random stream is the same whichever is used.
    """
    f = ARRIVAL_SOC_MIXTURE
    pick = rng.random(size) < f["w"]
    out = np.where(pick, rng.beta(f["a1"], f["b1"], size), rng.beta(f["a2"], f["b2"], size))
    if cfg is None or cfg.soc_arrival_mean is None:
        return out
    a, b, lo = _soc_beta_params(cfg)
    return lo + (1.0 - lo) * rng.beta(a, b, size)


def _soc_beta_params(cfg: SimConfig) -> tuple[float, float, float]:
    """Beta shape (a, b) on [soc_min, 1] for the requested mean and standard deviation.

    A beta with mean m and concentration nu has a = m nu, b = (1 - m) nu and
    sd^2 = m (1 - m) / (nu + 1). A combination that would put the peak of the density at an
    edge (a <= 1 or b <= 1) is refused: a lower mean needs a narrower spread.
    """
    lo = float(cfg.soc_min)
    m = (float(cfg.soc_arrival_mean) - lo) / (1.0 - lo)
    sd = (0.16 if cfg.soc_arrival_sd is None else float(cfg.soc_arrival_sd)) / (1.0 - lo)
    if not 0.0 < m < 1.0:
        raise ValueError(f"soc_arrival_mean={cfg.soc_arrival_mean} must lie strictly between "
                         f"soc_min={lo} and 1")
    nu = m * (1.0 - m) / (sd * sd) - 1.0
    a, b = m * nu, (1.0 - m) * nu
    if a <= 1.0 or b <= 1.0:
        sd_max = float(np.sqrt(m * (1.0 - m) / (1.0 / min(m, 1.0 - m) + 1.0)))
        raise ValueError(f"soc_arrival_mean={cfg.soc_arrival_mean} with soc_arrival_sd="
                         f"{cfg.soc_arrival_sd} gives Beta(a={a:.2f}, b={b:.2f}); a peak inside "
                         f"the range needs both above 1. Narrow the spread to about "
                         f"{sd_max * (1.0 - lo):.2f} or raise the mean.")
    return float(a), float(b), lo


# ---------------------------------------------------------------- pack size
def _capacity_weights(cfg: SimConfig) -> np.ndarray:
    """A normal distribution with mean ``cfg.capacity_mean_kwh`` and standard deviation
    ``cfg.capacity_sd_kwh``, discretised on ``cfg.capacity_grid_kwh`` and normalised."""
    g = np.asarray(cfg.capacity_grid_kwh, dtype=np.float64)
    w = np.exp(-0.5 * ((g - cfg.capacity_mean_kwh) / cfg.capacity_sd_kwh) ** 2)
    tot = float(w.sum())
    return w / tot if tot > 0 else np.full(g.size, 1.0 / g.size)


def draw_capacity(cfg: SimConfig, rng: np.random.Generator) -> float:
    """Draw one pack size, kWh."""
    g = np.asarray(cfg.capacity_grid_kwh, dtype=np.float64)
    return float(rng.choice(g, p=_capacity_weights(cfg)))


# ---------------------------------------------------------------- admitting a vehicle
def start_clock(fleet: FleetState, idx: np.ndarray) -> None:
    """Start the session clock and zero the meters of newly admitted vehicles."""
    if len(idx) == 0:
        return
    fleet.stated_steps[idx] = fleet.est_departure_steps[idx]
    fleet.elapsed_steps[idx] = 0
    fleet.delivered_kwh[idx] = 0.0
    fleet.stopped[idx] = False
    fleet.stopped_dis[idx] = False
    fleet.pack_hat_kwh[idx] = 0.0
    fleet.pack_peak_kwh[idx] = 0.0


def draw_hidden_battery(fleet: FleetState, idx: np.ndarray, cfg: SimConfig) -> None:
    """Draw the hidden battery state of newly admitted vehicles: resistance and cooling factors,
    arrival temperature, and polarization relaxation time. Reset the controller's estimates."""
    if len(idx) == 0:
        return
    g = fleet.hetero_rng
    fleet.r_mult[idx] = g.uniform(1.0 - cfg.r_spread, 1.0 + cfg.r_spread, len(idx))
    fleet.cool_mult[idx] = g.uniform(1.0 - cfg.th_cool_spread, 1.0 + cfg.th_cool_spread, len(idx))
    t_arr = g.normal(cfg.t_arrival_mean_c, cfg.t_arrival_sd_c, len(idx))
    fleet.temp_k[idx] = 273.15 + np.clip(t_arr, cfg.t_amb_c - 10.0, cfg.derate_t_start_c - 3.0)
    fleet.temp_hat_k[idx] = 273.15 + cfg.t_arrival_mean_c
    fleet.pol_tau_mult[idx] = g.uniform(1.0 - cfg.pol_tau_spread, 1.0 + cfg.pol_tau_spread, len(idx))
    fleet.v_pol[idx] = 0.0
    fleet.v_pol_hat[idx] = 0.0
    from .physics import refresh_reported_limits
    refresh_reported_limits(fleet, cfg, idx)
    fleet.taper_knee[idx] = 1.0


def set_arrival_plan(fleet: FleetState, idx: np.ndarray, cfg: SimConfig) -> None:
    """Fix the baseline plan of newly admitted vehicles: the plan the fleet would follow with no
    market. A flat rate until the requested energy is delivered:

        r_k = min(cbar_k, e_k / (eta_c * f * T_k))

    | symbol | meaning | units |
    |---|---|---|
    | r_k | plan rate of vehicle k | kW |
    | cbar_k | charger rating | kW |
    | e_k | energy the driver requested | kWh |
    | eta_c | charge efficiency, `cfg.charge_efficiency` | — |
    | f | share of the stated stay the plan uses, `cfg.plan_stay_fraction`, in (0, 1] | — |
    | T_k | the stated stay | h |

    With f < 1 the plan finishes before the stated time, a margin against early departures.
    """
    if len(idx) == 0:
        return
    e_need = np.maximum(fleet.request_kwh[idx], 0.0)
    steps = np.floor(fleet.est_departure_steps[idx] * float(cfg.plan_stay_fraction))
    t_h = np.maximum(steps, 1) * cfg.dt_hours
    fleet.plan_rate_kw[idx] = np.minimum(fleet.max_charge_kw[idx], e_need / (cfg.charge_efficiency * t_h))
    fleet.plan_left_kwh[idx] = e_need


def _admit(fleet: FleetState, schedule: SessionSchedule, upto_step: int, cfg: SimConfig) -> None:
    """Admit every pending session that has arrived by ``upto_step`` into a free charger.
    A session that finds every charger busy is dropped and counted."""
    while schedule.cursor < len(schedule.arrival_step) and schedule.arrival_step[schedule.cursor] <= upto_step:
        i = schedule.cursor
        free = np.where(~fleet.connected)[0]
        if len(free) == 0:
            schedule.n_dropped += 1
        else:
            s = int(free[0])
            fleet.soc[s] = schedule.soc_arrival[i]
            fleet.target_soc[s] = min(schedule.soc_arrival[i] + schedule.energy_kwh[i] / schedule.capacity_kwh[i],
                                      cfg.soc_max - 0.01)
            fleet.departure_steps[s] = schedule.dwell_act_steps[i]
            fleet.est_departure_steps[s] = schedule.dwell_est_steps[i]
            if schedule.driver_type is not None:
                fleet.true_type[s] = int(schedule.driver_type[i])
            fleet.battery_kwh[s] = schedule.capacity_kwh[i]
            # each session gets the next rating in the cycle of charger_choices
            fleet.max_charge_kw[s] = float(cfg.charger_choices[schedule.cursor % len(cfg.charger_choices)])
            fleet.request_kwh[s] = float(schedule.energy_kwh[i])
            one = np.array([s])
            start_clock(fleet, one)
            draw_hidden_battery(fleet, one, cfg)
            set_arrival_plan(fleet, one, cfg)
        schedule.cursor += 1


# ---------------------------------------------------------------- fleet lifecycle
def init_fleet(cfg: SimConfig, rng: np.random.Generator, schedule: SessionSchedule,
               hetero_rng: np.random.Generator) -> FleetState:
    """An empty fleet, then the sessions present at the start of the episode.
    ``hetero_rng`` is the random stream for the hidden battery state."""
    n = cfg.n_chargers
    fleet = FleetState(
        soc=np.full(n, cfg.soc_min, dtype=np.float64),
        target_soc=np.full(n, cfg.soc_target_departure, dtype=np.float64),
        departure_steps=np.zeros(n, dtype=np.int64),
        est_departure_steps=np.zeros(n, dtype=np.int64),
        battery_kwh=np.full(n, cfg.battery_kwh, dtype=np.float64),
        max_charge_kw=np.full(n, cfg.max_charge_kw, dtype=np.float64),
        stated_steps=np.zeros(n, dtype=np.int64),
        elapsed_steps=np.zeros(n, dtype=np.int64),
        true_type=np.full(n, -1, dtype=np.int64),
    )
    fleet.plan_rate_kw = np.zeros(n, dtype=np.float64)
    fleet.plan_left_kwh = np.zeros(n, dtype=np.float64)
    fleet.temp_k = np.full(n, 273.15 + cfg.t_amb_c)
    fleet.r_mult = np.ones(n)
    fleet.cool_mult = np.ones(n)
    fleet.limit_c_kw = fleet.max_charge_kw.copy()
    fleet.limit_d_kw = fleet.max_charge_kw * cfg.discharge_rating_frac
    fleet.temp_hat_k = np.full(n, 273.15 + cfg.t_amb_c)
    fleet.v_pol = np.zeros(n)
    fleet.pol_tau_mult = np.ones(n)
    fleet.v_pol_hat = np.zeros(n)
    fleet.hetero_rng = hetero_rng
    fleet.taper_knee = np.ones(n, dtype=np.float64)
    fleet.request_kwh = np.zeros(n, dtype=np.float64)
    fleet.delivered_kwh = np.zeros(n, dtype=np.float64)
    fleet.stopped = np.zeros(n, dtype=bool)
    fleet.stopped_dis = np.zeros(n, dtype=bool)
    fleet.pack_hat_kwh = np.zeros(n)
    fleet.pack_peak_kwh = np.zeros(n)
    _admit(fleet, schedule, 0, cfg)
    return fleet


def unmet_kwh_per_car(fleet: FleetState, cfg: SimConfig) -> np.ndarray:
    """Energy owed to each driver if the vehicle left now, kWh: requested energy minus the metered
    pack energy, and 0 once the vehicle has stopped accepting charge (it is full)."""
    owed = np.maximum(fleet.request_kwh - fleet.pack_hat_kwh, 0.0)
    return np.where(fleet.stopped, 0.0, owed)


def advance_departures_and_arrivals(fleet: FleetState, cfg: SimConfig, rng: np.random.Generator,
                                    schedule: SessionSchedule, next_step: int) -> List[DepartureRecord]:
    """Count every connected vehicle down one step, settle the vehicles that leave, admit new arrivals."""
    was_connected = fleet.connected
    fleet.departure_steps[was_connected] -= 1
    fleet.est_departure_steps[was_connected] = np.maximum(fleet.est_departure_steps[was_connected] - 1, 0)
    fleet.elapsed_steps[was_connected] += 1

    departures: List[DepartureRecord] = []
    depart_idx = np.where(was_connected & (fleet.departure_steps <= 0))[0]
    for i in depart_idx:
        est_left = int(fleet.est_departure_steps[i])
        departures.append(DepartureRecord(
            ev_index=int(i),
            soc_at_departure=float(fleet.soc[i]),
            target_soc=float(fleet.target_soc[i]),
            unmet_kwh=float(unmet_kwh_per_car(fleet, cfg)[i]),
            early=est_left > 0,
            est_error_steps=est_left,
        ))
        fleet.departure_steps[i] = 0
        fleet.est_departure_steps[i] = 0

    _admit(fleet, schedule, next_step, cfg)
    return departures
