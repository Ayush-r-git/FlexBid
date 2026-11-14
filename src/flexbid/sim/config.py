"""Every setting of the simulator, in one dataclass.

The defaults are the 50-charger hub with 6-hour stays in the `base` market setting. `flexbid.make_config` builds the other scenarios; any field can be overridden there or
here. Units are given in each comment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields


@dataclass
class SimConfig:
    # ------------------------------------------------------------------ clock
    seed: int | None = 10                    # default seed of a Simulator without an explicit reset seed
    episode_length: int = 288                # market intervals per episode (288 x 15 min = 3 days)
    market_interval_min: int = 15            # length of one market interval, min
    dispatch_interval_min: int = 5           # length of one dispatch step, min; must divide the interval
    episode_start_hour: float | None = 12.0  # hour of day the episode starts; None = a random 5-minute step

    # ------------------------------------------------------------------ market
    market_lead_intervals: int = 0           # bid lead L, intervals: a bid placed at t is delivered at t + L
    p_min: float = -10.0                     # lowest clearing price, price units
    p_max: float = 10.0                      # highest clearing price, price units
    price_unit: float = 0.45                 # value of one price unit, $/kWh
    imbalance_price: float = 0.25            # charge for missing the commitment, $/kWh
    undelivered_energy_price: float = 1.0    # charge for energy a departing vehicle did not receive, $/kWh
    bid_scale: float = 1200.0                # scale of the last bid in the bid's observation, price units
    anchor_scale: float = 2.0                # the anchor moves +-anchor_scale around the last price, price units
    logistic_beta: float = 1.0               # steepness of this fleet's bid curve, 1 per price unit
    d_sigma_kw: float = 6000.0               # scale of the CAISO requirement anomaly, kW
    d_noise_std: float = 3.0                 # noise added to the requirement, kW
    d_band_fraction: float = 0.12            # the requirement's median size, as a share of the whole market's nominal capacity
    d_elasticity_kw_per_pu: float = 300.0    # price response of the requirement, kW per price unit
    # rivals
    n_competitors: int = 3                   # rival aggregators
    comp_capacity_scale: float = 1.0         # each rival's size relative to this fleet's nominal capacity
    comp_down_ratio: float = 1.0             # each rival's room to increase charging, relative to its room to reduce
    comp_beta: float = 1.0                   # rivals' curve steepness, 1 per price unit
    comp_heterogeneity: float = 0.3          # rivals' size and steepness vary by +- this share
    comp_p0_init_sd: float = 1.0             # spread of an unused initial anchor draw, price units (kept for the random stream)
    comp_exo_mean: float = 0.1               # mean of the rivals' mean anchors, price units
    comp_exo_mean_sd: float = 0.2            # spread of the rivals' mean anchors, price units
    comp_exo_sd: float = 0.15                # spread of each rival's anchor path, price units
    comp_exo_phi: float = 0.98               # persistence of each rival's anchor path per interval
    b_other_base: float = 800.0              # an unused AR(1) series drawn at reset, kept for the random stream
    b_other_rho: float = 0.9                 # its persistence
    b_other_noise_std: float = 60.0          # its noise

    # ------------------------------------------------------------------ site and chargers
    n_chargers: int = 50                     # chargers (stalls)
    charger_choices: tuple = (50.0, 100.0, 150.0)   # charger ratings, assigned in turn to arriving sessions, kW
    discharge_rating_frac: float = 0.5       # discharge limit as a share of the charger rating
    station_size: int = 4                    # chargers per station
    station_bus_kw: float = 250.0            # DC bus limit of one station, kW
    station_limit_method: str = "projection" # how the dispatch QP respects the bus: "projection" or "exact"
    conv_loss_frac_at_rated: float = 0.04    # converter loss at the rating, share of the power
    max_charge_kw: float = 11.0              # rating of an empty charger before its first session, kW
    battery_kwh: float = 50.0                # pack size of an empty charger before its first session, kWh

    # ------------------------------------------------------------------ arrivals and stays
    arr_rate_per_day: float = 150.0          # arrivals per day
    arr_morning_hour: float = 9.0            # centre of the morning arrival peak, h
    arr_evening_hour: float = 18.0           # centre of the evening arrival peak, h
    arr_kappa: float = 2.0                   # concentration of the peaks
    arr_morning_weight: float = 0.6          # share of peak arrivals in the morning peak
    arr_night_share: float = 0.2             # share of arrivals in the night window
    arr_night_start_hour: float = 22.0       # start of the night window, h
    arr_night_end_hour: float = 6.0          # end of the night window, h
    arr_history_days: float = 1.0            # days of arrivals drawn before the start, so chargers are in use at t = 0
    arr_day_amplitude: float = 0.25          # day-to-day variation of the arrival rate
    arr_day_period_days: float = 7.0         # period of that variation, days
    dwell_mean_h: float = 6.0                # mean stay, h
    dwell_log_sd: float = 0.75               # log standard deviation of the stay
    dwell_mix: tuple | None = None           # optional mixture of stays: ((mean h, weight), ...); its mean must equal dwell_mean_h
    dwell_err_shrink: float = 0.7            # how closely a stated stay tracks the true one; 1 = fully
    dwell_err_log_mean: float = -0.08        # bias of the stated stay when no driver types are set
    dwell_err_log_sd: float = 0.3            # noise of the stated stay when no driver types are set
    dwell_err_max_ratio: float = 2.0         # stated / true stay is kept within [1/R, R]
    # hidden driver types: (name, share, bias of log stated stay, noise of log stated stay)
    dwell_types: tuple | None = (("on_time", 0.4, -0.07, 0.12), ("early", 0.25, 0.224, 0.12),
                                 ("overstay", 0.35, -0.28, 0.12))
    synth_request_margin: float = 0.9        # a driver asks for at most this share of what its charger can deliver in the stated stay

    # ------------------------------------------------------------------ vehicles
    soc_min: float = 0.1                     # lowest state of charge
    soc_max: float = 0.99                    # highest state of charge
    soc_target_departure: float = 0.9        # target state of charge at departure
    soc_arrival_mean: float | None = 0.38    # mean state of charge at arrival; None = the fitted beta mixture
    soc_arrival_sd: float | None = 0.16      # its standard deviation
    capacity_grid_kwh: tuple = tuple(float(x) for x in range(50, 125, 5))   # possible pack sizes, kWh
    capacity_mean_kwh: float = 90.0          # mean of the pack-size distribution, kWh
    capacity_sd_kwh: float = 18.0            # its standard deviation, kWh
    charge_efficiency: float = 0.95          # charge efficiency, grid to pack
    discharge_efficiency: float = 0.95       # discharge efficiency, pack to grid
    plan_stay_fraction: float = 1.0          # share of the stated stay the baseline plan uses, in (0, 1]
    full_accept_kw: float = 0.1              # a vehicle accepting less than this is reported as full, kW

    # ------------------------------------------------------------------ battery
    cell_capacity_ah: float = 4.85           # cell capacity, Ah
    cell_v_max: float = 4.2                  # cell voltage limit when charging, V
    cell_v_min: float = 3.0                  # cell voltage limit when discharging, V
    r25_mohm: float = 50.0                   # cell resistance at 25 °C, mohm
    r_ea_kj: float = 20.0                    # activation energy of the resistance, kJ/mol
    r_spread: float = 0.3                    # hidden per-vehicle resistance factor varies by +- this share
    th_tau_min: float = 35.0                 # thermal time constant, min
    th_rise_1c_k: float = 10.0               # steady temperature rise at a 1 C charge, K
    th_cool_spread: float = 0.3              # hidden per-vehicle cooling factor varies by +- this share
    t_amb_c: float = 25.0                    # ambient temperature, °C
    t_arrival_mean_c: float = 32.0           # mean cell temperature at arrival, °C
    t_arrival_sd_c: float = 4.0              # its standard deviation, °C
    derate_t_start_c: float = 48.0           # thermal derating starts, °C
    derate_t_full_c: float = 58.0            # thermal derating reaches its floor, °C
    derate_floor: float = 0.3                # lowest derating factor
    plating_c_at_25: float = 2.0             # highest charge rate before lithium plating at 25 °C, 1/h
    plating_c_at_0: float = 1.0              # the same at 0 °C, 1/h
    pol_r_mohm: float = 25.0                 # polarization resistance, mohm
    pol_tau_min: float = 20.0                # polarization relaxation time, min
    pol_tau_spread: float = 0.5              # hidden per-vehicle relaxation factor varies by +- this share

    # ------------------------------------------------------------------ battery wear
    battery_capital_usd_per_kwh: float = 150.0   # pack price that wear is charged at, $/kWh
    q_ref_ah: float = 5000.0                 # reference accumulated throughput of the cycle-fade slope, Ah
    t_ref_days: float = 365.0                # reference age of the calendar-fade slope, days
    schmalstieg_v_offset: float = 3.695      # voltage offset of the cycle-fade model, V
    wear_t_ref_c: float = 35.0               # temperature at which the wear factor g is 1, °C
    wear_t_min_c: float = 25.0               # temperature of least wear, °C
    wear_ea_hot_kj: float = 37.0             # activation energy of wear above wear_t_min_c, kJ/mol
    wear_ea_cold_kj: float = 55.0            # activation energy of wear below it, kJ/mol
    wang_d: float = -0.0067                  # rate term of the wear factor, per K and per 1/h
    wang_e: float = 2.35                     # rate term of the wear factor, per 1/h
    wear_c_ref: float = 1.0                  # rate at which the wear factor g is 1, 1/h
    merit_urgency_weight_usd_per_kwh: float = 1.0   # weight of urgency in the per-vehicle cost scores, $/kWh

    # ------------------------------------------------------------------ derived (not arguments)
    K: int = field(init=False, default=0)                     # dispatch steps per market interval
    dt_hours: float = field(init=False, default=0.0)          # one dispatch step, h
    rated_kw_max: float = field(init=False, default=0.0)      # highest charger rating, kW
    rated_kw_mean: float = field(init=False, default=0.0)     # mean charger rating, kW
    max_deviation_kw: float = field(init=False, default=0.0)  # full scale of a dispatch action, kW
    power_scale_kw: float = field(init=False, default=0.0)    # nominal fleet power: chargers x mean rating, kW
    d_scale: float = field(init=False, default=0.0)           # scale of the requirement in observations, kW

    def __post_init__(self) -> None:
        if self.market_interval_min % self.dispatch_interval_min != 0:
            raise ValueError("market_interval_min must be a multiple of dispatch_interval_min")
        if int(self.market_lead_intervals) < 0:
            raise ValueError("market_lead_intervals must be >= 0")
        if not 0.0 < float(self.plan_stay_fraction) <= 1.0:
            raise ValueError("plan_stay_fraction must be in (0, 1]")
        if self.station_limit_method not in ("projection", "exact"):
            raise ValueError(f"station_limit_method must be 'projection' or 'exact', got {self.station_limit_method!r}")
        if len(self.charger_choices) == 0:
            raise ValueError("charger_choices must not be empty")
        if not 0.0 < self.discharge_rating_frac <= 1.0:
            raise ValueError("discharge_rating_frac must be in (0, 1]")
        if not 0.0 <= self.soc_min < self.soc_target_departure < self.soc_max <= 1.0:
            raise ValueError("need 0 <= soc_min < soc_target_departure < soc_max <= 1")
        if self.dwell_types is not None:
            if len(self.dwell_types) == 0 or any(len(t) != 4 or t[1] < 0 or t[3] < 0 for t in self.dwell_types):
                raise ValueError("dwell_types must be (name, share >= 0, bias, noise >= 0) tuples")
        if self.dwell_mix is not None:
            if len(self.dwell_mix) == 0 or any(len(c) != 2 or c[0] <= 0 or c[1] < 0 for c in self.dwell_mix):
                raise ValueError("dwell_mix must be (mean stay h > 0, weight >= 0) pairs")
            w = sum(c[1] for c in self.dwell_mix)
            mean = sum(c[0] * c[1] for c in self.dwell_mix) / w
            if abs(mean - self.dwell_mean_h) > 1e-6 * max(1.0, mean):
                raise ValueError(f"dwell_mean_h ({self.dwell_mean_h}) must equal the dwell_mix mean ({mean:.6g})")
        self.K = self.market_interval_min // self.dispatch_interval_min
        self.dt_hours = self.dispatch_interval_min / 60.0
        ratings = [float(r) for r in self.charger_choices]
        self.rated_kw_max = max(ratings)
        self.rated_kw_mean = sum(ratings) / len(ratings)
        self.max_deviation_kw = 2.0 * self.rated_kw_max
        self.power_scale_kw = float(self.n_chargers) * self.rated_kw_mean
        self.d_scale = float(self.d_sigma_kw)

    def physics_fingerprint(self) -> str:
        """A short hash of every setting except the seed. Two runs with different fingerprints
        did not use the same simulator."""
        blob = ";".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if f.init and f.name != "seed")
        return hashlib.sha256(blob.encode()).hexdigest()[:12]
