"""What the learned policies observe.

Both observations are built from the controller's view (`sim.cfleet`) and the market's public
outcome only. Neither reads a hidden battery parameter, a driver's type or a true departure.

**Bid observation.** `bid_obs_size(cfg)` numbers, the same for any fleet size:

| part | numbers | what |
|---|---|---|
| `sim.market_obs()` | 18 | time of day, the last market outcome, the fleet's room and urgency, its cost curve |
| `departure_features` | 4 | how many vehicles are near or past their stated departure |
| `fixed_commitments` | L | commitments already fixed for intervals t .. t+L-1, under a bid lead of L intervals |

**Dispatch observation.** A flat vector, read by the deep set (`flexbid.policies.nets`):

    [ vehicle features (n x 13), connected mask (n), global features (9), extra inputs (k) ]

| symbol | meaning | units |
|---|---|---|
| n | number of chargers | — |
| k | number of extra inputs: 0, or 4 when the dispatch sees the commitments | — |
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ the bid
N_BID_OBS = 22          # 18 market numbers and 4 departure features


def bid_obs_size(cfg):
    """Width of the bid observation: 22, plus one per interval of bid lead."""
    return N_BID_OBS + int(cfg.market_lead_intervals)


def departure_features(sim):
    """Four departure features, from stated departures and the fitted departure model only.

    | # | feature | units |
    |---|---|---|
    | 0 | share of connected vehicles past 60 % of their own stated stay | — |
    | 1 | share of connected vehicles past their stated departure | — |
    | 2 | energy still owed by the vehicles in feature 0, over the fleet's total pack energy | — |
    | 3 | mean time past the stated departure over the vehicles that overstay, over the mean stated stay | — |
    """
    from .departures import DepartureModel
    f, cfg = sim.cfleet, sim.cfg
    c = f.connected
    if not c.any() or f.elapsed_steps is None:
        return np.zeros(4, dtype=np.float32)
    scale = DepartureModel.fit(cfg).scale_steps(cfg)
    stated = np.maximum(f.stated_steps[c], 1).astype(np.float64)
    ratio = f.elapsed_steps[c] / stated
    over = f.overstay_steps[c].astype(np.float64)
    owed = (np.maximum(f.target_soc - f.soc, 0.0) * f.battery_kwh)[c]
    past60 = ratio >= 0.6
    return np.array([past60.mean(), (over > 0).mean(),
                     float(owed[past60].sum()) / float(f.battery_kwh.sum()),
                     float(over[over > 0].mean() / scale) if (over > 0).any() else 0.0],
                    dtype=np.float32)


def fixed_commitments(sim, offsets):
    """Commitments already fixed for intervals t + j, j in `offsets`, over `cfg.power_scale_kw`.

    0 where none is fixed. Read it before or after the interval's market step: it always
    returns the intervals still to be delivered.
    """
    scale = max(sim.cfg.power_scale_kw, 1e-9)
    out = []
    for j in offsets:
        cl = sim.pending.get(sim.t + j)
        if cl is None and j == 0 and sim.phase == "dispatch":
            cl = sim.clearing            # interval t's commitment, already taken from the queue
        out.append(cl.commitment_kw / scale if cl is not None else 0.0)
    return np.asarray(out, dtype=np.float32)


def bid_observation(sim):
    """The bid's observation at the start of interval t, before its bid."""
    parts = [sim.market_obs(), departure_features(sim)]
    L = int(sim.cfg.market_lead_intervals)
    if L > 0:
        parts.append(fixed_commitments(sim, range(L)))
    return np.concatenate(parts).astype(np.float32)


# ------------------------------------------------------------------ the dispatch
CAR_FEATURES = ("soc", "owed", "time_left", "time_spent", "lim_c", "lim_d", "temp",
                "last_p", "rating", "station_load", "plan_rate", "urgency", "pack")
N_CAR = len(CAR_FEATURES)
GLOBAL_FEATURES = ("commitment", "band_hi", "band_lo", "price", "sin_h", "cos_h", "substep",
                   "occupied", "episode_t")
N_GLOB = len(GLOBAL_FEATURES)

#: extra inputs of the dispatch network: name -> number of inputs
DISPATCH_EXTRA = {None: 0, "commitments": 4}
#: the commitment inputs have this fixed width, whatever the lead: the last requirement, then
#: the commitments fixed for t+1 .. t+3 (0 where none is fixed)
COMMITMENT_INPUTS = 4


def car_features(sim):
    """(n, 13) per-vehicle features and the (n,) connected mask.

    | feature | meaning | scale |
    |---|---|---|
    | soc | state of charge | — |
    | owed | target state of charge minus state of charge, at least 0 | — |
    | time_left | dispatch steps to the stated departure | / 144 (12 h) |
    | time_spent | steps since arrival over stated steps | — |
    | lim_c, lim_d | reported charge and discharge limits over the charger rating | — |
    | temp | estimated temperature above ambient | / 10 K |
    | last_p | last power over the charger rating | — |
    | rating | charger rating | / 150 kW |
    | station_load | last power of the vehicle's station over the station bus limit | — |
    | plan_rate | baseline plan over the charger rating | — |
    | urgency | power needed to reach the target by the stated departure, over the rating, at most 3 | — |
    | pack | pack size | / 90 kWh |

    Every feature is a share or a scaled quantity, so one network reads any fleet. Empty
    chargers read 0.
    """
    from flexbid.sim.physics import baseline_charge, station_index
    cfg, f = sim.cfg, sim.cfleet
    n, c = cfg.n_chargers, f.connected
    rating = np.maximum(f.max_charge_kw, 1e-9)
    owed = np.maximum(f.target_soc - f.soc, 0.0)
    left = np.maximum(f.est_departure_steps, 0) / 144.0
    spent = f.elapsed_steps / np.maximum(f.stated_steps, 1)
    lim_c = (f.charge_cap_kw / rating) if f.charge_cap_kw is not None else np.ones(n)
    lim_d = (f.discharge_cap_kw / rating) if f.discharge_cap_kw is not None else np.ones(n)
    temp = ((f.temp_hat_k - 273.15 - cfg.t_amb_c) / 10.0) if f.temp_hat_k is not None else np.zeros(n)
    ld = sim.last_dispatch
    P = ld.power_kw if ld is not None else np.zeros(n)
    st = station_index(n, cfg)
    cap = cfg.station_bus_kw or (cfg.station_size * cfg.rated_kw_mean)
    load = np.bincount(st, weights=np.where(c, P, 0.0), minlength=int(st.max()) + 1) / cap
    cb = baseline_charge(f, cfg)
    steps = np.maximum(f.est_departure_steps.astype(np.float64), 1.0)
    req = owed * f.battery_kwh / (cfg.charge_efficiency * steps * cfg.dt_hours + 1e-9)
    X = np.stack([f.soc, owed, left, spent, lim_c, lim_d, temp, P / rating,
                  f.max_charge_kw / 150.0, load[st], cb / rating, np.minimum(req / rating, 3.0),
                  f.battery_kwh / 90.0], axis=1)
    X = np.where(c[:, None], np.clip(X, -10.0, 10.0), 0.0)
    return X.astype(np.float32), c.astype(np.float32)


def global_features(sim):
    """9 fleet-wide numbers: this interval's commitment, its deliverable bounds and price,
    the time of day, the step inside the interval, the share of chargers occupied, and the
    time since the episode started."""
    cfg, cl = sim.cfg, sim.clearing
    scale = max(cfg.power_scale_kw, 1e-9)
    hour = ((sim.start_minute + sim.t * cfg.market_interval_min) % 1440) / 1440.0 * 2 * np.pi
    q = cl.commitment_kw / scale if cl is not None else 0.0
    hi = cl.commitment_bound_hi_kw / scale if cl is not None else 0.0
    lo = -cl.commitment_bound_lo_kw / scale if cl is not None else 0.0
    price = cl.price / max(cfg.p_max, 1e-9) if cl is not None else 0.0
    occ = float(np.mean(sim.cfleet.connected))
    g = [q, hi, lo, price, np.sin(hour), np.cos(hour), sim.k / cfg.K, occ, sim.t / cfg.episode_length]
    return np.clip(np.asarray(g, dtype=np.float32), -10.0, 10.0)


def dispatch_extra(sim, extra):
    """The extra inputs of the dispatch network, or an empty array.

    "commitments": the last requirement as the bid sees it (`market_obs()[2]`), then the
    commitments already fixed for t+1 .. t+3. The same information about the next hour that
    the bid has, given to the dispatch network instead of a message.
    """
    if extra is None:
        return np.zeros(0, dtype=np.float32)
    if extra == "commitments":
        req = np.float32(sim.d_prev / sim.cfg.d_scale)
        return np.concatenate([[req], fixed_commitments(sim, range(1, COMMITMENT_INPUTS))]).astype(np.float32)
    raise ValueError(f"unknown dispatch extra {extra!r}; use one of {list(DISPATCH_EXTRA)}")


def dispatch_observation(sim, extra=None):
    """[vehicles (n x 13), mask (n), globals (9), extra inputs], flat. The deep set finds n from the width."""
    X, m = car_features(sim)
    x = np.concatenate([X.reshape(-1), m, global_features(sim)]).astype(np.float32)
    if extra is None:
        return x
    return np.concatenate([x, np.clip(dispatch_extra(sim, extra), -10.0, 10.0)]).astype(np.float32)


def dispatch_obs_size(cfg, extra=None):
    return cfg.n_chargers * (N_CAR + 1) + N_GLOB + DISPATCH_EXTRA[extra]


def split_dispatch_obs(flat, n_extra=0):
    """Inverse of `dispatch_observation` for a batch: (B, w) -> X (B, n, 13), mask (B, n),
    globals (B, 9 + n_extra). The extra inputs are appended to the globals."""
    w = flat.shape[-1] - n_extra
    n = (w - N_GLOB) // (N_CAR + 1)
    X = flat[..., : n * N_CAR].reshape(*flat.shape[:-1], n, N_CAR)
    m = flat[..., n * N_CAR: n * N_CAR + n]
    g = flat[..., n * N_CAR + n:]
    return X, m, g
