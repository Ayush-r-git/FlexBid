"""The session generator: when vehicles arrive, how long they stay, and what they state.

ARRIVALS
--------
A non-homogeneous Poisson process on the clock. The intensity is a flat floor plus a morning
and an evening bump:

    lambda(h) = R * [ w_f / 24 + (1 - w_f) * (w_m * vm(h; mu_m, kappa) + (1 - w_m) * vm(h; mu_e, kappa)) ]

| symbol | meaning | units |
|---|---|---|
| h | hour of day, wraps at 24 | h |
| R | arrivals per day, `cfg.arr_rate_per_day` | vehicles per day |
| vm | von Mises density on the 24-hour circle, integrating to 1 | 1 per h |
| mu_m, mu_e | centres of the morning and evening bumps | h |
| kappa | bump concentration; larger is narrower | — |
| w_f | weight of the flat floor, solved so the night window gets `cfg.arr_night_share` of arrivals | — |
| w_m | weight of the morning bump among the two bumps | — |

The rate also varies from day to day by a factor 1 + A cos(2 pi k / P + phase), with A =
`cfg.arr_day_amplitude`, P = `cfg.arr_day_period_days` and a random phase per episode.

STAYS
-----
Lognormal with mean `cfg.dwell_mean_h` and log standard deviation `cfg.dwell_log_sd`, or a
mixture of such lognormals (`cfg.dwell_mix`).

STATED STAYS
------------
Each driver states a stay that is a shrinkage of the true one towards the fleet's median stay,
with a bias and noise set by the driver's hidden type (`cfg.dwell_types`):

    log(stated) = log(m) + b * (log(actual) - log(m)) + mu_type + sd_type * z

| symbol | meaning | units |
|---|---|---|
| m | the fleet's median stay | min |
| b | how closely the statement tracks the truth, `cfg.dwell_err_shrink`; 1 = fully | — |
| mu_type | the type's bias; negative means the driver understates | — |
| sd_type | the type's noise | — |
| z | a standard normal draw | — |

The ratio of stated to actual stay is kept within [1/R, R], R = `cfg.dwell_err_max_ratio`,
by redrawing the statements outside it.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig

MINUTES_PER_DAY = 1440
_GRID = np.arange(0.0, 24.0, 1.0 / 60.0)      # one-minute clock grid


def _von_mises(h, mu, kappa):
    """Circular density on 24 h, integrating to 1 over the day."""
    v = np.exp(kappa * np.cos(2.0 * np.pi * (h - mu) / 24.0))
    return v / (np.trapezoid(v, h) if hasattr(np, "trapezoid") else np.trapz(v, h))


def arrival_density(cfg: SimConfig, w_flat: float | None = None) -> np.ndarray:
    """Arrival density on a one-minute grid of the day, 1 per hour. `w_flat` overrides the
    solved flat weight; the solver uses it."""
    if w_flat is None:
        w_flat = solve_flat_weight(cfg)
    m = _von_mises(_GRID, cfg.arr_morning_hour, cfg.arr_kappa)
    e = _von_mises(_GRID, cfg.arr_evening_hour, cfg.arr_kappa)
    rest = 1.0 - w_flat
    dens = w_flat / 24.0 + rest * (cfg.arr_morning_weight * m
                                   + (1.0 - cfg.arr_morning_weight) * e)
    area = np.trapezoid(dens, _GRID) if hasattr(np, "trapezoid") else np.trapz(dens, _GRID)
    return dens / area


def night_share(cfg: SimConfig, w_flat: float) -> float:
    """Fraction of arrivals inside the night window, which wraps midnight."""
    d = arrival_density(cfg, w_flat)
    lo, hi = cfg.arr_night_start_hour, cfg.arr_night_end_hour
    mask = (_GRID >= lo) | (_GRID < hi) if lo > hi else (_GRID >= lo) & (_GRID < hi)
    num = np.trapezoid(d * mask, _GRID) if hasattr(np, "trapezoid") else np.trapz(d * mask, _GRID)
    return float(num)


def solve_flat_weight(cfg: SimConfig) -> float:
    """Bisect the flat weight so the night share equals `cfg.arr_night_share`. The share rises
    with the flat weight, so bisection is well posed."""
    target = float(cfg.arr_night_share)
    lo, hi = 0.0, 1.0
    if night_share(cfg, hi) < target:
        return 1.0                      # even a flat day cannot reach it
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if night_share(cfg, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def dwell_params(cfg: SimConfig) -> tuple[float, float]:
    """(mu, sigma) of the lognormal stay, in log hours, so that its mean is `cfg.dwell_mean_h`."""
    m, s = float(cfg.dwell_mean_h), float(cfg.dwell_log_sd)
    return float(np.log(m) - 0.5 * s * s), s


def day_rate_factor(cfg: SimConfig, day_index: np.ndarray,
                    phase: float = 0.0) -> np.ndarray:
    """Day-to-day factor on the arrival rate, mean 1: 1 + A cos(2 pi k / P + phase).

    | symbol | meaning | units |
    |---|---|---|
    | k | day index within the episode | days |
    | A | `cfg.arr_day_amplitude`; 0 means no day-to-day variation | — |
    | P | `cfg.arr_day_period_days` | days |
    | phase | drawn per episode, so the busy days fall at a different place each time | rad |
    """
    A = float(cfg.arr_day_amplitude)
    if A == 0.0:
        return np.ones_like(np.asarray(day_index, dtype=float))
    P = float(cfg.arr_day_period_days)
    return 1.0 + A * np.cos(2.0 * np.pi * np.asarray(day_index, float) / P + phase)


def sample_synth_schedule(cfg: SimConfig, rng: np.random.Generator,
                          start_minute: float):
    """Draw every session of the episode, including the vehicles already plugged in at the start.
    Every draw comes from `rng`, so one seed reproduces the episode exactly."""
    from .fleet import SessionSchedule, arrival_soc, draw_capacity

    dt_min = float(cfg.dispatch_interval_min)
    episode_min = float(cfg.episode_length) * float(cfg.market_interval_min)
    days = episode_min / MINUTES_PER_DAY

    dens = arrival_density(cfg)
    cdf = np.cumsum(dens); cdf /= cdf[-1]
    mu_d, sd_d = dwell_params(cfg)

    # The count first, then the times: exact for a non-homogeneous Poisson process. Arrivals
    # are also drawn for `arr_history_days` before the start, so the chargers are already in
    # use at t = 0. The extra day slot covers the shift by `start_minute`.
    hist_days = float(cfg.arr_history_days)
    n_slots = int(np.ceil(days + hist_days)) + 1
    n = int(rng.poisson(cfg.arr_rate_per_day * n_slots))
    if n == 0:
        z = np.zeros(0)
        return SessionSchedule(z.astype(np.int64), z.astype(np.int64),
                               z.astype(np.int64), z, z, z)

    hod = np.interp(rng.random(n), cdf, _GRID)              # hour of day
    # Day slot, weighted by the multi-day factor so the rate varies across
    # days as well as across hours.
    day_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    w_day = day_rate_factor(cfg, np.arange(n_slots) - hist_days, day_phase)
    w_day = np.maximum(w_day, 0.0)
    day_idx = rng.choice(n_slots, size=n, p=w_day / w_day.sum())
    # Absolute minute relative to the episode start, negative = before it.
    abs_min = (day_idx - hist_days) * MINUTES_PER_DAY + hod * 60.0 - start_minute

    # Mixture of stays: each arrival draws its component by weight, then a lognormal with that
    # component's mean. mu_d, from the overall mean, still sets the fleet's median stay.
    mix = cfg.dwell_mix
    if mix is not None:
        from dataclasses import replace
        mus = np.array([dwell_params(replace(cfg, dwell_mean_h=float(m), dwell_mix=None))[0] for m, _ in mix])
        w = np.array([float(c[1]) for c in mix]); w = w / w.sum()
        comp = rng.choice(len(mix), size=n, p=w)
        mu_i = mus[comp]
    else:
        mu_i = mu_d
    dwell_h = np.exp(mu_i + sd_d * rng.standard_normal(n))
    dwell_min = dwell_h * 60.0
    still_here = abs_min + dwell_min > 0.0
    within = abs_min < episode_min
    keep = still_here & within
    abs_min, dwell_min = abs_min[keep], dwell_min[keep]
    n = int(keep.sum())
    if n == 0:
        z = np.zeros(0)
        return SessionSchedule(z.astype(np.int64), z.astype(np.int64),
                               z.astype(np.int64), z, z, z)

    socs = arrival_soc(rng, n, cfg)
    caps = np.array([draw_capacity(cfg, rng) for _ in range(n)])
    kwh = np.maximum(cfg.soc_target_departure - socs, 0.0) * caps

    # Vehicles already on the plug at t = 0 enter at step 0 with what is left.
    started_before = abs_min < 0.0
    frac_left = np.where(started_before,
                         np.clip((abs_min + dwell_min) / dwell_min, 0.0, 1.0), 1.0)
    rem_act = np.where(started_before, abs_min + dwell_min, dwell_min)
    kwh = kwh * frac_left
    socs = np.where(started_before, cfg.soc_target_departure - kwh / caps, socs)

    # stated stay, see the module docstring; the driver types are drawn after the true stays
    b = float(cfg.dwell_err_shrink)
    ref_min = float(np.exp(mu_d)) * 60.0
    types = cfg.dwell_types
    if types:
        w = np.array([t[1] for t in types], dtype=np.float64)
        kind = rng.choice(len(types), size=n, p=w / w.sum())
        err_mu = np.array([t[2] for t in types], dtype=np.float64)[kind]
        err_sd = np.array([t[3] for t in types], dtype=np.float64)[kind]
    else:
        kind = None
        err_mu, err_sd = cfg.dwell_err_log_mean, cfg.dwell_err_log_sd
    err_sd = np.broadcast_to(np.asarray(err_sd, dtype=np.float64), (n,))
    base = (np.log(ref_min) + b * (np.log(rem_act) - np.log(ref_min))
            + err_mu)
    log_est = base + err_sd * rng.standard_normal(n)

    # keep 1/R <= stated/actual <= R by redrawing the statements outside it
    R = cfg.dwell_err_max_ratio
    if R is not None:
        lim, la = float(np.log(R)), np.log(rem_act)
        for _ in range(60):
            bad = np.abs(log_est - la) > lim
            if not bad.any():
                break
            log_est[bad] = (base[bad]
                            + err_sd[bad] * rng.standard_normal(int(bad.sum())))
        log_est = np.clip(log_est, la - lim, la + lim)
    rem_est = np.maximum(np.exp(log_est), dt_min)

    arrival = np.maximum(np.floor(np.maximum(abs_min, 0.0) / dt_min), 0).astype(np.int64)
    order = np.argsort(arrival, kind="stable")

    # A driver asks for no more than its own charger can deliver in the stated stay, times
    # `cfg.synth_request_margin`. Ratings follow the same cycle `fleet._admit` uses.
    choices = tuple(cfg.charger_choices)
    p_kw_sorted = np.array([choices[k % len(choices)] for k in range(n)], dtype=np.float64)
    p_kw = np.empty(n); p_kw[order] = p_kw_sorted
    deliverable = float(cfg.synth_request_margin) * p_kw * float(cfg.charge_efficiency) * rem_est / 60.0
    kwh = np.minimum(kwh, deliverable)
    return SessionSchedule(
        arrival_step=arrival[order],
        dwell_act_steps=np.maximum(np.ceil(rem_act / dt_min), 1).astype(np.int64)[order],
        dwell_est_steps=np.maximum(np.ceil(rem_est / dt_min), 1).astype(np.int64)[order],
        energy_kwh=kwh[order],
        capacity_kwh=caps[order],
        soc_arrival=np.clip(socs, cfg.soc_min, 1.0)[order],
        driver_type=None if kind is None else kind[order],
    )
