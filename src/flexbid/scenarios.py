"""Scenarios: one fleet and one market setting, turned into a simulator configuration.

    from flexbid import make_config
    cfg = make_config("hub-6h")                                     # no bid lead
    cfg = make_config("hub-6h", market="lead", market_lead_min=60)  # 60-minute bid lead
    cfg = make_config("hub-12h", n_chargers=100, undelivered_energy_price=1.5)

Any keyword that is not a scenario argument is passed to `SimConfig` unchanged, so every
simulator setting can be changed here.

Fleets. The arrival rate keeps 75 % of chargers occupied on average:
arrivals per day = 0.75 x chargers x 24 / mean stay.

| fleet | chargers | mean stay | arrivals per day |
|---|---|---|---|
| `hub-6h` | 50 | 6 h | 150 |
| `hub-12h` | 50 | 12 h | 75 |
| `hub-mixed` | 50 | half 6 h, half 12 h (mean 9 h) | 100 |
| `depot-15` | 15 | 6 h | 45 |

Markets.

| setting | `base` | `lead` |
|---|---|---|
| bid lead | 0 min | 0 min unless `market_lead_min` is given |
| station limits in the dispatch QP | projection | exact |
| share of the stated stay the baseline plan uses | 1.0 | 0.8 |
"""
from __future__ import annotations

from .sim.config import SimConfig

OCCUPANCY = 0.75                          # mean share of chargers occupied
MIXED_STAYS = ((6.0, 0.5), (12.0, 0.5))   # (mean stay in h, share of drivers)

#: name: (chargers, mean stay in h, mixture of stays or None)
FLEETS = {
    "hub-6h": (50, 6.0, None),
    "hub-12h": (50, 12.0, None),
    "hub-mixed": (50, 9.0, MIXED_STAYS),
    "depot-15": (15, 6.0, None),
}

#: name: settings of that market
MARKETS = {
    "base": dict(station_limit_method="projection", plan_stay_fraction=1.0),
    "lead": dict(station_limit_method="exact", plan_stay_fraction=0.8),
}

MARKET_INTERVAL_MIN = 15


def arrivals_per_day(n_chargers: int, mean_stay_h: float) -> float:
    """Arrivals per day that keep OCCUPANCY of the chargers occupied on average."""
    return OCCUPANCY * n_chargers * 24.0 / mean_stay_h


def make_config(fleet: str = "hub-6h", market: str = "base", *, market_lead_min: int = 0,
                n_chargers: int | None = None, **overrides) -> SimConfig:
    """The configuration of one scenario.

    | argument | meaning | units |
    |---|---|---|
    | `fleet` | one of `FLEETS` | — |
    | `market` | `"base"` or `"lead"`, see `MARKETS` | — |
    | `market_lead_min` | bid lead, a multiple of 15: the bid placed at interval t is cleared at once for interval t + lead / 15 | min |
    | `n_chargers` | fleet size; the arrival rate scales with it | — |
    | `overrides` | any `SimConfig` field | — |
    """
    if fleet not in FLEETS:
        raise KeyError(f"unknown fleet {fleet!r}; known: {sorted(FLEETS)}")
    if market not in MARKETS:
        raise KeyError(f"unknown market {market!r}; known: {sorted(MARKETS)}")
    if market_lead_min < 0 or market_lead_min % MARKET_INTERVAL_MIN:
        raise ValueError("market_lead_min must be a non-negative multiple of 15")
    n, stay, mix = FLEETS[fleet]
    n = int(n_chargers) if n_chargers is not None else n
    kw = dict(MARKETS[market])
    kw.update(n_chargers=n, dwell_mean_h=stay, dwell_mix=mix, arr_rate_per_day=arrivals_per_day(n, stay),
              market_lead_intervals=market_lead_min // MARKET_INTERVAL_MIN)
    kw.update(overrides)
    return SimConfig(**kw)
