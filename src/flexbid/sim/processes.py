"""The system requirement d_t, sampled once per episode.

d_t is built from real CAISO net load: 42 days in six seasonal windows of 2025, at 5-minute
resolution, stored as a z-scored anomaly in `data/caiso_netload.npz`. Each episode draws a
window and a start time, averages the trace over each market interval, scales it by
`cfg.d_sigma_kw` (kW) and adds noise of standard deviation `cfg.d_noise_std` (kW). The
anomaly is two-sided: in surplus hours d_t < 0 and the market asks for more consumption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SimConfig

MINUTES_PER_DAY = 1440
_DATA_DIR = Path(__file__).resolve().parent / "data"
_TRACE_CACHE: dict = {}


def time_features(
    cfg: SimConfig, t: int, k: int, start_minute: float
) -> tuple[float, float]:
    """(sin, cos) of the time of day at interval t, dispatch step k."""
    minute = (
        start_minute + t * cfg.market_interval_min + k * cfg.dispatch_interval_min
    ) % MINUTES_PER_DAY
    theta = 2.0 * np.pi * minute / MINUTES_PER_DAY
    return float(np.sin(theta)), float(np.cos(theta))


class DemandTrace:
    """The packaged CAISO net-load anomaly windows (z-scored, 5-minute steps)."""

    def __init__(self, path: Path):
        raw = np.load(path, allow_pickle=False)
        self.minutes_per_step = int(raw["minutes_per_step"])
        self.windows = [
            raw[f"window_{i}"].astype(np.float64)
            for i in range(int(raw["n_windows"]))
        ]

    @classmethod
    def packaged(cls) -> "DemandTrace":
        path = _DATA_DIR / "caiso_netload.npz"
        if str(path) not in _TRACE_CACHE:
            _TRACE_CACHE[str(path)] = cls(path)
        return _TRACE_CACHE[str(path)]

    def sample_episode(
        self, cfg: SimConfig, rng: np.random.Generator
    ) -> tuple[np.ndarray, float]:
        """Draw (d_t per market interval in kW, start minute of the day).

        With `cfg.episode_start_hour` set, the episode starts at that hour on a random day of
        the window; with None, at a random 5-minute step."""
        if cfg.market_interval_min % self.minutes_per_step != 0:
            raise ValueError(
                "market_interval_min must be a multiple of the trace resolution"
            )
        steps_per_interval = cfg.market_interval_min // self.minutes_per_step
        needed = cfg.episode_length * steps_per_interval

        w = self.windows[int(rng.integers(len(self.windows)))]
        if len(w) < needed:
            raise ValueError("demand trace window shorter than the episode")
        hi = len(w) - needed + 1
        if getattr(cfg, "episode_start_hour", None) is None:
            start = int(rng.integers(hi))
        else:
            # offsets that land on the requested hour are a whole day apart
            steps_per_day = MINUTES_PER_DAY // self.minutes_per_step
            off = int(round(float(cfg.episode_start_hour) * 60.0
                            / self.minutes_per_step)) % steps_per_day
            cand = np.arange(off, hi, steps_per_day)
            if len(cand) == 0:
                raise ValueError(
                    f"episode_start_hour={cfg.episode_start_hour} leaves no "
                    f"usable offset: the window holds {len(w)} steps and the "
                    f"episode needs {needed}"
                )
            start = int(cand[rng.integers(len(cand))])

        z = w[start : start + needed].reshape(cfg.episode_length, steps_per_interval)
        d = z.mean(axis=1) * cfg.d_sigma_kw
        d = d + rng.normal(0.0, cfg.d_noise_std, size=cfg.episode_length)
        start_minute = float((start * self.minutes_per_step) % MINUTES_PER_DAY)
        return d.astype(np.float64), start_minute


def sample_demand(cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """The episode's requirement per market interval (kW) and its start minute of the day."""
    return DemandTrace.packaged().sample_episode(cfg, rng)


def sample_competitor_bids(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """An AR(1) series of length `cfg.episode_length`. The market does not use it. The
    simulator still draws it at every reset, because it advances the rivals' random stream,
    and every episode after it depends on that stream's position."""
    T = cfg.episode_length
    B = np.zeros(T, dtype=np.float64)
    B[0] = cfg.b_other_base + rng.normal(0.0, cfg.b_other_noise_std)
    for t in range(1, T):
        B[t] = cfg.b_other_base + cfg.b_other_rho * (B[t - 1] - cfg.b_other_base) \
            + rng.normal(0.0, cfg.b_other_noise_std)
    return B
