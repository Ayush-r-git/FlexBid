"""The logistic bid curve, the price solver and the rival aggregators.

Every participant bids

    G(p) = -H + (Q + H) * sigma(beta * (p - p0))

| symbol | meaning | units |
|---|---|---|
| p | price | price units |
| Q | most the participant offers to reduce charging | kW |
| H | most the participant offers to increase charging | kW |
| beta | steepness of the curve | 1 per price unit |
| p0 | the anchor: the price at the middle of the curve | price units |
| G(p) | quantity the participant must deliver at price p, always in [-H, Q] | kW |
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SimConfig

__all__ = ["LogisticBid", "sigma", "G", "dG", "S", "dS", "band", "clear_price", "ClearFailure",
           "CompetitorPool"]


def sigma(x):
    """The logistic function, safe for large |x|."""
    x = np.clip(np.asarray(x, dtype=float), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class LogisticBid:
    """One participant's curve. Fields as in the module table."""

    Q: float
    H: float
    beta: float
    p0: float


def G(bid: LogisticBid, p: float) -> float:
    """Committed quantity at price p, kW."""
    return float(-bid.H + (bid.Q + bid.H) * sigma(bid.beta * (p - bid.p0)))


def dG(bid: LogisticBid, p: float) -> float:
    """Slope of one curve, kW per price unit."""
    s = sigma(bid.beta * (p - bid.p0))
    return float((bid.Q + bid.H) * bid.beta * s * (1.0 - s))


def S(bids, p: float) -> float:
    """Sum of all curves at price p, kW."""
    return float(sum(G(b, p) for b in bids))


def dS(bids, p: float) -> float:
    """Slope of the sum, kW per price unit."""
    return float(sum(dG(b, p) for b in bids))


def band(bids) -> tuple[float, float]:
    """The open range of quantities the market can clear, kW: (-sum H, sum Q)."""
    return -float(sum(b.H for b in bids)), float(sum(b.Q for b in bids))


class ClearFailure(RuntimeError):
    """The requirement lies outside the market's range, so no price exists (inelastic demand only)."""


def clear_price(bids, d_t: float, tol: float = 1e-9, max_iter: int = 200,
                elasticity: float = 0.0) -> tuple[float, int]:
    """Solve S(p) + elasticity * p = d_t for p. Returns (p, iterations).

    | argument | meaning | units |
    |---|---|---|
    | `d_t` | the requirement | kW |
    | `elasticity` | price response of the requirement | kW per price unit |

    S is strictly increasing, so the root is unique. Newton steps are kept inside a
    bracket and fall back to bisection, because the slope underflows to exactly 0 in
    the flat tails of the curves.
    """
    eps = float(elasticity)
    lo_q, hi_q = band(bids)
    if eps <= 0.0 and not (lo_q < d_t < hi_q):
        raise ClearFailure(f"d_t={d_t:.3f} kW outside the market's range ({lo_q:.3f}, {hi_q:.3f}) kW")

    # bracket the root by doubling outward from the mean anchor
    c = float(np.mean([b.p0 for b in bids])) if bids else 0.0
    lo, hi = c - 1.0, c + 1.0
    F = lambda q: S(bids, q) + eps * q
    for _ in range(200):
        if F(lo) <= d_t:
            break
        lo -= max(1.0, abs(lo))
    for _ in range(200):
        if F(hi) >= d_t:
            break
        hi += max(1.0, abs(hi))

    p = 0.5 * (lo + hi)
    for it in range(1, max_iter + 1):
        f = F(p) - d_t
        if abs(f) <= tol:
            return float(p), it
        if f > 0.0:
            hi = p
        else:
            lo = p
        slope = dS(bids, p) + eps
        step = p - f / slope if slope > 0.0 else 0.5 * (lo + hi)
        p = step if lo < step < hi else 0.5 * (lo + hi)
    return float(p), max_iter


@dataclass
class CompetitorPool:
    """The rival aggregators. Each has its own curve; their anchors follow a random path.

    Each rival's anchor follows an AR(1) path drawn in full at the start of the episode:

        p0_j(t+1) = mu_j + phi * (p0_j(t) - mu_j) + sd * sqrt(1 - phi^2) * z

    | symbol | meaning | units |
    |---|---|---|
    | mu_j | rival j's mean anchor, drawn once per episode | price units |
    | phi | persistence of the path, `cfg.comp_exo_phi` | — |
    | sd | spread of the path, `cfg.comp_exo_sd` | price units |
    | z | a standard normal draw | — |

    The path is fixed before the first bid, so nothing this fleet does can move the rivals.
    """

    Q: np.ndarray            # each rival's room to reduce charging, kW
    H: np.ndarray            # each rival's room to increase charging, kW
    beta: np.ndarray         # each rival's steepness, 1 per price unit
    p0: np.ndarray           # each rival's current anchor, price units
    path: np.ndarray = field(default=None, repr=False)   # anchor path, shape (T + 1, n)
    _t: int = 0

    @classmethod
    def build(cls, cfg: SimConfig, rng: np.random.Generator, own_capacity_kw: float) -> "CompetitorPool":
        """Draw the rivals. Each rival's size is `cfg.comp_capacity_scale` times this fleet's
        nominal capacity `own_capacity_kw` (kW), varied by +-`cfg.comp_heterogeneity`."""
        n = int(cfg.n_competitors)
        scale = float(cfg.comp_capacity_scale) * float(own_capacity_kw)
        spread = rng.uniform(1.0 - cfg.comp_heterogeneity, 1.0 + cfg.comp_heterogeneity, size=n)
        beta = float(cfg.comp_beta) * rng.uniform(1.0 - cfg.comp_heterogeneity,
                                                  1.0 + cfg.comp_heterogeneity, n)
        # This draw is overwritten by the path below, but it advances the stream. It is kept so
        # that every random number after it, and so every published result, stays the same.
        rng.normal(0.0, float(cfg.comp_p0_init_sd), n)
        g = np.random.default_rng(int(rng.integers(2**62)))
        T = int(cfg.episode_length) + 1
        phi, sd = float(cfg.comp_exo_phi), float(cfg.comp_exo_sd)
        mu = g.normal(float(cfg.comp_exo_mean), float(cfg.comp_exo_mean_sd), n)
        path = np.empty((T, n))
        path[0] = mu + sd * g.standard_normal(n)
        innov = sd * np.sqrt(max(1.0 - phi * phi, 0.0))
        for t in range(1, T):
            path[t] = mu + phi * (path[t - 1] - mu) + innov * g.standard_normal(n)
        return cls(Q=scale * spread, H=scale * spread * float(cfg.comp_down_ratio), beta=beta,
                   p0=path[0].copy(), path=path)

    def bids(self) -> list[LogisticBid]:
        return [LogisticBid(float(q), float(h), float(b), float(a))
                for q, h, b, a in zip(self.Q, self.H, self.beta, self.p0)]

    def advance(self) -> None:
        """Move every rival one step along its anchor path. Called after each clearing."""
        self._t = min(self._t + 1, len(self.path) - 1)
        self.p0 = self.path[self._t].copy()
