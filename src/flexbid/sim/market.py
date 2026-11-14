"""Market clearing and the money of one dispatch step.

Every participant, this fleet and its rivals, bids a logistic supply curve:

    G_i(p) = -H_i + (Q_i + H_i) * sigma(beta_i * (p - p0_i))

| symbol | meaning | units |
|---|---|---|
| G_i(p) | flexibility participant i delivers at price p; positive means less charging | kW |
| Q_i | most it offers to reduce charging | kW |
| H_i | most it offers to increase charging | kW |
| beta_i | steepness of the curve | 1 per price unit |
| p0_i | the anchor: the price at the middle of the curve | price units |
| sigma | the logistic function, 1 / (1 + exp(-x)) | — |

The market clears at the price p* where the curves meet the requirement, after the
requirement's own response to price:

    sum_i G_i(p*) = d_t - eps * p*

| symbol | meaning | units |
|---|---|---|
| d_t | the system requirement of interval t, signed | kW |
| eps | price response of the requirement, `cfg.d_elasticity_kw_per_pu` | kW per price unit |
| p* | the clearing price | price units |

One price unit is worth `cfg.price_unit` dollars per kWh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig


@dataclass
class ClearingResult:
    """One market clearing.

    | field | meaning | units |
    |---|---|---|
    | `bid` | this fleet's anchor p0 | price units |
    | `price` | clearing price p*, clipped to [p_min, p_max] | price units |
    | `commitment_kw` | this fleet's commitment G(p*) | kW |
    | `commitment_bound_lo_kw`, `commitment_bound_hi_kw` | -H and Q as offered | kW |
    | `d_t` | the system requirement before its price response | kW |
    | `B_minus_i` | the residual slope this fleet faces, S'(p*) + eps - G'(p*), where S is the sum of all curves | kW per price unit |
    | `anchor`, `beta` | the offered curve's anchor and steepness | price units, 1 per price unit |
    | `Q_kw`, `H_kw` | the offered room in each direction | kW |
    | `newton_iters` | iterations the price solve used | — |
    | `d_clamped` | True if the requirement was pulled inside the market's range (inelastic demand only) | — |
    | `d_cleared_kw` | the quantity the market cleared, d_t - eps * p*; this is what is published | kW |
    """

    bid: float
    price: float
    commitment_kw: float
    commitment_bound_lo_kw: float
    commitment_bound_hi_kw: float
    d_t: float
    B_minus_i: float
    anchor: float | None = None
    beta: float | None = None
    Q_kw: float | None = None
    H_kw: float | None = None
    newton_iters: int | None = None
    d_clamped: bool = False
    d_cleared_kw: float | None = None


def revenue_usd(price: float, realized_flex_kw: float, cfg: SimConfig) -> float:
    """Payment for one dispatch step, $: price x price_unit x delivered flexibility x step length.
    With a negative price, delivering negative flexibility (charging more) earns money."""
    return price * cfg.price_unit * realized_flex_kw * cfg.dt_hours


def imbalance_charge_usd(realized_flex_kw: float, commitment_kw: float, cfg: SimConfig) -> float:
    """Charge for missing the commitment in one dispatch step, $: imbalance price x |gap| x step length."""
    return cfg.imbalance_price * abs(realized_flex_kw - commitment_kw) * cfg.dt_hours


def unmet_energy_charge_usd(unmet_kwh: float, cfg: SimConfig) -> float:
    """Charge for energy departing vehicles did not receive, $."""
    return cfg.undelivered_energy_price * unmet_kwh


def clear_logistic(anchor_normalized: float, Q_kw: float, H_kw: float, rival_bids, d_t: float,
                   cfg: SimConfig, *, q_frac: float = 1.0, h_frac: float = 1.0,
                   beta_mult: float = 1.0, anchor_ref: float = 0.0) -> ClearingResult:
    """Clear one interval.

    | argument | meaning | units |
    |---|---|---|
    | `anchor_normalized` | the bid's anchor, in [-1, 1]; the anchor is `anchor_ref` + this x `cfg.anchor_scale` | — |
    | `Q_kw`, `H_kw` | the fleet's deliverable room in each direction | kW |
    | `rival_bids` | the rivals' `LogisticBid` curves | — |
    | `d_t` | the system requirement, signed | kW |
    | `q_frac`, `h_frac` | share of each side that is offered, in [0, 1] | — |
    | `beta_mult` | multiplier on the curve's steepness `cfg.logistic_beta` | — |
    | `anchor_ref` | the price the anchor is measured from (the last cleared price) | price units |

    The price is found by safeguarded Newton steps. The commitment is read off the
    fleet's own curve at the clipped price, so it always lies in [-H, Q].
    """
    from .logistic import LogisticBid, G as _G, dG as _dG, dS as _dS, band as _band, clear_price

    p0 = float(anchor_ref) + float(np.clip(anchor_normalized, -1.0, 1.0)) * cfg.anchor_scale
    qf = float(np.clip(q_frac, 0.0, 1.0))
    hf = float(np.clip(h_frac, 0.0, 1.0))
    beta_eff = max(float(cfg.logistic_beta) * float(beta_mult), 1e-6)
    Q_bid, H_bid = float(Q_kw) * qf, float(H_kw) * hf
    own = LogisticBid(max(Q_bid, 1e-6), max(H_bid, 1e-6), beta_eff, p0)
    bids = [own] + list(rival_bids)

    eps = float(cfg.d_elasticity_kw_per_pu or 0.0)
    lo_q, hi_q = _band(bids)
    if eps > 0.0:
        # elastic demand: a clearing price always exists
        d_used, clamped = float(d_t), False
    else:
        # inelastic demand: a requirement outside the market's range has no price,
        # so it is pulled just inside the range
        margin = 1e-6 * max(1.0, hi_q - lo_q)
        d_used = float(np.clip(d_t, lo_q + margin, hi_q - margin))
        clamped = d_used != float(d_t)

    p_raw, iters = clear_price(bids, d_used, elasticity=eps)
    price = float(np.clip(p_raw, cfg.p_min, cfg.p_max))
    q_committed = _G(own, price)
    d_cleared = float(d_used) - eps * price
    residual_slope = _dS(bids, price) + eps - _dG(own, price)

    return ClearingResult(
        bid=float(p0), price=price, commitment_kw=q_committed,
        commitment_bound_lo_kw=-H_bid, commitment_bound_hi_kw=Q_bid, d_t=float(d_t),
        B_minus_i=float(residual_slope), anchor=p0, beta=beta_eff, Q_kw=Q_bid, H_kw=H_bid,
        newton_iters=int(iters), d_clamped=bool(clamped), d_cleared_kw=d_cleared,
    )
