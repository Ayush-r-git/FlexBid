"""How a learned bid's outputs become a bid.

A learned bid outputs four numbers in [-1, 1]. They map to the four numbers of a bid:

| output | bid number | map |
|---|---|---|
| x_0 | anchor | the same number |
| x_1 | q_frac, share of the room to reduce charging that is offered | `frac_from_action`, onto [0.02, 1] |
| x_2 | h_frac, share of the room to increase charging that is offered | `frac_from_action`, onto [0.02, 1] |
| x_3 | beta_mult, multiplier on the curve's steepness | 2^x_3, onto [0.5, 2] |

A bid that sends a message has a fifth output, see `flexbid.policies.message`.
"""
from __future__ import annotations

import numpy as np

# frac_from_action maps [-1, 1] onto exactly [0.02, 1], the range the fixed bids of the rule menu use
_S_LO, _S_HI = 1.0 / (1.0 + np.exp(3.0)), 1.0 / (1.0 + np.exp(-3.0))


def frac_from_action(x):
    """Offer share in [0.02, 1] from an output in [-1, 1]: a logistic of slope 3, rescaled."""
    s = 1.0 / (1.0 + np.exp(-3.0 * float(np.clip(x, -1.0, 1.0))))
    return 0.02 + 0.98 * (s - _S_LO) / (_S_HI - _S_LO)


def action_from_frac(y):
    """Inverse of `frac_from_action`, clipped to [-1, 1]."""
    u = (float(np.clip(y, 0.02, 1.0)) - 0.02) / 0.98 * (_S_HI - _S_LO) + _S_LO
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    return float(np.clip(np.log(u / (1.0 - u)) / 3.0, -1.0, 1.0))


def bid_to_action(bid):
    """A bid (anchor, q_frac, h_frac, beta_mult) as the four outputs in [-1, 1]."""
    a, q, h, beta = bid
    return np.array([np.clip(a, -1, 1), action_from_frac(q), action_from_frac(h),
                     np.clip(np.log(beta) / np.log(2.0), -1, 1)], dtype=np.float32)


def action_to_bid(x):
    """Four outputs in [-1, 1] as a bid (anchor, q_frac, h_frac, beta_mult)."""
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    return (float(x[0]), float(frac_from_action(x[1])), float(frac_from_action(x[2])),
            float(np.exp(x[3] * np.log(2.0))))
