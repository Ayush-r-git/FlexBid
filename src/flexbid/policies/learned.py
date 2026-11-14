"""Learned policies, and one way to describe any policy so it can be rebuilt in a worker process.

**Policy descriptions.** `make_bid` and `make_dispatch` accept:

| description | bid | dispatch |
|---|---|---|
| a rule name, e.g. "fixed:8" or "qp-depth:3e-3" | `rules.bid_rule` | `rules.dispatch_rule` |
| "mpc:<N>" | `mpc.MPCBid` with lookahead N intervals | — |
| {"path": "bid.zip", "message": ..., "eta": ...} | `LearnedBid` | — |
| {"path": "dispatch.pt", "extra": ...} | — | `LearnedDispatch` |
| a callable | used as it is | used as it is |

A dict is a plain description, so it can be sent to a worker process. A learned network is
loaded once per `make_*` call.
"""
from __future__ import annotations

import numpy as np

from .encoding import action_to_bid
from .features import DISPATCH_EXTRA, bid_observation, dispatch_observation
from .message import SCORE_SCALE, computed_message, message_credit, message_from_action, set_message


def powers_from_scores(sim, y):
    """Vehicle powers from scores: credit_k = κ y_k (plus the message credit), then the dispatch QP.

    Returns (P, action): powers in kW and the simulator action that gives them.
    """
    from flexbid.sim.physics import baseline_charge, power_to_action, qp_allocation_power
    cfg, f = sim.cfg, sim.cfleet
    cb = baseline_charge(sim.cfleet, cfg)
    q = sim.clearing.commitment_kw if sim.clearing is not None else 0.0
    s = np.where(f.connected, SCORE_SCALE * np.asarray(y, dtype=np.float64), 0.0)
    mc = message_credit(sim)
    if mc is not None:
        s = s + mc
    P = qp_allocation_power(f, cb, q, cfg, credit_extra=s)
    return P, power_to_action(P, cb, cfg)


class LearnedBid:
    """A trained bid (a stable-baselines3 PPO file), acting greedily.

    | argument | meaning |
    |---|---|
    | path | the saved PPO bid, `.zip` |
    | message | how the message is set each interval: None (no message), "fresh", "fixed", "computed" or "off". Default: "fresh" if the bid has a fifth output, else None |
    | eta | step size of the computed message, score units per kW; needed for "computed" |
    """

    def __init__(self, path, message="auto", eta=None):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")
        self.n_out = int(self.model.action_space.shape[0])
        if message == "auto":
            message = "fresh" if self.n_out > 4 else None
        if message in ("fresh", "fixed") and self.n_out < 5:
            raise ValueError(f"message {message!r} needs a bid with a fifth output; {path} has {self.n_out}")
        if message == "computed" and eta is None:
            raise ValueError("the computed message needs its step size eta")
        self.message, self.eta = message, eta

    def __call__(self, sim):
        a = self.model.predict(bid_observation(sim), deterministic=True)[0]
        if self.message in ("fresh", "fixed"):
            set_message(sim, message_from_action(a[4]), self.message)
        elif self.message == "computed":
            set_message(sim, computed_message(sim, float(self.eta)), "fresh")
        elif self.message == "off":
            set_message(sim, 0.0, "off")
        return action_to_bid(a[:4])


class LearnedDispatch:
    """A trained deep-set dispatch (a `.pt` state dict), acting greedily.

    `extra`: None, or "commitments" if the network also reads the last requirement and the
    commitments already fixed (`features.dispatch_extra`).
    """

    def __init__(self, path, extra=None):
        from .nets import load_dispatch_net
        self.extra = extra
        self.net = load_dispatch_net(path, DISPATCH_EXTRA[extra])

    def scores(self, sim):
        return self.net.mean_np(dispatch_observation(sim, self.extra))

    def __call__(self, sim):
        return powers_from_scores(sim, self.scores(sim))[1]


def make_bid(spec):
    """A bid policy, callable as `policy(sim) -> (anchor, q_frac, h_frac, beta_mult)`."""
    if callable(spec):
        return spec
    if isinstance(spec, dict):
        if "rule" in spec:
            return make_bid(spec["rule"])
        if "mpc" in spec:
            from .mpc import MPCBid
            return MPCBid(spec["mpc"])
        return LearnedBid(spec["path"], spec.get("message", "auto"), spec.get("eta"))
    if isinstance(spec, str) and spec.startswith("mpc:"):
        from .mpc import MPCBid
        return MPCBid(int(spec.split(":")[1]))
    from .rules import bid_rule
    return lambda sim: bid_rule(spec, sim)


def make_dispatch(spec):
    """A dispatch policy, callable as `policy(sim) -> action`, one number per charger."""
    if callable(spec):
        return spec
    if isinstance(spec, dict):
        if "rule" in spec:
            return make_dispatch(spec["rule"])
        return LearnedDispatch(spec["path"], spec.get("extra"))
    from .rules import dispatch_rule
    return dispatch_rule(spec)
