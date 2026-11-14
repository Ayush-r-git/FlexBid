"""The two training environments. Each trains one level; the other level is a fixed policy.

| environment | one step | action | the other level |
|---|---|---|---|
| `BidEnv` | one 15-minute market interval | 4 numbers in [-1, 1], or 5 with a message | a dispatch policy |
| `DispatchEnv` | one 5-minute dispatch step | one score per charger, in [-5, 5] | a bid policy |

Both pay the same money, on their own clock:

    dispatch reward per step  = (money + shaping) / 100
    bid reward per interval   = the sum of its 3 dispatch rewards
    at the end of an episode, both also get the settlement of the vehicles still plugged in / 100

| term | meaning | units |
|---|---|---|
| money | revenue − imbalance − wear − undelivered energy of the step | $ |
| shaping | change in the undelivered-energy liability of the connected vehicles; it sums to almost 0 over an episode and only speeds up learning | $ |
| 100 | `REWARD_DIV`: dollars per reward unit | $ |

Episode i of an environment uses the seed `first_seed + i`. The other level is given as a policy
description (see `flexbid.policies.learned`), so an environment can be rebuilt in a worker.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flexbid.policies.encoding import action_to_bid
from flexbid.policies.features import (DISPATCH_EXTRA, bid_obs_size, bid_observation,
                                       dispatch_obs_size, dispatch_observation)
from flexbid.policies.learned import make_bid, make_dispatch, powers_from_scores
from flexbid.policies.message import message_from_action, set_message
from flexbid.sim.simulator import Simulator

REWARD_DIV = 100.0
SCORE_BOUND = 5.0       # learned dispatch scores are clipped here


def step_reward(outcome):
    """Reward of one dispatch step: (money + shaping) / REWARD_DIV."""
    c = outcome.components
    return (float(c["economic_usd"]) + float(c["shaping_usd"])) / REWARD_DIV


def place_bid(sim, bid):
    a, q, h, beta = bid
    sim.market_step(float(a), q_frac=float(q), h_frac=float(h), beta_mult=float(beta))


def _new_sim(cfg, seed):
    sim = Simulator(cfg)        # a fresh simulator per episode
    sim.reset(int(seed))
    return sim


class BidEnv(gym.Env):
    """The bid learns; the dispatch is the fixed policy `dispatch`.

    `message`: None, or "fresh" or "fixed" (see `flexbid.policies.message`). With a message the
    action has a fifth number, the message in output units.
    """

    def __init__(self, cfg, dispatch, first_seed, message=None):
        if message not in (None, "fresh", "fixed"):
            raise ValueError(f"a bid learns the message 'fresh' or 'fixed', not {message!r}")
        self.cfg, self.message, self.first_seed = cfg, message, int(first_seed)
        self.dispatch = make_dispatch(dispatch)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(bid_obs_size(cfg),), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5 if message else 4,), dtype=np.float32)
        self._i, self.sim = 0, None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed = self.first_seed + self._i if seed is None else seed
        self._i += 1
        self.sim = _new_sim(self.cfg, ep_seed)
        self.money = 0.0
        return bid_observation(self.sim), {}

    def step(self, action):
        sim, cfg = self.sim, self.cfg
        if self.message:
            set_message(sim, message_from_action(action[4]), self.message)
        place_bid(sim, action_to_bid(action[:4]))
        r = 0.0
        for _ in range(cfg.K):
            o = sim.dispatch_step(self.dispatch(sim))
            r += step_reward(o)
            self.money += float(o.components["economic_usd"])
            if sim.truncated:
                break
        if sim.truncated:
            term = sim.settle()
            self.money += term
            return (np.zeros(bid_obs_size(cfg), np.float32), float(r + term / REWARD_DIV), False, True,
                    {"money_usd": self.money})
        return bid_observation(sim), float(r), False, False, {}


class DispatchEnv(gym.Env):
    """The dispatch learns; the bid is the fixed policy `bid`.

    `extra`: None, or "commitments" for the extra inputs of `features.dispatch_extra`.
    """

    def __init__(self, cfg, bid, first_seed, extra=None):
        if extra not in DISPATCH_EXTRA:
            raise ValueError(f"unknown dispatch extra {extra!r}")
        self.cfg, self.extra, self.first_seed = cfg, extra, int(first_seed)
        self.bid = make_bid(bid)
        n = cfg.n_chargers
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(dispatch_obs_size(cfg, extra),), dtype=np.float32)
        self.action_space = spaces.Box(-SCORE_BOUND, SCORE_BOUND, shape=(n,), dtype=np.float32)
        self._i, self.sim = 0, None

    def _bid_if_due(self):
        if self.sim.phase == "market" and not self.sim.truncated:
            place_bid(self.sim, self.bid(self.sim))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed = self.first_seed + self._i if seed is None else seed
        self._i += 1
        self.sim = _new_sim(self.cfg, ep_seed)
        self.money = 0.0
        self._bid_if_due()
        return dispatch_observation(self.sim, self.extra), {}

    def step(self, action):
        sim = self.sim
        _, act = powers_from_scores(sim, np.asarray(action, dtype=np.float64))
        o = sim.dispatch_step(act)
        r = step_reward(o)
        self.money += float(o.components["economic_usd"])
        if sim.truncated:
            term = sim.settle()
            self.money += term
            return dispatch_observation(sim, self.extra), float(r + term / REWARD_DIV), False, True, {"money_usd": self.money}
        self._bid_if_due()
        return dispatch_observation(sim, self.extra), float(r), False, False, {}
