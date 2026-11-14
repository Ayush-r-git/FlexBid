"""The departure model a policy may use.

The true departure process, with its hidden driver types, is never shown to a policy. A
policy may use each vehicle's stated departure, its own past, and this model. The model is fitted
on the sessions of the fit episodes `FIT_SEEDS`, which no tuning, validation or test list uses.

It is the standard single-error model:

    log(actual) ~ N(mu_d, sd_d)
    log(stated) = log(m) + b (log(actual) - log(m)) + mu + sd u,     u ~ N(0, 1)

| symbol | meaning | units |
|---|---|---|
| mu_d, sd_d | mean and standard deviation of the log of the actual stay | log h |
| m | reference stay, exp(mu_d) | h |
| b | how closely the stated stay tracks the actual one (least-squares slope) | — |
| mu | systematic bias of the stated stay (least-squares intercept) | — |
| sd | residual noise of the stated stay | — |
| mean_stated_h | mean stated stay; the scale of the departure features | h |

With hidden driver types this model is misspecified on purpose: one Gaussian error cannot
describe a mixture of on-time, early and late drivers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Episode seeds the departure model and the MPC's forecast are fitted on. Disjoint from every other seed list.
FIT_SEEDS = tuple(range(5000, 5020))

_CACHE: dict = {}


@dataclass(frozen=True)
class DepartureModel:
    mu_d: float
    sd_d: float
    b: float
    mu: float
    sd: float
    mean_stated_h: float
    n_sessions: int

    @classmethod
    def fit(cls, cfg, seeds=FIT_SEEDS):
        """Fit on the sessions of the episodes `seeds`. Cached per simulator configuration."""
        from flexbid.sim.simulator import Simulator
        key = (cfg.physics_fingerprint(), tuple(seeds))
        if key in _CACHE:
            return _CACHE[key]
        act, est = [], []
        for d in seeds:
            s = Simulator(cfg, estimator=False)
            s.reset(d)
            act.append(s.schedule.dwell_act_steps)
            est.append(s.schedule.dwell_est_steps)
        a_h = np.concatenate(act) * cfg.dt_hours
        e_h = np.concatenate(est) * cfg.dt_hours
        la, le = np.log(a_h), np.log(e_h)
        mu_d, sd_d = float(la.mean()), float(la.std())
        x, y = la - mu_d, le - mu_d
        b = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
        mu = float(y.mean() - b * x.mean())
        sd = float(np.std(y - (b * x + mu)))
        m = cls(mu_d, sd_d, b, mu, sd, float(e_h.mean()), len(a_h))
        _CACHE[key] = m
        return m

    def scale_steps(self, cfg):
        """The mean stated stay in dispatch steps, at least 1."""
        return max(self.mean_stated_h / cfg.dt_hours, 1.0)
