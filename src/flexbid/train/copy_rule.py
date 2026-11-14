"""Start a learned level as a copy of the best rule.

1. **Record.** Run the best rule pair for a few episodes. At every decision of the learned level,
   keep its observation, the rule's choice in the learned level's output, and the discounted
   return that followed.
2. **Copy.** Fit the network's greedy output to the rule's choices and its value to the returns.

Then PPO starts from a policy that already earns about what the rule earns.

The dispatch copy reads the rule's credit per vehicle: the extra term the rule adds to the dispatch
QP, divided by κ (`SCORE_SCALE`). The plain `qp` adds none, so its copy is a score of 0, which is
exact. `qp-rate` changes the wear instead and `qp-few` moves discharge after the QP, so their
copies are only approximate. The copy check on the validation episodes reports how close it is.
"""
from __future__ import annotations

import numpy as np

from flexbid.policies.encoding import bid_to_action
from flexbid.policies.features import bid_observation, dispatch_observation
from flexbid.policies.message import SCORE_SCALE
from flexbid.policies.rules import bid_rule, dispatch_rule

from .envs import REWARD_DIV, _new_sim, place_bid, step_reward

GAMMA_DISPATCH = 0.995                 # discount per 5-minute step
GAMMA_BID = GAMMA_DISPATCH ** 3        # discount per 15-minute interval


def rule_scores(rule, sim):
    """The dispatch rule's action, and its credit per vehicle as scores (0 where it adds none)."""
    import flexbid.sim.physics as ph
    seen = {}
    orig = ph.qp_allocation_power

    def spy(*a, **kw):
        if kw.get("credit_extra") is not None:
            seen["credit"] = np.asarray(kw["credit_extra"], dtype=np.float64)
        return orig(*a, **kw)
    ph.qp_allocation_power = spy
    try:
        act = rule(sim)
    finally:
        ph.qp_allocation_power = orig
    f = sim.cfleet
    y = np.where(f.connected, seen.get("credit", np.zeros(sim.cfg.n_chargers)) / SCORE_SCALE, 0.0)
    return act, y.astype(np.float32)


def record(cfg, bid_name, dispatch_name, level, episodes, first_seed):
    """Observations, the rule's choices and the discounted returns of `level` ("bid" or "dispatch")."""
    rule = dispatch_rule(dispatch_name)
    obs, tgt, rew, ends = [], [], [], []
    for e in range(episodes):
        sim = _new_sim(cfg, first_seed + e)
        while not sim.truncated:
            if sim.phase == "market":
                bid = bid_rule(bid_name, sim)
                if level == "bid":
                    obs.append(bid_observation(sim)); tgt.append(bid_to_action(bid)); rew.append(0.0)
                place_bid(sim, bid)
            if level == "bid":
                act = rule(sim)
            else:
                x = dispatch_observation(sim)
                act, y = rule_scores(rule, sim)
                obs.append(x); tgt.append(y); rew.append(0.0)
            rew[-1] += step_reward(sim.dispatch_step(act))
        rew[-1] += sim.settle() / REWARD_DIV
        ends.append(len(rew))
    gamma = GAMMA_BID if level == "bid" else GAMMA_DISPATCH
    ret = np.zeros(len(rew), np.float32)
    start = 0
    for end in ends:
        g = 0.0
        for i in range(end - 1, start - 1, -1):
            g = rew[i] + gamma * g
            ret[i] = g
        start = end
    return np.asarray(obs, np.float32), np.asarray(tgt, np.float32), ret


def copy_into(model, level, obs, tgt, ret, grad_steps=4000, lr=1e-3, batch=256):
    """Fit the PPO model's greedy output to `tgt` and its value to `ret`.

    A fixed number of gradient steps, with the step size decayed to 0 on a cosine, so a small
    recording still gets a full fit. Returns the loss every grad_steps / 8 steps.
    """
    import torch as th
    pol = model.policy
    opt = th.optim.Adam(pol.parameters(), lr=lr)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=grad_steps)
    O, T, R = (th.as_tensor(x) for x in (obs, tgt, ret))
    n, log, report = len(O), [], max(grad_steps // 8, 1)
    la = lv = 0.0
    for step in range(grad_steps):
        idx = th.randint(0, n, (min(batch, n),))
        o, t, r = O[idx], T[idx], R[idx]
        if level == "bid":
            mean = pol.get_distribution(o).distribution.mean
            v = pol.predict_values(o).squeeze(-1)
            l_a = ((mean - t) ** 2).mean()
        else:
            mean, v, m = pol.net(o)
            v = v.squeeze(-1)
            l_a = (((mean - t) ** 2) * m).sum() / m.sum().clamp(min=1.0)
        l_v = ((v - r) ** 2).mean()
        opt.zero_grad()
        (l_a + 0.5 * l_v).backward()
        opt.step()
        sched.step()
        la += l_a.item()
        lv += l_v.item()
        if (step + 1) % report == 0:
            log.append(dict(step=step + 1, action_mse=la / report, value_mse=lv / report))
            la = lv = 0.0
    return log
