"""The message: one number per market interval, sent from the bid into the dispatch QP.

A learned dispatch policy outputs a score y_k per vehicle. The dispatch QP then sets each vehicle's
power. At every 5-minute step it solves

    min_P  Σ_k [ a_k |P_k| + b_k P_k² − κ (y_k + θ_t w_k) P_k ]
    s.t.   Σ_k P_k = Σ_k c_k − q_t,     ℓ_k ≤ P_k ≤ u_k

| symbol | meaning | units |
|---|---|---|
| k | a connected vehicle | — |
| P_k | power of vehicle k, negative when discharging | kW |
| a_k, b_k | linear and quadratic battery-wear cost of vehicle k | $/kW, $/kW² per step |
| κ | `SCORE_SCALE`, turns a score into money | $/kW per step per score unit |
| y_k | the dispatch network's score for vehicle k | score units |
| θ_t | the message of interval t, at most `MESSAGE_MAX` in size | score units |
| w_k | vehicle k's urgency, standardized over the connected vehicles (`urgency_weights`) | — |
| c_k | vehicle k's baseline plan | kW |
| q_t | the commitment of interval t, positive when the fleet charges less | kW |
| ℓ_k, u_k | vehicle k's power limits, including its deadline floor | kW |

θ_t > 0 moves charging toward vehicles that leave soon with small packs. Their deadline floors
drop, so the fleet keeps more room to reduce charging later. θ_t < 0 does the opposite and keeps
more room to increase charging later. So θ_t acts as a price on the fleet's future room.

**Sources of θ_t.** A learned bid sends it as a fifth output x_4 in [-1, 1], θ_t = MESSAGE_MAX x_4.

| timing | when the message acts |
|---|---|
| "fresh" | in the interval in which the bid chooses it |
| "fixed" | it is sent together with the bid, so it acts L intervals later, when that bid is delivered |
| "computed" | not learned: `computed_message`, a price-update rule on the commitments already fixed |
| "off" | never: θ_t = 0 |

The simulator holds the message of the current interval in `sim.message`. A rule dispatch
ignores it. With θ_t = 0 the QP is exactly the QP without a message.
"""
from __future__ import annotations

import copy

import numpy as np

#: κ, $/kW per step per score unit. A score noise of 0.001 $/kW per step already costs about
#: 2 $/day, so scores must map to credits near the wear cost, about 1e-3 $/kW per step.
SCORE_SCALE = 0.01
#: largest message, score units. A constant message of this size already moves the money by
#: tens of $/day, so the useful range is small.
MESSAGE_MAX = 0.03
#: exploration noise of a new message output at the start of training, in output units
MESSAGE_STD_INIT = 0.2
W_CLIP = 3.0
TIMINGS = ("fresh", "fixed", "computed", "off")
#: the computed message shrinks by this factor in an interval with no projected shortfall
COMPUTED_DECAY = 0.5


def message_from_action(x):
    """θ_t from the bid's fifth output x in [-1, 1]."""
    return MESSAGE_MAX * float(np.clip(x, -1.0, 1.0))


def urgency_weights(sim):
    """w_k, 0 for empty chargers:

        w_k = clip( z( log( Δt / (max(T_k, Δt) E_k) ) ), −3, 3 )

    | symbol | meaning | units |
    |---|---|---|
    | Δt | one dispatch step | h |
    | T_k | time to vehicle k's stated departure | h |
    | E_k | vehicle k's pack size | kWh |
    | z(·) | standardization over the connected vehicles: mean 0, standard deviation 1 | — |

    The log and the clip keep a few vehicles about to leave from taking the whole credit.
    """
    cfg, f = sim.cfg, sim.cfleet
    c = f.connected
    w = np.zeros(cfg.n_chargers)
    if not c.any():
        return w
    T = np.maximum(f.est_departure_steps.astype(np.float64), 1.0) * cfg.dt_hours
    r = np.log(cfg.dt_hours / (T[c] * np.maximum(f.battery_kwh[c], 1e-9)))
    sd = r.std()
    w[c] = np.clip((r - r.mean()) / sd, -W_CLIP, W_CLIP) if sd > 1e-9 else 0.0
    return w


def message_credit(sim):
    """κ θ_t w_k per vehicle, $/kW per step, or None when no message acts."""
    theta = float(sim.message)
    if theta == 0.0:
        return None
    return SCORE_SCALE * theta * urgency_weights(sim)


def set_message(sim, theta, timing="fresh"):
    """Set the message of this interval. Call it once per interval, before the bid is placed.

    "fixed" under a bid lead of L intervals: θ chosen now is stored for interval t + L, and
    the one stored L intervals ago acts now (0 in the first L intervals). At L = 0 it equals
    "fresh". "off" sets 0.
    """
    if timing == "off":
        sim.message = 0.0
        return
    L = int(sim.cfg.market_lead_intervals)
    if timing == "fixed" and L > 0:
        queue = sim.policy_state.setdefault("message_queue", {})
        queue[sim.t + L] = float(theta)
        theta = float(queue.pop(sim.t, 0.0))
    elif timing not in ("fresh", "fixed"):
        raise ValueError(f"unknown message timing {timing!r}")
    sim.message = float(theta)


# ------------------------------------------------------------------ the computed message
def projected_room(sim, offsets):
    """Room to reduce (Q) and to increase (H) charging at the start of interval t + j, j in
    `offsets`, if every vehicle followed its baseline plan from now on. kW, two arrays.

    Uses stated departures only: a vehicle leaves the projection when its stated time ends. Reads no
    future arrival and no actual departure.
    """
    from flexbid.sim.physics import baseline_charge, feasible_flex_range
    cfg = sim.cfg
    f = copy.deepcopy(sim.cfleet)
    stated = np.maximum(f.est_departure_steps.astype(np.int64), 1)
    f.departure_steps = np.where(f.connected, stated, 0)
    f.est_departure_steps = np.where(f.connected, stated, 0)
    eta, dt = cfg.charge_efficiency, cfg.dt_hours
    Q, H, done = [], [], 0
    for j in sorted(offsets):
        while done < j * cfg.K:
            P = baseline_charge(f, cfg)
            f.soc = np.where(f.connected, np.minimum(f.soc + P * eta * dt / np.maximum(f.battery_kwh, 1e-9), cfg.soc_max), f.soc)
            if f.plan_left_kwh is not None:
                f.plan_left_kwh = np.maximum(f.plan_left_kwh - P * eta * dt, 0.0)
            f.departure_steps = np.maximum(f.departure_steps - 1, 0)
            f.est_departure_steps = np.maximum(f.est_departure_steps - 1, 0)
            done += 1
        lo, hi = feasible_flex_range(f, baseline_charge(f, cfg), cfg)
        Q.append(hi)
        H.append(-lo)
    return np.array(Q), np.array(H)


def computed_message(sim, eta):
    """The computed message: a rule with no learning that protects room for the commitments
    already fixed. Called once per interval, before the bid.

        s_t = max_j (q_{t+j} − Q_{t+j})⁺ − max_j (−q_{t+j} − H_{t+j})⁺,     j = 1 .. L−1
        θ_t = clip(θ_{t−1} + η s_t, −MESSAGE_MAX, MESSAGE_MAX)   if s_t ≠ 0
              COMPUTED_DECAY θ_{t−1}                              otherwise

    | symbol | meaning | units |
    |---|---|---|
    | q_{t+j} | commitment fixed for interval t+j, positive when the fleet charges less | kW |
    | Q_{t+j}, H_{t+j} | projected room to reduce and to increase charging at t+j (`projected_room`) | kW |
    | s_t | shortfall to reduce minus shortfall to increase | kW |
    | η | step size, with its sign; tune it on validation episodes (`flexbid.train.tune_computed_eta`) | score units per kW |

    At a lead of 0 or 1 interval no later commitment is fixed, so θ_t = 0.
    """
    from .features import fixed_commitments
    L = int(sim.cfg.market_lead_intervals)
    prev = float(sim.policy_state.get("computed_message", 0.0))
    if L < 2:
        return 0.0
    offs = list(range(1, L))
    q = fixed_commitments(sim, offs) * sim.cfg.power_scale_kw
    Q, H = projected_room(sim, offs)
    short_reduce = float(np.max(np.maximum(q - Q, 0.0)))
    short_increase = float(np.max(np.maximum(-q - H, 0.0)))
    s = short_reduce - short_increase
    th = prev + eta * s if s != 0.0 else COMPUTED_DECAY * prev
    th = float(np.clip(th, -MESSAGE_MAX, MESSAGE_MAX))
    sim.policy_state["computed_message"] = th
    return th
