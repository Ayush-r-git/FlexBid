"""The simulator: an EV fleet that bids its flexibility into a real-time market.

The simulator is a state machine with two calls per market interval:

1. ``market_step(anchor, q_frac=..., h_frac=..., beta_mult=...)`` once. The fleet's bid is
   cleared against the rivals' bids and the system requirement, and a commitment is fixed.
2. ``dispatch_step(action)`` exactly ``cfg.K`` times. Each call sets every charger's power for
   one dispatch step and settles the money of that step.

Calls out of this order raise an error. An episode lasts ``cfg.episode_length`` intervals.

**Bid lead.** With ``cfg.market_lead_intervals = L``, the bid placed at interval ``t`` is
cleared at once against the requirement of interval ``t + L``, with the fleet's capacity
measured at ``t``. Its price and commitment are then fixed, and the fleet delivers the
commitment at ``t + L``. The first ``L`` intervals of an episode carry no commitment. Bids for
intervals after the end of the episode are dropped. With ``L = 0`` the bid is delivered in the
interval it clears.

**Money of one dispatch step,** in $:

    money = revenue - imbalance - wear - undelivered

| term | meaning |
|---|---|
| revenue | clearing price times the flexibility actually delivered |
| imbalance | `imbalance_price` times the gap between commitment and delivery |
| wear | battery cycle and calendar wear of every connected vehicle |
| undelivered | `undelivered_energy_price` times the energy a departing vehicle did not receive |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import SimConfig
from .fleet import DepartureRecord, FleetState, advance_departures_and_arrivals, init_fleet
from .market import (
    ClearingResult,
    clear_logistic,
    imbalance_charge_usd,
    revenue_usd,
    unmet_energy_charge_usd,
)
from .physics import (
    apply_dispatch,
    baseline_charge,
    build_view,
    converter_eta,
    degradation_cost_usd,
    feasible_flex_range,
    flex_cost_curve,
    taper_limit_kw,
    unmet_liability_usd,
    urgency_stats,
)
from .processes import sample_competitor_bids, sample_demand, time_features

#: The independent random streams one seed is split into. Append only: a stream's position
#: is its identity, so a new name at the end leaves every existing stream unchanged.
#:   demand    the system requirement d_t and the episode's start time
#:   rivals    the rival aggregators
#:   fleet     the arrival schedule and the vehicles present at the start
#:   arrivals  draws made during the episode
#:   hetero    hidden per-vehicle battery parameters
RNG_STREAMS = ("demand", "rivals", "fleet", "arrivals", "hetero")


def episode_streams(entropy: int, episode: int) -> Dict[str, np.random.Generator]:
    """One independent generator per stream, for episode ``episode`` of seed ``entropy``."""
    return {
        name: np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(episode, i)))
        for i, name in enumerate(RNG_STREAMS)
    }


def _no_commitment() -> ClearingResult:
    """An interval without a commitment (the first L intervals under a bid lead).
    Price 0 and commitment 0: the fleet follows its baseline, earns nothing and owes nothing."""
    return ClearingResult(
        bid=0.0, price=0.0, commitment_kw=0.0,
        commitment_bound_lo_kw=0.0, commitment_bound_hi_kw=0.0,
        d_t=0.0, B_minus_i=0.0, Q_kw=0.0, H_kw=0.0, d_cleared_kw=0.0,
    )


@dataclass
class StepOutcome:
    """Result of one dispatch step. ``components`` holds the money, in $ for this step:

    | key | meaning |
    |---|---|
    | `revenue_usd` | payment for delivered flexibility |
    | `imbalance_usd` | charge for missing the commitment |
    | `degradation_usd` | battery wear |
    | `unmet_usd` | charge for energy not delivered to departing vehicles |
    | `unmet_kwh` | that energy, kWh |
    | `economic_usd` | revenue - imbalance - degradation - unmet, the money of the step |
    | `shaping_usd` | change in the undelivered-energy liability of the connected vehicles; a learning aid, not money |
    """

    components: Dict[str, float]
    departures: List[DepartureRecord]
    interval_done: bool
    truncated: bool


class Simulator:
    """The two-timescale simulator. See the module docstring."""

    def __init__(self, cfg: Optional[SimConfig] = None, *, estimator: bool = True):
        """``estimator`` attaches the controller's temperature estimator
        (`estimator.TemperatureEstimator`) at every reset."""
        self.cfg = cfg or SimConfig()
        self.use_estimator = bool(estimator)
        self.estimator = None
        self._entropy = int(self.cfg.seed) if self.cfg.seed is not None \
            else int(np.random.SeedSequence().entropy)
        self._next_episode = 0
        #: Replays this episode exactly when passed to ``reset(episode_key=...)``.
        self.episode_key = None
        self._needs_reset = True

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: Optional[int] = None, *, episode_key=None) -> Dict:
        """Start a new episode.

        ``seed`` given: episode 0 of that seed. ``seed`` None: the next episode of the current
        seed. ``episode_key`` given: replay that episode exactly.
        """
        cfg = self.cfg
        if episode_key is None:
            if seed is not None:
                self._entropy, self._next_episode = int(seed), 0
            episode_key = (self._entropy, self._next_episode)
            self._next_episode += 1
        st = episode_streams(*episode_key)
        self.rng = st["arrivals"]
        self.episode_key = episode_key

        self.t = 0                 # market interval
        self.k = 0                 # dispatch step inside the interval, 0 .. K-1
        self.phase = "market"      # "market": waiting for a bid; "dispatch": waiting for powers
        self.truncated = False

        # the system requirement d_t and the episode's start time of day
        self.d_process, self.start_minute = sample_demand(cfg, st["demand"])
        # This draw is not used by the market, but it advances the rivals' stream. It is kept so
        # that every random number after it, and so every published result, stays the same.
        sample_competitor_bids(cfg, st["rivals"])
        from .logistic import CompetitorPool
        own_nom = float(cfg.power_scale_kw)
        self.competitors = CompetitorPool.build(cfg, st["rivals"], own_nom)
        # Rescale the requirement so its median size is d_band_fraction of the whole market's
        # nominal capacity: this fleet plus every rival.
        total_up = own_nom + float(np.sum(self.competitors.Q))
        med = float(np.median(np.abs(self.d_process)))
        if med > 1e-9:
            self.d_process = self.d_process * (float(cfg.d_band_fraction) * total_up / med)

        from .synth_fleet import sample_synth_schedule
        self.schedule = sample_synth_schedule(cfg, st["fleet"], self.start_minute)
        self.fleet: FleetState = init_fleet(cfg, st["fleet"], self.schedule, hetero_rng=st["hetero"])
        self.global_step = 0       # dispatch steps since the start of the episode

        # last published market outcome, read by the bid's observation; zero before the first clearing
        self.d_prev = 0.0
        self.p_prev = 0.0
        self.b_prev = 0.0
        self.q_prev = 0.0

        #: the clearing delivered in the current interval
        self.clearing: Optional[ClearingResult] = None
        #: the most recent clearing, whichever interval it is for
        self.last_cleared: Optional[ClearingResult] = None
        #: clearings fixed for future intervals, by interval index
        self.pending: Dict[int, ClearingResult] = {}
        #: every price cleared this episode, in clearing order
        self.cleared_prices: List[float] = []
        self.last_dispatch = None
        #: the message the bid sends into the dispatch QP for the current interval, in score units.
        #: 0 means no message. Only a learned dispatch reads it (flexbid.policies.message).
        self.message = 0.0
        #: scratch space for policies that keep state within one episode
        self.policy_state: Dict = {}
        self._needs_reset = False
        from .estimator import TemperatureEstimator
        self.estimator = TemperatureEstimator(cfg) if self.use_estimator else None
        if self.estimator is not None:
            self.estimator.admit(self.fleet, cfg)
        return {"t": 0, "num_connected": int(np.sum(self.fleet.connected)),
                "start_minute": self.start_minute}

    def settle(self) -> float:
        """Settle the vehicles still plugged in at the end of the episode, $ (see `settlement`).
        Energy ahead of or behind the plan is valued at the episode's mean cleared price."""
        from .settlement import terminal_usd
        p = self.cleared_prices
        p_ref = float(np.mean(p)) * self.cfg.price_unit if p else 0.0
        return float(terminal_usd(self, p_ref))

    @property
    def cfleet(self) -> FleetState:
        """The fleet as a controller sees it: observable quantities and the controller's
        estimates, never the hidden truth. Every controller should read this."""
        return build_view(self.fleet, self.cfg)

    # ------------------------------------------------------------------ observation
    def market_obs(self) -> np.ndarray:
        """18 numbers for the bid level, the same for any fleet size.

        [sin and cos of time of day, last requirement, last price, last bid, last commitment,
        share of chargers occupied, energy owed over total pack energy, lowest and highest
        deliverable flexibility, mean and 90th-percentile urgency, 6 points of the fleet's
        flexibility cost curve]. Power quantities are divided by ``cfg.power_scale_kw``.
        """
        cfg = self.cfg
        fleet = self.cfleet
        sin_t, cos_t = time_features(cfg, self.t, 0, self.start_minute)
        c_base = baseline_charge(fleet, cfg)
        q_lo, q_hi = feasible_flex_range(fleet, c_base, cfg)
        u_mean, u_p90 = urgency_stats(fleet, cfg)
        e_need = np.maximum(fleet.target_soc - fleet.soc, 0.0) * fleet.battery_kwh
        e_need_frac = float(np.sum(e_need * fleet.connected)) / float(np.sum(fleet.battery_kwh))
        curve = flex_cost_curve(fleet, c_base, cfg)
        return np.concatenate([[
            sin_t, cos_t,
            self.d_prev / cfg.d_scale,
            self.p_prev / cfg.p_max,
            self.b_prev / cfg.bid_scale,
            self.q_prev / cfg.power_scale_kw,
            float(np.sum(fleet.connected)) / fleet.n,
            e_need_frac,
            q_lo / cfg.power_scale_kw,
            q_hi / cfg.power_scale_kw,
            u_mean, u_p90,
        ], curve]).astype(np.float32)

    # ------------------------------------------------------------------ dynamics
    def market_step(self, bid_normalized: float, *, q_frac: float = 1.0, h_frac: float = 1.0,
                    beta_mult: float = 1.0) -> ClearingResult:
        """Clear the market with the fleet's bid and fix the commitment.

        | argument | meaning | units |
        |---|---|---|
        | `bid_normalized` | the anchor: where the bid curve sits on the price axis, in [-1, 1] | — |
        | `q_frac` | share of the fleet's room to reduce charging that is offered, in [0, 1] | — |
        | `h_frac` | share of the fleet's room to increase charging that is offered, in [0, 1] | — |
        | `beta_mult` | multiplier on the steepness of the bid curve | — |

        Returns the clearing delivered in this interval. Under a bid lead that is the
        clearing fixed ``L`` intervals ago; the new bid is stored in ``pending``.
        """
        if self._needs_reset or self.truncated:
            raise RuntimeError("Episode over or not started: call reset().")
        if self.phase != "market":
            raise RuntimeError(f"market_step() called mid-interval (k={self.k}/{self.cfg.K}).")

        cfg = self.cfg
        cf = self.cfleet
        c_base = baseline_charge(cf, cfg)
        commitment_lo, commitment_hi = feasible_flex_range(cf, c_base, cfg)
        Q_t, H_t = commitment_hi, -commitment_lo
        anchor_ref = float(self.last_cleared.price) if self.last_cleared is not None else 0.0
        t_del = self.t + int(cfg.market_lead_intervals)
        if t_del < cfg.episode_length:
            cl = clear_logistic(
                float(bid_normalized), Q_t, H_t, self.competitors.bids(),
                float(self.d_process[t_del]), cfg,
                q_frac=float(q_frac), h_frac=float(h_frac), beta_mult=float(beta_mult),
                anchor_ref=anchor_ref,
            )
            self.competitors.advance()
            self.pending[t_del] = cl
            self.last_cleared = cl
            self.cleared_prices.append(float(cl.price))
        self.clearing = self.pending.pop(self.t, None)
        if self.clearing is None:
            self.clearing = _no_commitment()
        self.phase = "dispatch"
        self.k = 0
        return self.clearing

    def dispatch_step(self, action: np.ndarray) -> StepOutcome:
        """Apply one dispatch step.

        ``action`` has one number per charger in [-1, 1]: the deviation of its power from the
        baseline plan, as a share of ``cfg.max_deviation_kw``. Use
        ``physics.power_to_action`` to turn powers in kW into this form.
        """
        if self._needs_reset or self.truncated:
            raise RuntimeError("Episode over or not started: call reset().")
        if self.phase != "dispatch":
            raise RuntimeError("dispatch_step() called before market_step().")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.fleet.n,):
            raise ValueError(f"dispatch action must have shape {(self.fleet.n,)}, got {action.shape}")

        cfg = self.cfg
        clearing = self.clearing
        f = self.fleet
        soc_before = f.soc.copy()
        phi_before = unmet_liability_usd(f, cfg, f.connected)
        c_base = baseline_charge(f, cfg)
        result = apply_dispatch(f, action, c_base, cfg, ctrl_fleet=self.cfleet)

        conn = f.connected
        # the controller's own energy ledger: metered power with nominal efficiencies
        ecv = converter_eta(f, cfg, result.power_kw)
        f.pack_hat_kwh = f.pack_hat_kwh + np.where(conn, (
            cfg.charge_efficiency * ecv * result.charge_kw
            - result.discharge_kw / (cfg.discharge_efficiency * ecv)) * cfg.dt_hours, 0.0)
        f.pack_peak_kwh = np.maximum(f.pack_peak_kwh, f.pack_hat_kwh)
        # observable meters: energy through the meter, and whether a vehicle has stopped accepting power
        f.delivered_kwh = f.delivered_kwh + np.where(conn, result.power_kw, 0.0) * cfg.dt_hours
        f.stopped = conn & ((taper_limit_kw(f, cfg) < cfg.full_accept_kw) | (f.soc >= cfg.soc_max - 1e-9))
        if f.stopped_dis is not None:
            f.stopped_dis = conn & (f.soc <= cfg.soc_min + 1e-9)
        # the declared plan ends when the vehicle stops accepting power
        f.plan_left_kwh = np.where(f.stopped, 0.0, f.plan_left_kwh)
        self.last_dispatch = result
        if self.estimator is not None:
            self.estimator.observe(f, result, cfg)
        # the arrival plan runs down on its own clock, whatever was done
        f.plan_left_kwh = np.maximum(f.plan_left_kwh - c_base * cfg.charge_efficiency * cfg.dt_hours, 0.0)

        deg_usd, _, _ = degradation_cost_usd(soc_before, f.soc, result.charge_kw, result.discharge_kw, f, cfg)

        self.global_step += 1
        departures = advance_departures_and_arrivals(f, cfg, self.rng, self.schedule, self.global_step)
        if self.estimator is not None:
            self.estimator.admit(f, cfg)

        unmet_kwh = float(sum(rec.unmet_kwh for rec in departures))
        unmet_usd = unmet_energy_charge_usd(unmet_kwh, cfg)
        # Potential-based shaping with discount 1: the change in what the connected vehicles would
        # owe if they all left now. It telescopes over an episode, so it moves no money.
        shaping_usd = unmet_liability_usd(f, cfg, f.connected) - phi_before

        rev_usd = revenue_usd(clearing.price, result.realized_flex_kw, cfg)
        imb_usd = imbalance_charge_usd(result.realized_flex_kw, clearing.commitment_kw, cfg)
        economic_usd = rev_usd - imb_usd - deg_usd - unmet_usd

        self.k += 1
        interval_done = self.k >= cfg.K
        if interval_done:
            # the most recent published clearing, for the next bid's observation
            pub = self.last_cleared if self.last_cleared is not None else clearing
            self.d_prev = pub.d_t
            self.p_prev = pub.price
            self.b_prev = pub.bid
            self.q_prev = pub.commitment_kw
            self.t += 1
            self.phase = "market"
            self.truncated = self.t >= cfg.episode_length

        return StepOutcome(
            components={
                "revenue_usd": rev_usd,
                "imbalance_usd": imb_usd,
                "degradation_usd": deg_usd,
                "unmet_usd": unmet_usd,
                "unmet_kwh": unmet_kwh,
                "economic_usd": economic_usd,
                "shaping_usd": shaping_usd,
                "commitment_kw": float(clearing.commitment_kw),
                "realized_flex_kw": float(result.realized_flex_kw),
                "price": float(clearing.price),
                "total_power_kw": float(np.sum(result.power_kw)),
            },
            departures=departures,
            interval_done=interval_done,
            truncated=self.truncated,
        )
