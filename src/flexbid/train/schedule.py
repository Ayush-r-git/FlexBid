"""Training schedules, the lists of episode seeds, and the numbering of training episodes.

**Seed lists.** A seed fixes one episode, which lasts three days of simulated time.

| list | seeds | used for |
|---|---|---|
| tuning | 300–309 | choosing the best rule pair |
| validation | 100–129 | choosing the checkpoint or round of a learned policy |
| test | 200–229 | read once, at the end of a training step |
| fit | 5000–5019 | the departure model, and the MPC's forecast and water value |
| training | from 1,000,000,000 up | PPO episodes, see `training_seed` |

The trainer stops if a tuning or validation list holds a test episode.

**Presets.** `SCHEDULES["standard"]` trains each step for a fixed budget. `SCHEDULES["long"]`
adds long training to every hierarchical pair: rounds that continue until the validation money
stops improving.
`SCHEDULES["smoke"]` runs every step on a few short episodes, to check that the code runs; its
policies mean nothing.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

TUNE_SEEDS = tuple(range(300, 310))
VALIDATION_SEEDS = tuple(range(100, 130))
TEST_SEEDS = tuple(range(200, 230))


@dataclass(frozen=True)
class Rounds:
    """Alternating rounds: in each round the bid trains, then the dispatch.

    Training stops at `max_rounds`, or once at least `min_rounds` rounds are done and no round in
    the last `patience` rounds beat the best validation money so far by more than `tol`.
    With `min_rounds == max_rounds` the number of rounds is fixed.
    """
    episodes_per_level: int        # episodes per level per round
    min_rounds: int
    max_rounds: int
    patience: int = 8              # rounds
    tol: float = 3.0               # $/day


@dataclass(frozen=True)
class TrainSchedule:
    """Every training setting. Change one with `dataclasses.replace(schedule, field=value)`."""
    name: str = "standard"
    # one level learned, the other the best rule (learned-bid, learned-dispatch)
    single_episodes: int = 9000            # episodes
    monitor_every: int = 900               # episodes between checkpoints
    copy_episodes: int = 60                # rule episodes recorded for the copy of the rule
    copy_grad_steps: int = 4000            # gradient steps of the copy
    copy_tol: float = 0.10                 # stop if the copy is off by more than this share of |rule money|
    # both levels learned (bid-and-dispatch, hierarchical)
    rounds: Rounds = Rounds(1200, 5, 5)
    # continuing a hierarchical pair: with a message, with commitment inputs, or long training
    continuation: Rounds = Rounds(1200, 5, 5)
    continue_without_message: bool = False   # also continue a plain hierarchical pair
    # PPO step sizes
    lr_bid: float = 1e-4
    lr_dispatch: float = 3e-5
    # episode seeds
    tune_seeds: tuple = TUNE_SEEDS
    validation_seeds: tuple = VALIDATION_SEEDS
    test_seeds: tuple = TEST_SEEDS
    computed_eta_seeds: tuple = tuple(range(100, 110))   # where the computed message's step size is tuned
    computed_etas: tuple = (-1e-2, -3e-3, -1e-3, -3e-4, -1e-4, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    # rule search: None means every rule (and the cautious bids under a bid lead)
    bid_rules: tuple | None = None
    dispatch_rules: tuple | None = None
    # parallel work
    n_envs: int = 16                       # environments per PPO learner
    workers: int = 16                      # processes for evaluation


SCHEDULES = {
    "standard": TrainSchedule(),
    "long": TrainSchedule(name="long",
                          continuation=Rounds(1000, 12, 40, patience=8, tol=3.0),
                          continue_without_message=True),
    "smoke": TrainSchedule(name="smoke", single_episodes=2, monitor_every=1, copy_episodes=1,
                           copy_grad_steps=20, copy_tol=float("inf"),
                           rounds=Rounds(1, 1, 1), continuation=Rounds(1, 1, 2, patience=1),
                           continue_without_message=True,
                           tune_seeds=(300, 301), validation_seeds=(100, 101), test_seeds=(200, 201),
                           computed_eta_seeds=(100,), computed_etas=(-1e-3, 1e-3),
                           bid_rules=("fixed:8", "balanced"), dispatch_rules=("qp", "qp-depth:3e-3"),
                           n_envs=1, workers=1),
}


def get_schedule(schedule, **changes):
    """A schedule from a preset name or a TrainSchedule, with `changes` applied."""
    s = SCHEDULES[schedule] if isinstance(schedule, str) else schedule
    return dataclasses.replace(s, **changes) if changes else s


def assert_not_test(seeds, what, test_seeds=TEST_SEEDS):
    """Stop if a tuning or validation list holds a test episode."""
    bad = sorted(set(int(d) for d in seeds) & set(int(d) for d in test_seeds))
    if bad:
        raise ValueError(f"{what} holds test episodes {bad[:5]}; test episodes are read only at the end")


# ------------------------------------------------------------------ training episodes
TRAIN_SEED_BASE = 1_000_000_000
STEPS = ("learned-bid", "learned-dispatch", "bid-and-dispatch", "hierarchical",
         "continue", "continue-fresh", "continue-fixed", "continue-commitments", "copy")
_SEED, _STEP, _LEVEL, _ROUND, _ENV = 10**9, 10**8, 5 * 10**7, 10**6, 5 * 10**4


def training_seed(seed, step, level, round_, env):
    """The seed of the first training episode of one environment. Seeds of different runs, steps, levels,
    rounds and environments never overlap, as long as an environment plays fewer than 50,000
    episodes per round, a level has fewer than 50 rounds and a learner has fewer than 20
    environments."""
    if not (0 <= env < 20 and 0 <= round_ < 50):
        raise ValueError("at most 20 environments and 50 rounds")
    return (TRAIN_SEED_BASE + _SEED * int(seed) + _STEP * STEPS.index(step) + _LEVEL * int(level)
            + _ROUND * int(round_) + _ENV * int(env))


def episodes_per_env(episodes, n_envs):
    n = -(-int(episodes) // int(n_envs))
    if n >= _ENV:
        raise ValueError(f"{n} episodes per environment is too many for the seed numbering")
    return n
