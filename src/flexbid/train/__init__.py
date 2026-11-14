"""Training of the learned policies. Needs the `train` extra:
`pip install "flexbid[train]"`.

    from flexbid import make_config
    from flexbid.train import train

    cfg = make_config("hub-6h", market="lead", market_lead_min=60)
    train("hierarchical", cfg, "runs/hub-6h-lead60/seed0", seed=0, schedule="long", message="fresh")

| name | what |
|---|---|
| `train` | run every step a policy kind needs (`pipeline`) |
| `KINDS`, `SCHEDULES`, `TrainSchedule`, `Rounds` | the policy kinds, and the training settings with their presets |
| `rule_search` | the best rule pair on the tuning episodes |
| `train_single`, `train_rounds` | the two training procedures, for custom pipelines |
| `evaluate_pair`, `paired` | money of a policy pair on a list of days, and a paired comparison |
| `BidEnv`, `DispatchEnv` | the gymnasium environments of the two levels |
| `tune_computed_eta` | the step size of the computed message |

Command line: `python -m flexbid.train --help`.
"""
from .evaluate import evaluate_pair, make_pool, paired
from .pipeline import KINDS, MESSAGES, train, tune_computed_eta
from .rounds import train_rounds
from .rule_search import rule_search
from .schedule import SCHEDULES, Rounds, TrainSchedule, get_schedule
from .single import train_single


def __getattr__(name):
    if name in ("BidEnv", "DispatchEnv"):
        from . import envs
        return getattr(envs, name)
    raise AttributeError(name)


__all__ = ["train", "KINDS", "MESSAGES", "SCHEDULES", "TrainSchedule", "Rounds", "get_schedule", "rule_search",
           "train_single", "train_rounds", "evaluate_pair", "paired", "make_pool", "tune_computed_eta",
           "BidEnv", "DispatchEnv"]
