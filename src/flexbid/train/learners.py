"""PPO settings of both levels, vectorized environments, saving, and the run log.

| setting | bid | dispatch |
|---|---|---|
| network | MLP, 2 x 256 | deep set, 64 units (`flexbid.policies.nets`) |
| rollout per environment | one episode: 288 intervals | one episode: 864 steps |
| minibatch | one episode | one episode |
| epochs per rollout | 5 | 5 |
| discount | 0.995³ per interval | 0.995 per step |
| GAE λ, clip range, entropy, target KL | 0.95, 0.1, 0, 0.03 | 0.95, 0.1, 0, 0.03 |
| starting exploration noise | 0.1 per output | 0.1 per score |
| step size | `schedule.lr_bid`, 1e-4 | `schedule.lr_dispatch`, 3e-5 |
"""
from __future__ import annotations

import functools
import json
import pathlib
import time

import numpy as np

from .copy_rule import GAMMA_BID, GAMMA_DISPATCH


def new_bid_model(env, cfg, lr, seed):
    from stable_baselines3 import PPO
    from flexbid.policies.nets import BID_NET_ARCH
    return PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu", learning_rate=lr,
               n_steps=cfg.episode_length, batch_size=cfg.episode_length, n_epochs=5,
               gamma=GAMMA_BID, gae_lambda=0.95, clip_range=0.1, ent_coef=0.0, target_kl=0.03,
               policy_kwargs=dict(net_arch=list(BID_NET_ARCH), log_std_init=float(np.log(0.1))))


def new_dispatch_model(env, cfg, lr, seed, n_extra=0):
    from stable_baselines3 import PPO
    from flexbid.policies.nets import DeepSetPPOPolicy
    steps = cfg.episode_length * cfg.K
    return PPO(DeepSetPPOPolicy, env, seed=seed, verbose=0, device="cpu", learning_rate=lr,
               n_steps=steps, batch_size=steps, n_epochs=5,
               gamma=GAMMA_DISPATCH, gae_lambda=0.95, clip_range=0.1, ent_coef=0.0, target_kl=0.03,
               policy_kwargs=dict(log_std_init=float(np.log(0.1)), n_extra=n_extra))


def _bid_env(cfg, dispatch, first_seed, message):
    from .envs import BidEnv
    return BidEnv(cfg, dispatch, first_seed, message)


def _dispatch_env(cfg, bid, first_seed, extra):
    from .envs import DispatchEnv
    return DispatchEnv(cfg, bid, first_seed, extra)


def make_venv(level, cfg, partner, first_seeds, option=None):
    """Vectorized environments of one level, one per first seed. `partner` describes the other
    level. `option` is the message timing (bid) or the extra inputs (dispatch)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    mk = _bid_env if level == "bid" else _dispatch_env
    fns = [functools.partial(mk, cfg, partner, d, option) for d in first_seeds]
    return VecMonitor(SubprocVecEnv(fns) if len(fns) > 1 else DummyVecEnv(fns))


def save_bid(model, path):
    model.save(str(path))
    return str(path)


def save_dispatch(model, path):
    import torch as th
    th.save(model.policy.net.state_dict(), str(path))
    return str(path)


def steps_per_episode(level, cfg):
    return cfg.episode_length * (1 if level == "bid" else cfg.K)


class RunLog:
    """log.txt, monitor.jsonl (one validation point per line) and result.json of one training step."""

    def __init__(self, folder):
        self.dir = pathlib.Path(folder)
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"{time.strftime('%F %T')}  {msg}"
        print(line, flush=True)
        with open(self.dir / "log.txt", "a") as fh:
            fh.write(line + "\n")

    def point(self, pt, first=False):
        with open(self.dir / "monitor.jsonl", "w" if first else "a") as fh:
            fh.write(json.dumps(pt) + "\n")

    def result(self, res):
        (self.dir / "result.json").write_text(json.dumps(res, indent=1))
        return res
