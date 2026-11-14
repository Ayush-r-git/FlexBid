"""The two networks, and how a trained network is widened for a new input or output.

**Bid.** A stable-baselines3 `MlpPolicy` with two hidden layers of 256 units (`BID_NET_ARCH`).
Its input is the bid observation (`features.bid_observation`). Its outputs are four numbers in
[-1, 1] (`encoding.action_to_bid`), and a fifth one, the message, when it sends one.

**Dispatch.** A deep set. One shared network scores every vehicle, so the same weights run on
any number of chargers:

    z   = Σ_k m_k φ(x_k) / max(Σ_k m_k, 1)      fleet summary, blind to the order of vehicles
    y_k = ψ(x_k, z, g)                           one score per vehicle
    V   = ρ(z, g)                                value of the fleet's state

| symbol | meaning | units |
|---|---|---|
| x_k | vehicle k's 13 features (`features.car_features`) | — |
| m_k | 1 if charger k holds a vehicle, else 0 | — |
| g | the 9 global features, then the extra inputs, if any | — |
| φ, ψ, ρ | small networks, two hidden layers of 64 units | — |
| y_k | vehicle k's score; the dispatch QP turns scores into powers (`learned.powers_from_scores`) | score units |

For PPO, each connected vehicle's score is a Gaussian around y_k with one shared spread. The
log-probability sums over connected vehicles only, so empty chargers add nothing.

**Widening.** A later training step may need a wider network than the step before: a fifth
bid output for the message, more bid inputs under a bid lead, or extra dispatch inputs. The
new weights start at zero. So the widened network acts exactly like the one it came from, and
training starts from that policy.
"""
from __future__ import annotations

import math

import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

from .features import N_CAR, N_GLOB, split_dispatch_obs

HID = 64
BID_NET_ARCH = [256, 256]


def _mlp(i, o, hid=HID, last_act=False):
    layers = [nn.Linear(i, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, o)]
    if last_act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DeepSetNet(nn.Module):
    """φ, ψ, ρ and the shared log-spread. `n_extra` extra global inputs come last in the observation."""

    def __init__(self, n_extra=0, log_std_init=-1.0):
        super().__init__()
        self.n_extra = int(n_extra)
        self.phi = _mlp(N_CAR, HID, last_act=True)
        self.psi = _mlp(N_CAR + HID + N_GLOB + self.n_extra, 1)
        self.rho = _mlp(HID + N_GLOB + self.n_extra, 1)
        self.log_std = nn.Parameter(th.tensor(float(log_std_init)))

    def forward(self, flat):
        X, m, g = split_dispatch_obs(flat, self.n_extra)      # (B,n,13) (B,n) (B,9+extra)
        e = self.phi(X)                                       # (B,n,H)
        z = (e * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
        n = X.shape[1]
        inp = th.cat([X, z.unsqueeze(1).expand(-1, n, -1), g.unsqueeze(1).expand(-1, n, -1)], dim=-1)
        mean = self.psi(inp).squeeze(-1) * m                  # (B,n), 0 on empty chargers
        value = self.rho(th.cat([z, g], dim=-1))              # (B,1)
        return mean, value, m

    @th.no_grad()
    def mean_np(self, flat):
        """Greedy scores for one observation, any fleet size."""
        mean, _, _ = self.forward(th.as_tensor(flat, dtype=th.float32)[None])
        return mean[0].numpy()


def load_dispatch_net(path, n_extra=0):
    net = DeepSetNet(n_extra)
    net.load_state_dict(th.load(path, map_location="cpu"))
    net.eval()
    return net


def _masked_log_prob(mean, log_std, m, a):
    var = th.exp(2 * log_std)
    lp = -0.5 * ((a - mean) ** 2) / var - log_std - 0.5 * math.log(2 * math.pi)
    return (lp * m).sum(-1)


def _masked_entropy(log_std, m):
    return ((0.5 + 0.5 * math.log(2 * math.pi) + log_std) * m).sum(-1)


class DeepSetPPOPolicy(ActorCriticPolicy):
    """A stable-baselines3 policy whose network, distribution and value are the deep set."""

    def __init__(self, *args, log_std_init=-1.0, n_extra=0, **kwargs):
        self._ds_log_std_init = log_std_init
        self._n_extra = int(n_extra)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        self.net = DeepSetNet(self._n_extra, self._ds_log_std_init)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs, deterministic=False):
        mean, value, m = self.net(obs.float())
        a = mean if deterministic else mean + th.exp(self.net.log_std) * th.randn_like(mean)
        a = a * m
        return a, value, _masked_log_prob(mean, self.net.log_std, m, a)

    def evaluate_actions(self, obs, actions):
        mean, value, m = self.net(obs.float())
        return value, _masked_log_prob(mean, self.net.log_std, m, actions), _masked_entropy(self.net.log_std, m)

    def predict_values(self, obs):
        return self.net(obs.float())[1]

    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)[0]

    def get_distribution(self, obs):
        raise NotImplementedError("use forward or evaluate_actions; the distribution is masked per vehicle")


# ------------------------------------------------------------------ widening
def widen_dispatch(state_dict, n_extra):
    """A DeepSetNet with `n_extra` extra inputs and the weights of `state_dict` (which may have
    fewer). The new inputs get zero weight, so the network gives the same scores."""
    net = DeepSetNet(n_extra)
    sd = net.state_dict()
    for key, v in state_dict.items():
        if sd[key].shape == v.shape:
            sd[key] = v.clone()
        elif key in ("psi.0.weight", "rho.0.weight") and sd[key].shape[1] > v.shape[1]:
            t = th.zeros_like(sd[key])
            t[:, : v.shape[1]] = v
            sd[key] = t
        else:
            raise ValueError(f"cannot widen {key}: {tuple(v.shape)} -> {tuple(sd[key].shape)}")
    net.load_state_dict(sd)
    return net


def widen_bid(src, env, learning_rate, seed):
    """A new PPO bid for `env` with the weights of the saved bid `src` (a path or a PPO model).

    The new PPO has the hyperparameters of `src` and a fresh optimizer. Where `env` is wider:

    * more inputs (a bid lead): the new input columns of the first layer of the policy and
      value networks get zero weight;
    * more outputs (the message): the new outputs get zero weight and bias, so the message is 0,
      and exploration noise `MESSAGE_STD_INIT`.

    So the widened bid acts exactly like `src`.
    """
    from stable_baselines3 import PPO
    from .message import MESSAGE_STD_INIT
    old = PPO.load(src, device="cpu") if isinstance(src, (str, bytes)) or hasattr(src, "__fspath__") else src
    new = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu", learning_rate=learning_rate,
              n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs,
              gamma=old.gamma, gae_lambda=old.gae_lambda, clip_range=0.1, ent_coef=old.ent_coef,
              target_kl=old.target_kl, policy_kwargs=dict(old.policy_kwargs))
    so, sn = old.policy.state_dict(), new.policy.state_dict()
    n_old = so["log_std"].shape[0]
    first = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
    for k, v in so.items():
        if sn[k].shape == v.shape:
            sn[k] = v.clone()
        elif k in first and sn[k].shape[0] == v.shape[0] and sn[k].shape[1] > v.shape[1]:
            t = th.zeros_like(sn[k])
            t[:, : v.shape[1]] = v
            sn[k] = t
        elif k in ("action_net.weight", "action_net.bias", "log_std") and sn[k].shape[0] > v.shape[0]:
            t = th.zeros_like(sn[k])
            t[:n_old] = v
            if k == "log_std":
                t[n_old:] = float(np.log(MESSAGE_STD_INIT))
            sn[k] = t
        else:
            raise ValueError(f"cannot widen {k}: {tuple(v.shape)} -> {tuple(sn[k].shape)}")
    new.policy.load_state_dict(sn)
    return new
