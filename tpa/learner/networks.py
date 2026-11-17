"""Actor and twin critics, Eqs. (48)-(50).

State (Eq. 48 with h(t) dropped):
    s_t = [S_sys (49); tf.flat (32 x 17); r (d); sigma_p (1)]   -> 658 dims (A0)
    memory-free arms: [S_sys; tf.flat]                         -> 593 dims
The last task feature is the valid-mask bit. Empty task slots are masked out of
the action seen by the critic and out of the log-probability.

Action per task slot: 2M numbers in [-1, 1] (tanh-squashed Gaussian).
Columns [0, M) are the preference scores a_ij; columns [M, 2M) map to the
allocation fraction f~_ij = (a + 1) / 2 in DRM.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------- dimensions
def s_sys_dim(cfg) -> int:
    """S_sys size: 6M + 2Z + 2 + M (= 49 for M=5, Z=6)."""
    M, Z = cfg.scenario.n_uav, cfg.scenario.n_zones
    return 6 * M + 2 * Z + 2 + M


def task_feat_dim(cfg) -> int:
    """Task features: size, cycles, slack, priority, complexity, zone one-hot (Z),
    rate per UAV (M), valid bit (= 17)."""
    M, Z = cfg.scenario.n_uav, cfg.scenario.n_zones
    return 5 + Z + M + 1


def state_dim(cfg, has_memory: bool) -> int:
    """Eq. (48) dimension: 658 with memory, 593 without."""
    base = s_sys_dim(cfg) + cfg.task.q_max * task_feat_dim(cfg)
    return base + (cfg.memory.d + 1 if has_memory else 0)


def action_dim(cfg) -> int:
    """Per task slot: (a_ij, f~_ij) for each UAV = 2M."""
    return 2 * cfg.scenario.n_uav


def param_count(module: nn.Module) -> int:
    """Number of trainable parameters (used for A10 parameter matching)."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _mlp(sizes: list[int], layer_norm: bool = False, last_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for k in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[k], sizes[k + 1]))
        is_last = k == len(sizes) - 2
        if not is_last or last_act:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[k + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _flat_state(S_sys, tf, r, sigma, has_memory: bool) -> torch.Tensor:
    parts = [S_sys, tf.reshape(tf.shape[0], -1)]
    if has_memory:
        if r is None or sigma is None:
            raise ValueError("memory arm requires r and sigma")
        parts += [r, sigma.reshape(sigma.shape[0], 1)]
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------- actor
class Actor(nn.Module):
    """Eq. (49): pi = f_policy(f_att(f_enc(s), r)).

    f_enc : MLP [state -> 512 -> 256 -> 128] with LayerNorm.
    f_att : one cross-attention layer (query f_enc, keys/values r as
            n_heads tokens of d/n_heads), then MLP [128, 64]. Zeros(64) without memory.
    task  : per-task MLP 17 -> 64.
    head  : per task [proc 64; task 64; f_enc 128] = 256 -> [256, 128] -> (mean, log_std) x 2M.
    """

    def __init__(self, cfg, has_memory: bool):
        super().__init__()
        L = cfg.learner
        self.has_memory = bool(has_memory)
        self.M = cfg.scenario.n_uav
        self.q_max = cfg.task.q_max
        self.adim = action_dim(cfg)
        self.in_dim = state_dim(cfg, has_memory)
        enc = list(L.actor_enc)
        self.f_enc = _mlp([self.in_dim] + enc, layer_norm=True)
        enc_out = enc[-1]
        mem_sizes = list(L.actor_mem)
        self.proc_dim = mem_sizes[-1]
        if self.has_memory:
            d = cfg.memory.d
            self.n_tok = cfg.memory.n_heads
            if d % self.n_tok:
                raise ValueError("memory.d must be divisible by memory.n_heads")
            self.tok_dim = d // self.n_tok
            self.att = nn.MultiheadAttention(embed_dim=enc_out, num_heads=cfg.memory.n_heads,
                                             kdim=self.tok_dim, vdim=self.tok_dim,
                                             batch_first=True)
            self.proc = _mlp([enc_out] + mem_sizes)
        self.f_task = _mlp([task_feat_dim(cfg), L.task_enc])
        head_in = self.proc_dim + L.task_enc + enc_out
        self.head = _mlp([head_in] + list(L.actor_head))
        self.out = nn.Linear(list(L.actor_head)[-1], 2 * self.adim)
        self.log_std_min = float(L.get("log_std_min", -20.0))
        self.log_std_max = float(L.get("log_std_max", 2.0))

    def forward(self, S_sys, tf, r=None, sigma=None):
        """Returns (mean, log_std), each (B, q_max, 2M)."""
        B = S_sys.shape[0]
        h = self.f_enc(_flat_state(S_sys, tf, r, sigma, self.has_memory))     # (B,128)
        if self.has_memory:
            kv = r.reshape(B, self.n_tok, self.tok_dim)
            att, _ = self.att(h.unsqueeze(1), kv, kv, need_weights=False)      # (B,1,128)
            proc = self.proc(att.squeeze(1))                                   # (B,64)
        else:
            proc = h.new_zeros(B, self.proc_dim)
        te = self.f_task(tf)                                                   # (B,32,64)
        Q = tf.shape[1]
        z = torch.cat([proc.unsqueeze(1).expand(B, Q, -1), te,
                       h.unsqueeze(1).expand(B, Q, -1)], dim=-1)               # (B,32,256)
        o = self.out(self.head(z))
        mean, log_std = o[..., :self.adim], o[..., self.adim:]
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    @staticmethod
    def mask_of(tf) -> torch.Tensor:
        return tf[..., -1]

    def sample(self, S_sys, tf, r=None, sigma=None, noise=None, generator=None):
        """Reparameterised tanh-Gaussian sample.

        Returns a (B, q_max, 2M) in [-1, 1] and logp (B,), the log-density of
        the squashed action summed over action dims of **valid** task slots only.
        `noise` (same shape as a) may be supplied (e.g. drawn from a numpy
        Generator); otherwise `generator` (torch.Generator) is used.
        """
        mean, log_std = self.forward(S_sys, tf, r, sigma)
        std = log_std.exp()
        if noise is None:
            noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype,
                                device=mean.device)
        u = mean + std * noise
        a = torch.tanh(u)
        # Gaussian log-density + tanh change of variables (stable form)
        lp = (-0.5 * noise.pow(2) - log_std - 0.5 * math.log(2.0 * math.pi)
              - 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u)))
        mask = self.mask_of(tf)
        logp = (lp.sum(-1) * mask).sum(-1)
        return a, logp

    def deterministic(self, S_sys, tf, r=None, sigma=None):
        """Evaluation action: tanh(mean)."""
        mean, _ = self.forward(S_sys, tf, r, sigma)
        return torch.tanh(mean)


# ---------------------------------------------------------------- critics
class Critic(nn.Module):
    """Eq. (50): Q = f_Q[f_state(s); f_act(a); f_mem(r)].

    f_state: MLP [S_sys + tf.flat (+sigma) -> 256 -> 128]  (task features included)
    f_act  : MLP [masked, flattened action 320 -> 128 -> 64]
    f_mem  : MLP [d -> 64 -> 32] or zeros(32) without memory
    f_Q    : MLP [224 -> 256 -> 128 -> 1]
    """

    def __init__(self, cfg, has_memory: bool):
        super().__init__()
        L = cfg.learner
        self.has_memory = bool(has_memory)
        sd = s_sys_dim(cfg) + cfg.task.q_max * task_feat_dim(cfg) + (1 if has_memory else 0)
        self.f_state = _mlp([sd] + list(L.critic_state))
        self.f_act = _mlp([cfg.task.q_max * action_dim(cfg)] + list(L.critic_act))
        mem_sizes = list(L.critic_mem)
        self.mem_out = mem_sizes[-1]
        if self.has_memory:
            self.f_mem = _mlp([cfg.memory.d] + mem_sizes)
        q_in = list(L.critic_state)[-1] + list(L.critic_act)[-1] + self.mem_out
        self.f_Q = _mlp([q_in] + list(L.critic_q) + [1], last_act=False)

    def forward(self, S_sys, tf, a, r=None, sigma=None):
        B = S_sys.shape[0]
        parts = [S_sys, tf.reshape(B, -1)]
        if self.has_memory:
            parts.append(sigma.reshape(B, 1))
        hs = self.f_state(torch.cat(parts, -1))
        mask = tf[..., -1:]
        ha = self.f_act((a * mask).reshape(B, -1))
        hm = self.f_mem(r) if self.has_memory else hs.new_zeros(B, self.mem_out)
        return self.f_Q(torch.cat([hs, ha, hm], -1)).squeeze(-1)

    def param_count(self) -> int:
        return param_count(self)


class TwinCritic(nn.Module):
    """Two independent critics (clipped double-Q)."""

    def __init__(self, cfg, has_memory: bool):
        super().__init__()
        self.q1 = Critic(cfg, has_memory)
        self.q2 = Critic(cfg, has_memory)
        self.has_memory = bool(has_memory)

    def forward(self, S_sys, tf, a, r=None, sigma=None):
        return self.q1(S_sys, tf, a, r, sigma), self.q2(S_sys, tf, a, r, sigma)

    def param_count(self) -> int:
        return param_count(self)


def to_tensor(x, device="cpu") -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
