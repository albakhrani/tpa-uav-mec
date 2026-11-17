"""SAC with twin critics and trained memory parameters; Eq. (51), Alg. 2.

Per update (one gradient step):
  r_t     = recon_fn(refs,  grad=True)    (graph into the memory parameters)
  r_{t+1} = recon_fn(refs2, grad=False)   (true s_{t+1}, under no_grad)
  r_{t-1} = recon_fn(refs_prev, grad=False)
  critic loss  sum_i (Q_i(s,a,r_t) - y)^2,  y = rew + gamma (min_i Qtarg_i(s',a',r') - alpha logp')
  actor loss   alpha logp(a~|s) - min_i Q_i(s,a~,r_t) + lambda_mem L_mem          (Eq. 51)
  L_mem        ||r_t - r_{t-1}||^2 + max(0, ||r_t - r_{t-1}|| - Delta_max)^2       (Eq. 43)
  alpha loss   -log_alpha (logp + H_target),  H_target = -0.5 x (#valid action dims)
Both losses are back-propagated once together; critic parameters are frozen
while the actor loss is built, so the actor loss reaches only the actor and the
memory parameters, and the critic loss reaches only the critics and the memory
parameters. Each optimiser (actor, critics, memory, alpha) steps once.
Episode ends are time-limit truncations: the target always bootstraps.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
from torch import nn

from .networks import action_dim


class SAC:
    def __init__(self, cfg, actor: nn.Module, critics: nn.Module,
                 mem_params: nn.Module | None, device="cpu",
                 rng: np.random.Generator | None = None):
        L = cfg.learner
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critics = critics.to(self.device)
        self.critics_targ = copy.deepcopy(self.critics)
        for p in self.critics_targ.parameters():
            p.requires_grad_(False)
        self.mem_params = mem_params.to(self.device) if mem_params is not None else None
        self.has_memory = bool(getattr(actor, "has_memory", False))
        # rollout inference (act) runs on a CPU mirror of the actor; on a CPU
        # learner the mirror IS the actor (bit-identical to the single-device path).
        self._cpu = torch.device("cpu")
        if self.device.type == "cpu":
            self.actor_cpu = self.actor
        else:
            self.actor_cpu = copy.deepcopy(self.actor).to(self._cpu)
            for p in self.actor_cpu.parameters():
                p.requires_grad_(False)
        # policy-noise stream (numpy, CRN rule); the controller passes streams.policy
        self.rng = rng if rng is not None else np.random.Generator(np.random.PCG64(0))
        self.gamma = float(L.gamma)
        self.tau = float(L.tau)
        self.lambda_mem = float(L.lambda_mem)
        self.delta_max = float(cfg.memory.delta_max)
        self.grad_clip = float(L.grad_clip)
        self.ent_scale = float(L.target_entropy_scale)
        self.adim = action_dim(cfg)
        lr = float(L.lr)
        self.log_alpha = torch.tensor(float(np.log(L.alpha0)), device=self.device,
                                      requires_grad=True)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critics.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.opt_mem = (torch.optim.Adam(self.mem_params.parameters(), lr=lr)
                        if self.mem_params is not None else None)
        self.n_updates = 0

    # ------------------------------------------------------------ helpers
    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def _noise(self, shape, device=None) -> torch.Tensor:
        z = self.rng.standard_normal(size=tuple(shape))
        return torch.as_tensor(z, dtype=torch.float32,
                               device=self.device if device is None else device)

    @torch.no_grad()
    def sync_mirror(self) -> None:
        """Copy the learner actor's weights into the CPU rollout mirror."""
        if self.actor_cpu is not self.actor:
            for k, v in self.actor.state_dict().items():
                self.actor_cpu.state_dict()[k].copy_(v.to(self._cpu))

    def target_entropy(self, tf: torch.Tensor) -> torch.Tensor:
        """Per-sample target entropy: -0.5 x (number of unmasked action dims)."""
        n_valid = tf[..., -1].sum(-1)
        return -self.ent_scale * n_valid * self.adim

    @staticmethod
    def _prev_refs(refs):
        return [(s, t - 1) for s, t in refs]

    def random_action(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Warm-up action: uniform in [-1, 1]^(q_max x 2M) (2,000-slot warm-up in the default config)."""
        g = rng if rng is not None else self.rng
        return g.uniform(-1.0, 1.0, size=(self.cfg.task.q_max, self.adim))

    @torch.no_grad()
    def act(self, S_sys, tf, r=None, sigma=None, deterministic: bool = False) -> np.ndarray:
        """Rollout action for one state (numpy in, numpy (q_max, 2M) out), computed
        on the CPU mirror of the actor."""
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self._cpu).unsqueeze(0)
        Ss, T = t(S_sys), t(tf)
        rr = t(r) if (self.has_memory and r is not None) else None
        ss = t(np.reshape(sigma, (1,))) if self.has_memory else None
        if deterministic:
            a = self.actor_cpu.deterministic(Ss, T, rr, ss)
        else:
            a, _ = self.actor_cpu.sample(Ss, T, rr, ss, noise=self._noise(
                (1,) + tuple(T.shape[1:2]) + (self.adim,), device=self._cpu))
        return a.squeeze(0).cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------ targets
    @torch.no_grad()
    def td_target(self, batch: dict, r2: torch.Tensor | None) -> torch.Tensor:
        """y = rew + gamma (min_i Qtarg_i(s', a') - alpha logp'), a' ~ pi(.|s'),
        with the TRUE s' = (S_sys2, tf2, r(t+1), sigma2)."""
        S2, tf2 = batch["S_sys2"], batch["tf2"]
        sg2 = batch["sigma2"] if self.has_memory else None
        a2, logp2 = self.actor.sample(S2, tf2, r2, sg2, noise=self._noise(
            (S2.shape[0], tf2.shape[1], self.adim)))
        q1t, q2t = self.critics_targ(S2, tf2, a2, r2, sg2)
        alpha = self.log_alpha.exp()
        return batch["reward"] + self.gamma * (torch.min(q1t, q2t) - alpha * logp2)

    def l_mem(self, r_t: torch.Tensor, r_prev: torch.Tensor):
        """Eq. (43) soft penalty. Returns (L_mem mean, violation rate)."""
        dr = r_t - r_prev
        n2 = dr.pow(2).sum(-1)
        n = torch.linalg.vector_norm(dr, dim=-1)
        hinge = torch.clamp(n - self.delta_max, min=0.0)
        viol = (n.detach() > self.delta_max).float().mean()
        return (n2 + hinge.pow(2)).mean(), viol

    def losses(self, batch: dict, recon_fn: Callable | None):
        """Build (critic_loss, actor_loss, alpha_loss, info) without stepping."""
        S, tf, a = batch["S_sys"], batch["tf"], batch["a"]
        refs = batch.get("refs")
        if self.has_memory:
            if recon_fn is None:
                raise ValueError("memory arm requires recon_fn")
            r_t = recon_fn(refs, True)
            with torch.no_grad():
                r2 = recon_fn(batch["refs2"], False)
                r_prev = recon_fn(self._prev_refs(refs), False)
            sg = batch["sigma"]
        else:
            r_t = r2 = r_prev = sg = None

        y = self.td_target(batch, r2)
        q1, q2 = self.critics(S, tf, a, r_t, sg)
        critic_loss = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()

        # actor loss: critic parameters frozen, gradient still flows to a~ and r_t
        req = [p.requires_grad for p in self.critics.parameters()]
        for p in self.critics.parameters():
            p.requires_grad_(False)
        try:
            a_new, logp = self.actor.sample(S, tf, r_t, sg, noise=self._noise(a.shape))
            q1p, q2p = self.critics(S, tf, a_new, r_t, sg)
        finally:
            for p, rq in zip(self.critics.parameters(), req):
                p.requires_grad_(rq)
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * logp - torch.min(q1p, q2p)).mean()
        info = {}
        if self.has_memory:
            lm, viol = self.l_mem(r_t, r_prev)
            actor_loss = actor_loss + self.lambda_mem * lm
            info.update(l_mem=float(lm.detach()), dr_viol=float(viol))
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy(tf))).mean()
        info.update(q1=float(q1.detach().mean()), y=float(y.mean()),
                    logp=float(logp.detach().mean()))
        return critic_loss, actor_loss, alpha_loss, info

    # ------------------------------------------------------------ update
    def update(self, batch: dict, recon_fn: Callable | None) -> dict:
        critic_loss, actor_loss, alpha_loss, info = self.losses(batch, recon_fn)
        opts = [self.opt_actor, self.opt_critic, self.opt_alpha]
        if self.opt_mem is not None:
            opts.append(self.opt_mem)
        for o in opts:
            o.zero_grad(set_to_none=True)
        (critic_loss + actor_loss + alpha_loss).backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.grad_clip)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        if self.mem_params is not None:
            nn.utils.clip_grad_norm_(self.mem_params.parameters(), self.grad_clip)
        for o in opts:
            o.step()
        self.polyak()
        self.sync_mirror()
        self.n_updates += 1
        info.update(critic_loss=float(critic_loss.detach()),
                    actor_loss=float(actor_loss.detach()),
                    alpha_loss=float(alpha_loss.detach()), alpha=self.alpha)
        return info

    @torch.no_grad()
    def polyak(self) -> None:
        for p, pt in zip(self.critics.parameters(), self.critics_targ.parameters()):
            pt.mul_(1.0 - self.tau).add_(p, alpha=self.tau)

    # ------------------------------------------------------------ checkpoint
    def get_state(self) -> dict:
        st = {"actor": self.actor.state_dict(), "critics": self.critics.state_dict(),
              "critics_targ": self.critics_targ.state_dict(),
              "log_alpha": self.log_alpha.detach().clone(),
              "opt_actor": self.opt_actor.state_dict(),
              "opt_critic": self.opt_critic.state_dict(),
              "opt_alpha": self.opt_alpha.state_dict(),
              "rng": self.rng.bit_generator.state, "n_updates": self.n_updates}
        if self.mem_params is not None:
            st["mem_params"] = self.mem_params.state_dict()
            st["opt_mem"] = self.opt_mem.state_dict()
        return _to_cpu(st)

    def set_state(self, st: dict) -> None:
        self.actor.load_state_dict(st["actor"])
        self.critics.load_state_dict(st["critics"])
        self.critics_targ.load_state_dict(st["critics_targ"])
        with torch.no_grad():
            self.log_alpha.copy_(st["log_alpha"])
        self.opt_actor.load_state_dict(st["opt_actor"])
        self.opt_critic.load_state_dict(st["opt_critic"])
        self.opt_alpha.load_state_dict(st["opt_alpha"])
        self.rng.bit_generator.state = st["rng"]
        self.n_updates = int(st["n_updates"])
        if self.mem_params is not None and "mem_params" in st:
            self.mem_params.load_state_dict(st["mem_params"])
            self.opt_mem.load_state_dict(st["opt_mem"])
        self.sync_mirror()


def _to_cpu(obj):
    """Deep copy with every tensor moved to the CPU (device-independent checkpoints)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)
