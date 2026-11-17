"""B_HUANG: static-hover restriction of Huang et al., "Multiobjective Deep Reinforcement
Learning for Computation Offloading and Trajectory Control in UAV-Base-Station-Assisted
MEC", IEEE IoT-J 11(19), 2024.

Published method: multi-objective DRL that jointly learns computation offloading and UAV
trajectory control (continuous actions), trading off delay and energy objectives.
Restriction used here:
  * trajectory actions removed (fixed hover); only offloading/allocation is learned;
  * learner: DDPG -- deterministic actor, ONE critic, target copies of both, Gaussian
    exploration noise N(0, 0.1) added to tanh(mu) and clipped to [-1, 1];
  * state: S_sys + task window tau(t) (FIFO window, 593 dims), no memory, no sigma_p;
  * networks: the A9 actor body (Actor(cfg, has_memory=False), deterministic tanh(mean)
    head; log_std unused) and the A9 critic body (Critic(cfg, has_memory=False)), so the
    capacity difference to A9 is only twin -> single critic;
  * reward: the same scalarised reward as the TPA arms with FIXED weights w_base/4 (the
    multi-objective scalarisation without TPA's Eq. 59 adaptation) and the same Normalizer;
  * same replay size, batch, lr, gamma, tau, U and warm-up as SAC (cfg.learner);
  * (a_ij, f~_ij) are mapped by the executor: argmax_j a_ij over feasible pairs and
    f_method = f~_ij * cap_bound_j, f = clip(max(f_method, f_req), 0, F_res).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..control import task_features
from ..learner.networks import Actor, Critic, action_dim
from ..learner.replay import Replay
from .executor import BaseController, Executor, argmax_pref_choose, ftilde_alloc, opt


class DDPG:
    def __init__(self, cfg, actor: nn.Module, critic: nn.Module, rng, device="cpu"):
        L = cfg.learner
        self.cfg, self.device = cfg, torch.device(device)
        self.actor, self.critic = actor.to(self.device), critic.to(self.device)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        for p in list(self.actor_t.parameters()) + list(self.critic_t.parameters()):
            p.requires_grad_(False)
        self.rng = rng
        self.gamma, self.tau = float(L.gamma), float(L.tau)
        self.grad_clip = float(L.grad_clip)
        self.noise = float(opt(cfg, "baselines.ddpg_noise", 0.1))
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=float(L.lr))
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=float(L.lr))
        self.adim = action_dim(cfg)
        self.n_updates = 0

    def random_action(self) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=(self.cfg.task.q_max, self.adim))

    @torch.no_grad()
    def act(self, S, tf, explore: bool) -> np.ndarray:
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device).unsqueeze(0)
        a = self.actor.deterministic(t(S), t(tf)).squeeze(0).cpu().numpy().astype(np.float64)
        if explore:
            a = np.clip(a + self.rng.normal(0.0, self.noise, size=a.shape), -1.0, 1.0)
        return a

    def update(self, b: dict) -> dict:
        S, tf, a, rew, S2, tf2 = b["S_sys"], b["tf"], b["a"], b["reward"], b["S_sys2"], b["tf2"]
        with torch.no_grad():                        # truncation: always bootstrap
            y = rew + self.gamma * self.critic_t(S2, tf2, self.actor_t.deterministic(S2, tf2))
        lc = (self.critic(S, tf, a) - y).pow(2).mean()
        self.opt_c.zero_grad(set_to_none=True)
        lc.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_c.step()
        req = [p.requires_grad for p in self.critic.parameters()]
        for p in self.critic.parameters():
            p.requires_grad_(False)
        try:
            la = -self.critic(S, tf, self.actor.deterministic(S, tf)).mean()
        finally:
            for p, r in zip(self.critic.parameters(), req):
                p.requires_grad_(r)
        self.opt_a.zero_grad(set_to_none=True)
        la.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.opt_a.step()
        with torch.no_grad():
            for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1.0 - self.tau).add_(p, alpha=self.tau)
        self.n_updates += 1
        # The trainer's episode row reads the SAC keys of every update
        # (trainer._ep_row); DDPG has no entropy temperature, so alpha is reported as NaN
        return dict(critic_loss=float(lc.detach()), actor_loss=float(la.detach()),
                    alpha=float("nan"))

    def get_state(self) -> dict:
        return copy.deepcopy(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                                  actor_t=self.actor_t.state_dict(), critic_t=self.critic_t.state_dict(),
                                  opt_a=self.opt_a.state_dict(), opt_c=self.opt_c.state_dict(),
                                  rng=self.rng.bit_generator.state, n_updates=self.n_updates))

    def set_state(self, st: dict) -> None:
        self.actor.load_state_dict(st["actor"])
        self.critic.load_state_dict(st["critic"])
        self.actor_t.load_state_dict(st["actor_t"])
        self.critic_t.load_state_dict(st["critic_t"])
        self.opt_a.load_state_dict(st["opt_a"])
        self.opt_c.load_state_dict(st["opt_c"])
        self.rng.bit_generator.state = st["rng"]
        self.n_updates = int(st["n_updates"])


class HuangController(BaseController):
    fixed_weights = True
    LEARNED_STATE_KEYS = ("agent", "norm")    # eval_clone shares the DDPG learner

    def __init__(self, cfg, arm_key, streams, r_ref, h_min, device="cpu", timer=None, _shared=None):
        super().__init__(cfg, arm_key, streams, r_ref, h_min, device, timer)
        self.ex = Executor(cfg, arm_key, h_min)
        if _shared is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(streams.torch_seed())
                actor = Actor(cfg, False)
                critic = Critic(cfg, False)
            self.agent = DDPG(cfg, actor, critic, streams.policy, device)
            self.replay = Replay(cfg.learner.replay, streams.replay)
        else:
            self.agent = _shared
            self.replay = None
        self._pending = None
        self._cur = None

    def reset_segment(self, t0: int) -> None:
        self.seg_id += 1
        self._pending = None

    def start_episode(self) -> None:
        self._pending = None

    def act(self, obs, label, train):
        cfg = self.cfg
        with self.timer("drm"):
            win = self.ex.window(obs)
            tf = task_features(obs, win, cfg, self.r_ref)
        S = obs.S_sys
        ref = (self.seg_id, int(obs.t))
        if self._pending is not None and train and self.replay is not None:
            p = self._pending
            self.replay.push(dict(S_sys=p["S"], tf=p["tf"], a=p["a"], reward=p["reward"],
                                  sigma=0.0, ref=p["ref"], S_sys2=S, tf2=tf, sigma2=0.0, ref2=ref))
        self._pending = None
        with self.timer("actor"):
            if train and self.slots_seen < cfg.learner.warmup_slots:
                a = self.agent.random_action()
            else:
                a = self.agent.act(S, tf, explore=train)
        with self.timer("drm"):
            dec = self.ex.run(obs, win, argmax_pref_choose(a, self.ex.M),
                              ftilde_alloc(a, self.ex.M, np.asarray(obs.cap_bound, np.float64)))
        self._cur = dict(S=S, tf=tf, a=a, ref=ref)
        self.last = dict(n_window=len(win))
        return dec, 0.0

    def observe(self, obs, dec, outcome, train):
        J, rew = self._reward(dec, outcome, train)
        info = dict(J=J, reward=rew)
        if train:
            self._pending = dict(self._cur, reward=rew)
            self.slots_seen += 1
            L = self.cfg.learner
            if (self.slots_seen >= L.warmup_slots and self.slots_seen % int(L.update_every) == 0
                    and len(self.replay) >= L.batch):
                with self.timer("learner"):
                    info["update"] = self.agent.update(self.replay.sample(L.batch, device=self.device))
        return info

    def eval_clone(self, streams):
        c = HuangController(self.cfg, self.arm_key, streams, self.r_ref, self.h_min,
                            self.device, None, _shared=self.agent)
        c.norm.set_state(self.norm.get_state())
        return c

    def get_state(self) -> dict:
        st = self._base_state()
        st.update(agent=self.agent.get_state(),
                  replay=self.replay.get_state() if self.replay is not None else None,
                  pending=copy.deepcopy(self._pending))
        return st

    def set_state(self, st: dict) -> None:
        self._set_base_state(st)
        self.agent.set_state(st["agent"])
        if self.replay is not None and st["replay"] is not None:
            self.replay.set_state(st["replay"])
        self._pending = copy.deepcopy(st.get("pending"))
