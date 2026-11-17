"""A11: A9 + frame stack.

Identical to A9 (SAC with twin critics, reactive DRM: chi = 0, gamma_p = 0, omega_L = 1,
beta_b = 0, burst on; top-32 window by Eq. 52 priority; same objectives, Eq. 59 weights
with sigma_p = 0, warm-up, U, replay) except that the S_sys block of the state is
replaced by the last k = cfg.learner.frame_stack frames [S_sys(t-k+1); ...; S_sys(t)]
(oldest first; 8 x 49 = 392), zero-padded at the start of each memory segment
(reset_segment). Frames persist across episode boundaries inside a segment, as A0's
memory does. State = 392 + 32 x 17 = 936 dims.

The networks are the A9 networks (Actor/TwinCritic with has_memory=False) whose input
layers (actor f_enc[0], critic f_state[0]) are rebuilt for the wider S block; all other
layers are unchanged.

`MemlessSAC` is the A9-equivalent controller body; FrameStackController (A11) extends it
through the `s_dim` / `features` hooks (A10 has its own LSTMController in lstm_ac.py);
it mirrors tpa.control.TPAController with memory off.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import torch

from ..control import task_features
from ..drm.drm import DRM
from ..learner.networks import Actor, TwinCritic, _mlp, s_sys_dim, task_feat_dim
from ..learner.replay import Replay
from ..learner.sac import SAC
from .executor import BaseController


def widen_inputs(actor: Actor, critics: TwinCritic, cfg, s_dim: int) -> None:
    """Rebuild the first layer of actor f_enc and of each critic f_state for an S block of
    `s_dim` instead of S_sys (49). Call inside the seeded fork_rng context."""
    tf_flat = int(cfg.task.q_max) * task_feat_dim(cfg)
    L = cfg.learner
    actor.in_dim = s_dim + tf_flat
    actor.f_enc = _mlp([actor.in_dim] + list(L.actor_enc), layer_norm=True)
    for c in (critics.q1, critics.q2):
        c.f_state = _mlp([s_dim + tf_flat] + list(L.critic_state))


class MemlessSAC(BaseController):
    """A9-equivalent SAC controller with a hook `features(obs)` for the S block."""

    LEARNED_STATE_KEYS = ("sac", "norm")      # eval_clone shares self.sac

    def __init__(self, cfg, arm_key, streams, r_ref, h_min, device="cpu", timer=None, _shared=None):
        super().__init__(cfg, arm_key, streams, r_ref, h_min, device, timer)
        drm_flags = self.flags if self.flags.beta_c is not None else \
            replace(self.flags, beta_c=float(cfg.drm.beta_c))
        self.drm = DRM(cfg, drm_flags, arm_key)
        self.drm.h_min = float(h_min)
        if _shared is None:
            self.sac = self._build(cfg, streams, device)
            self.replay = Replay(cfg.learner.replay, streams.replay)
        else:
            self.sac = _shared
            self.replay = None
        self._pending = None
        self._cur = None

    # ---- hooks
    def s_dim(self) -> int:
        return s_sys_dim(self.cfg)

    def _build(self, cfg, streams, device) -> SAC:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(streams.torch_seed())
            actor = Actor(cfg, False)
            critics = TwinCritic(cfg, False)
            if self.s_dim() != s_sys_dim(cfg):
                widen_inputs(actor, critics, cfg, self.s_dim())
        return SAC(cfg, actor, critics, None, device=device, rng=streams.policy)

    def features(self, obs) -> np.ndarray:
        return obs.S_sys

    def _reset_features(self) -> None:
        pass

    # ---- protocol
    def reset_segment(self, t0: int) -> None:
        self.seg_id += 1
        self.drm.reset()
        self._reset_features()
        self._pending = None

    def start_episode(self) -> None:
        self._pending = None
        self.drm.burst.reset()

    def act(self, obs, label, train):
        cfg = self.cfg
        with self.timer("drm"):
            pi = self.drm.priority(obs, None, 0.0)
            win = self.drm.select_window(pi, obs)
            tf = task_features(obs, win, cfg, self.r_ref)
        S = self.features(obs)
        ref = (self.seg_id, int(obs.t))
        if self._pending is not None and train and self.replay is not None:
            p = self._pending
            self.replay.push(dict(S_sys=p["S"], tf=p["tf"], a=p["a"], reward=p["reward"],
                                  sigma=0.0, ref=p["ref"], S_sys2=S, tf2=tf, sigma2=0.0, ref2=ref))
        self._pending = None
        with self.timer("actor"):
            if train and self.slots_seen < cfg.learner.warmup_slots:
                a = self.sac.random_action()
            else:
                a = self.sac.act(S, tf, None, None, deterministic=not train)
        with self.timer("drm"):
            dec = self.drm.run(obs, win, a, None, 0.0)
            dec.burst_budget = np.asarray(dec.util_terms.get("burst_budget_frac",
                                                             np.zeros(cfg.scenario.n_uav)))
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
                    info["update"] = self.sac.update(self.replay.sample(L.batch, device=self.device), None)
        return info

    def eval_clone(self, streams):
        c = type(self)(self.cfg, self.arm_key, streams, self.r_ref, self.h_min,
                       self.device, None, _shared=self.sac)
        c.norm.set_state(self.norm.get_state())
        return c

    def _feature_state(self) -> dict:
        return {}

    def _set_feature_state(self, st: dict) -> None:
        pass

    def get_state(self) -> dict:
        st = self._base_state()
        st.update(sac=self.sac.get_state(), drm=self.drm.get_state(),
                  replay=self.replay.get_state() if self.replay is not None else None,
                  pending=copy.deepcopy(self._pending), feat=self._feature_state())
        return st

    def set_state(self, st: dict) -> None:
        self._set_base_state(st)
        self.sac.set_state(st["sac"])
        self.drm.set_state(st["drm"])
        if self.replay is not None and st["replay"] is not None:
            self.replay.set_state(st["replay"])
        self._pending = copy.deepcopy(st["pending"])
        self._set_feature_state(st["feat"])


class FrameStackController(MemlessSAC):
    """A11: S block = last k frames of S_sys, zero-padded at segment start."""

    def __init__(self, cfg, *a, **kw):
        self.k = int(cfg.learner.frame_stack)
        self.ds = s_sys_dim(cfg)
        self.frames = np.zeros((self.k, self.ds))
        self._last_t = None
        super().__init__(cfg, *a, **kw)

    def s_dim(self) -> int:
        return self.k * self.ds

    def _reset_features(self) -> None:
        self.frames = np.zeros((self.k, self.ds))
        self._last_t = None

    def features(self, obs) -> np.ndarray:
        if self._last_t != obs.t:                      # push once per slot
            self.frames = np.roll(self.frames, -1, axis=0)
            self.frames[-1] = obs.S_sys
            self._last_t = obs.t
        return self.frames.reshape(-1).copy()

    def _feature_state(self) -> dict:
        return dict(frames=self.frames.copy(), last_t=self._last_t)

    def _set_feature_state(self, st: dict) -> None:
        self.frames = np.asarray(st["frames"], np.float64).copy()
        self._last_t = st["last_t"]
