"""A10: LSTM actor-critic (tuning: LSTM width under parameter matching).

A9 plus implicit (recurrent) memory:
  * a shared LSTM torso reads x_t = [S_sys(t); tau_bar(t)] (the same 83-dim summary that
    forms A0's retrieval query) and carries (h, c) across slots; its output
    r_t = proj(h_t) in R^d takes the place of A0's retrieved content r(t);
  * actor and twin critics are A0's networks (Actor/TwinCritic with has_memory=True: the
    cross-attention memory processor in the actor, f_mem in the critics) fed with r_t and
    a constant sigma = 0 (there is no sigma_p in A10);
  * the torso is "shared" exactly as A0's memory parameters are: one module whose output
    feeds actor and critics and which is trained by both SAC losses (SAC mem_params group);
  * PARAMETER MATCHING: total = Actor(True) + TwinCritic(True) + LSTM(+proj) versus
    A0 = Actor(True) + TwinCritic(True) + MemoryParams; must be within +-5%
    (`param_counts(cfg)`; asserted in tests). Width H = baselines.lstm_hidden (default d);
    proj = identity if H == d else Linear(H, d);
  * reactive DRM as A9 (chi = 0, gamma_p = 0, omega_L = 1, beta_b = 0; burst on), top-32
    window by Eq. 52 priority; same objectives and Eq. 59 weights with sigma_p = 0;
  * no L_mem (lambda_mem = 0): Eq. 43 constrains explicit retrieved memory, not a hidden state.

R2D2-style sequence replay (Kapturowski et al. 2019):
  * the rollout stores each slot's (x, S_sys, tf, a, reward) and the LSTM state BEFORE the
    slot; sequences of L = burn_in + unroll + 1 consecutive slots of one episode are stored
    with their stored initial state (h0, c0), overlapping by burn_in + 1 (stride = unroll);
    the tail of an episode is stored zero-padded with a validity mask;
  * per update: n_seq = ceil(batch / unroll) sequences (so ~batch transitions, matching the
    SAC arms' transitions per update; baselines.lstm_seq_batch overrides); the LSTM is run
    from (h0, c0) over the burn-in slots under no_grad, then over the unroll (+1) slots with
    gradient; transition k (state k, next state k+1) uses r_k with grad and r_{k+1} detached
    (true s_{t+1}, as A0's recon under no_grad); SAC twin-critic / actor / alpha losses as
    tpa.learner.sac.SAC on the flattened valid transitions.
  * hidden state resets to zero at reset_segment (A0's memory resets there too) and carries
    across episodes inside a segment; sequences never cross an episode boundary.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from ..control import queue_summary, task_features
from ..drm.drm import DRM
from ..learner.networks import Actor, TwinCritic, param_count
from ..learner.sac import SAC
from ..memory import MemoryParams, q_dim
from .executor import BaseController, opt


class LSTMTorso(nn.Module):
    def __init__(self, cfg, hidden: int):
        super().__init__()
        self.d = int(cfg.memory.d)
        self.H = int(hidden)
        self.lstm = nn.LSTM(q_dim(cfg), self.H, batch_first=True)
        self.proj = nn.Identity() if self.H == self.d else nn.Linear(self.H, self.d)

    def forward(self, x, h0, c0):
        """x (B,T,in), h0/c0 (B,H) -> r (B,T,d), (hT, cT)."""
        out, (h, c) = self.lstm(x, (h0.unsqueeze(0), c0.unsqueeze(0)))
        return self.proj(out), (h.squeeze(0), c.squeeze(0))


def lstm_hidden(cfg) -> int:
    return int(opt(cfg, "baselines.lstm_hidden", int(cfg.memory.d)))


def param_counts(cfg) -> dict:
    """Parameter counts for the A10-vs-A0 matching (within +-5%)."""
    a = param_count(Actor(cfg, True))
    c = param_count(TwinCritic(cfg, True))
    m = param_count(MemoryParams(cfg, seed=0))
    t = param_count(LSTMTorso(cfg, lstm_hidden(cfg)))
    a0, a10 = a + c + m, a + c + t
    return dict(actor=a, critics=c, memory_params=m, lstm_torso=t, A0=a0, A10=a10,
                rel_diff=(a10 - a0) / a0)


class SeqReplay:
    """Fixed-capacity ring of length-L sequences (numpy float32), uniform sampling."""

    KEYS = ("x", "S", "tf", "a", "rew", "valid", "h0", "c0")

    def __init__(self, capacity: int, rng):
        self.capacity, self.rng = int(capacity), rng
        self.size = self.ptr = 0
        self.buf = None

    def __len__(self):
        return self.size

    def push(self, seq: dict) -> None:
        if self.buf is None:
            self.buf = {k: np.zeros((self.capacity,) + np.asarray(seq[k]).shape, np.float32)
                        for k in self.KEYS}
        for k in self.KEYS:
            self.buf[k][self.ptr] = seq[k]
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) -> dict:
        idx = self.rng.integers(0, self.size, size=int(n))
        return {k: torch.from_numpy(np.ascontiguousarray(v[idx])) for k, v in self.buf.items()}

    def get_state(self) -> dict:
        st = dict(capacity=self.capacity, size=self.size, ptr=self.ptr,
                  rng=copy.deepcopy(self.rng.bit_generator.state))
        if self.buf is not None:
            st["arrays"] = {k: v[: self.size].copy() for k, v in self.buf.items()}
            st["shapes"] = {k: v.shape[1:] for k, v in self.buf.items()}
        return st

    def set_state(self, st: dict) -> None:
        self.capacity, self.size, self.ptr = int(st["capacity"]), int(st["size"]), int(st["ptr"])
        self.rng.bit_generator.state = copy.deepcopy(st["rng"])
        if "arrays" in st:
            self.buf = {}
            for k, shp in st["shapes"].items():
                a = np.zeros((self.capacity,) + tuple(shp), np.float32)
                a[: self.size] = st["arrays"][k]
                self.buf[k] = a
        else:
            self.buf = None


class LSTMController(BaseController):
    LEARNED_STATE_KEYS = ("sac", "norm")      # eval_clone shares self.sac (torso included)

    def __init__(self, cfg, arm_key, streams, r_ref, h_min, device="cpu", timer=None, _shared=None):
        super().__init__(cfg, arm_key, streams, r_ref, h_min, device, timer)
        L = cfg.learner
        drm_flags = self.flags if self.flags.beta_c is not None else \
            replace(self.flags, beta_c=float(cfg.drm.beta_c))
        self.drm = DRM(cfg, drm_flags, arm_key)
        self.drm.h_min = float(h_min)
        self.burn, self.unroll = int(L.lstm_burn_in), int(L.lstm_unroll)
        self.seq_len = self.burn + self.unroll + 1
        self.n_seq = int(opt(cfg, "baselines.lstm_seq_batch",
                             max(1, math.ceil(int(L.batch) / self.unroll))))
        if _shared is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(streams.torch_seed())
                actor = Actor(cfg, True)
                critics = TwinCritic(cfg, True)
                self.torso = LSTMTorso(cfg, lstm_hidden(cfg))
            self.sac = SAC(cfg, actor, critics, self.torso, device=device, rng=streams.policy)
            self.sac.lambda_mem = 0.0                         # no Eq. 43 term on a hidden state
            self.replay = SeqReplay(max(1, int(L.replay) // self.unroll), streams.replay)
        else:
            self.sac, self.torso = _shared
            self.replay = None
        self.dev = torch.device(device)
        self.H = self.torso.H
        self._zero_hidden()
        self.steps: list = []
        self._cur = None

    # ------------------------------------------------------------ helpers
    def _zero_hidden(self):
        self.h = torch.zeros(1, self.H, device=self.dev)
        self.c = torch.zeros(1, self.H, device=self.dev)

    def _push_chunk(self, steps, n_valid):
        L, cfg = self.seq_len, self.cfg
        def stack(key, shape):
            out = np.zeros((L,) + shape, np.float32)
            for n, s in enumerate(steps[:n_valid]):
                out[n] = s[key]
            return out
        s0 = steps[0]
        x_dim = s0["x"].shape
        seq = dict(x=stack("x", x_dim), S=stack("S", s0["S"].shape), tf=stack("tf", s0["tf"].shape),
                   a=stack("a", s0["a"].shape),
                   rew=np.array([s.get("reward", 0.0) for s in steps[:n_valid]] + [0.0] * (L - n_valid),
                                np.float32),
                   valid=np.array([1.0] * n_valid + [0.0] * (L - n_valid), np.float32),
                   h0=s0["h0"], c0=s0["c0"])
        self.replay.push(seq)

    def _flush_tail(self):
        if self.replay is not None and len(self.steps) >= self.burn + 2:
            self._push_chunk(self.steps, len(self.steps))
        self.steps = []

    # ------------------------------------------------------------ protocol
    def reset_segment(self, t0: int) -> None:
        self._flush_tail()
        self.seg_id += 1
        self.drm.reset()
        self._zero_hidden()

    def start_episode(self) -> None:
        self._flush_tail()
        self.drm.burst.reset()

    def act(self, obs, label, train):
        cfg = self.cfg
        x = np.concatenate([obs.S_sys, queue_summary(obs, cfg, self.r_ref)])
        with self.timer("retrieval"):
            h0, c0 = self.h, self.c
            with torch.no_grad():
                xt = torch.as_tensor(x, dtype=torch.float32, device=self.dev).view(1, 1, -1)
                r, (self.h, self.c) = self.torso(xt, h0, c0)
            r = r.view(-1).cpu().numpy().astype(np.float64)
        with self.timer("drm"):
            pi = self.drm.priority(obs, None, 0.0)
            win = self.drm.select_window(pi, obs)
            tf = task_features(obs, win, cfg, self.r_ref)
        with self.timer("actor"):
            if train and self.slots_seen < cfg.learner.warmup_slots:
                a = self.sac.random_action()
            else:
                a = self.sac.act(obs.S_sys, tf, r, np.zeros(1), deterministic=not train)
        with self.timer("drm"):
            dec = self.drm.run(obs, win, a, None, 0.0)
            dec.burst_budget = np.asarray(dec.util_terms.get("burst_budget_frac",
                                                             np.zeros(cfg.scenario.n_uav)))
        self._cur = dict(x=x, S=np.asarray(obs.S_sys, np.float64), tf=tf, a=a,
                         h0=h0.view(-1).cpu().numpy().copy(), c0=c0.view(-1).cpu().numpy().copy())
        self.last = dict(n_window=len(win), r_l1=float(np.abs(r).sum()))
        return dec, 0.0

    def observe(self, obs, dec, outcome, train):
        J, rew = self._reward(dec, outcome, train)
        info = dict(J=J, reward=rew)
        if not train or self.replay is None:
            return info
        self.steps.append(dict(self._cur, reward=rew))
        if len(self.steps) == self.seq_len:
            self._push_chunk(self.steps, self.seq_len)
            self.steps = self.steps[self.unroll:]
        self.slots_seen += 1
        L = self.cfg.learner
        if (self.slots_seen >= L.warmup_slots and self.slots_seen % int(L.update_every) == 0
                and len(self.replay) >= self.n_seq):
            with self.timer("learner"):
                info["update"] = self.update()
        return info

    # ------------------------------------------------------------ learning
    def update(self) -> dict | None:
        b = self.replay.sample(self.n_seq)
        dev, B, U = self.dev, self.burn, self.unroll
        b = {k: v.to(dev) for k, v in b.items()}
        h, c = b["h0"], b["c0"]
        if B > 0:
            with torch.no_grad():                                       # burn-in
                _, (h, c) = self.torso(b["x"][:, :B], h, c)
        r_all, _ = self.torso(b["x"][:, B:], h, c)                      # (n, U+1, d), with grad
        val = b["valid"]
        m = (val[:, B:B + U] > 0) & (val[:, B + 1:B + U + 1] > 0)       # transition k valid
        if not bool(m.any()):
            return None
        r_t = r_all[:, :U][m]
        r_2 = r_all[:, 1:U + 1].detach()[m]
        sl, sl2 = slice(B, B + U), slice(B + 1, B + U + 1)
        n = int(m.sum())
        batch = dict(S_sys=b["S"][:, sl][m], tf=b["tf"][:, sl][m], a=b["a"][:, sl][m],
                     reward=b["rew"][:, sl][m], S_sys2=b["S"][:, sl2][m], tf2=b["tf"][:, sl2][m],
                     sigma=torch.zeros(n, 1, device=dev), sigma2=torch.zeros(n, 1, device=dev),
                     refs=[(0, i) for i in range(n)], refs2=[(1, i) for i in range(n)])

        def recon_fn(refs, grad):
            if grad:
                return r_t
            return r_2 if refs and refs[0][0] == 1 else r_t.detach()   # (t-1 only feeds L_mem = 0)

        info = self.sac.update(batch, recon_fn)
        info["n_transitions"] = n
        return info

    # ------------------------------------------------------------ evaluation / state
    def eval_clone(self, streams):
        c = LSTMController(self.cfg, self.arm_key, streams, self.r_ref, self.h_min,
                           self.device, None, _shared=(self.sac, self.torso))
        c.norm.set_state(self.norm.get_state())
        return c

    def get_state(self) -> dict:
        st = self._base_state()
        st.update(sac=self.sac.get_state(), drm=self.drm.get_state(),
                  replay=self.replay.get_state() if self.replay is not None else None,
                  h=self.h.clone(), c=self.c.clone(), steps=copy.deepcopy(self.steps))
        return st

    def set_state(self, st: dict) -> None:
        self._set_base_state(st)
        self.sac.set_state(st["sac"])                 # includes the torso (mem_params group)
        self.drm.set_state(st["drm"])
        if self.replay is not None and st["replay"] is not None:
            self.replay.set_state(st["replay"])
        self.h, self.c = st["h"].clone().to(self.dev), st["c"].clone().to(self.dev)
        self.steps = copy.deepcopy(st["steps"])
