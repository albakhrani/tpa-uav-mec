"""A12: zone x phase lookup controller. No learning.

A12 is the ONLY arm given the true cycle phase. By definition it needs
t mod P, so it reads the slot index t from SlotObs and the period P from
cfg.workload.period_slots:  bin(t) = floor((t mod P) / P * n_bins), n_bins = 24
(cfg.memory.hit_bins). This is a privileged input (an upper bound for a phase-indexed
lookup), not part of S_sys; no other arm sees t or the phase.

Table T[z, b] (Z x 24): EWM of OBSERVED zone arrivals (SlotObs.zone_arrivals) in phase
bin b, T <- (1 - lam) T + lam a_z(t), the first visit of a cell initialising it to the
observation. The table starts empty at every reset_segment (start of a training segment
or test run), like A0's memory.
Per slot:
  P(t)    = T[:, bin(t)] as a Z x 1 matrix of zone intensities, read BEFORE this slot's
            arrivals are written (a lookup of history); an unvisited cell falls back to
            the online zone mean obs.zone_mean[z] (no generator leakage);
  sigma_p = ||P - P_ref||_2 / ||P_ref||_2 clipped at sigma_max, with the P_ref rule:
            sigma_p = 0 during the first w slots of a segment, P_ref = mean of those w
            patterns, then P_ref <- (1 - alpha_ref) P_ref + alpha_ref P (2-norm of a Z x 1
            matrix = spectral norm);
  DRM     = the full pattern-aware DRM (pattern_terms True, burst on) fed with P, sigma_p,
            P_prev and P_ref; no actor: a_ij = 0 and
            f~_ij = clip(scale * c phi / (slack * F_max^j), 0, 1)   (deadline-proportional),
            slack = remaining deadline T_max - (t - t_arr).
Tunables (the common tuning budget of scripts/tuning.py): EWM rate lam (baselines.lookup_ewm),
theta_b (burst.theta_b, passed per controller), f~ scale (baselines.lookup_f_scale).
"""
from __future__ import annotations

import numpy as np

from ..drm.drm import DRM
from ..memory import EPS
from .executor import BaseController, opt


class LookupController(BaseController):
    # No learning; the table T starts empty at every reset_segment and every
    # evaluation run (module docstring), and eval_clone takes only the frozen normaliser
    LEARNED_STATE_KEYS = ("norm",)

    def __init__(self, cfg, arm_key, streams, r_ref, h_min, device="cpu", timer=None,
                 ewm=None, theta_b=None, f_scale=None):
        super().__init__(cfg, arm_key, streams, r_ref, h_min, device, timer)
        self.Z, self.M = int(cfg.scenario.n_zones), int(cfg.scenario.n_uav)
        self.period = int(cfg.workload.period_slots)
        self.n_bins = int(cfg.memory.hit_bins)
        self.w = int(cfg.memory.window_w)
        self.alpha_ref = float(cfg.memory.alpha_ref)
        self.sigma_max = float(cfg.memory.sigma_max)
        self.lam = float(opt(cfg, "baselines.lookup_ewm", 0.1) if ewm is None else ewm)
        self.f_scale = float(opt(cfg, "baselines.lookup_f_scale", 1.0) if f_scale is None else f_scale)
        self.theta_b = float(cfg.burst.theta_b if theta_b is None else theta_b)
        f = self.flags
        self.drm = DRM(cfg, {"pattern_terms": f.pattern_terms, "burst_on": f.burst_on,
                             "beta_c": float(cfg.drm.beta_c), "theta_b": self.theta_b}, arm_key)
        self.drm.h_min = float(h_min)
        self._reset_table()

    def _reset_table(self) -> None:
        self.T = np.zeros((self.Z, self.n_bins))
        self.seen = np.zeros((self.Z, self.n_bins), dtype=bool)
        self.n_seg = 0                     # slots seen in this segment
        self.warm: list = []               # first w patterns (P_ref warm-up)
        self.P_ref = None
        self.P_prev = None

    def phase_bin(self, t: int) -> int:
        """The true phase bin of slot t (privileged, A12 only)."""
        return int((int(t) % self.period) * self.n_bins // self.period)

    def predict(self, obs) -> np.ndarray:
        b = self.phase_bin(obs.t)
        zm = np.asarray(obs.zone_mean, np.float64)
        col = np.where(self.seen[:, b], self.T[:, b], zm)
        return col.reshape(self.Z, 1)

    def _significance(self, P) -> float:
        """Eq. (47) with the P_ref rule (warm-up w slots, then EWM alpha_ref)."""
        if self.n_seg < self.w:
            self.warm.append(P.copy())
            if len(self.warm) == self.w:
                self.P_ref = np.mean(self.warm, axis=0)
                self.warm = []
            return 0.0
        nref = float(np.linalg.norm(self.P_ref, 2))
        s = float(np.linalg.norm(P - self.P_ref, 2)) / max(nref, EPS)
        self.P_ref = (1.0 - self.alpha_ref) * self.P_ref + self.alpha_ref * P
        return float(min(s, self.sigma_max))

    def _update_table(self, obs) -> None:
        b = self.phase_bin(obs.t)
        za = np.asarray(obs.zone_arrivals, np.float64)
        new = ~self.seen[:, b]
        self.T[:, b] = np.where(new, za, (1.0 - self.lam) * self.T[:, b] + self.lam * za)
        self.seen[:, b] = True

    def f_tilde(self, obs, win) -> np.ndarray:
        """f~_ij = clip(scale c phi / (slack F_max^j), 0, 1) for window rows (q_max, M)."""
        out = np.zeros((int(self.cfg.task.q_max), self.M))
        f_max = np.asarray(obs.f_max, np.float64)
        dt = float(self.cfg.scenario.slot_s)
        for k, i in enumerate(win):
            tk = obs.queue[int(i)]
            slack = tk.deadline_s - (obs.t - tk.t_arr) * dt
            if slack <= 0:
                out[k] = 1.0
                continue
            out[k] = np.clip(self.f_scale * tk.cycles * tk.complexity / (slack * f_max), 0.0, 1.0)
        return out

    # ------------------------------------------------------------ protocol
    def reset_segment(self, t0: int) -> None:
        self.seg_id += 1
        self.drm.reset()
        self._reset_table()

    def start_episode(self) -> None:
        self.drm.burst.reset()

    def act(self, obs, label, train):
        with self.timer("mepr"):
            P = self.predict(obs)
            sigma = self._significance(P)
            self._update_table(obs)
            self.n_seg += 1
        with self.timer("drm"):
            pi = self.drm.priority(obs, P, sigma)
            win = self.drm.select_window(pi, obs)
            a = np.zeros((int(self.cfg.task.q_max), 2 * self.M))     # a_ij = 0
            a[:, self.M:] = 2.0 * self.f_tilde(obs, win) - 1.0        # f~ = (a + 1)/2
            P_prev = self.P_prev if self.P_prev is not None else P
            dec = self.drm.run(obs, win, a, P, sigma, P_prev=P_prev,
                               P_ref=self.P_ref if self.P_ref is not None else None)
            dec.burst_budget = np.asarray(dec.util_terms.get("burst_budget_frac", np.zeros(self.M)))
        self.P_prev = P.copy()
        self._sigma = sigma                                   # Eq. 59 sigma_p (BaseController)
        self.last = dict(sigma=sigma, P=P, n_window=len(win), phase_bin=self.phase_bin(obs.t))
        return dec, 0.0

    def observe(self, obs, dec, outcome, train):
        J, rew = self._reward(dec, outcome, train)
        if train:
            self.slots_seen += 1
        return dict(J=J, reward=rew)

    def eval_clone(self, streams):
        c = LookupController(self.cfg, self.arm_key, streams, self.r_ref, self.h_min, self.device,
                             None, ewm=self.lam, theta_b=self.theta_b, f_scale=self.f_scale)
        c.norm.set_state(self.norm.get_state())
        return c

    def get_state(self) -> dict:
        st = self._base_state()
        st.update(T=self.T.copy(), seen=self.seen.copy(), n_seg=self.n_seg,
                  warm=[p.copy() for p in self.warm],
                  P_ref=None if self.P_ref is None else self.P_ref.copy(),
                  P_prev=None if self.P_prev is None else self.P_prev.copy(),
                  drm=self.drm.get_state())
        return st

    def set_state(self, st: dict) -> None:
        self._set_base_state(st)
        self.T = np.asarray(st["T"], np.float64).copy()
        self.seen = np.asarray(st["seen"], bool).copy()
        self.n_seg = int(st["n_seg"])
        self.warm = [np.asarray(p, np.float64).copy() for p in st["warm"]]
        self.P_ref = None if st["P_ref"] is None else np.asarray(st["P_ref"], np.float64).copy()
        self.P_prev = None if st["P_prev"] is None else np.asarray(st["P_prev"], np.float64).copy()
        self.drm.set_state(st["drm"])
