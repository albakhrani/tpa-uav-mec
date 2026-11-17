"""Adaptive burst mode, Eqs. (57)-(58), with the paper's run-length cap T_b.

Decided after the DRM assignment loop at slot t (Alg. 3 line 15) and applied by
the simulator at slot t+1. B_j = 1 raises UAV j's capacity bound to
F_max eta (1 + rho_b) (simulator side) and draws Delta_E_b per slot.

Run-length rule: consecutive burst slots are capped by
T_b = min(T_max, (E_j - E_min) / (Delta_E_b * slot_s)); when the cap is reached
the run is ended and a cooldown of `cooldown_slots` follows (an addition to
Eqs. (57)-(58), stated in the paper). A run that ends because Eq. 57 turns
false gets no cooldown.
"""
from __future__ import annotations

import numpy as np


def _flag(flags, name, default):
    if flags is None:
        return default
    if isinstance(flags, dict):
        return flags.get(name, default)
    return getattr(flags, name, default)


def spectral_norm(A) -> float:
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return 0.0
    return float(np.linalg.norm(A, 2)) if A.ndim == 2 else float(np.linalg.norm(A))


def load_pressure(assign_terms: np.ndarray, f_max: np.ndarray, P, P_prev, P_ref,
                  alpha_b: float, beta_b: float) -> np.ndarray:
    """Eq. (58) with normalised (unit-free) terms:
    Phi_j = alpha_b sum_i x_ij pi_i f_ij / F_max^j + beta_b ||P - P_prev||_2 / ||P_ref||_2
    (spectral norms). assign_terms[j] = sum_i x_ij pi_i f_ij."""
    phi = alpha_b * np.asarray(assign_terms, dtype=np.float64) / np.asarray(f_max, dtype=np.float64)
    if beta_b != 0.0 and P is not None and P_prev is not None and P_ref is not None:
        nref = spectral_norm(P_ref)
        if nref > 0.0:
            phi = phi + beta_b * spectral_norm(np.asarray(P, np.float64) - np.asarray(P_prev, np.float64)) / nref
    return phi


class Burst:
    def __init__(self, cfg, flags=None):
        B = cfg.burst
        self.cfg = cfg
        self.on = bool(_flag(flags, "burst_on", True))
        pattern = bool(_flag(flags, "pattern_terms", True))
        self.theta_b = float(_flag(flags, "theta_b", B.theta_b))
        self.alpha_b = float(B.alpha_b)
        self.beta_b = float(B.beta_b) if pattern else 0.0       # A6 / memoryless: beta_b = 0
        self.de_slot = float(B.delta_e_w) * float(cfg.scenario.slot_s)   # J per burst slot
        self.t_max = int(B.t_max_slots)
        self.cool_n = int(B.cooldown_slots)
        M = cfg.scenario.n_uav
        self.active = np.zeros(M, dtype=bool)
        self.run_len = np.zeros(M, dtype=np.int64)
        self.cooldown = np.zeros(M, dtype=np.int64)
        self.last_phi = np.zeros(M)

    def reset(self) -> None:
        self.active[:] = False
        self.run_len[:] = 0
        self.cooldown[:] = 0
        self.last_phi[:] = 0.0

    def t_b(self, energy, e_min) -> np.ndarray:
        """Run-length cap T_b = min{T_max, (E_j - E_min)/Delta_E_b}, in slots."""
        room = np.maximum(np.asarray(energy, np.float64) - np.asarray(e_min, np.float64), 0.0)
        return np.minimum(float(self.t_max), np.floor(room / self.de_slot))

    def step(self, obs, decision, P, P_prev, P_ref, pi=None, f_max=None):
        """Returns (B_next (M,), budget_frac (M,)) for slot t+1.
        budget_frac = remaining burst budget (T_b - run_len)/T_max, clipped to [0, 1]."""
        M = len(obs.energy)
        if not self.on:                                         # A8: always 0
            self.last_phi = np.zeros(M)
            return np.zeros(M), np.zeros(M)
        if f_max is None:
            f_max = getattr(obs, "f_max")
        pi = np.zeros(len(obs.queue)) if pi is None else np.asarray(pi, np.float64)
        terms = np.zeros(M)
        for a in decision.assignments:
            terms[a.uav] += pi[a.task_index] * a.f
        phi = load_pressure(terms, f_max, P, P_prev, P_ref, self.alpha_b, self.beta_b)
        self.last_phi = phi
        want = (phi > self.theta_b) & (np.asarray(obs.energy) > np.asarray(obs.e_min))  # Eq. 57
        tb = self.t_b(obs.energy, obs.e_min)
        B_next = np.zeros(M)
        for j in range(M):
            if self.cooldown[j] > 0:
                self.cooldown[j] -= 1
                self.active[j] = False
                self.run_len[j] = 0
                continue
            if want[j]:
                if self.run_len[j] >= tb[j]:                    # cap reached: end + cooldown
                    if self.active[j]:
                        self.cooldown[j] = max(self.cool_n - 1, 0)
                    self.active[j] = False
                    self.run_len[j] = 0
                else:
                    B_next[j] = 1.0
                    self.active[j] = True
                    self.run_len[j] += 1
            else:
                self.active[j] = False
                self.run_len[j] = 0
        budget = np.clip((tb - self.run_len) / float(self.t_max), 0.0, 1.0)
        return B_next, budget

    def get_state(self) -> dict:
        return {"active": self.active.copy(), "run_len": self.run_len.copy(),
                "cooldown": self.cooldown.copy(), "last_phi": self.last_phi.copy()}

    def set_state(self, st: dict) -> None:
        self.active = np.asarray(st["active"], dtype=bool).copy()
        self.run_len = np.asarray(st["run_len"], dtype=np.int64).copy()
        self.cooldown = np.asarray(st["cooldown"], dtype=np.int64).copy()
        self.last_phi = np.asarray(st["last_phi"], dtype=np.float64).copy()
