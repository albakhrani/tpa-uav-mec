"""Per-slot objectives J1-J4 (Eqs. 30-33), min-max normalisation
(Eq. 34), adaptive weights (Eq. 59) and the reward (Eq. 51).

Sign convention: J~2 (utilisation, maximised) enters the reward and Eq. 59
negated, so every signed term is a cost:  c = (J~1, -J~2, J~3, J~4).
"""
from __future__ import annotations

import numpy as np

SIGNS = np.array([1.0, -1.0, 1.0, 1.0])


def slot_objectives(decision, outcome, mem_stats: dict | None, cfg) -> np.ndarray:
    """Single-slot J(t) = (J1, J2, J3, J4).

    J1 (Eq. 30) = sum of predicted T_total of tasks assigned at t.
    J2 (Eq. 31) = sum_j [b1 sum_i f_ij/F_max^j + b2 sum_i x_ij R_ij/R_max + b3 ||r||_1/R_mem],
                  R_mem = K_S + K_L. The per-UAV fractions come from decision.util_terms
                  ('cpu_frac', 'rate_frac', each (M,)); the memory term is inside sum_j (literal).
    J3 (Eq. 32) = sum_j [sum_i E_comp x + kappa_m ||r||_1]  -- literal, NO burst energy.
                  Uses outcome.comp_energy_j (M,) and outcome.mem_energy_j (charged per UAV).
    J4 (Eq. 33) = theta_S ||M_S||_F + theta_L ||M_L||_F + omega_p ||r(t) - r(t-1)||.
    mem_stats: {'l1': ||r||_1, 'fro_s', 'fro_l', 'dr_norm'}; None for memory-free arms.
    """
    M = cfg.scenario.n_uav
    O = cfg.objectives
    b1, b2, b3 = O.beta_util
    J1 = float(sum(a.t_total_pred for a in decision.assignments))
    ut = decision.util_terms or {}
    cpu = np.asarray(ut.get("cpu_frac", np.zeros(M)), dtype=np.float64)
    rate = np.asarray(ut.get("rate_frac", np.zeros(M)), dtype=np.float64)
    ms = mem_stats or {}
    r_mem = cfg.memory.k_short + cfg.memory.k_long
    l1 = float(ms.get("l1", 0.0))
    J2 = float(np.sum(b1 * cpu + b2 * rate + b3 * l1 / r_mem))
    J3 = float(np.sum(outcome.comp_energy_j) + M * outcome.mem_energy_j)
    th_s, th_l = O.theta_mem
    J4 = float(th_s * ms.get("fro_s", 0.0) + th_l * ms.get("fro_l", 0.0)
               + O.omega_p * ms.get("dr_norm", 0.0))
    return np.array([J1, J2, J3, J4], dtype=np.float64)


class Normalizer:
    """Eq. (34) min-max normalisation.

    Before `norm_freeze_slots` observations: running min/max of the values seen
    so far (including the current one). At exactly `norm_freeze_slots`
    observations, lo/hi are set to the (norm_lo_pct, norm_hi_pct) percentiles of
    those observations and frozen for the rest of training and for evaluation.
    J~ = clip((J - lo)/(hi - lo), 0, 1); a degenerate range (hi <= lo) gives J~ = 0.
    """

    def __init__(self, cfg, n: int = 4):
        L = cfg.learner
        self.freeze_n = int(L.norm_freeze_slots)
        self.lo_pct, self.hi_pct = float(L.norm_lo_pct), float(L.norm_hi_pct)
        self.n = n
        self.buf: list[np.ndarray] = []
        self.lo = np.full(n, np.inf)
        self.hi = np.full(n, -np.inf)
        self.frozen = False
        self.count = 0

    def observe(self, J) -> None:
        if self.frozen:
            return
        J = np.asarray(J, dtype=np.float64)
        self.buf.append(J.copy())
        self.count += 1
        self.lo = np.minimum(self.lo, J)
        self.hi = np.maximum(self.hi, J)
        if self.count >= self.freeze_n:
            arr = np.stack(self.buf)
            self.lo = np.percentile(arr, self.lo_pct, axis=0)
            self.hi = np.percentile(arr, self.hi_pct, axis=0)
            self.frozen = True
            self.buf = []

    def normalize(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        rng = self.hi - self.lo
        ok = np.isfinite(rng) & (rng > 0.0)
        out = np.zeros(self.n)
        out[ok] = np.clip((J[ok] - self.lo[ok]) / rng[ok], 0.0, 1.0)
        return out

    def __call__(self, J, update: bool = True) -> np.ndarray:
        if update:
            self.observe(J)
        return self.normalize(J)

    def get_state(self) -> dict:
        return {"buf": np.stack(self.buf) if self.buf else np.zeros((0, self.n)),
                "lo": self.lo.copy(), "hi": self.hi.copy(), "frozen": self.frozen,
                "count": self.count}

    def set_state(self, st: dict) -> None:
        self.buf = [row.copy() for row in np.asarray(st["buf"])]
        self.lo = np.asarray(st["lo"], dtype=np.float64).copy()
        self.hi = np.asarray(st["hi"], dtype=np.float64).copy()
        self.frozen = bool(st["frozen"])
        self.count = int(st["count"])


class AdaptiveWeights:
    """Eq. (59), literal:
        w_k(t) = w_k^base exp(alpha_k sigma_p(t)) gamma_k(t),
        gamma_k = exp(-beta_k dJ_k) / sum_i exp(-beta_i dJ_i),
        dJ_k(t) = EWM(c_k)(t-1) - c_k(t),  c = signed normalised costs (J~2 negated).
    No renormalisation. EWM(t) = ewm * EWM(t-1) + (1 - ewm) c(t), initialised to the
    first c (so dJ = 0 and gamma is uniform on the first call).
    fixed=True (A7): w_k = w_k^base / 4, constant.
    """

    def __init__(self, cfg, fixed: bool = False):
        O = cfg.objectives
        self.w_base = np.asarray(O.w_base, dtype=np.float64)
        self.alpha_k = np.broadcast_to(np.asarray(O.alpha_k, dtype=np.float64), (4,)).copy()
        self.beta_k = np.broadcast_to(np.asarray(O.beta_k, dtype=np.float64), (4,)).copy()
        self.lam = float(O.ewm)
        self.fixed = bool(fixed)
        self.ewm: np.ndarray | None = None

    def step(self, Jn, sigma_p: float) -> np.ndarray:
        """Weights for slot t given normalised J~(t) (unsigned, as returned by
        Normalizer) and sigma_p(t); then updates the EWM."""
        c = SIGNS * np.asarray(Jn, dtype=np.float64)
        if self.fixed:
            w = self.w_base / 4.0
        else:
            dJ = np.zeros(4) if self.ewm is None else self.ewm - c
            e = -self.beta_k * dJ
            g = np.exp(e - e.max())
            g = g / g.sum()
            w = self.w_base * np.exp(self.alpha_k * float(sigma_p)) * g
        self.ewm = c.copy() if self.ewm is None else self.lam * self.ewm + (1.0 - self.lam) * c
        return w

    def get_state(self) -> dict:
        return {"ewm": None if self.ewm is None else self.ewm.copy()}

    def set_state(self, st: dict) -> None:
        self.ewm = None if st["ewm"] is None else np.asarray(st["ewm"], dtype=np.float64).copy()


def reward(Jn, w) -> float:
    """R(t) = -sum_k w_k c_k with c = (J~1, -J~2, J~3, J~4) (Eq. 51)."""
    return float(-np.sum(np.asarray(w) * SIGNS * np.asarray(Jn, dtype=np.float64)))
