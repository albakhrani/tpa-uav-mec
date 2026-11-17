"""MEPR -- Memory-Enhanced Pattern Recognition (Alg. 1; Eqs. 44, 47).

f_encode : f_tau = e_tau = tanh(W_S x_tau), the per-slot encoding written to M_S
           (Alg. 1 l.3). The window is tau = t-w .. t (w+1 rows, fewer at the
           start of a segment); A2 (no M_S): the current slot only.
f_corr   : c_tau = max(0, cos(f_tau, EWM of the previous `lag` features in the window)),
           EWM weights exp(-0.5 j), j = 0 for the most recent predecessor;
           c_tau = 1 for a row with no predecessor. c_tau is DETACHED.
f_agg    : P_z(t) = mean_tau c_tau * a~_z(tau) * f_tau, a~_z = A[tau, z] / zone_mean[z].
           P(t) in R^{Z x d} (one row per zone). B_max = w + 1 never binds.
sigma_p  : ||P - P_ref||_2 / ||P_ref||_2 with the SPECTRAL norm, clipped at sigma_max.
P_ref    : sigma_p = 0 and no consolidation during the first w steps of a segment;
           then P_ref = mean of those w patterns; afterwards P_ref <- (1-a) P_ref + a P
           (a = memory.alpha_ref; default.toml sets 1/3600, one over the workload period),
           updated after sigma_p is computed.
v_t      : zone-resolved candidate vector in R^{Z+M}:
           v_t = [a_bar_t ; o_t], a_bar_t[z] = mean over the same window rows as P of
           a~_z(tau) = A[tau, z] / zone_mean_t[z] (zone_mean of the CURRENT slot for every
           row, as in compute_P_batch), o_t = uav_occupancy_t (M,) passed by the caller
           (zeros if None). compute_v_batch is the pure torch function reused at replay.
v_ref    : kept exactly like P_ref (None during warm-up, mean of the w warm-up vectors,
           then EWM with alpha_ref, updated after use); v_ref_used = the reference of
           this slot (None during warm-up), logged per event for "novelty" reconstruction.
"""
from __future__ import annotations

import numpy as np
import torch

from . import EPS, MemoryFlags


def compute_P_batch(F: torch.Tensor, A: torch.Tensor, zone_mean: torch.Tensor,
                    mask: torch.Tensor, lag: int) -> torch.Tensor:
    """Eq. (44) / Alg. 1 lines 2-10, batched. Torch pure function (reused at replay).

    F (B, n, d) features, A (B, n, Z) zone arrivals, zone_mean (B, Z), mask (B, n) bool.
    Valid rows must be right-aligned (padding at the front), oldest first.
    Returns P (B, Z, d).
    """
    B, n, d = F.shape
    mf = mask.to(F.dtype)
    with torch.no_grad():                                   # c_tau is detached
        Fd = F.detach() * mf[..., None]
        num = torch.zeros_like(Fd)
        den = torch.zeros(B, n, dtype=F.dtype, device=F.device)
        for j in range(1, min(lag, n - 1) + 1):
            wj = float(np.exp(-0.5 * (j - 1)))
            num[:, j:] += wj * Fd[:, :-j]
            den[:, j:] += wj * mf[:, :-j]
        has = den > 0
        ref = num / torch.clamp(den, min=EPS)[..., None]
        cos = (Fd * ref).sum(-1) / (torch.linalg.vector_norm(Fd, dim=-1)
                                    * torch.linalg.vector_norm(ref, dim=-1) + EPS)
        c = torch.where(has, torch.clamp(cos, min=0.0), torch.ones_like(cos)) * mf
    a_t = A * mf[..., None] / torch.clamp(zone_mean, min=EPS)[:, None, :]      # (B, n, Z)
    cnt = torch.clamp(mf.sum(dim=1), min=1.0)                                  # (B,)
    P = torch.einsum("bn,bnz,bnd->bzd", c, a_t, F) / cnt[:, None, None]
    return P


def compute_v_batch(A: torch.Tensor, zone_mean: torch.Tensor, mask: torch.Tensor,
                    uav_occ: torch.Tensor) -> torch.Tensor:
    """Zone-resolved candidate vector v_t, batched. Torch pure function (reused at replay),
    float32 arithmetic.

    A (B, n, Z) zone arrivals, zone_mean (B, Z), mask (B, n) bool (valid rows),
    uav_occ (B, M). Returns v (B, Z + M) = [mean over valid rows of A / zone_mean ; uav_occ].
    """
    mf = mask.to(A.dtype)
    a_t = A * mf[..., None] / torch.clamp(zone_mean, min=EPS)[:, None, :]      # (B, n, Z)
    cnt = torch.clamp(mf.sum(dim=1), min=1.0)                                  # (B,)
    a_bar = a_t.sum(dim=1) / cnt[:, None]
    return torch.cat([a_bar, uav_occ.to(A.dtype)], dim=-1)


def compute_v(A_win, zone_mean, uav_occ) -> torch.Tensor:
    """Zone-resolved candidate vector v_t (Z + M,) float32 from A_win (n, Z), zone_mean (Z,),
    uav_occ (M,)."""
    A = torch.as_tensor(np.asarray(A_win, dtype=np.float32)) if not isinstance(A_win, torch.Tensor) \
        else A_win.to(torch.float32)
    dev = A.device
    zm = torch.as_tensor(np.asarray(zone_mean, dtype=np.float32), device=dev) \
        if not isinstance(zone_mean, torch.Tensor) else zone_mean.to(device=dev, dtype=torch.float32)
    uo = torch.as_tensor(np.asarray(uav_occ, dtype=np.float32), device=dev) \
        if not isinstance(uav_occ, torch.Tensor) else uav_occ.to(device=dev, dtype=torch.float32)
    mask = torch.ones(1, A.shape[0], dtype=torch.bool, device=dev)
    return compute_v_batch(A[None], zm[None], mask, uo[None])[0]


def compute_P(F_win, A_win, zone_mean, lag: int) -> torch.Tensor:
    """Eq. (44): P(t) (Z, d) from F_win (n, d), A_win (n, Z), zone_mean (Z,)."""
    F = F_win if isinstance(F_win, torch.Tensor) else torch.as_tensor(np.asarray(F_win, np.float32))
    F = F.to(torch.float32)
    dev = F.device
    A = torch.as_tensor(np.asarray(A_win, dtype=np.float32), device=dev) \
        if not isinstance(A_win, torch.Tensor) else A_win.to(device=dev, dtype=torch.float32)
    zm = torch.as_tensor(np.asarray(zone_mean, dtype=np.float32), device=dev) \
        if not isinstance(zone_mean, torch.Tensor) else zone_mean.to(device=dev, dtype=torch.float32)
    mask = torch.ones(1, F.shape[0], dtype=torch.bool, device=dev)
    return compute_P_batch(F[None], A[None], zm[None], mask, lag)[0]


def spectral_ratio(P: np.ndarray, P_ref: np.ndarray, sigma_max: float) -> float:
    """Eq. (47): min(||P - P_ref||_2 / max(||P_ref||_2, eps), sigma_max), spectral norm."""
    num = float(np.linalg.norm(P - P_ref, 2))
    den = max(float(np.linalg.norm(P_ref, 2)), EPS)
    return min(num / den, float(sigma_max))


class MEPR:
    """Alg. 1, run every slot. Keeps its own window buffer of (e_tau, zone arrivals).

        P, sigma_p, cons = mepr.step_buffered(e_t, zone_arr_t, zone_mean_t, uav_occ_t)
    or  mepr.push(e_t, zone_arr_t); P, sigma_p, cons = mepr.step(*mepr.window(), zone_mean_t,
                                                                  uav_occ_t)

    P is returned as float64 numpy (Z, d) (the float32 torch value is in .P_torch).
    The slot's zone-resolved vector v_t is in .v (float64 numpy (Z+M,)) / .v_torch
    (float32) and its reference in .v_ref_used (None during warm-up); uav_occ_t = None
    means zeros(M). `cons` is always False when the arm has no consolidation path (A1)."""

    def __init__(self, cfg, flags: MemoryFlags):
        mc = cfg.memory
        self.M = int(cfg.scenario.n_uav)
        self.w = int(mc.window_w)
        self.lag = int(mc.corr_lag)
        self.sigma_max = float(mc.sigma_max)
        self.alpha_ref = float(mc.alpha_ref)
        self.win_len = self.w + 1 if flags.use_ms else 1           # A2: current slot only
        self.theta = float(mc.theta_p) if flags.gate_on else float("-inf")   # A5
        self.can_consolidate = bool(flags.use_ml or flags.flat)    # A1: no consolidation path
        self.reset()

    def reset(self) -> None:
        self.F_buf: list[np.ndarray] = []
        self.A_buf: list[np.ndarray] = []
        self.n_steps = 0
        self.P_sum = None
        self.P_ref = None
        self.P_ref_used = None      # the reference sigma_p was measured against this slot
        self.P = None
        self.P_prev = None
        self.P_torch = None
        self.sigma_p = 0.0
        self.v_sum = None           # v_ref warm-up accumulator (mirrors P_sum)
        self.v_ref = None
        self.v_ref_used = None
        self.v = None
        self.v_torch = None

    def push(self, e, zone_arr) -> None:
        e = e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else np.asarray(e)
        self.F_buf.append(np.asarray(e, np.float32).reshape(-1).copy())
        self.A_buf.append(np.asarray(zone_arr, np.float64).copy())
        if len(self.F_buf) > self.win_len:
            self.F_buf.pop(0)
            self.A_buf.pop(0)

    def window(self):
        return np.stack(self.F_buf), np.stack(self.A_buf)

    @torch.no_grad()
    def step(self, F_win, A_win, zone_mean, uav_occ=None):
        """Alg. 1: P(t) (Eq. 44), sigma_p (Eq. 47), consolidate = sigma_p > theta_p;
        v(t) from the same window rows and v_ref with the P_ref rules."""
        P_t = compute_P(F_win, A_win, zone_mean, self.lag)
        P = P_t.double().cpu().numpy()
        uo = np.zeros(self.M) if uav_occ is None else np.asarray(uav_occ, np.float64).reshape(self.M)
        v_t = compute_v(A_win, zone_mean, uo)
        v = v_t.double().cpu().numpy()
        self.n_steps += 1
        cons = False
        if self.n_steps <= self.w:                                  # warm-up
            sigma = 0.0
            self.P_ref_used = None
            self.v_ref_used = None
            self.P_sum = P.copy() if self.P_sum is None else self.P_sum + P
            self.v_sum = v.copy() if self.v_sum is None else self.v_sum + v
            if self.n_steps == self.w:
                self.P_ref = self.P_sum / self.w
                self.v_ref = self.v_sum / self.w
        else:
            if self.P_ref is None:                                  # only if w == 0
                self.P_ref = P.copy()
            if self.v_ref is None:
                self.v_ref = v.copy()
            self.P_ref_used = self.P_ref.copy()
            self.v_ref_used = self.v_ref.copy()
            sigma = spectral_ratio(P, self.P_ref, self.sigma_max)
            cons = self.can_consolidate and sigma > self.theta
            self.P_ref = (1.0 - self.alpha_ref) * self.P_ref + self.alpha_ref * P
            self.v_ref = (1.0 - self.alpha_ref) * self.v_ref + self.alpha_ref * v
        self.P_prev, self.P, self.P_torch, self.sigma_p = self.P, P, P_t, sigma
        self.v, self.v_torch = v, v_t
        return P, sigma, bool(cons)

    def step_buffered(self, e, zone_arr, zone_mean, uav_occ=None):
        self.push(e, zone_arr)
        return self.step(*self.window(), zone_mean, uav_occ)

    def get_state(self) -> dict:
        c = lambda a: None if a is None else np.array(a, copy=True)  # noqa: E731
        return dict(F_buf=[f.copy() for f in self.F_buf], A_buf=[a.copy() for a in self.A_buf],
                    n_steps=self.n_steps, P_sum=c(self.P_sum), P_ref=c(self.P_ref), P=c(self.P),
                    P_ref_used=c(self.P_ref_used),
                    P_prev=c(self.P_prev),
                    P_torch=None if self.P_torch is None else self.P_torch.cpu().numpy().copy(),
                    sigma_p=self.sigma_p,
                    v_sum=c(self.v_sum), v_ref=c(self.v_ref), v_ref_used=c(self.v_ref_used),
                    v=c(self.v),
                    v_torch=None if self.v_torch is None else self.v_torch.cpu().numpy().copy())

    def set_state(self, s: dict) -> None:
        c = lambda a: None if a is None else np.array(a, copy=True)  # noqa: E731
        self.F_buf = [np.array(f, np.float32) for f in s["F_buf"]]
        self.A_buf = [np.array(a, np.float64) for a in s["A_buf"]]
        self.n_steps = int(s["n_steps"])
        self.P_sum, self.P_ref, self.P, self.P_prev = c(s["P_sum"]), c(s["P_ref"]), c(s["P"]), c(s["P_prev"])
        self.P_ref_used = c(s.get("P_ref_used"))
        self.P_torch = None if s["P_torch"] is None else torch.as_tensor(s["P_torch"])
        self.sigma_p = float(s["sigma_p"])
        self.v_sum, self.v_ref = c(s.get("v_sum")), c(s.get("v_ref"))
        self.v_ref_used, self.v = c(s.get("v_ref_used")), c(s.get("v"))
        vt = s.get("v_torch")
        self.v_torch = None if vt is None else torch.as_tensor(vt)
