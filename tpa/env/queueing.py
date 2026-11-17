"""Delay model, Eqs. (13)-(16), and compute energy, Eq. (18). Shared by the
simulator and DRM so prediction and realization use one formula."""
from __future__ import annotations

import math


def t_trans(size_bits: float, phi: float, rate_bps: float, cfg) -> float:
    """Eq. (14): T_trans = s (1 + varrho phi) / R."""
    if rate_bps <= 0.0:
        return math.inf
    return size_bits * (1.0 + cfg.task.tx_overhead * phi) / rate_bps


def t_comp(cycles: float, phi: float, f: float) -> float:
    """Eq. (15) with eta applied once, to capacity: T_comp = c phi / f."""
    if f <= 0.0:
        return math.inf
    return cycles * phi / f


def f_required(cycles: float, phi: float, time_budget_s: float) -> float:
    """Minimal f (cycles/s) that finishes c*phi cycles within time_budget_s."""
    if time_budget_s <= 0.0:
        return math.inf
    return cycles * phi / time_budget_s


def mm1_wait(lam_eff: float, mu: float, eta: float) -> float:
    """Eq. (16): T_queue = rho / (mu eta - lam), rho = lam / (mu eta). Predictor only.
    Returns +inf when the queue is unstable (lam >= mu eta)."""
    if lam_eff <= 0.0:
        return 0.0                       # empty arrival stream: no wait (avoids cold-start deadlock)
    cap = mu * eta
    if cap <= 0.0 or lam_eff >= cap:
        return math.inf
    rho = lam_eff / cap
    return rho / (cap - lam_eff)


def predict_total_time(size_bits: float, cycles: float, phi: float, rate_bps: float,
                       f: float, lam_eff: float, mu: float, eta: float,
                       t_mem_s: float, cfg) -> float:
    """Eq. (13): T_total = T_trans + T_comp + T_queue + T_mem."""
    return (t_trans(size_bits, phi, rate_bps, cfg) + t_comp(cycles, phi, f)
            + mm1_wait(lam_eff, mu, eta) + t_mem_s)


def comp_energy(cycles: float, f: float, phi: float, cfg) -> float:
    """Eq. (18): E_comp = kappa_c c f^2 phi (J)."""
    return cfg.uav.kappa_c * cycles * f * f * phi


def t_mem_seconds(cfg, arm_key: str) -> float:
    """T_mem (Eq. 13), measured then frozen per arm; 0 if not calibrated."""
    tm = cfg.t_mem_ms
    return float(tm.get(arm_key, tm.get("A0", 0.0))) / 1e3 if arm_key not in NO_MEMORY_ARMS else 0.0


NO_MEMORY_ARMS = {"A9", "A10", "A11", "A12", "B_ZHOU", "B_QIN", "B_HUANG", "B_SONG"}
