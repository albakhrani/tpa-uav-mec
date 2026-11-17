"""Energy model, Eqs. (17)-(18), plus log-only hover/propulsion, uplink and downlink.

Eq. (17) as implemented:
    E_j(t+1) = E_j(t) - sum_i E_comp_ij x_ij - kappa_m ||r(t)||_1 - Delta_E_b delta_t B_j(t)
with E_harvest = 0. E_comp is charged at assignment.
"""
from __future__ import annotations

import math

import numpy as np

from .queueing import comp_energy  # noqa: F401  Eq. (18), re-exported (defined in queueing.py)

def _zeng_terms(cfg):
    """Rotary-wing constants of Zeng, Xu, Zhang, IEEE TWC 2019 from cfg.uav.zeng
    (disc area of n_rotors rotors, 4 by default); mass from cfg.uav.mass_kg. Log only."""
    p = cfg.uav.zeng.to_dict()
    area = p["n_rotors"] * math.pi * p["radius_m"] ** 2
    p0 = (p["delta"] / 8.0) * p["rho"] * p["s"] * area * p["omega_rad_s"] ** 3 * p["radius_m"] ** 3
    w = float(cfg.uav.mass_kg) * p["g"]
    pi = (1.0 + p["k_corr"]) * w ** 1.5 / math.sqrt(2.0 * p["rho"] * area)
    u_tip = p["omega_rad_s"] * p["radius_m"]
    v0 = math.sqrt(w / (2.0 * p["rho"] * area))
    return p, area, p0, pi, u_tip, v0


def propulsion_power(v: float, cfg) -> float:
    """Zeng 2019, Eq. (12): P(V) = P0(1 + 3V^2/U_tip^2)
    + P_i (sqrt(1 + V^4/(4 v0^4)) - V^2/(2 v0^2))^(1/2) + 0.5 d0 rho s A V^3  (W)."""
    p, area, p0, pi, u_tip, v0 = _zeng_terms(cfg)
    v = float(v)
    blade = p0 * (1.0 + 3.0 * v * v / (u_tip * u_tip))
    induced = pi * math.sqrt(math.sqrt(1.0 + v ** 4 / (4.0 * v0 ** 4)) - v * v / (2.0 * v0 * v0))
    parasite = 0.5 * p["d0"] * p["rho"] * p["s"] * area * v ** 3
    return blade + induced + parasite


def hover_power(cfg) -> float:
    """Zeng 2019 at V = 0: P0 + P_i (W). Log only (excluded from objectives)."""
    return propulsion_power(0.0, cfg)


def uplink_energy(ptx, seconds):
    """Device-side uplink energy P_i * T (J)."""
    return np.asarray(ptx) * np.asarray(seconds)


def downlink_rate(h, eta, cfg):
    """Result downlink rate over the reciprocal channel at P_U, no interference
    (log-only model): eta_j B log2(1 + P_U h / (N0 B))."""
    c = cfg.channel
    snr = cfg.uav.downlink_power_w * np.asarray(h) / (c.noise_psd_w_hz * c.bandwidth_hz)
    return np.asarray(eta) * c.bandwidth_hz * np.log2(1.0 + snr)


def downlink_energy(size_bits, rate, cfg):
    """UAV transmit energy for the result: P_U * s_out / R_down, s_out = downlink_frac * s (J).
    `size_bits` is the task input size s_i."""
    s_out = cfg.uav.downlink_frac * np.asarray(size_bits)
    return cfg.uav.downlink_power_w * s_out / np.asarray(rate)


def step_energy(E: np.ndarray, comp_j: np.ndarray, mem_energy_j: float, burst: np.ndarray, cfg) -> np.ndarray:
    """Eq. (17), one slot: E - sum E_comp x - kappa_m||r||_1 (per UAV) - Delta_E_b delta_t B_j.
    E_harvest = 0 (manuscript)."""
    burst_j = cfg.burst.delta_e_w * cfg.scenario.slot_s * np.asarray(burst, dtype=float)
    return np.asarray(E) - np.asarray(comp_j) - float(mem_energy_j) - burst_j
