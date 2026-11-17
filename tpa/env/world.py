"""Static scenario layout, Eq. (1) V(t) = U u D.

Everything here is drawn once per world from the `world` stream, in a fixed
order, so the world of a seed is identical across arms and blocks.

Units: positions m, f_max cycles/s, battery_j J (converted from Wh), ptx W.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WH_TO_J = 3600.0          # unit conversion, not a model parameter
GHZ = 1e9                 # unit conversion


@dataclass
class World:
    """Eq. (1): the vertex set U (UAVs) and D (devices) plus their fixed attributes.

    alpha is ONE path-loss exponent per world, kappa is per link (N,M), and
    device_ptx is per device; all fixed for the world."""
    zone_centers: np.ndarray   # (Z,2) m
    zone_weights: np.ndarray   # (Z,)  omega_z, Eq. 11
    zone_phase: np.ndarray     # (Z,)  varphi_z, one phase per zone
    device_pos: np.ndarray     # (N,2) initial horizontal positions q_i(0), m
    device_ptx: np.ndarray     # (N,)  P_i, W
    uav_pos: np.ndarray        # (M,2) horizontal hover positions p_bar_j, m
    f_max: np.ndarray          # (M,)  F_max^j, cycles/s
    eta: np.ndarray            # (M,)  eta_j, time-invariant
    battery_j: np.ndarray      # (M,)  E_j(0), J
    alpha: float               # path-loss exponent, one per world
    kappa: np.ndarray          # (N,M) Rician factor per link

    @property
    def n_devices(self) -> int:
        return int(self.device_pos.shape[0])

    @property
    def n_uav(self) -> int:
        return int(self.uav_pos.shape[0])

    @property
    def n_zones(self) -> int:
        return int(self.zone_centers.shape[0])


def zone_of(pos: np.ndarray, world: World) -> np.ndarray:
    """Zone of a position = Voronoi cell of the zone centres (nearest centre).

    pos (N,2) or (2,) -> int array (N,) or int."""
    p = np.asarray(pos, dtype=float)
    single = p.ndim == 1
    p2 = p[None, :] if single else p
    d2 = ((p2[:, None, :] - world.zone_centers[None, :, :]) ** 2).sum(-1)
    z = np.argmin(d2, axis=1)
    return int(z[0]) if single else z


def uav_grid(cfg) -> np.ndarray:
    """UAV hover positions on a coverage grid at [lo, hi] of the side:
    ceil(sqrt(M)) x ceil(sqrt(M)) grid, first M points."""
    sc = cfg.scenario
    m = int(sc.n_uav)
    side_n = int(math.ceil(math.sqrt(m)))
    g = np.linspace(sc.uav_grid_lo, sc.uav_grid_hi, side_n) * sc.area_side_m
    gx, gy = np.meshgrid(g, g)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)[:m].astype(float)


def build_world(cfg, rng: np.random.Generator) -> World:
    """Draw one world. Fixed draw order (do not reorder: it changes every world):
    zone centres, zone weights, zone phases, device home zones, cluster flags,
    Gaussian offsets (all N), uniform positions (all N), P_i, F_max, eta, E(0),
    alpha, kappa."""
    sc, ch, uv, wl = cfg.scenario, cfg.channel, cfg.uav, cfg.workload
    n, m, z = int(sc.n_devices), int(sc.n_uav), int(sc.n_zones)
    side = float(sc.area_side_m)

    zone_centers = rng.uniform(sc.zone_center_lo, sc.zone_center_hi, size=(z, 2)) * side
    zone_weights = rng.uniform(*wl.zone_weight, size=z)
    zone_phase = rng.uniform(0.0, 2.0 * math.pi, size=z)

    home = rng.integers(0, z, size=n)                      # cluster centre per device
    clustered = rng.random(n) < sc.device_cluster_frac
    gauss = zone_centers[home] + sc.device_cluster_sigma_frac * side * rng.standard_normal((n, 2))
    unif = rng.uniform(0.0, side, size=(n, 2))
    device_pos = np.where(clustered[:, None], gauss, unif)
    device_pos = np.clip(device_pos, 0.0, side)

    device_ptx = rng.uniform(*ch.tx_power_w, size=n)
    f_max = rng.uniform(*uv.compute_ghz, size=m) * GHZ
    eta = rng.uniform(*uv.eta, size=m)
    battery_j = rng.uniform(*uv.battery_wh, size=m) * WH_TO_J
    alpha = float(rng.uniform(*ch.path_loss_alpha))
    kappa = rng.uniform(*ch.rician_kappa, size=(n, m))

    return World(zone_centers=zone_centers, zone_weights=zone_weights, zone_phase=zone_phase,
                 device_pos=device_pos, device_ptx=device_ptx, uav_pos=uav_grid(cfg),
                 f_max=f_max, eta=eta, battery_j=battery_j, alpha=alpha, kappa=kappa)
