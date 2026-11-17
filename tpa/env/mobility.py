"""Mobility models (robustness workloads).

* `static`: devices and UAVs never move (default; Eq. 3 fixed deployments).
* `GaussMarkov`: device velocity v(t+1) = a v(t) + (1-a) v_bar + sigma sqrt(1-a^2) w(t),
  w ~ N(0, I_2), per device; v_bar has magnitude gm_mean_speed and a per-device
  direction drawn once. Reflecting boundaries on [0, side]^2 (position mirrored,
  the velocity and mean-velocity component normal to the wall flipped).
  Consumes exactly N*2 normals per step from the `mobility` stream.
* `Patrol`: each UAV flies a circle of radius patrol_radius_m about its hover
  point p_bar_j at patrol_speed_ms (deterministic; evenly spaced start angles).
"""
from __future__ import annotations

import math

import numpy as np


class GaussMarkov:
    def __init__(self, cfg, rng: np.random.Generator, n: int | None = None):
        wl = cfg.workload
        self.rng = rng
        self.a = float(wl.gm_alpha)
        self.sigma = float(wl.gm_sigma)
        self.side = float(cfg.scenario.area_side_m)
        self.dt = float(cfg.scenario.slot_s)
        n = int(cfg.scenario.n_devices if n is None else n)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
        self.v_bar = float(wl.gm_mean_speed) * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self.v = self.v_bar.copy()

    def step(self, device_pos: np.ndarray) -> np.ndarray:
        a = self.a
        w = self.rng.standard_normal(self.v.shape)
        self.v = a * self.v + (1.0 - a) * self.v_bar + self.sigma * math.sqrt(1.0 - a * a) * w
        p = device_pos + self.v * self.dt
        lo, hi = 0.0, self.side
        for _ in range(2):                       # a reflection can at most overshoot once more
            below, above = p < lo, p > hi
            out = below | above
            p = np.where(below, 2 * lo - p, np.where(above, 2 * hi - p, p))
            self.v = np.where(out, -self.v, self.v)
            self.v_bar = np.where(out, -self.v_bar, self.v_bar)
        return np.clip(p, lo, hi)

    def get_state(self) -> dict:
        return {"v": self.v.copy(), "v_bar": self.v_bar.copy(), "rng": self.rng.bit_generator.state}

    def set_state(self, s: dict) -> None:
        self.v, self.v_bar = s["v"].copy(), s["v_bar"].copy()
        self.rng.bit_generator.state = s["rng"]


class Patrol:
    def __init__(self, cfg, world):
        uv = cfg.uav
        self.center = world.uav_pos.copy()
        self.r = float(uv.patrol_radius_m)
        self.v = float(uv.patrol_speed_ms)
        self.dt = float(cfg.scenario.slot_s)
        m = self.center.shape[0]
        self.theta0 = 2.0 * math.pi * np.arange(m) / m

    def uav_pos(self, t: int) -> np.ndarray:
        ang = self.theta0 + (self.v / self.r) * t * self.dt
        return self.center + self.r * np.stack([np.cos(ang), np.sin(ang)], axis=1)

    def speed(self) -> float:
        return self.v
