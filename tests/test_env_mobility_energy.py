"""Mobility (Gauss-Markov, patrol) and log-only energy models."""
import math

import numpy as np
import pytest

from tpa.config import load_config
from tpa.env.energy import (downlink_energy, downlink_rate, hover_power, propulsion_power,
                            step_energy, uplink_energy)
from tpa.env.mobility import GaussMarkov, Patrol
from tpa.env.world import build_world
from tpa.rng import Streams

cfg = load_config()


def test_gauss_markov_bounds_and_fixed_consumption():
    side = cfg.scenario.area_side_m
    g1 = GaussMarkov(cfg, np.random.Generator(np.random.PCG64(1)))
    g2 = GaussMarkov(cfg, np.random.Generator(np.random.PCG64(1)))
    p1 = np.full((200, 2), 10.0)
    p2 = np.full((200, 2), 2500.0)
    for _ in range(3000):
        p1 = g1.step(p1); p2 = g2.step(p2)
        assert p1.min() >= 0 and p1.max() <= side
    assert g1.rng.bit_generator.state == g2.rng.bit_generator.state
    speed = np.linalg.norm(g2.v, axis=1).mean()
    assert 0.5 < speed < 3.0


def test_patrol_circle():
    w = build_world(cfg, Streams(1).world)
    pt = Patrol(cfg, w)
    p0, p1 = pt.uav_pos(0), pt.uav_pos(1)
    assert np.allclose(np.linalg.norm(p0 - w.uav_pos, axis=1), 200.0)
    chord = 2 * 200 * math.sin(10.0 / 200 / 2)
    assert np.allclose(np.linalg.norm(p1 - p0, axis=1), chord)
    assert pt.speed() == 10.0


def test_zeng_hover_and_propulsion():
    rho, s, R, Om, k, d = 1.225, 0.05, 0.4, 300.0, 0.1, 0.012
    A = 4 * math.pi * R * R
    W = 5.0 * 9.81
    hand = d / 8 * rho * s * A * Om ** 3 * R ** 3 + (1 + k) * W ** 1.5 / math.sqrt(2 * rho * A)
    assert hover_power(cfg) == pytest.approx(hand, rel=1e-12)
    assert propulsion_power(10.0, cfg) != hover_power(cfg)


def test_step_energy_and_log_energies():
    E = np.array([100.0, 100.0])
    out = step_energy(E, np.array([1.0, 0.0]), 0.5, np.array([0, 1]), cfg)
    assert out.tolist() == [98.5, 100.0 - 0.5 - 40.0]
    assert uplink_energy(0.5, 2.0) == pytest.approx(1.0)
    r = downlink_rate(1e-10, 0.9, cfg)
    assert r == pytest.approx(0.9 * 1e7 * math.log2(1 + 0.5 * 1e-10 / 4e-14))
    assert downlink_energy(8e6, r, cfg) == pytest.approx(0.5 * 0.8e6 / r)
