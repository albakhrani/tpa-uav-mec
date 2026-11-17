"""Eqs. (2), (4)-(9): channel model vs hand-computed values."""
import math

import numpy as np
import pytest

from tpa.config import load_config
from tpa.env import channel as ch
from tpa.env.world import World, build_world, zone_of
from tpa.rng import Streams

cfg = load_config()
C = cfg.channel
H = cfg.scenario.hover_altitude_m


def tiny_world(kappa, alpha, ptx, eta, dev, uav):
    n, m = len(dev), len(uav)
    return World(zone_centers=np.zeros((1, 2)), zone_weights=np.ones(1), zone_phase=np.zeros(1),
                 device_pos=np.asarray(dev, float), device_ptx=np.asarray(ptx, float),
                 uav_pos=np.asarray(uav, float), f_max=np.ones(m), eta=np.asarray(eta, float),
                 battery_j=np.ones(m), alpha=alpha, kappa=np.asarray(kappa, float).reshape(n, m))


def test_eq6_los_probability_hand():
    # theta = 90 deg directly below, d_ground = 0
    p = ch.los_probability(90.0, 0.0, cfg)
    hand = 1.0 / (1.0 + 9.61 * math.exp(-0.16 * (90.0 - 9.61))) * math.exp(-2e-4 * 0.0)
    assert p == pytest.approx(hand, rel=1e-12)
    assert p == pytest.approx(0.99997511, abs=1e-7)
    # 45 deg at 100 m ground distance
    p2 = ch.los_probability(45.0, 100.0, cfg)
    hand2 = 1.0 / (1.0 + 9.61 * math.exp(-0.16 * 35.39)) * math.exp(-0.02)
    assert p2 == pytest.approx(hand2, rel=1e-12)


def test_eq5_env_factor_hand():
    assert ch.env_factor(0.25, cfg) == pytest.approx(0.25 * 1.0 + 0.75 * 0.2)
    assert ch.env_factor(1.0, cfg) == pytest.approx(1.0)
    assert ch.env_factor(0.0, cfg) == pytest.approx(0.2)


def test_eq4_channel_gain_hand():
    # device at (0,0), UAV at (300,400) -> d_ground 500, d^2 = 250000 + 10000
    w = tiny_world(kappa=[[10.0]], alpha=3.0, ptx=[0.5], eta=[0.9], dev=[[0, 0]], uav=[[300, 400]])
    xi = np.array([[1.3]])
    h = ch.channel_gain(w.device_pos, w.uav_pos, w, xi, cfg)[0, 0]
    d2 = 500.0 ** 2 + H ** 2
    theta = math.degrees(math.atan2(H, 500.0))
    plos = 1.0 / (1.0 + 9.61 * math.exp(-0.16 * (theta - 9.61))) * math.exp(-2e-4 * 500.0)
    psi = plos * 1.0 + (1 - plos) * 0.2
    hand = 1e-4 * (10.0 / d2) ** 1.5 * 1.3 * psi
    assert h == pytest.approx(hand, rel=1e-12)


def test_eq2_viable():
    h = np.array([[2e-11, 1e-11, 0.5e-11]])
    assert ch.viable(h, cfg).tolist() == [[True, True, False]]


def test_eq8_system_noise_hand():
    z = ch.system_noise(np.array([0, 3]), np.array([0.8, 0.9]), cfg)
    assert z[0] == pytest.approx(1e-12 * 0.2)
    assert z[1] == pytest.approx(1e-12 * 0.1 + 3e-13)


def test_eq9_interference_hand():
    # 3 devices, 2 UAVs. Active: dev0->U0, dev1->U0, dev2->U1
    h = np.array([[1e-10, 2e-10], [3e-10, 4e-10], [5e-10, 6e-10]])
    p = np.array([1.0, 0.5, 0.2])
    act = np.array([[1, 0], [1, 0], [0, 1]], bool)
    I = ch.interference(h, p, act, cfg)
    phi = 0.1
    # I_00: co-channel k != 0 on U0: dev1 -> 0.5*3e-10; cross: sum over l=1, k: dev2 -> 0.2*6e-10
    assert I[0, 0] == pytest.approx(0.5 * 3e-10 + phi * 0.2 * 6e-10)
    assert I[1, 0] == pytest.approx(1.0 * 1e-10 + phi * 0.2 * 6e-10)
    # I_21: co-channel on U1 excluding itself: none; cross from U0 uploads: dev0, dev1 gains to U0
    assert I[2, 1] == pytest.approx(phi * (1.0 * 1e-10 + 0.5 * 3e-10))
    # non-active pair (dev2, U0): would-be interference = all U0 uploads + cross from U1
    assert I[2, 0] == pytest.approx(1.0 * 1e-10 + 0.5 * 3e-10 + phi * 0.2 * 6e-10)


def test_eq7_rate_hand():
    h = np.array([[1e-10, 2e-10], [3e-10, 4e-10], [5e-10, 6e-10]])
    p = np.array([1.0, 0.5, 0.2])
    eta = np.array([0.8, 0.95])
    act = np.array([[1, 0], [1, 0], [0, 1]], bool)
    R = ch.rate(h, p, act, eta, cfg)
    N0B = 4e-21 * 1e7
    I00 = 0.5 * 3e-10 + 0.1 * 0.2 * 6e-10
    z0 = 1e-12 * 0.2 + 1e-13 * 2
    hand00 = 0.8 * 1e7 * math.log2(1 + 1.0 * 1e-10 / (N0B + I00 + z0))
    assert R[0, 0] == pytest.approx(hand00, rel=1e-12)
    # not-active pair (dev2, U0) counts itself in zeta: n_0 + 1 = 3
    I20 = 1.0 * 1e-10 + 0.5 * 3e-10 + 0.1 * 0.2 * 6e-10
    z0p = 1e-12 * 0.2 + 1e-13 * 3
    hand20 = 0.8 * 1e7 * math.log2(1 + 0.2 * 5e-10 / (N0B + I20 + z0p))
    assert R[2, 0] == pytest.approx(hand20, rel=1e-12)
    # no interferers: plain SNR
    R0 = ch.rate(h, p, np.zeros_like(act), eta, cfg)
    hand = 0.95 * 1e7 * math.log2(1 + 0.2 * 6e-10 / (N0B + 1e-12 * 0.05 + 1e-13))
    assert R0[2, 1] == pytest.approx(hand, rel=1e-12)


def test_fading_unit_mean_and_full_draw():
    rng = np.random.Generator(np.random.PCG64(0))
    kappa = np.full((200, 5), 10.0)
    xs = np.stack([ch.draw_fading(rng, kappa) for _ in range(200)])
    assert xs.shape == (200, 200, 5)
    assert xs.mean() == pytest.approx(1.0, abs=0.01)
    # exactly 2*N*M normals per call
    r1 = np.random.Generator(np.random.PCG64(5))
    r2 = np.random.Generator(np.random.PCG64(5))
    ch.draw_fading(r1, kappa)
    r2.standard_normal((2, 200, 5))
    assert r1.bit_generator.state == r2.bit_generator.state


def test_world_draws_and_voronoi():
    st = Streams(1)
    w = build_world(cfg, st.world)
    assert isinstance(w.alpha, float) and 2.5 <= w.alpha <= 3.5
    assert w.kappa.shape == (200, 5) and w.kappa.min() >= 5 and w.kappa.max() <= 15
    assert w.device_ptx.shape == (200,) and 0.1 <= w.device_ptx.min() and w.device_ptx.max() <= 1.0
    assert w.battery_j.min() >= 50 * 3600 and w.battery_j.max() <= 100 * 3600
    # same seed -> same world, different block irrelevant
    w2 = build_world(cfg, Streams(1, "test").world)
    assert np.array_equal(w.kappa, w2.kappa) and w.alpha == w2.alpha
    z = zone_of(w.device_pos, w)
    d = np.linalg.norm(w.device_pos[:, None] - w.zone_centers[None], axis=-1)
    assert np.array_equal(z, d.argmin(1))
