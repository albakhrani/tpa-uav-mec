"""Burst mode (Eqs. 57-58) unit tests."""
import numpy as np
import pytest

from tpa.config import load_config
from tpa.drm import DRM, Burst, load_pressure
from tpa.types import Assignment, Decision, SlotObs, Task

CFG = load_config()
M = CFG.scenario.n_uav
Z = CFG.scenario.n_zones
D = CFG.memory.d


def mk_task(tid, zone=0):
    return Task(tid=tid, device=tid, zone=zone, t_arr=0, size_bits=1e6, cycles=1e9,
                deadline_s=2.0, priority=0.5, complexity=1.0,
                loss_u=np.zeros(CFG.task.max_tx_attempts), truth_bin=-1)


def mk_obs(tasks, energy=1e5, e_min=0.0):
    Q = len(tasks)
    cap = np.full(M, 4e9)
    return SlotObs(t=0, S_sys=np.zeros(49), queue=list(tasks),
                   rates_pred=np.full((Q, M), 1e8), h_now=np.ones((Q, M)),
                   zone_arrivals=np.zeros(Z), zone_mean=np.ones(Z), residual_f=cap.copy(),
                   cap_bound=cap, energy=np.full(M, float(energy)) if np.isscalar(energy) else np.asarray(energy, float),
                   e_min=np.full(M, e_min), burst=np.zeros(M), lam_eff=np.zeros(M),
                   mu=np.full(M, 10.0))


def P_diag(v):
    P = np.zeros((Z, D))
    P[0, 0] = v
    return P


def test_eq58_load_pressure_hand():
    terms = np.zeros(M)
    terms[0] = 0.5 * 2e9                                   # pi * f
    P, P_prev, P_ref = P_diag(2.0), np.zeros((Z, D)), P_diag(4.0)
    phi = load_pressure(terms, np.full(M, 4e9), P, P_prev, P_ref, 1.0, 0.5)
    assert phi[0] == pytest.approx(0.25 + 0.5 * 2.0 / 4.0)
    assert phi[1] == pytest.approx(0.25)
    # spectral (not Frobenius) norm: rank-1 matrix of ones (6 x 64): ||.||_2 = sqrt(6*64)
    ones = np.ones((Z, D))
    phi2 = load_pressure(np.zeros(M), np.full(M, 4e9), ones, np.zeros((Z, D)), np.eye(Z, D), 1.0, 0.5)
    assert phi2[0] == pytest.approx(0.5 * np.sqrt(Z * D) / 1.0)


def test_eq57_indicator_hand():
    b = Burst(CFG, {"theta_b": 0.4})
    obs = mk_obs([mk_task(0)], energy=[1e5, 1e5, 0.0, 1e5, 1e5])
    dec = Decision(assignments=[Assignment(0, 0, 2e9, 1.0)])
    pi = np.array([0.5])
    B, _ = b.step(obs, dec, P_diag(2.0), np.zeros((Z, D)), P_diag(4.0), pi=pi, f_max=np.full(M, 4e9))
    # Phi = [0.5, 0.25, 0.25, 0.25, 0.25] > 0.4 only for UAV 0
    assert list(B) == [1, 0, 0, 0, 0]
    b2 = Burst(CFG, {"theta_b": 0.1})
    B2, _ = b2.step(obs, dec, P_diag(2.0), np.zeros((Z, D)), P_diag(4.0), pi=pi, f_max=np.full(M, 4e9))
    assert list(B2) == [1, 1, 0, 1, 1]                      # UAV 2 has E <= E_min


def test_tb_cap_and_cooldown():
    b = Burst(CFG, {"theta_b": 0.0})
    de = CFG.burst.delta_e_w * CFG.scenario.slot_s
    obs = mk_obs([], energy=3 * de, e_min=0.0)              # T_b = 3 slots
    assert np.all(b.t_b(obs.energy, obs.e_min) == 3)
    seq = []
    for _ in range(3 + CFG.burst.cooldown_slots + 2):
        B, _ = b.step(obs, Decision(), P_diag(1.0), np.zeros((Z, D)), P_diag(1.0),
                      f_max=np.full(M, 4e9))
        seq.append(B[0])
    ones = [k for k, v in enumerate(seq) if v == 1]
    assert ones == [0, 1, 2, 3 + CFG.burst.cooldown_slots, 4 + CFG.burst.cooldown_slots]
    # T_max cap with ample energy
    b = Burst(CFG, {"theta_b": 0.0})
    assert np.all(b.t_b(np.full(M, 1e9), np.zeros(M)) == CFG.burst.t_max_slots)


def test_a8_always_zero_and_a0_can_turn_on():
    P, P_prev, P_ref = P_diag(5.0), np.zeros((Z, D)), P_diag(1.0)
    obs = mk_obs([mk_task(0)])
    a = np.zeros((CFG.task.q_max, 2 * M))
    kw = dict(f_max=np.full(M, 4e9), eta=np.full(M, 0.9))
    a0 = DRM(CFG, {}, "A0", **kw)
    a8 = DRM(CFG, {"burst_on": False}, "A8", **kw)
    d0 = a0.run(obs, np.array([0]), a, P, 0.5, P_prev=P_prev, P_ref=P_ref)
    d8 = a8.run(obs, np.array([0]), a, P, 0.5, P_prev=P_prev, P_ref=P_ref)
    assert d0.burst_next.sum() == M                          # pattern term 0.5*5/1 > 0.75
    assert np.all(d8.burst_next == 0)
    for _ in range(50):
        d8 = a8.run(obs, np.array([0]), a, P, 0.5, P_prev=P_prev, P_ref=P_ref)
        assert np.all(d8.burst_next == 0)
    # A6 (beta_b = 0): the pattern-change term is removed
    a6 = DRM(CFG, {"pattern_terms": False}, "A6", **kw)
    d6 = a6.run(obs, np.array([0]), a, P, 0.5, P_prev=P_prev, P_ref=P_ref)
    assert np.all(d6.burst_next == 0)


def test_burst_state_roundtrip():
    b = Burst(CFG, {"theta_b": 0.0})
    obs = mk_obs([])
    for _ in range(5):
        b.step(obs, Decision(), P_diag(1.0), np.zeros((Z, D)), P_diag(1.0), f_max=np.full(M, 4e9))
    b2 = Burst(CFG, {"theta_b": 0.0})
    b2.set_state(b.get_state())
    for _ in range(3):
        r1 = b.step(obs, Decision(), P_diag(1.0), np.zeros((Z, D)), P_diag(1.0), f_max=np.full(M, 4e9))
        r2 = b2.step(obs, Decision(), P_diag(1.0), np.zeros((Z, D)), P_diag(1.0), f_max=np.full(M, 4e9))
        assert np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1])
