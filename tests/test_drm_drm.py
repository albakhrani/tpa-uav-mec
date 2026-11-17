"""DRM (Alg. 3, Eqs. 52-56) unit tests with hand-computed values."""
import math

import numpy as np
import pytest

from tpa.config import load_config
from tpa.drm import DRM
from tpa.env.queueing import f_required
from tpa.types import SlotObs, Task

CFG = load_config()
M = CFG.scenario.n_uav
Z = CFG.scenario.n_zones


def mk_task(tid, zone=0, t_arr=0, size_bits=1e6, cycles=1e9, deadline=2.0, prio=0.5, phi=1.0):
    return Task(tid=tid, device=tid, zone=zone, t_arr=t_arr, size_bits=size_bits, cycles=cycles,
                deadline_s=deadline, priority=prio, complexity=phi,
                loss_u=np.zeros(CFG.task.max_tx_attempts), truth_bin=-1)


def mk_obs(tasks, t=0, rate=1e8, cap=4e9, residual=None, energy=1e5, e_min=0.0,
           lam=0.0, mu=10.0):
    Q = len(tasks)
    cap_b = np.full(M, cap)
    return SlotObs(
        t=t, S_sys=np.zeros(49), queue=list(tasks),
        rates_pred=np.full((Q, M), rate), h_now=np.full((Q, M), 1.0),
        zone_arrivals=np.zeros(Z), zone_mean=np.ones(Z),
        residual_f=cap_b.copy() if residual is None else np.asarray(residual, float),
        cap_bound=cap_b, energy=np.full(M, energy), e_min=np.full(M, e_min),
        burst=np.zeros(M), lam_eff=np.full(M, lam), mu=np.full(M, mu))


def mk_drm(flags=None):
    return DRM(CFG, flags or {}, "A0", f_max=np.full(M, 4e9), eta=np.full(M, 0.9))


def P_two_rows(a=3.0, b=6.0):
    P = np.zeros((Z, CFG.memory.d))
    P[0, 0] = a
    P[1, 1] = b
    return P


def act(n, pref=None, ftilde=0.0):
    """Action rows: pref (M,) in [-1,1]; f~ -> a = 2 f~ - 1."""
    a = np.zeros((CFG.task.q_max, 2 * M))
    if pref is not None:
        a[:n, :M] = pref
    a[:n, M:] = 2.0 * ftilde - 1.0
    return a


# ---------------------------------------------------------------- Eqs. 52-53
def test_eq52_eq53_priority_hand():
    drm = mk_drm()
    task = mk_task(0, zone=0, t_arr=8, deadline=4.0, prio=0.5)
    obs = mk_obs([task], t=10)
    P = P_two_rows(3.0, 6.0)
    chi = drm.pattern_priority(obs, P, 0.4)
    assert chi[0] == pytest.approx(3.0 / 6.0 * 0.4)                           # Eq. 53
    pi = drm.priority(obs, P, 0.4)
    hand = 0.4 * 0.5 + 0.3 * 0.2 + 0.3 * math.exp(-(4.0 - 2.0) / 4.0)          # Eq. 52
    assert pi[0] == pytest.approx(hand, rel=1e-12)
    assert pi[0] == pytest.approx(0.441959198, rel=1e-8)


def test_select_window_top32_ties_by_tid():
    drm = mk_drm()
    tasks = [mk_task(100 - k) for k in range(40)]          # tids 100..61, queue order
    obs = mk_obs(tasks)
    pi = np.zeros(40)
    pi[5] = 2.0
    win = drm.select_window(pi, obs)
    assert len(win) == 32 and win[0] == 5
    rest = [obs.queue[i].tid for i in win[1:]]
    assert rest == sorted(rest)                             # ties by ascending tid
    assert set(rest) == set(range(61, 92)) - {95}          # smallest tids win ties


# ---------------------------------------------------------------- Eq. 54
def test_eq54_allocation_hand():
    drm = mk_drm()
    assert drm.allocation(0.5, 4e9, 0.5) == pytest.approx(0.5 * 4e9 * (1 + 0.3 * 0.5))
    # through Alg. 3: one task in zone 0 with ||P_0||/max = 0.5, f~ = 0.5
    obs = mk_obs([mk_task(0, zone=0)])
    dec = drm.run(obs, np.array([0]), act(1, ftilde=0.5), P_two_rows(3.0, 6.0), 0.0)
    assert len(dec.assignments) == 1
    assert dec.assignments[0].f == pytest.approx(2.3e9)


# ---------------------------------------------------------------- Eq. 56
def test_eq56_load_index_hand():
    drm = mk_drm()
    P = np.zeros((Z, CFG.memory.d))
    P[0, 0] = 3.0
    P[1, 1] = 4.0                                            # ||P||_F = 5
    drm.zone_mix[0, 0] = 1.0                                 # P_0 = row 0 -> 3/5
    drm.zone_mix[1, 0] = drm.zone_mix[1, 1] = 1.0           # P_1 = [1.5, 2] -> 2.5/5
    share = drm.uav_pattern_share(P)
    assert share[0] == pytest.approx(0.6) and share[1] == pytest.approx(0.5)
    assert share[2] == 0.0
    cap = np.full(M, 4e9)
    res = cap.copy()
    res[0] = 2e9
    L = drm.load_index(cap, res, share)
    assert L[0] == pytest.approx(0.6 * 0.5 + 0.4 * 0.6)
    assert L[1] == pytest.approx(0.4 * 0.5)
    # normalised by the current (burst-raised) bound: in [0, 1]
    assert np.all((L >= 0) & (L <= 1))


# ---------------------------------------------------------------- all-zero f~: every satisfiable task assigned
def test_all_zero_ftilde_assigns_every_satisfiable_task():
    drm = mk_drm()
    tasks = [mk_task(k, cycles=2e8 + 1e7 * k, deadline=2.0 + 0.1 * k) for k in range(10)]
    obs = mk_obs(tasks)
    win = drm.select_window(drm.priority(obs, None, 0.0), obs)
    dec = drm.run(obs, win, act(len(win), ftilde=0.0), None, 0.0)
    assert sorted(a.task_index for a in dec.assignments) == list(range(10))
    for asg in dec.assignments:
        t = tasks[asg.task_index]
        budget = t.deadline_s - t.size_bits * (1 + CFG.task.tx_overhead * t.complexity) / 1e8
        assert asg.f == pytest.approx(f_required(t.cycles, t.complexity, budget))
        assert asg.t_total_pred <= t.deadline_s + 1e-9
    # infeasible tasks (deadline cannot be met at full residual) are re-queued, not assigned
    big = mk_task(99, cycles=1e12, deadline=1.0)
    obs2 = mk_obs([big])
    dec2 = drm.run(obs2, np.array([0]), act(1, ftilde=1.0), None, 0.0)
    assert dec2.assignments == []


def test_rate_and_viability_infeasible():
    drm = mk_drm()
    obs = mk_obs([mk_task(0)], rate=CFG.channel.r_min_mbps * 1e6 * 0.5)
    assert drm.run(obs, np.array([0]), act(1), None, 0.0).assignments == []
    obs = mk_obs([mk_task(0)])
    obs.h_now[:] = CFG.channel.h_min * 0.1
    assert drm.run(obs, np.array([0]), act(1), None, 0.0).assignments == []


# ---------------------------------------------------------------- utility argmax + residuals
def test_utility_argmax_and_residual_update():
    drm = mk_drm()
    A = mk_task(0, prio=1.0)
    B = mk_task(1, prio=0.0)
    res = np.full(M, 4e9)
    res[2] = 1e9
    obs = mk_obs([A, B], residual=res)
    a = act(2, ftilde=0.0)
    a[0, :M] = [0, 0, 1, 0, 0]
    a[1, :M] = [0, 0, 1, 0.5, 0]
    win = drm.select_window(drm.priority(obs, None, 0.0), obs)
    assert list(win) == [0, 1]
    dec = drm.run(obs, win, a, None, 0.0)
    by_task = {x.task_index: x for x in dec.assignments}
    assert by_task[0].uav == 2
    f_req = f_required(1e9, 1.0, 2.0 - 1e6 * 1.1 / 1e8)
    assert by_task[0].f == pytest.approx(f_req)
    # residual on UAV 2 now 1e9 - f_req < f_req -> task B cannot use UAV 2, goes to UAV 3
    assert by_task[1].uav == 3
    assert dec.load_index[2] == pytest.approx(0.6 * (4e9 - (1e9 - f_req)) / 4e9)
    assert dec.load_index[3] == pytest.approx(0.6 * f_req / 4e9)
    assert dec.util_terms["cpu_frac"][2] == pytest.approx(f_req / 4e9)


def test_capacity_never_exceeded():
    drm = mk_drm()
    tasks = [mk_task(k, cycles=1e9, deadline=1.5) for k in range(32)]
    obs = mk_obs(tasks)
    win = drm.select_window(drm.priority(obs, None, 0.0), obs)
    dec = drm.run(obs, win, act(32, ftilde=1.0), None, 0.0)
    used = np.zeros(M)
    for x in dec.assignments:
        used[x.uav] += x.f
    assert np.all(used <= obs.residual_f * (1 + 1e-12))
    assert 0 < len(dec.assignments) < 32


# ---------------------------------------------------------------- A6 flags
def test_a6_removes_pattern_terms():
    P = P_two_rows(3.0, 6.0)
    full, a6 = mk_drm(), mk_drm({"pattern_terms": False})
    task = mk_task(0, zone=0, t_arr=8, deadline=4.0, prio=0.5)
    obs = mk_obs([task], t=10)
    assert np.all(a6.pattern_priority(obs, P, 0.4) == 0.0)
    assert a6.priority(obs, P, 0.4)[0] == pytest.approx(0.4 * 0.5 + 0.3 * math.exp(-0.5))
    assert full.priority(obs, P, 0.4)[0] > a6.priority(obs, P, 0.4)[0]
    assert a6.gamma_p == 0.0 and a6.omega_l == 1.0 and a6.burst.beta_b == 0.0
    a6.zone_mix[:] = 1.0
    assert np.all(a6.uav_pattern_share(P) == 0.0)
    obs = mk_obs([mk_task(0, zone=0)])
    dec = a6.run(obs, np.array([0]), act(1, ftilde=0.5), P, 0.4)
    assert dec.assignments[0].f == pytest.approx(0.5 * 4e9)        # no (1 + gamma_p ...) boost


def test_state_roundtrip():
    drm = mk_drm()
    obs = mk_obs([mk_task(k) for k in range(5)])
    drm.run(obs, np.arange(5), act(5, ftilde=0.2), P_two_rows(), 0.3, P_prev=np.zeros((Z, 64)),
            P_ref=P_two_rows())
    st = drm.get_state()
    d2 = mk_drm()
    d2.set_state(st)
    assert np.array_equal(d2.zone_mix, drm.zone_mix) and d2.p_max == drm.p_max
    o1 = drm.run(obs, np.arange(5), act(5, ftilde=0.2), P_two_rows(), 0.3)
    o2 = d2.run(obs, np.arange(5), act(5, ftilde=0.2), P_two_rows(), 0.3)
    assert [(a.task_index, a.uav, a.f) for a in o1.assignments] == \
        [(a.task_index, a.uav, a.f) for a in o2.assignments]
    assert np.array_equal(o1.burst_next, o2.burst_next)
