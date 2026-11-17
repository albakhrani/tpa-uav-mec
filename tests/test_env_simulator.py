"""Simulator: S_sys, CRN, state round-trip, deadline/late accounting, reservation,
energy (Eq. 17), burst capacity, slot-wise uploads, packet loss."""
import hashlib
import math

import numpy as np
import pytest

from _env_helpers import greedy_decision, hash_slot, make_sim, run_hash
from tpa.env import channel as ch
from tpa.env.queueing import comp_energy, mm1_wait, predict_total_time, t_comp, t_trans
from tpa.env.simulator import s_sys_layout
from tpa.types import Assignment, Decision, Task


def mk_task(tid, device=0, t_arr=0, size_bits=8e6, cycles=1e9, deadline=1.9, phi=1.0, loss_u=None, zone=0):
    return Task(tid=tid, device=device, zone=zone, t_arr=t_arr, size_bits=size_bits, cycles=cycles,
                deadline_s=deadline, priority=0.5, complexity=phi,
                loss_u=np.ones(8) if loss_u is None else np.asarray(loss_u, float), truth_bin=0)


def quiet_sim(R=1e7, **kw):
    """No arrivals (load 0) and a constant rate R for every pair."""
    sim, cfg, w = make_sim(1, load=0.0, **kw)
    sim._rates = lambda h, active: np.full((sim.N, sim.M), R)
    return sim, cfg, w


NONE = Decision()


# ------------------------------------------------------------------ Eqs. 13-16 (shared formulas)
def test_eq14_16_queueing_hand():
    from tpa.config import load_config
    cfg = load_config()
    assert t_trans(8e6, 1.5, 1e7, cfg) == pytest.approx(8e6 * 1.15 / 1e7)
    assert t_comp(1e9, 1.5, 3e9) == pytest.approx(0.5)
    # rho = 2/(5*0.8) = 0.5, wait = 0.5/(4-2)
    assert mm1_wait(2.0, 5.0, 0.8) == pytest.approx(0.25)
    assert mm1_wait(4.0, 5.0, 0.8) == math.inf
    assert predict_total_time(8e6, 1e9, 1.5, 1e7, 3e9, 2.0, 5.0, 0.8, 0.001, cfg) == \
        pytest.approx(0.92 + 0.5 + 0.25 + 0.001)
    assert comp_energy(1e9, 2e9, 1.5, cfg) == pytest.approx(0.6)          # Eq. 18


# ------------------------------------------------------------------ S_sys
def test_s_sys_49_dims_no_time():
    sim, cfg, w = make_sim(1)
    names = s_sys_layout(5, 6)
    assert len(names) == 49
    assert not any(("time" in n) or ("phase" in n) or ("slot" in n) for n in names)
    for t in range(30):
        obs = sim.begin_slot(t)
        assert obs.S_sys.shape == (49,) and np.isfinite(obs.S_sys).all()
        sim.end_slot(greedy_decision(obs), 0.0)
    assert obs.rates_pred.shape == (len(obs.queue), 5) and obs.h_now.shape == (len(obs.queue), 5)


def test_s_sys_burst_budget_from_decision():
    sim, cfg, w = quiet_sim()
    sim.begin_slot(0)
    d = Decision(burst_next=np.array([1, 0, 0, 0, 1]), burst_budget=np.array([0.5, 0, 0, 0, 0.25]))
    sim.end_slot(d, 0.0)
    obs = sim.begin_slot(1)
    per = obs.S_sys[:30].reshape(5, 6)
    assert per[:, 4].tolist() == [1, 0, 0, 0, 1]
    assert per[:, 5].tolist() == [0.5, 0, 0, 0, 0.25]


# ------------------------------------------------------------------ CRN
@pytest.mark.parametrize("kw", [dict(), dict(mobility="gauss_markov", packet_loss=0.1),
                                dict(workload_kind="mmpp"), dict(workload_kind="burst", patrol=True)])
def test_crn_env_level(kw):
    s1, _, _ = make_sim(7, **kw)
    s2, _, _ = make_sim(7, **kw)
    h1 = run_hash(s1, lambda o: NONE, 300)
    h2 = run_hash(s2, lambda o: greedy_decision(o, frac=0.1), 300)
    assert h1 == h2
    for n in ("arrivals", "channel", "mobility"):
        assert getattr(s1.streams, n).bit_generator.state == getattr(s2.streams, n).bit_generator.state
    # the policies did differ
    assert len(s1.inflight) == 0 and len(s2.inflight) > 0


# ------------------------------------------------------------------ state round-trip
def _outcome_digest(sim, T, policy):
    h = hashlib.sha256()
    for _ in range(T):
        obs = sim.begin_slot(sim.next_t)
        hash_slot(sim, h)
        h.update(obs.S_sys.tobytes())
        out = sim.end_slot(policy(obs), 0.123)
        for k in out.completed + out.late + out.failed:
            h.update(np.array([k.tid, k.t_done, k.t_upload_done, k.uav], float).tobytes())
        h.update(sim.energy.tobytes())
        h.update(out.cpu_util.tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("kw", [dict(), dict(mobility="gauss_markov", packet_loss=0.1, workload_kind="pareto")])
def test_state_round_trip(kw):
    sim, cfg, w = make_sim(3, **kw)
    pol = lambda o: greedy_decision(o, frac=0.2)
    for t in range(80):
        sim.end_slot(pol(sim.begin_slot(t)), 0.1)
    st = sim.get_state()
    d1 = _outcome_digest(sim, 60, pol)
    sim.set_state(st)
    d2 = _outcome_digest(sim, 60, pol)
    fresh, _, _ = make_sim(3, **kw)
    fresh.set_state(st)
    d3 = _outcome_digest(fresh, 60, pol)
    assert d1 == d2 == d3


# ------------------------------------------------------------------ deadline / late
@pytest.mark.parametrize("deadline,ok", [(1.9, True), (1.85, False)])
def test_deadline_and_late_accounting(deadline, ok):
    sim, cfg, w = quiet_sim(R=1e7)
    sim.queue.append(mk_task(0, deadline=deadline))
    obs = sim.begin_slot(0)
    f = 1e9
    sim.end_slot(Decision([Assignment(0, 0, f, 0.0)]), 0.0)
    obs = sim.begin_slot(1)
    out = sim.end_slot(NONE, 0.0)
    # t_done = T_mem(0) + s(1+0.1 phi)/R + c phi/f = 0.88 + 1.0
    done = out.completed + out.late
    assert len(done) == 1 and done[0].t_done == pytest.approx(1.88, abs=1e-12)
    assert done[0].t_upload_done == pytest.approx(0.88, abs=1e-12)
    assert (len(out.completed) == 1) == ok and (len(out.late) == 1) == (not ok)
    assert len(sim.inflight) == 0 and sim.reserved().sum() == 0


def test_queue_wait_counts_and_expiry():
    sim, cfg, w = quiet_sim(R=1e7)
    sim.queue.append(mk_task(0, deadline=2.5))
    sim.queue.append(mk_task(1, device=1, deadline=1.0))
    sim.begin_slot(0); sim.end_slot(NONE, 0.0)
    obs = sim.begin_slot(1)                     # task 1: 1 - 0 > 1.0 is False -> still queued
    assert [k.tid for k in obs.queue] == [0, 1]
    sim.end_slot(Decision([Assignment(0, 2, 1e9, 0.0)]), 0.0)   # assigned after 1 slot of waiting
    obs = sim.begin_slot(2)                     # task 1 expires now
    assert [k.tid for k in obs.queue] == []
    out = sim.end_slot(NONE, 0.0)
    assert [k.tid for k in out.failed] == [1]
    assert [k.tid for k in out.late] == [0] and out.late[0].t_done == pytest.approx(2.88)


def test_overflow_drops_oldest():
    sim, cfg, w = quiet_sim()
    for i in range(200):
        sim.queue.append(mk_task(i, device=i % 200, t_arr=0, deadline=10.0))
    obs = sim.begin_slot(0)
    assert len(obs.queue) == 4 * 32
    out = sim.end_slot(NONE, 0.0)
    assert sorted(k.tid for k in out.failed) == list(range(72))


# ------------------------------------------------------------------ reservation (Eq. 37 over time)
def test_reservation_capacity():
    sim, cfg, w = quiet_sim(R=1e6)                # 8.8 s upload
    for i in range(3):
        sim.queue.append(mk_task(i, device=i, deadline=100.0))
    obs = sim.begin_slot(0)
    cap = obs.cap_bound[0]
    assert cap == pytest.approx(w.f_max[0] * w.eta[0])
    with pytest.raises(AssertionError, match="exceeds residual"):
        sim.end_slot(Decision([Assignment(0, 0, 0.6 * cap, 0.0), Assignment(1, 0, 0.6 * cap, 0.0)]), 0.0)
    # fresh run: reserve 0.6 cap; residual 0.4 cap stays reserved across slots until done
    sim, cfg, w = quiet_sim(R=1e6)
    for i in range(3):
        sim.queue.append(mk_task(i, device=i, deadline=100.0))
    obs = sim.begin_slot(0)
    sim.end_slot(Decision([Assignment(0, 0, 0.6 * cap, 0.0)]), 0.0)
    for t in range(1, 9):
        obs = sim.begin_slot(t)
        assert obs.residual_f[0] == pytest.approx(0.4 * cap)
        assert obs.S_sys[2] == pytest.approx(0.6)          # reserved fraction of UAV 0
        out = sim.end_slot(NONE, 0.0)
        assert out.cpu_util[0] == pytest.approx(0.6)
    # upload done at 8.8 s, compute 1e9/(0.6 cap) after that
    t_done = 8.8 + 1e9 / (0.6 * cap)
    t = 9
    while sim.inflight:
        obs = sim.begin_slot(t); out = sim.end_slot(NONE, 0.0); t += 1
    assert (out.completed + out.late)[0].t_done == pytest.approx(t_done)
    assert sim.residual()[0] == pytest.approx(cap)


# ------------------------------------------------------------------ energy (Eqs. 17-18)
def test_energy_decrement_eq17():
    sim, cfg, w = quiet_sim()
    sim.queue.append(mk_task(0, cycles=5e8, phi=1.2, deadline=10.0))
    obs = sim.begin_slot(0)
    E0 = sim.energy.copy()
    f = 2e9
    out = sim.end_slot(Decision([Assignment(0, 3, f, 0.0)]), mem_energy_j=0.05)
    ec = 1e-28 * 5e8 * f * f * 1.2
    assert out.comp_energy_j[3] == pytest.approx(ec)
    expect = E0 - 0.05
    expect[3] -= ec
    assert np.allclose(sim.energy, expect, rtol=0, atol=1e-9)
    # next slot: only memory energy
    sim.begin_slot(1)
    sim.end_slot(NONE, mem_energy_j=0.02)
    expect -= 0.02
    assert np.allclose(sim.energy, expect, rtol=0, atol=1e-9)
    sim.reset_battery()
    assert np.array_equal(sim.energy, w.battery_j)


# ------------------------------------------------------------------ burst
def test_burst_capacity_and_energy():
    sim, cfg, w = quiet_sim(R=1e6)
    sim.queue.append(mk_task(0, deadline=100.0))
    sim.begin_slot(0)
    sim.end_slot(Decision(burst_next=np.array([1, 0, 0, 0, 0])), 0.0)   # takes effect at t+1
    obs = sim.begin_slot(1)
    nominal = w.f_max * w.eta
    assert obs.cap_bound[0] == pytest.approx(nominal[0] * 1.25)
    assert obs.cap_bound[1] == pytest.approx(nominal[1])
    assert obs.burst.tolist() == [True, False, False, False, False]
    E0 = sim.energy.copy()
    f = 1.2 * nominal[0]                             # above nominal, below burst bound
    out = sim.end_slot(Decision([Assignment(0, 0, f, 0.0)], burst_next=np.array([1, 0, 0, 0, 0])), 0.0)
    assert out.burst_energy_j.tolist() == [40.0, 0, 0, 0, 0]
    ec = comp_energy(1e9, f, 1.0, cfg)
    assert sim.energy[0] == pytest.approx(E0[0] - ec - 40.0)
    assert sim.energy[1] == pytest.approx(E0[1])
    assert out.cpu_util[0] == pytest.approx(1.2)
    # burst off: bound drops to nominal, over-commitment shows as zero residual
    sim.begin_slot(2)
    sim.end_slot(Decision(burst_next=np.zeros(5)), 0.0)
    obs = sim.begin_slot(3)
    assert obs.cap_bound[0] == pytest.approx(nominal[0]) and obs.residual_f[0] == 0.0


# ------------------------------------------------------------------ uploads
def test_multislot_upload_and_serial_radio():
    sim, cfg, w = quiet_sim(R=4e6)                  # 8.8e6 bits -> 2.2 s each
    sim.queue.append(mk_task(0, device=5, cycles=1e8, deadline=100.0))
    sim.queue.append(mk_task(1, device=5, cycles=1e8, deadline=100.0))
    sim.begin_slot(0)
    sim.end_slot(Decision([Assignment(0, 0, 1e9, 0.0), Assignment(1, 1, 1e9, 0.0)]), 0.0)
    done = []
    for t in range(1, 8):
        sim.begin_slot(t); out = sim.end_slot(NONE, 0.0); done += out.completed
    up = {k.tid: k.t_upload_done for k in done}
    assert up[0] == pytest.approx(2.2) and up[1] == pytest.approx(4.4)
    assert {k.tid: k.t_done for k in done}[1] == pytest.approx(4.5)


def test_interference_uses_uploads_in_progress():
    from tpa.config import load_config
    c = load_config(overrides={"channel.interference_set": "in_progress", "channel.access": "shared"})
    sim, cfg, w = make_sim(1, load=0.0, cfg=c)
    for i in range(3):
        sim.queue.append(mk_task(i, device=i, size_bits=5e7, deadline=100.0))
    sim.begin_slot(0)
    sim.end_slot(Decision([Assignment(0, 0, 1e8, 0.0), Assignment(1, 0, 1e8, 0.0)]), 0.0)
    act = sim._active()
    assert act[0, 0] and act[1, 0] and act.sum() == 2
    obs = sim.begin_slot(1)                          # task 2 still queued: predicted with 2 interferers
    R = ch.rate(sim.h_now, w.device_ptx, act, w.eta, cfg)
    assert obs.rates_pred[0] == pytest.approx(R[2])
    R_free = ch.rate(sim.h_now, w.device_ptx, np.zeros_like(act), w.eta, cfg)
    assert obs.rates_pred[0, 0] < R_free[2, 0]


def test_packet_loss_retransmits_next_slot():
    sim, cfg, w = quiet_sim(R=1e7, packet_loss=0.5)
    sim.queue.append(mk_task(0, deadline=100.0, loss_u=[0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]))
    sim.begin_slot(0)
    sim.end_slot(Decision([Assignment(0, 0, 1e9, 0.0)]), 0.0)
    assert sim.inflight[0].attempts == 1 and sim.inflight[0].bits_left > 0
    done = []
    for t in range(1, 4):
        sim.begin_slot(t); out = sim.end_slot(NONE, 0.0); done += out.completed
    assert done[0].t_upload_done == pytest.approx(1.88) and done[0].t_done == pytest.approx(2.88)
    assert done[0].attempts == 2


def test_flush_reports_flushed():
    sim, cfg, w = quiet_sim(R=1e6)
    sim.queue.append(mk_task(0, deadline=100.0))
    sim.queue.append(mk_task(1, device=1, deadline=100.0))
    sim.begin_slot(0)
    sim.end_slot(Decision([Assignment(0, 0, 1e9, 0.0)]), 0.0)
    fl = sim.flush()
    assert [k.tid for k in fl["flushed_inflight"]] == [0] and [k.tid for k in fl["flushed_queue"]] == [1]
    assert sim.reserved().sum() == 0 and not sim.queue


def test_obs_fields_and_cold_start_mu():
    sim, cfg, w = make_sim(1)
    obs = sim.begin_slot(0)
    assert np.array_equal(obs.f_max, w.f_max) and np.array_equal(obs.eta, w.eta)
    assert (obs.lam_eff == 0).all()
    assert obs.mu == pytest.approx(w.f_max / (505e6 * 1.0))
    assert (obs.mu > 0).all()
    assert mm1_wait(0.0, 0.0, 0.9) == 0.0
