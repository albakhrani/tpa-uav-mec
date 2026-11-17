"""Environment features: workload.load_basis, SlotObs.h_min / rate_fn, channel.access = fdma_per_uav."""
import hashlib
import math

import numpy as np
import pytest

from _env_helpers import greedy_decision, make_sim, run_hash
from tpa.config import load_config
from tpa.env import channel as ch
from tpa.env.workload import (Workload, lambda_cpu, lambda_radio, mean_offered_bits,
                              mean_offered_cycles)
from tpa.env.world import World, build_world
from tpa.rng import Streams
from tpa.types import Assignment, Decision
from test_env_simulator import _outcome_digest, mk_task

BASE = load_config()


def cfg_with(**kw):
    keys = {"basis": "workload.load_basis", "access": "channel.access", "iset": "channel.interference_set"}
    return BASE.with_overrides({keys[k]: v for k, v in kw.items()})


# ------------------------------------------------------------------ load_basis
def test_lambda_cpu_radio_hand():
    cfg = BASE
    w = build_world(cfg, Streams(9001, "val").world)
    assert lambda_cpu(w, cfg) == pytest.approx(np.sum(w.f_max * w.eta) / (505e6 * 1.0))
    assert mean_offered_bits(cfg) == pytest.approx(2.75 * 8e6 * 1.1)
    h = ch.channel_gain(w.device_pos, w.uav_pos, w, np.ones_like(w.kappa), cfg)
    best = h.argmax(1)
    cap = 0.0
    for j in range(5):
        sel = best == j
        if sel.any():
            snr = np.median(w.device_ptx[sel] * h[sel, j] / (4e-21 * 1e7))
            cap += w.eta[j] * 1e7 * math.log2(1 + snr)
    assert lambda_radio(w, cfg) == pytest.approx(cap / (2.75 * 8e6 * 1.1), rel=1e-12)


@pytest.mark.parametrize("basis", ["binding", "compute"])
def test_load_basis_rate_normalisation(basis):
    cfg = cfg_with(basis=basis)
    st = Streams(9002, "val")
    w = build_world(cfg, st.world)
    W = Workload(cfg, w, st.arrivals, "sinusoidal", 0.25)
    lc, lr = lambda_cpu(w, cfg), lambda_radio(w, cfg)
    assert W.lambda_cpu == lc and W.lambda_radio == lr
    assert W.binding == ("cpu" if lc <= lr else "radio")
    P = cfg.workload.period_slots
    tot = np.mean([W.rates(t, w.device_pos).sum() for t in range(P)]) / cfg.scenario.slot_s
    target = 0.25 * (min(lc, lr) if basis == "binding" else lc)
    assert tot == pytest.approx(target, rel=1e-9)
    with pytest.raises(ValueError):
        Workload(cfg.with_overrides({"workload.load_basis": "bogus"}), w, st.arrivals, "sinusoidal", 0.25)


# ------------------------------------------------------------------ SlotObs.h_min, rate_fn
@pytest.mark.parametrize("iset,access", [("slot_assigned", "shared"), ("slot_assigned", "fdma_per_uav"),
                                         ("in_progress", "shared"), ("in_progress", "fdma_per_uav")])
def test_rate_fn_pure_and_consistent(iset, access):
    cfg = cfg_with(iset=iset, access=access)
    sim, _, w = make_sim(4, cfg=cfg)
    for t in range(15):
        sim.end_slot(greedy_decision(sim.begin_slot(t), frac=0.1), 0.0)
    obs = sim.begin_slot(15)
    assert obs.h_min == sim.h_min and len(obs.queue) >= 3
    st_before = sim.get_state()
    in_prog = sim._active()
    pairs = [(0, 1), (2, 1), (1, 3)]
    R = obs.rate_fn(pairs)
    act = (in_prog.copy() if iset == "in_progress" else np.zeros_like(in_prog))
    for i, j in pairs:
        act[obs.queue[i].device, j] = True
    qdev = [k.device for k in obs.queue]
    assert np.allclose(R, ch.rate_access(sim.h_now, w.device_ptx, act, w.eta, cfg)[qdev], rtol=1e-12)
    assert np.allclose(obs.rate_fn([]), obs.rates_pred, rtol=1e-12)
    # pure: no RNG consumed, no state changed (compare digests of the continuation)
    for n in ("arrivals", "channel", "mobility"):
        assert getattr(sim.streams, n).bit_generator.state == st_before["streams"][n]
    d_after = sim.get_state()
    assert hashlib.sha256(repr(d_after["streams"]).encode()).digest() == \
        hashlib.sha256(repr(st_before["streams"]).encode()).digest()
    # slot_assigned: the realized fixed rates equal rate_fn of the full slot-t assignment set
    if iset == "slot_assigned":
        res = obs.residual_f.copy()
        dec = []
        for i, j in pairs:
            f = 0.01 * obs.cap_bound[j]
            if f <= res[j]:
                res[j] -= f; dec.append(Assignment(i, j, f, 0.0))
        R_all = obs.rate_fn([(a.task_index, a.uav) for a in dec])
        tids = [obs.queue[a.task_index].tid for a in dec]
        sim.end_slot(Decision(dec), 0.0)
        checked = 0
        for a, tid in zip(dec, tids):
            if tid in sim.fixed_rate:          # still uploading after this slot
                assert sim.fixed_rate[tid] == pytest.approx(R_all[a.task_index, a.uav], rel=1e-12)
                checked += 1
        assert checked >= 1
    else:
        sim.end_slot(Decision(), 0.0)


# ------------------------------------------------------------------ fdma_per_uav (model change)
def test_fdma_rate_hand():
    cfg = cfg_with(access="fdma_per_uav")
    h = np.array([[1e-10, 2e-10], [3e-10, 4e-10], [5e-10, 6e-10]])
    p = np.array([1.0, 0.5, 0.2])
    eta = np.array([0.8, 0.95])
    act = np.array([[1, 0], [1, 0], [0, 1]], bool)          # n_0 = 2, n_1 = 1
    R = ch.rate_access(h, p, act, eta, cfg)
    N0, B, phi = 4e-21, 1e7, 0.1
    # (0,0): band B/2, no intra-cell term; cross from cell 1 (dev2, n_1=1) received at UAV 0: P_2 h_20
    cross0 = phi * (0.2 * 5e-10 / 1)
    z0 = 1e-12 * 0.2 + 1e-13 * 2
    hand00 = 0.8 * (B / 2) * math.log2(1 + 1.0 * 1e-10 / (N0 * B / 2 + cross0 + z0))
    assert R[0, 0] == pytest.approx(hand00, rel=1e-12)
    # (2,1): band B/1; cross from cell 0 (dev0, dev1, n_0=2) received at UAV 1
    cross1 = phi * (1.0 * 2e-10 + 0.5 * 4e-10) / 2
    z1 = 1e-12 * 0.05 + 1e-13 * 1
    hand21 = 0.95 * B * math.log2(1 + 0.2 * 6e-10 / (N0 * B + cross1 + z1))
    assert R[2, 1] == pytest.approx(hand21, rel=1e-12)
    # would-be pair (2,0): joins cell 0 -> n = 3, zeta with 3
    z0p = 1e-12 * 0.2 + 1e-13 * 3
    hand20 = 0.8 * (B / 3) * math.log2(1 + 0.2 * 5e-10 / (N0 * B / 3 + cross0 + z0p))
    assert R[2, 0] == pytest.approx(hand20, rel=1e-12)
    # no interferers: each pair alone in its cell
    R0 = ch.rate_access(h, p, np.zeros_like(act), eta, cfg)
    assert R0[1, 1] == pytest.approx(0.95 * B * math.log2(1 + 0.5 * 4e-10 / (N0 * B + 1e-12 * 0.05 + 1e-13)))
    with pytest.raises(ValueError):
        ch.rate_access(h, p, act, eta, cfg_with(access="bogus"))


@pytest.mark.parametrize("access", ["shared", "fdma_per_uav"])
def test_crn_and_round_trip_access(access):
    cfg = cfg_with(access=access)
    s1, _, _ = make_sim(9003, block="val", cfg=cfg, packet_loss=0.05)
    s2, _, _ = make_sim(9003, block="val", cfg=cfg, packet_loss=0.05)
    assert run_hash(s1, lambda o: Decision(), 150) == run_hash(s2, lambda o: greedy_decision(o, frac=0.1), 150)
    pol = lambda o: greedy_decision(o, frac=0.2)
    st = s2.get_state()
    d1 = _outcome_digest(s2, 40, pol)
    fresh, _, _ = make_sim(9003, block="val", cfg=cfg, packet_loss=0.05)
    fresh.set_state(st)
    assert _outcome_digest(fresh, 40, pol) == d1


def test_sim_exposes_load_diagnostics():
    sim, cfg, w = make_sim(9001, block="val")
    wl = sim.workload
    assert wl.lambda_cpu > 0 and wl.lambda_radio > 0 and wl.binding in ("cpu", "radio")
    with pytest.raises(ValueError):
        make_sim(1, cfg=cfg_with(access="bogus"))


# ------------------------------------------------------------------ SlotObs.device_busy_s (serial radio)
@pytest.mark.parametrize("iset", ["in_progress", "slot_assigned"])
def test_device_busy_seconds_hand(iset):
    cfg = cfg_with(iset=iset)
    sim, _, w = make_sim(1, load=0.0, cfg=cfg)
    sim._rates = lambda h, active: np.full((sim.N, sim.M), 1e6)     # 1 Mb/s every pair
    sim.queue += [mk_task(0, device=7, deadline=100.0), mk_task(1, device=7, deadline=100.0),
                  mk_task(2, device=9, deadline=100.0)]
    obs = sim.begin_slot(0)
    assert obs.device_busy_s == {}                                   # nothing in flight yet
    sim.end_slot(Decision([Assignment(0, 0, 1e9, 0.0), Assignment(1, 2, 1e9, 0.0)]), 0.0)
    obs = sim.begin_slot(1)
    # task 0: 8.8e6 bits, 1 s sent -> 7.8 s left; task 1 queued behind the same radio: 8.8 s
    assert set(obs.device_busy_s) == {7}
    assert obs.device_busy_s[7] == pytest.approx(7.8 + 8.8, rel=1e-12)
    sim.end_slot(Decision(), 0.0)
    st = sim.get_state()
    o1 = sim.begin_slot(2); b1 = dict(o1.device_busy_s); sim.end_slot(Decision(), 0.0)
    assert b1[7] == pytest.approx(6.8 + 8.8, rel=1e-12)
    sim.set_state(st)
    o2 = sim.begin_slot(2)
    assert o2.device_busy_s == b1


def test_device_busy_crn_and_pure():
    s1, _, _ = make_sim(9002, block="val")
    s2, _, _ = make_sim(9002, block="val")
    busy_seen = []

    def pol(o):
        busy_seen.append(len(o.device_busy_s))
        return greedy_decision(o, frac=0.1)
    assert run_hash(s1, lambda o: Decision(), 120) == run_hash(s2, pol, 120)
    assert max(busy_seen) > 0
