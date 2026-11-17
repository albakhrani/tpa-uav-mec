"""Eqs. (10)-(12): workload model; one phase per zone; trace families."""
import math

import numpy as np
import pytest

from tpa.config import load_config
from tpa.env.workload import Workload, mean_offered_cycles
from tpa.env.world import build_world, zone_of
from tpa.rng import Streams

cfg = load_config(overrides={"workload.load_basis": "compute"})  # compute-calibration tests


def wl(seed=1, kind="sinusoidal", load=0.55, **kw):
    st = Streams(seed)
    w = build_world(cfg, st.world)
    return Workload(cfg, w, st.arrivals, kind, load, **kw), w


def test_eq11_rate_hand():
    W, w = wl()
    t = 1234
    z = zone_of(w.device_pos, w)
    fleet = float(np.sum(w.f_max * w.eta))
    base = 0.55 * fleet / (505e6 * 1.0) * 1.0 / 200 / np.mean(w.zone_weights[z])
    psi = 1 + 0.4 * np.sin(2 * math.pi * t / 3600 + w.zone_phase[z])
    assert np.allclose(W.rates(t, w.device_pos), base * psi * w.zone_weights[z], rtol=1e-12)


def test_one_phase_per_zone():
    W, w = wl()
    z = zone_of(w.device_pos, w)
    for t in (0, 900, 1800, 2700):
        ratio = W.rates(t, w.device_pos) / (W.base * w.zone_weights[z])
        for k in range(w.n_zones):
            r = ratio[z == k]
            if len(r):
                assert np.ptp(r) < 1e-12          # identical psi within a zone


def test_base_rate_normalisation():
    W, w = wl(load=0.85)
    P = cfg.workload.period_slots
    tot = np.mean([W.rates(t, w.device_pos).sum() for t in range(P)])   # tasks/slot
    fleet = float(np.sum(w.f_max * w.eta))
    assert tot * mean_offered_cycles(cfg) / cfg.scenario.slot_s == pytest.approx(0.85 * fleet, rel=1e-9)


def test_eq10_poisson_counts_mean():
    W, w = wl()
    lam = sum(W.rates(t, w.device_pos).sum() for t in range(400))
    n = sum(len(W.step(t, w.device_pos)) for t in range(400))
    assert abs(n - lam) < 5 * math.sqrt(lam)


def test_measured_zone_amplitude_seeds_1_3():
    A = cfg.workload.amplitude
    for seed in (1, 2, 3):
        W, w = wl(seed)
        state = W.get_state()
        amp = W.measured_zone_amplitude()
        z = zone_of(w.device_pos, w)
        for k in range(w.n_zones):
            if (z == k).sum() >= 5:
                assert abs(amp[k] - A) < 0.06, (seed, k, amp)
        # helper must not consume the workload streams
        s2 = W.get_state()
        assert s2["rng"] == state["rng"] and s2["next_tid"] == state["next_tid"]


def test_task_profile_eq12_and_loss_u():
    W, w = wl()
    tasks = [k for t in range(50) for k in W.step(t, w.device_pos)]
    assert len(tasks) > 0
    for k in tasks:
        assert 0.5 * 8e6 <= k.size_bits <= 5 * 8e6
        assert 10e6 <= k.cycles <= 1000e6
        assert 1.0 <= k.deadline_s <= 10.0 and 0 <= k.priority <= 1 and 0.5 <= k.complexity <= 1.5
        assert k.loss_u.shape == (cfg.task.max_tx_attempts,)
        assert k.zone == zone_of(w.device_pos[k.device], w)
        assert k.truth_bin == int((k.t_arr % 3600) * 24 // 3600)
    assert len({k.tid for k in tasks}) == len(tasks)


def test_shifted_and_burst():
    Ws, w = wl(kind="shifted", horizon=1000)
    Wb, _ = wl(kind="sinusoidal")
    Ws.step(0, w.device_pos); Wb.step(0, w.device_pos)
    assert np.allclose(Ws.rates(10, w.device_pos), Wb.rates(10, w.device_pos))
    z = zone_of(w.device_pos, w)
    psi = 1 + 0.4 * np.sin(2 * math.pi * 600 / 3600 + w.zone_phase[z] + math.pi / 3)
    assert np.allclose(Ws.rates(600, w.device_pos), Ws.base * psi * w.zone_weights[z])
    Wu, _ = wl(kind="burst")
    on = 0
    for t in range(3600):
        Wu._advance(t)
        on += Wu.in_burst(t)
    assert on == 360


def test_mmpp_pareto_mean_and_trace_hook(tmp_path):
    for kind, tol in (("mmpp", 0.15), ("pareto", None)):
        W, w = wl(kind=kind)
        tot = 0.0
        T = 20000
        for t in range(T):
            W._advance(t)
            r = W.rates(t, w.device_pos)
            assert (r >= 0).all()
            tot += r.sum()
        target = 0.55 * float(np.sum(w.f_max * w.eta)) / mean_offered_cycles(cfg)
        if tol:
            assert tot / T == pytest.approx(target, rel=tol)
        assert W.truth_bin(5) == -1
    with pytest.raises(ValueError, match="trace_path"):
        wl(kind="trace")
    arr = np.abs(np.random.Generator(np.random.PCG64(0)).standard_normal((50, 6))) + 0.1
    p = tmp_path / "tr.npy"
    np.save(p, arr)
    c2 = cfg.with_overrides({"workload.trace_path": str(p)})
    st = Streams(1)
    w = build_world(c2, st.world)
    W = Workload(c2, w, st.arrivals, "trace", 0.55)
    assert len(W.step(0, w.device_pos)) >= 0
    assert W.trace_rates.mean(axis=0) == pytest.approx(np.ones(6))
