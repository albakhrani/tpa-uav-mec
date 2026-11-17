"""Equivalence test: the fast Algorithm 3 path (cfg.drm.impl = "fast") must
produce bit-identical X*(t), F*(t), admission decisions, burst flags and reward sequences
to the reference loop (impl = "reference") for EVERY arm, on 3 validation blocks of 500
slots each with the adopted scenario (25 UAVs; runs/d3/adopted.json). Any difference fails.

Blocks: validation seeds 9001 / 9002 / 9003 (validation streams, test_run 0; never a test
seed), loads heavy / medium / heavy. Each block runs the full simulator + controller twice
from identical streams and identical (untrained, deterministic) controller clones, once per
implementation, so any divergence compounds through the state and is caught at the first slot.
Compared per slot: assignments (task_index, uav, f, t_total_pred), burst_next, burst_budget,
load_index, reward, util_terms (admit_rate, admit_delay, cpu_frac, rate_frac, r_max,
n_feasible_pairs, burst_phi, eta_res, eta_res_target), and the simulator outcome (completed /
failed / late tids, energy). The reference's diagnostic count n_admission_checks is the one
field allowed to differ. Branch coverage counts (feasible, refused, re-queued, expired,
aborted, burst on/off transitions, raises) are written to runs/d8/equivalence/<arm>.json.

Row-restricted rate prediction (channel.rate_fdma_rows) is also checked directly against
rate_fdma on random interferer sets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpa.arms import ARMS                                   # noqa: E402
from tpa.config import load_config                          # noqa: E402
from tpa.env import channel as ch                           # noqa: E402
from tpa.env.simulator import Simulator                     # noqa: E402
from tpa.env.world import build_world                       # noqa: E402
from tpa.factory import make_controller                     # noqa: E402
from tpa.rng import Streams                                 # noqa: E402

ADOPTED = json.loads((ROOT / "runs" / "d3" / "adopted.json").read_text())["overrides"]
BLOCKS = [(9001, 2), (9002, 1), (9003, 2)]                  # (validation seed, load index)
N_SLOTS = int(__import__("os").environ.get("D8_SLOTS", "500"))
OUT = ROOT / "runs" / "d8" / "equivalence"


def _cfg(impl: str):
    ov = dict(ADOPTED)
    ov["drm.impl"] = impl
    return load_config(overrides=ov)


def _run(arm: str, impl: str, seed: int, load_idx: int, n: int):
    """One block; returns (per-slot records, branch counts)."""
    return _run_cfg(arm, _cfg(impl), seed, load_idx, n)


def _run_cfg(arm: str, cfg, seed: int, load_idx: int, n: int):
    torch.set_num_threads(1)
    world = build_world(cfg, Streams(seed, "train").world)
    st = Streams(seed, "val", test_run=0)
    load = float(cfg.workload.loads[load_idx])
    R = cfg.run
    t0 = int(R.test_seed_offset_slots) // 2 + int(R.val_slots) // 3
    sim0 = Simulator(cfg, world, st, arm_key=arm, load=load, t0=t0, horizon=n)
    ctrl = make_controller(cfg, arm, Streams(seed, "train"), sim0.r_ref, sim0.h_min).eval_clone(st)
    sim = Simulator(cfg, world, st, arm_key=arm, load=load, t0=t0, horizon=n)
    ctrl.reset_segment(t0)
    ctrl.start_episode()
    recs = []
    cov = dict(feasible=0, admission_checks=0, assignments=0, refused=0, requeued=0, expired=0,
               aborted=0, overflow=0, burst_on_slots=0, burst_transitions=0, raises=0, window=0)
    prev_b = None
    for t in range(t0, t0 + n):
        obs = sim.begin_slot(t)
        tb = sim.truth_phase_bin(t)
        dec, me = ctrl.act(obs, tb, train=False)
        out = sim.end_slot(dec, mem_energy_j=me)
        info = ctrl.observe(obs, dec, out, train=False)
        ut = dec.util_terms or {}
        asg = [(int(a.task_index), int(a.uav), float(a.f), float(a.t_total_pred)) for a in dec.assignments]
        rec = dict(
            asg=asg,
            burst=None if dec.burst_next is None else np.asarray(dec.burst_next, float).tobytes(),
            budget=None if dec.burst_budget is None else np.asarray(dec.burst_budget, float).tobytes(),
            load=None if dec.load_index is None else np.asarray(dec.load_index, float).tobytes(),
            reward=float(info.get("reward", np.nan)) if isinstance(info, dict) else None,
            admit_rate=tuple(float(x) for x in ut.get("admit_rate", ())),
            admit_delay=tuple(float(x) for x in ut.get("admit_delay", ())),
            cpu_frac=np.asarray(ut.get("cpu_frac", ()), float).tobytes(),
            rate_frac=np.asarray(ut.get("rate_frac", ()), float).tobytes(),
            r_max=float(ut.get("r_max", np.nan)),
            n_feas=int(ut.get("n_feasible_pairs", -1)),
            phi=np.asarray(ut.get("burst_phi", ()), float).tobytes(),
            eta_res=float(ut.get("eta_res", np.nan)),
            eta_tgt=float(ut.get("eta_res_target", np.nan)),
            t_admit=tuple(float(x) for x in ut.get("t_total_admit", ())),
            completed=tuple(k.tid for k in out.completed),
            failed=tuple(k.tid for k in out.failed),
            late=tuple(k.tid for k in out.late),
            energy=np.asarray(sim.energy, float).tobytes(),
        )
        recs.append(rec)
        nw = int(getattr(ctrl, "last", {}).get("n_window", 0))
        cov["window"] += nw
        cov["feasible"] += int(ut.get("n_feasible_pairs", 0))
        cov["admission_checks"] += int(ut.get("n_admission_checks", 0))
        cov["assignments"] += len(asg)
        cov["refused"] += max(int(ut.get("n_admission_checks", 0)) - len(asg), 0)
        cov["requeued"] += max(nw - len(asg), 0)
        li = sim.last_info
        cov["expired"] += int(li.get("n_expired", 0))
        cov["aborted"] += int(li.get("n_aborted", 0))
        cov["overflow"] += int(li.get("n_overflow", 0))
        b = None if dec.burst_next is None else np.asarray(dec.burst_next, bool)
        if b is not None:
            cov["burst_on_slots"] += int(b.sum())
            if prev_b is not None:
                cov["burst_transitions"] += int(np.sum(b != prev_b))
            prev_b = b
    drm = getattr(ctrl, "drm", None) or getattr(getattr(ctrl, "ex", None), "drm", None)
    cov["raises"] = int(getattr(drm, "n_raises", 0))
    return recs, cov


def _diff(ra, rb):
    for t, (a, b) in enumerate(zip(ra, rb)):
        for k in a:
            va, vb = a[k], b[k]
            same = (np.array_equal(va, vb, equal_nan=True) if isinstance(va, float)
                    else va == vb)
            if not same:
                return t, k, va, vb
    return None


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_fast_equals_reference(arm):
    OUT.mkdir(parents=True, exist_ok=True)
    cov_all = {}
    for seed, li in BLOCKS:
        ref, cov_r = _run(arm, "reference", seed, li, N_SLOTS)
        fast, cov_f = _run(arm, "fast", seed, li, N_SLOTS)
        d = _diff(ref, fast)
        assert d is None, f"{arm} seed {seed} load idx {li}: first difference at slot {d[0]} field {d[1]!r}: {d[2]!r} != {d[3]!r}"
        assert len(ref) == len(fast) == N_SLOTS
        cov_all[f"seed{seed}_load{li}"] = dict(reference=cov_r, fast=cov_f)
    (OUT / f"{arm}.json").write_text(json.dumps(dict(arm=arm, slots=N_SLOTS, blocks=cov_all), indent=1))


def test_rate_rows_bit_identical_to_full():
    cfg = _cfg("fast")
    world = build_world(cfg, Streams(9001, "train").world)
    rng = np.random.Generator(np.random.PCG64(123))
    N, M = world.kappa.shape
    xi = ch.draw_fading(rng, world.kappa)
    h = ch.channel_gain(world.device_pos, world.uav_pos, world, xi, cfg)
    p_kj = world.device_ptx[:, None] * h
    for trial in range(50):
        k = int(rng.integers(0, 40))
        act = np.zeros((N, M), dtype=bool)
        for _ in range(k):
            act[int(rng.integers(N)), int(rng.integers(M))] = True
        rows = rng.choice(N, size=int(rng.integers(1, 15)), replace=False)
        full = ch.rate_fdma(h, world.device_ptx, act, world.eta, cfg)
        part = ch.rate_fdma_rows(h, world.device_ptx, act, world.eta, cfg, rows, p_kj)
        assert part.tobytes() == full[rows].tobytes()
        part2 = ch.rate_fdma_rows(h, world.device_ptx, act, world.eta, cfg, rows)
        assert part2.tobytes() == full[rows].tobytes()


BURST_ARMS = [a for a in sorted(ARMS) if ARMS[a].burst_on and a != "A15"]


@pytest.mark.parametrize("arm", BURST_ARMS)
def test_fast_equals_reference_with_burst_forced(arm):
    """On untrained validation blocks theta_b = 0.75 is never exceeded, so the burst
    on/off branch of Algorithm 3 (l.15, Burst.step: want / cap / cooldown) was not exercised by
    the blocks above. This block lowers burst.theta_b to 0.05 (a diagnostic override, never a
    campaign value) so that burst turns on and off repeatedly, and requires bit-identity again."""
    import os
    n = int(os.environ.get("D8_BURST_SLOTS", "200"))
    OUT.mkdir(parents=True, exist_ok=True)

    def run(impl):
        ov = dict(ADOPTED)
        ov["drm.impl"] = impl
        ov["burst.theta_b"] = 0.05
        cfg = load_config(overrides=ov)
        return _run_cfg(arm, cfg, 9001, 2, n)
    ref, cov_r = run("reference")
    fast, cov_f = run("fast")
    d = _diff(ref, fast)
    assert d is None, f"{arm} burst-forced: first difference at slot {d[0]} field {d[1]!r}: {d[2]!r} != {d[3]!r}"
    assert cov_r["burst_on_slots"] > 0 and cov_r["burst_transitions"] > 0, cov_r
    p = OUT / f"{arm}.json"
    rec = json.loads(p.read_text()) if p.exists() else dict(arm=arm, blocks={})
    rec["burst_forced_theta_b_0.05"] = dict(slots=n, reference=cov_r, fast=cov_f)
    p.write_text(json.dumps(rec, indent=1))


def test_fdma_scan_entries_bit_identical_to_rate_fdma():
    """channel.FdmaScan.entries (one dgemm per candidate on the candidate's own w_kl) equals
    rate_fdma on the extended interferer set, bytes-equal, at random entries."""
    cfg = _cfg("fast")
    world = build_world(cfg, Streams(9002, "train").world)
    rng = np.random.Generator(np.random.PCG64(321))
    N, M = world.kappa.shape
    xi = ch.draw_fading(rng, world.kappa)
    h = ch.channel_gain(world.device_pos, world.uav_pos, world, xi, cfg)
    p_kj = world.device_ptx[:, None] * h
    for trial in range(60):
        act = np.zeros((N, M), dtype=bool)
        for _ in range(int(rng.integers(0, 30))):
            act[int(rng.integers(N)), int(rng.integers(M))] = True
        scan = ch.FdmaScan(h, world.device_ptx, act, world.eta, cfg, p_kj)
        for _ in range(5):
            d, j = int(rng.integers(N)), int(rng.integers(M))
            if trial % 7 == 0:                       # sometimes an already-active pair
                on = np.argwhere(act)
                if len(on):
                    d, j = map(int, on[int(rng.integers(len(on)))])
            devs = rng.integers(0, N, size=int(rng.integers(1, 13)))
            uavs = rng.integers(0, M, size=len(devs))
            devs[-1], uavs[-1] = d, j
            act2 = act.copy()
            act2[d, j] = True
            full = ch.rate_fdma(h, world.device_ptx, act2, world.eta, cfg)
            got = scan.entries(d, j, devs, uavs)
            assert got.tobytes() == full[devs, uavs].tobytes()
        rows = rng.choice(N, size=7, replace=False)
        full0 = ch.rate_fdma(h, world.device_ptx, act, world.eta, cfg)
        assert scan.rows(rows).tobytes() == full0[rows].tobytes()
