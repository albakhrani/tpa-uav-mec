"""Frozen-policy evaluation (out-of-sample protocol).

For each load and each of cfg.run.test_runs test runs: a fresh test-block stream set
Streams(seed, "test", k) (same world as training), a simulator starting at the
chronologically later slot t0 = test_seed_offset_slots + k * test_slots, and an evaluation
clone of the trained controller with empty memory. No gradient, no replay, deterministic
actions. Returns per-run summary rows and per-minute rows (row_kind = "run" / "minute").
The same routine evaluates on the validation block for checkpoint selection.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..env.simulator import Simulator
from ..env.world import build_world
from ..rng import Streams
from .metrics import SlotAccumulator


def run_block(ctrl_eval, cfg, world, streams, arm_key, workload, load, t0, n_slots,
              env_kw=None, label=None):
    """Run one frozen evaluation block. Returns (summary_row, minute_rows, timer_summary)."""
    env_kw = env_kw or {}
    sim = Simulator(cfg, world, streams, arm_key=arm_key, workload_kind=workload, load=load,
                    t0=t0, horizon=n_slots, **env_kw)
    ctrl_eval.reset_segment(t0)
    ctrl_eval.start_episode()
    has_mem = getattr(ctrl_eval, "has_mem", False)
    hit_defined = has_mem and not getattr(getattr(ctrl_eval, "mflags", None), "mean_pool", False) \
        and workload in ("sinusoidal", "shifted", "burst")
    acc = SlotAccumulator(cfg, has_mem, hit_defined)
    cap_nom = world.f_max * world.eta
    theta_p = float(cfg.memory.theta_p)
    n = n_slots
    act_ms = np.zeros(n)
    arr = np.zeros(n)
    done = np.zeros(n)
    detect = np.zeros(n, dtype=bool)
    burst_on = np.zeros(n, dtype=bool)
    w0 = time.perf_counter()
    prev_sig_above, prev_burst = False, False
    for i, t in enumerate(range(t0, t0 + n_slots)):
        obs = sim.begin_slot(t)
        tb = sim.truth_phase_bin(t)
        a0 = time.perf_counter_ns()
        dec, me = ctrl_eval.act(obs, tb, train=False)
        act_ms[i] = (time.perf_counter_ns() - a0) / 1e6          # decision-path wall-clock
        out = sim.end_slot(dec, mem_energy_j=me)
        info = ctrl_eval.observe(obs, dec, out, train=False)
        last = getattr(ctrl_eval, "last", {})
        acc.add(t, obs, dec, out, last, tb, info,
                ctrl_eval.mem_fill() if hasattr(ctrl_eval, "mem_fill") else 0.0, cap_nom)
        arr[i], done[i] = out.arrivals, len(out.completed)
        # a detection is a TRANSITION - sigma_p crossing theta_p upwards or a
        # burst activation turning on - not merely sigma_p being above threshold
        sig_above = float(last.get("sigma", 0.0)) > theta_p
        b_on = bool(dec.burst_next is not None and np.any(dec.burst_next))
        detect[i] = (sig_above and not prev_sig_above) or (b_on and not prev_burst)
        prev_sig_above, prev_burst = sig_above, b_on
        burst_on[i] = bool(getattr(sim.workload, "in_burst", lambda _t: False)(t))
    dist = disturbance_metrics(act_ms, arr, done, detect, burst_on)
    row = acc.summary()
    row.update(wall_s=time.perf_counter() - w0, **dist)
    if hasattr(ctrl_eval, "lt_occupancy"):
        row["lt_occupancy"] = ctrl_eval.lt_occupancy()
    tags = dict(label or {})
    for r in [row] + acc.rows_min:
        r.update(tags)
    return row, acc.rows_min, ctrl_eval.timer.summary()


def disturbance_metrics(act_ms, arr, done, detect, burst_on, pre=300, win=60, tol=0.02) -> dict:
    """Recovery latency and control-loop reaction time, plus service recovery.

    reaction_ms_p50/p90 : per-decision wall-clock of the control action (act()).
    recovery_ms         : mean, over disturbance onsets (sudden-burst family), of the
                          decision wall-clock in the first slot at/after onset in which the
                          controller registers the disturbance (sigma_p > theta_p or a burst
                          activation) -- detect + recompute + apply. detect_delay_slots: mean
                          slots from onset to that detection.
    service_recovery_slots : slots after onset until the 60-slot rolling completion ratio is
                          back within 2 pp of the 300-slot pre-onset level, after first falling
                          below it (0 if it never falls). Simulation time, not wall-clock.
    Wall-clock values are only meaningful in exclusive, P-core-pinned timing runs.
    """
    out = dict(reaction_ms_p50=float(np.median(act_ms)) if len(act_ms) else math.nan,
               reaction_ms_p90=float(np.percentile(act_ms, 90)) if len(act_ms) else math.nan,
               recovery_ms=math.nan, detect_delay_slots=math.nan,
               service_recovery_slots=math.nan, n_disturbances=0,
               false_detect_rate=float(np.mean(detect[~burst_on])) if np.any(~burst_on) else math.nan)
    onsets = [i for i in range(1, len(burst_on)) if burst_on[i] and not burst_on[i - 1]]
    rec, dly, srv = [], [], []
    for o in onsets:
        hits = np.nonzero(detect[o:])[0]
        if len(hits):
            rec.append(act_ms[o + hits[0]])
            dly.append(int(hits[0]))
        lo = max(0, o - pre)
        if arr[lo:o].sum() <= 0:
            continue
        ref = done[lo:o].sum() / arr[lo:o].sum()
        fell, k_rec = False, None
        for k in range(0, len(arr) - o - win):
            a_, d_ = arr[o + k:o + k + win].sum(), done[o + k:o + k + win].sum()
            ratio = d_ / a_ if a_ > 0 else ref
            if ratio < ref - tol:
                fell = True
            elif fell:
                k_rec = k
                break
        srv.append(0 if not fell else (k_rec if k_rec is not None else math.nan))
    if onsets:
        out.update(n_disturbances=len(onsets),
                   recovery_ms=float(np.mean(rec)) if rec else math.nan,
                   detect_delay_slots=float(np.mean(dly)) if dly else math.nan,
                   service_recovery_slots=float(np.nanmean(srv)) if srv and not all(
                       math.isnan(x) for x in srv) else math.nan)
    return out


def evaluate(ctrl, cfg, seed: int, arm_key: str, workloads=("sinusoidal",), loads=None,
             env_kw=None, block: str = "test") -> list[dict]:
    """Test-block evaluation of a trained controller. Returns all rows (run + minute)."""
    loads = list(cfg.workload.loads) if loads is None else list(loads)
    world = build_world(cfg, Streams(seed, "train").world)        # same world as training
    R = cfg.run
    rows = []
    n_runs = int(R.test_runs) if block in ("test", "val_full") else 1
    n_slots = int(R.test_slots) if block in ("test", "val_full") else int(R.val_slots)
    stream_block = "val" if block == "val_full" else block      # val_full: test-length runs on val traces
    for wl in workloads:
        for load in loads:
            for k in range(n_runs):
                # val_full uses validation traces other than the checkpoint-selection trace (k=0)
                st = Streams(seed, stream_block, test_run=k + 1 if block == "val_full" else k)
                # validation block starts elsewhere in the cycle phase than test run 0
                base = int(R.test_seed_offset_slots) if block == "test" else int(R.test_seed_offset_slots) // 2 + n_slots // 3
                t0 = base + k * n_slots
                ce = ctrl.eval_clone(st)
                label = dict(arm=arm_key, seed=seed, workload=wl, load=float(load),
                             test_run=k, block=block)
                run_row, minute_rows, tim = run_block(ce, cfg, world, st, arm_key, wl, load,
                                                      t0, n_slots, env_kw, label)
                for key, v in tim.items():
                    run_row[f"t_{key}_ms"] = v["mean_ms"]
                rows.append(run_row)
                rows.extend(minute_rows)
    return rows


def validate(ctrl, cfg, seed: int, arm_key: str) -> float:
    """Validation-block score for checkpoint selection: medium-load completion over one
    segment of the same seed's validation block."""
    loads = list(cfg.workload.loads)
    med = loads[len(loads) // 2]
    rows = evaluate(ctrl, cfg, seed, arm_key, loads=[med], block="val")
    run = [r for r in rows if r["row_kind"] == "run"]
    return float(np.nanmean([r["completion"] for r in run]))
