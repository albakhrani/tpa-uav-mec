"""No demand-cycle phase and no slot index in the state of Eq. (48) or in any baseline's
inputs (A12 excepted: it reads t mod P by design).

Two simulators that differ ONLY in the segment start slot (t0 = 0 vs t0 = 2P + P/3, a
non-multiple of P) see the same arrivals, channels and positions (same CRN streams; the
sinusoidal zone modulation psi_z(t) is patched to 1 so the generator itself has no
t-dependence). Then every SlotObs field other than `t` and `queue[*].t_arr`, the task
features / queue summary of tpa.control, the retrieved vector r(t), sigma_p and the
decisions of every controller except A12 must coincide slot for slot. The harness
`label` (= sim.truth_phase_bin(t), which DOES differ between the two offsets: bin 0/1
vs bin 8) must change nothing but the memory-entry tag reported in rinfo["top_label"].
"""
from __future__ import annotations

import numpy as np
import pytest

from tpa.config import load_config
from tpa.control import queue_summary, task_features
from tpa.env.simulator import Simulator
from tpa.env.world import build_world
from tpa.factory import make_controller
from tpa.rng import Streams

SEED = 3                       # train / val streams only (no test-block trace is read)
T = 120                        # slots per rollout
SMALL = {"scenario.n_devices": 60}
# Tolerance. The simulator keeps continuous slot-time on ABSOLUTE values (end_slot:
# st = max(float(t), t_upload_done), fin = st + need, cycles_left -= f (t_end - st) dt), so
# at t ~ 8400 the last bit of a float64 (ulp ~ 1.8e-12) moves backlog / reserved_frac by
# ~1e-11 relative and, through a learned controller's continuous f = frac * F_max (1e9
# cycles/s), the reservation by ~5e-11 relative. That is rounding, not a phase input; an
# absolute 1e-9 on 1e9-scale quantities would be below float64 resolution. Everything is
# therefore compared at RTOL = 1e-9 (plus atol 1e-9 for O(1) quantities); assignment
# (task, UAV) pairs, ages, burst decisions and memory tags are compared exactly.
RTOL = 1e-9
ATOL = 1e-9

OBS_FIELDS = ("S_sys", "rates_pred", "h_now", "zone_arrivals", "zone_mean", "residual_f",
              "cap_bound", "energy", "e_min", "burst", "lam_eff", "mu", "f_max", "eta",
              "uav_occupancy")
TASK_FIELDS = ("tid", "device", "zone", "size_bits", "cycles", "deadline_s", "priority",
               "complexity")


def _cfg():
    return load_config(overrides=SMALL)


def _dec_key(dec):
    return tuple((int(a.task_index), int(a.uav), float(a.f)) for a in dec.assignments)


def _rollout(arm_key: str, t0: int, label_mode: str = "truth", eval_clone: bool = True):
    """Run `arm_key` from slot t0 for T slots; return per-slot records of controller inputs,
    outputs and the obs fields."""
    cfg = _cfg()
    Z = int(cfg.scenario.n_zones)
    st = Streams(SEED, "train")
    world = build_world(cfg, st.world)
    sim = Simulator(cfg, world, st, arm_key=arm_key, load=float(cfg.workload.loads[1]),
                    t0=t0, horizon=T)
    # remove the generator's own t-dependence: psi_z(t) = 1 (same Poisson intensities from
    # either start slot); truth_bin(t) still reports the (now meaningless) cycle phase bin
    sim.workload.zone_psi = lambda t: np.ones(Z)
    ctrl = make_controller(cfg, arm_key, Streams(SEED, "train"), sim.r_ref, sim.h_min)
    if eval_clone:
        ctrl = ctrl.eval_clone(Streams(SEED, "val"))
    ctrl.reset_segment(t0)
    ctrl.start_episode()
    rec = []
    for t in range(t0, t0 + T):
        obs = sim.begin_slot(t)
        label = sim.truth_phase_bin(t) if label_mode == "truth" else 0
        dec, me = ctrl.act(obs, label, train=False)
        out = sim.end_slot(dec, mem_energy_j=me)
        ctrl.observe(obs, dec, out, train=False)
        last = getattr(ctrl, "last", {}) or {}
        r = None
        if getattr(ctrl, "has_mem", False):
            r = np.asarray(ctrl.r_log[ctrl.seg_id][int(t)], np.float64)
        n_win = int(last.get("n_window", 0))
        win = list(range(len(obs.queue)))[:n_win]          # proxy window (same length)
        rec.append(dict(
            t=int(t), label=int(label),
            obs={k: np.asarray(getattr(obs, k), np.float64).copy() for k in OBS_FIELDS},
            ages=np.array([t - k.t_arr for k in obs.queue], np.int64),
            tasks=np.array([[float(getattr(k, f)) for f in TASK_FIELDS] for k in obs.queue],
                           np.float64).reshape(len(obs.queue), len(TASK_FIELDS)),
            tf=task_features(obs, win, cfg, sim.r_ref),
            qs=queue_summary(obs, cfg, sim.r_ref),
            dec=_dec_key(dec),
            burst_next=None if dec.burst_next is None else np.asarray(dec.burst_next, float),
            mem_energy=float(me),
            r=r, sigma=float(last.get("sigma", 0.0)),
            top_label=(last.get("rinfo") or {}).get("top_label", None),
            rinfo_empty=(last.get("rinfo") or {}).get("empty", None),
        ))
    return rec


def _assert_same_inputs(ra, rb):
    """Every controller input except the slot index itself coincides."""
    assert len(ra) == len(rb) == T
    for a, b in zip(ra, rb):
        assert a["t"] != b["t"]                                   # the only thing that differs
        for k in OBS_FIELDS:
            np.testing.assert_allclose(a["obs"][k], b["obs"][k], rtol=RTOL, atol=ATOL, err_msg=k)
        np.testing.assert_array_equal(a["ages"], b["ages"])      # t - t_arr is offset-free
        np.testing.assert_array_equal(a["tasks"], b["tasks"])    # generator output identical
        np.testing.assert_allclose(a["tf"], b["tf"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(a["qs"], b["qs"], rtol=RTOL, atol=ATOL)


def _assert_same_outputs(ra, rb, check_r: bool):
    for a, b in zip(ra, rb):
        pa, pb = [(i, j) for i, j, _ in a["dec"]], [(i, j) for i, j, _ in b["dec"]]
        assert pa == pb, f"assignment pairs differ at slot {a['t']} vs {b['t']}"
        fa, fb = np.array([f for _, _, f in a["dec"]]), np.array([f for _, _, f in b["dec"]])
        np.testing.assert_allclose(fa, fb, rtol=RTOL, atol=ATOL)
        if a["burst_next"] is not None or b["burst_next"] is not None:
            np.testing.assert_array_equal(a["burst_next"], b["burst_next"])
        assert a["mem_energy"] == pytest.approx(b["mem_energy"], rel=RTOL, abs=ATOL)
        assert a["sigma"] == pytest.approx(b["sigma"], rel=RTOL, abs=ATOL)
        if check_r:
            np.testing.assert_allclose(a["r"], b["r"], rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------ generator sanity
def test_offsets_differ_in_phase_bin_only():
    cfg = _cfg()
    P = int(cfg.workload.period_slots)
    t0b = 2 * P + P // 3
    assert t0b % P != 0
    st = Streams(SEED, "train")
    world = build_world(cfg, st.world)
    sim = Simulator(cfg, world, st, arm_key="A0", load=0.55)
    assert sim.truth_phase_bin(0) != sim.truth_phase_bin(t0b)


# ------------------------------------------------------------------ Eq. (48) state, A0 / A9
@pytest.mark.parametrize("arm_key", ["A0", "A9"])
def test_tpa_state_and_decisions_are_start_offset_invariant(arm_key):
    cfg = _cfg()
    P = int(cfg.workload.period_slots)
    t0b = 2 * P + P // 3
    ra = _rollout(arm_key, 0)
    rb = _rollout(arm_key, t0b)
    # the harness label really differs between the two runs (bins 0/1 vs bin 8)
    assert set(x["label"] for x in ra).isdisjoint(set(x["label"] for x in rb))
    _assert_same_inputs(ra, rb)
    _assert_same_outputs(ra, rb, check_r=(arm_key == "A0"))
    if arm_key == "A0":
        assert all(x["r"] is not None for x in ra)
        assert any(np.abs(x["r"]).sum() > 0 for x in ra)          # memory was actually read
        # the tag follows the label (diagnostic only) while r(t) and the decisions do not
        tags_a = {x["top_label"] for x in ra if x["rinfo_empty"] is False}
        tags_b = {x["top_label"] for x in rb if x["rinfo_empty"] is False}
        assert tags_a and tags_b and tags_a.isdisjoint(tags_b)


def test_label_argument_does_not_enter_r_or_decision():
    """A0 at the same start slot, label = truth bin vs label = 0: identical r(t), sigma_p and
    decisions; only rinfo['top_label'] follows the label."""
    ra = _rollout("A0", 0, label_mode="truth")
    rb = _rollout("A0", 0, label_mode="zero")
    _assert_same_inputs_same_t(ra, rb)
    _assert_same_outputs(ra, rb, check_r=True)
    tags_b = {x["top_label"] for x in rb if x["rinfo_empty"] is False}
    assert tags_b <= {0}
    tags_a = {x["top_label"] for x in ra if x["rinfo_empty"] is False}
    assert tags_a and all(0 <= v < int(_cfg().memory.hit_bins) for v in tags_a)


def _assert_same_inputs_same_t(ra, rb):
    for a, b in zip(ra, rb):
        assert a["t"] == b["t"]
        for k in OBS_FIELDS:
            np.testing.assert_allclose(a["obs"][k], b["obs"][k], rtol=RTOL, atol=ATOL, err_msg=k)
        np.testing.assert_array_equal(a["ages"], b["ages"])


# ------------------------------------------------------------------ baselines and controls
@pytest.mark.parametrize("arm_key", ["A10", "A11", "B_ZHOU", "B_QIN", "B_HUANG", "B_SONG"])
def test_baseline_decisions_are_start_offset_invariant(arm_key):
    cfg = _cfg()
    P = int(cfg.workload.period_slots)
    t0b = 2 * P + P // 3
    ra = _rollout(arm_key, 0)
    rb = _rollout(arm_key, t0b)
    _assert_same_inputs(ra, rb)
    _assert_same_outputs(ra, rb, check_r=False)


def test_a12_is_the_declared_exception():
    """A12 reads t mod P on purpose; its phase bin differs between the two offsets."""
    from tpa.baselines import LookupController
    cfg = _cfg()
    P = int(cfg.workload.period_slots)
    t0b = 2 * P + P // 3
    st = Streams(SEED, "train")
    world = build_world(cfg, st.world)
    sim = Simulator(cfg, world, st, arm_key="A12", load=0.55)
    c = make_controller(cfg, "A12", st, sim.r_ref, sim.h_min)
    assert isinstance(c, LookupController)
    assert c.phase_bin(0) != c.phase_bin(t0b)
    assert c.phase_bin(t0b) == c.phase_bin(t0b + P)
