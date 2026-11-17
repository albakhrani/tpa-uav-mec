"""System tests: determinism, checkpoint-resume, common random numbers,
and that every ablation changes what it claims to change."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from _sys_helpers import row, tiny_cfg
from tpa.control import TPAController
from tpa.env.simulator import Simulator
from tpa.env.world import build_world
from tpa.rng import Streams
from tpa.run import checkpoint as ck
from tpa.run.logio import read_all
from tpa.run.trainer import SimulatedInterrupt, run


def _digest_nets(path):
    st = ck.load(path)["ctrl"]
    h = hashlib.sha256()
    sac = st["sac"]
    for k in sorted(sac):
        v = sac[k]
        if isinstance(v, dict):
            for kk in sorted(v):
                t = v[kk]
                if torch.is_tensor(t):
                    h.update(t.detach().cpu().numpy().tobytes())
    if "params" in st:
        for k in sorted(st["params"]):
            h.update(st["params"][k].cpu().numpy().tobytes())
    return h.hexdigest()


def _train_table(d):
    df = read_all(d / "train")
    drop = [c for c in df.columns if c.startswith("t_") or c == "wall_s"]
    return df.drop(columns=drop).sort_values("episode").reset_index(drop=True)


@pytest.mark.parametrize("arm", ["A0", "A9"])
def test_determinism_same_seed_same_output(tmp_path, arm):
    cfg = tiny_cfg()
    a = run(row(tmp_path / "a", arm, evaluate=True), cfg)
    b = run(row(tmp_path / "b", arm, evaluate=True), cfg)
    da, db = tmp_path / "a" / f"{arm}-s1", tmp_path / "b" / f"{arm}-s1"
    assert _train_table(da).equals(_train_table(db))
    assert _digest_nets(ck.final_path(da)) == _digest_nets(ck.final_path(db))
    ta = read_all(da / "test")
    tb = read_all(db / "test")
    wall = ("wall_s", "reaction_ms_p50", "reaction_ms_p90", "recovery_ms")   # wall-clock, not simulated
    cols = [c for c in ta.columns if not c.startswith("t_") and c not in wall]
    assert ta[cols].equals(tb[cols])
    assert a["val_curve"] == b["val_curve"]


@pytest.mark.parametrize("arm", ["A0", "A3", "A9"])
def test_checkpoint_resume_bitwise(tmp_path, arm):
    cfg = tiny_cfg()
    run(row(tmp_path / "straight", arm, evaluate=False), cfg)
    r = row(tmp_path / "resumed", arm, evaluate=False, _interrupt_after=2)
    with pytest.raises(SimulatedInterrupt):
        run(r, cfg)
    r.pop("_interrupt_after")
    run(r, cfg)                                       # resumes from the rolling checkpoint
    ds, dr = tmp_path / "straight" / f"{arm}-s1", tmp_path / "resumed" / f"{arm}-s1"
    assert _train_table(ds).equals(_train_table(dr))
    assert _digest_nets(ck.final_path(ds)) == _digest_nets(ck.final_path(dr))


def _env_hash(cfg, arm, seed=1, T=150):
    st = Streams(seed, "train")
    world = build_world(cfg, st.world)
    sim = Simulator(cfg, world, st, arm_key=arm, load=0.55, t0=0, horizon=T)
    try:
        from tpa.factory import make_controller
        ctrl = make_controller(cfg, arm, st, sim.r_ref, sim.h_min)
    except ImportError:
        ctrl = TPAController(cfg, arm, st, sim.r_ref, sim.h_min)
    ctrl.reset_segment(0)
    ctrl.start_episode()
    h_arr, h_ch = hashlib.sha256(), hashlib.sha256()
    for t in range(T):
        obs = sim.begin_slot(t)
        for tk in sim.last_arrivals:
            h_arr.update(np.array([tk.tid, tk.device, tk.size_bits, tk.cycles, tk.deadline_s,
                                   tk.priority, tk.complexity]).tobytes())
            h_arr.update(np.asarray(tk.loss_u).tobytes())
        h_ch.update(np.asarray(sim.xi).tobytes())
        dec, me = ctrl.act(obs, sim.truth_phase_bin(t), train=True)
        out = sim.end_slot(dec, mem_energy_j=me)
        ctrl.observe(obs, dec, out, train=True)
    return h_arr.hexdigest(), h_ch.hexdigest()


def test_common_random_numbers_across_arms():
    cfg = tiny_cfg()
    arms = ["A0", "A3", "A8", "A9", "A14"]
    try:
        import tpa.factory  # noqa: F401
        arms += ["A12", "B_ZHOU", "B_QIN"]
    except ImportError:
        pass
    ref = _env_hash(cfg, "A0")
    for a in arms[1:]:
        assert _env_hash(cfg, a) == ref, f"arrivals/channel differ for {a}"


# ---------------------------------------------------------------- ablations
def _drive(cfg, arm, T=120, seed=1, load=0.85):
    st = Streams(seed, "train")
    world = build_world(cfg, st.world)
    sim = Simulator(cfg, world, st, arm_key=arm, load=load, t0=0, horizon=T)
    c = TPAController(cfg, arm, st, sim.r_ref, sim.h_min)
    c.reset_segment(0)
    c.start_episode()
    log = dict(burst=0, cons=0, top_labels=[], chi_max=0.0, decisions=[])
    for t in range(T):
        obs = sim.begin_slot(t)
        dec, me = c.act(obs, sim.truth_phase_bin(t), train=True)
        log["burst"] += int(np.sum(dec.burst_next)) if dec.burst_next is not None else 0
        log["cons"] += int(c.last["cons"])
        log["top_labels"].append((c.last.get("rinfo") or {}).get("top_label", -1))
        log["decisions"].append(tuple((a.task_index, a.uav, round(a.f, 3)) for a in dec.assignments))
        if c.has_mem and c.last.get("P") is not None and len(obs.queue):
            log["chi_max"] = max(log["chi_max"], float(np.max(
                c.drm.pattern_priority(obs, c.last["P"], c.last["sigma"]))))
        out = sim.end_slot(dec, mem_energy_j=me)
        c.observe(obs, dec, out, train=True)
    return c, log


def test_ablations_change_what_they_claim():
    cfg = tiny_cfg(**{"burst.theta_b": 0.05})        # low threshold so burst can trigger
    c0, l0 = _drive(cfg, "A0")
    # A8: burst never on; A0 burst does turn on with this threshold
    _, l8 = _drive(cfg, "A8")
    assert l0["burst"] > 0 and l8["burst"] == 0
    assert l0["decisions"] != l8["decisions"]
    # A1: never consolidates / no long-term slots; A0 consolidates
    c1, l1 = _drive(cfg, "A1")
    assert l0["cons"] > 0 and c1.lt_occupancy() == 0
    # A2: no short-term entries
    c2, _ = _drive(cfg, "A2")
    assert c2.mem.S_fill == 0 and c0.mem.S_fill > 0
    # A3: flat store used, no separate levels
    c3, _ = _drive(cfg, "A3")
    assert c3.mem.F_fill > 0 and c3.lt_occupancy() == 0 and c3.mem.S_fill == 0
    # A4: uniform pooling -> no top-1 entry
    _, l4 = _drive(cfg, "A4")
    assert all(x == -1 for x in l4["top_labels"]) and any(x >= 0 for x in l0["top_labels"])
    # A5: consolidates every slot after the MEPR warm-up
    _, l5 = _drive(cfg, "A5")
    assert l5["cons"] == 120 - cfg.memory.window_w
    # A6: pattern priority chi is zero; A0 has chi > 0
    _, l6 = _drive(cfg, "A6")
    assert l6["chi_max"] == 0.0 and l0["chi_max"] > 0.0
    # A7: fixed weights w_base/4
    c7, _ = _drive(cfg, "A7")
    w7 = c7.weights.step(np.full(4, 0.5), 2.0)
    assert np.allclose(w7, np.asarray(cfg.objectives.w_base) / 4)
    w0 = c0.weights.step(np.full(4, 0.5), 2.0)
    assert not np.allclose(w0, np.asarray(cfg.objectives.w_base) / 4)
    # A9: no memory at all
    c9, _ = _drive(cfg, "A9")
    assert c9.mem is None and c9.params is None
    # A13: channel term weight active
    c13, _ = _drive(cfg, "A13")
    assert c13.drm.beta_c == pytest.approx(0.25) and c0.drm.beta_c == 0.0
    # A5 vs A0 can only differ when A0's gate actually closes. At the default theta_p the
    # gate stays open in nearly every post-warm-up slot here, so the behavioural A5 check
    # uses a threshold at which A0 gates.
    cfg_g = tiny_cfg(**{"burst.theta_b": 0.05, "memory.theta_p": 0.6})
    _, l0g = _drive(cfg_g, "A0")
    _, l5g = _drive(cfg_g, "A5")
    assert l0g["cons"] < l5g["cons"] and l0g["decisions"] != l5g["decisions"]
    # A13's channel term can only change the argmax when a task has >= 2 feasible UAVs.
    # Under h_min_rule="fixed" (h_min = 1e-11) no task-slot has two in this setting,
    # so the behavioural A13 check uses the from_r_min rule.
    cfg_r = tiny_cfg(**{"burst.theta_b": 0.05, "channel.h_min_rule": "from_r_min"})
    _, l0r = _drive(cfg_r, "A0")
    _, l13r = _drive(cfg_r, "A13")
    assert l0r["decisions"] != l13r["decisions"]
    # every other ablation's decisions differ from A0 somewhere
    for arm in ["A1", "A2", "A3", "A4", "A6", "A7", "A9", "A14"]:
        _, la = _drive(cfg, arm)
        assert la["decisions"] != l0["decisions"], f"{arm} identical to A0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_learner_cpu_rollout_path(tmp_path):
    """Memory arm with updates on CUDA and decisions on the CPU mirror trains, keeps
    the rollout memory params in sync with the learner copy, and evaluates."""
    cfg = tiny_cfg()
    res = run(row(tmp_path, "A0", device="cuda", evaluate=True), cfg)
    d = tmp_path / "A0-s1"
    tr = read_all(d / "train")
    assert tr["n_updates"].sum() > 0 and res["test_rows"] > 0
    st = ck.load(ck.final_path(d))["ctrl"]
    assert all(v.device.type == "cpu" for v in st["params"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_learner_determinism_and_resume(tmp_path):
    """With the CUDA learner (deterministic algorithms on), the same seed gives the same
    training log, and interrupt + resume reproduces it."""
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = tiny_cfg()
    run(row(tmp_path / "a", "A0", device="cuda", evaluate=False), cfg)
    run(row(tmp_path / "b", "A0", device="cuda", evaluate=False), cfg)
    r = row(tmp_path / "c", "A0", device="cuda", evaluate=False, _interrupt_after=2)
    with pytest.raises(SimulatedInterrupt):
        run(r, cfg)
    r.pop("_interrupt_after")
    run(r, cfg)
    ta, tb, tc = (_train_table(tmp_path / x / "A0-s1") for x in ("a", "b", "c"))
    assert ta.equals(tb), "CUDA learner not deterministic across identical runs"
    assert ta.equals(tc), "CUDA learner resume not reproducible"


def test_checkpoint_resume_bitwise_novelty(tmp_path):
    """Bitwise resume also under the novelty candidate representation."""
    cfg = tiny_cfg(**{"memory.candidate_repr": "novelty", "memory.rho_merge": 0.95})
    run(row(tmp_path / "straight", "A0", evaluate=False), cfg)
    r = row(tmp_path / "resumed", "A0", evaluate=False, _interrupt_after=2)
    with pytest.raises(SimulatedInterrupt):
        run(r, cfg)
    r.pop("_interrupt_after")
    run(r, cfg)
    ds, dr = tmp_path / "straight" / "A0-s1", tmp_path / "resumed" / "A0-s1"
    assert _train_table(ds).equals(_train_table(dr))
    assert _digest_nets(ck.final_path(ds)) == _digest_nets(ck.final_path(dr))
