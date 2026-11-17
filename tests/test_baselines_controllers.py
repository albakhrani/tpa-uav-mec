"""Baselines (B_*) and controls (A10-A12) -- protocol, determinism, CRN,
executor guarantees, A10 parameter matching."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from tpa.baselines import Executor, LookupController, fifo_window, param_counts
from tpa.config import load_config
from tpa.drm.drm import slot_admissible
from tpa.env.queueing import comp_energy
from tpa.env.simulator import Simulator
from tpa.env.world import build_world
from tpa.factory import make_controller
from tpa.rng import Streams

from _env_helpers import hash_slot

ARMS = ["B_ZHOU", "B_QIN", "B_HUANG", "B_SONG", "A10", "A11", "A12"]

# small, fast configuration: fewer devices; learning starts early so updates run in 200 slots
SMALL = {"scenario.n_devices": 60, "learner.warmup_slots": 40, "learner.batch": 16,
         "learner.replay": 2000, "learner.norm_freeze_slots": 100,
         "learner.lstm_burn_in": 4, "learner.lstm_unroll": 8}


def small_cfg(**extra):
    return load_config(overrides={**SMALL, **extra})


def make(arm_key, seed=3, cfg=None, load=0.85):
    cfg = cfg or small_cfg()
    st = Streams(seed, "train")
    world = build_world(cfg, st.world)
    sim = Simulator(cfg, world, st, arm_key=arm_key, load=load)
    ctrl = make_controller(cfg, arm_key, st, sim.r_ref, sim.h_min)
    return sim, ctrl, cfg


def dec_key(dec):
    return tuple((a.task_index, a.uav, round(a.f, 3)) for a in dec.assignments)


def run(sim, ctrl, t0, T, train=True, ep_len=None, rec=None, hasher=None):
    keys, n_upd = [], 0
    for t in range(t0, t0 + T):
        if ep_len and t > t0 and (t - t0) % ep_len == 0:
            sim.flush()
            ctrl.start_episode()
        obs = sim.begin_slot(t)
        if hasher is not None:
            hash_slot(sim, hasher)
        res = obs.residual_f.copy()
        dec, me = ctrl.act(obs, sim.truth_phase_bin(t), train)
        used = np.zeros(len(res))
        for a in dec.assignments:
            used[a.uav] += a.f
        assert np.all(used <= res * (1 + 1e-9) + 1e-6), "executor/DRM exceeded residual capacity"
        out = sim.end_slot(dec, me)
        info = ctrl.observe(obs, dec, out, train)
        n_upd += "update" in info and info["update"] is not None
        keys.append(dec_key(dec))
    return keys, n_upd


@pytest.mark.parametrize("arm_key", ARMS)
def test_runs_200_slots(arm_key):
    sim, ctrl, cfg = make(arm_key)
    ctrl.reset_segment(0)
    ctrl.start_episode()
    keys, n_upd = run(sim, ctrl, 0, 200, ep_len=50)
    assert sum(len(k) for k in keys) > 0, "no task was ever assigned"
    if arm_key in ("B_HUANG", "A10", "A11"):
        assert n_upd > 0, "no gradient update happened"
    if arm_key == "B_SONG":
        assert n_upd >= 0
    # frozen evaluation clone runs too
    ev = ctrl.eval_clone(Streams(3, "val"))
    ev.reset_segment(0)
    ev.start_episode()
    sim2, _, _ = make(arm_key, seed=3)
    run(sim2, ev, 0, 30, train=False)


@pytest.mark.parametrize("arm_key", ARMS)
def test_state_round_trip_identical_decisions(arm_key):
    sim, ctrl, cfg = make(arm_key)
    ctrl.reset_segment(0)
    ctrl.start_episode()
    run(sim, ctrl, 0, 70, ep_len=50)
    s_sim, s_ctrl = sim.get_state(), ctrl.get_state()
    k1, u1 = run(sim, ctrl, 70, 40, ep_len=None)
    # restore into the same objects
    sim.set_state(s_sim)
    ctrl.set_state(s_ctrl)
    k2, u2 = run(sim, ctrl, 70, 40, ep_len=None)
    assert k1 == k2 and u1 == u2
    if arm_key in ("B_HUANG", "A10", "A11"):
        assert u1 > 0, "round trip should cover gradient updates"


def test_song_es_iteration_and_budget():
    cfg = small_cfg()
    sim, ctrl, _ = make("B_SONG", cfg=cfg)
    ctrl.reset_segment(0)
    ctrl.start_episode()
    th0 = ctrl.theta.copy()
    _, n_upd = run(sim, ctrl, 0, 2 * ctrl.pairs * 10 + 1, ep_len=10)
    assert n_upd == 1 and ctrl.n_iter == 1 and not np.allclose(ctrl.theta, th0)
    assert ctrl.n_iterations_budget == cfg.run.episodes // (2 * ctrl.pairs)


@pytest.mark.parametrize("arm_key", ["B_ZHOU", "B_QIN"])
def test_rule_baselines_deterministic(arm_key):
    a = run(*make(arm_key)[:2], 0, 120)[0]
    b = run(*make(arm_key)[:2], 0, 120)[0]
    assert a == b and sum(len(k) for k in a) > 0


def test_crn_arrivals_identical_across_arms():
    digests = {}
    for arm_key in ARMS:
        sim, ctrl, _ = make(arm_key, seed=5)
        ctrl.reset_segment(0)
        ctrl.start_episode()
        h = hashlib.sha256()
        run(sim, ctrl, 0, 120, hasher=h)
        digests[arm_key] = h.hexdigest()
    assert len(set(digests.values())) == 1, digests


# ---------------------------------------------------------------- executor
def _collect_obs(seed=4, T=40, load=0.85):
    cfg = small_cfg()
    sim, _, _ = make("B_ZHOU", seed=seed, cfg=cfg, load=load)
    ex = Executor(cfg, "B_ZHOU", sim.h_min)
    out = []
    for t in range(T):
        obs = sim.begin_slot(t)
        out.append((obs, ex))
        dec = ex.run(obs, ex.window(obs), lambda i, k, c: c[0][0], lambda j, it, F0: [0.0] * len(it))
        sim.end_slot(dec, 0.0)
    return cfg, out


def test_executor_zero_allocation_assigns_satisfiable_tasks():
    """An all-zero method allocation still assigns every task that is satisfiable
    against the residuals left by the tasks served before it (f = max(0, f_req))."""
    cfg, data = _collect_obs()
    n_total = 0
    for obs, ex in data:
        win = ex.window(obs)
        dec = ex.run(obs, win, lambda i, k, c: c[0][0], lambda j, it, F0: [0.0] * len(it))
        got = {a.task_index for a in dec.assignments}
        # replay the sequential satisfiability check
        F, E = obs.residual_f.copy(), obs.energy.copy()
        # mirror the incremental-interference rates when the simulator provides rate_fn (slot_assigned)
        ob = replace(obs, rates_pred=np.asarray(obs.rates_pred, float).copy(), rate_fn=None)
        pairs, admitted = [], []
        for i in win:
            ok = [ex.drm._pair(ob, int(i), j, F, E, obs.eta) for j in range(ex.M)]
            if obs.rate_fn is not None:          # residual-rate-budget slot admission
                raises_of = {}
                for j in range(ex.M):
                    if ok[j][0]:
                        adm = slot_admissible(obs, admitted, (int(i), j), cfg, t_mem_s=ex.t_mem,
                                              F_res=F, E_res=E, eta=obs.eta, h_min=ex.drm.h_min,
                                              allow_raise=True)                 # raise-admission, not refuse mode
                        ok[j] = (adm.ok, adm.f_req, 0.0)
                        raises_of[j] = adm.raises
            feas = [j for j in range(ex.M) if ok[j][0]]
            if feas:
                assert int(i) in got, "admissible task not assigned under f_method = 0"
                j = feas[0]
                if obs.rate_fn is not None:
                    for p, fn in raises_of[j].items():
                        ai, aj, af = admitted[p]
                        tk = obs.queue[ai]
                        F[aj] -= fn - af
                        E[aj] -= (comp_energy(tk.cycles, fn, tk.complexity, cfg)
                                  - comp_energy(tk.cycles, af, tk.complexity, cfg))
                        admitted[p] = (ai, aj, fn)
                F[j] -= ok[j][1]
                if obs.rate_fn is not None:
                    admitted.append((int(i), j, ok[j][1]))
                    pairs.append((int(i), j))
                    ob.rates_pred = np.asarray(obs.rate_fn(list(pairs)), float)
                tk = obs.queue[int(i)]
                E[j] -= comp_energy(tk.cycles, ok[j][1], tk.complexity, cfg)
        for a in dec.assignments:          # zero method => exactly f_req
            ok, f_req, _ = ex.drm._pair(replace(obs, rate_fn=None), a.task_index, a.uav,
                                        obs.residual_f, obs.energy, obs.eta)
            assert a.f >= f_req * (1 - 1e-12) - 1e-9 or not ok
        n_total += len(dec.assignments)
    assert n_total > 0


def test_executor_never_exceeds_residual_with_greedy_allocation():
    cfg, data = _collect_obs(load=0.85)
    for obs, ex in data:
        dec = ex.run(obs, ex.window(obs), lambda i, k, c: c[0][0],
                     lambda j, it, F0: [1e12] * len(it))           # absurdly large f_method
        used = np.zeros(ex.M)
        for a in dec.assignments:
            used[a.uav] += a.f
        assert np.all(used <= obs.residual_f * (1 + 1e-9) + 1e-6)
        assert dec.util_terms["cpu_frac"].shape == (ex.M,)
        assert np.all(dec.burst_next == 0)


def test_fifo_window_order():
    cfg, data = _collect_obs(T=15)
    obs, _ = data[-1]
    win = fifo_window(obs, cfg.task.q_max)
    keys = [(obs.queue[i].t_arr, obs.queue[i].tid) for i in win]
    assert keys == sorted(keys) and len(win) == min(len(obs.queue), cfg.task.q_max)


# ---------------------------------------------------------------- A10 / A12 specifics
def test_a10_parameter_matched_to_a0():
    pc = param_counts(load_config())
    print(f"\nA0 params = {pc['A0']:,}  A10 params = {pc['A10']:,}  rel diff = {pc['rel_diff']:+.4f} "
          f"(actor {pc['actor']:,}, twin critic {pc['critics']:,}, MemoryParams {pc['memory_params']:,}, "
          f"LSTM torso {pc['lstm_torso']:,})")
    assert abs(pc["rel_diff"]) <= 0.05


def test_a10_controller_counts_match_helper():
    sim, ctrl, cfg = make("A10", cfg=load_config(overrides={"scenario.n_devices": 60}))
    from tpa.learner.networks import param_count
    n = param_count(ctrl.sac.actor) + param_count(ctrl.sac.critics) + param_count(ctrl.torso)
    assert n == param_counts(cfg)["A10"]


def test_a12_phase_and_table_reset():
    cfg = small_cfg()
    sim, ctrl, _ = make("A12", cfg=cfg)
    P = cfg.workload.period_slots
    assert ctrl.phase_bin(0) == 0 and ctrl.phase_bin(P - 1) == cfg.memory.hit_bins - 1
    assert ctrl.phase_bin(P) == 0 and ctrl.phase_bin(P // 2) == cfg.memory.hit_bins // 2
    ctrl.reset_segment(0)
    run(sim, ctrl, 0, 150)
    assert ctrl.seen.any() and ctrl.P_ref is not None          # w = 100 warm-up passed
    assert ctrl.last["sigma"] >= 0.0
    ctrl.reset_segment(150)
    assert not ctrl.seen.any() and ctrl.P_ref is None


def test_a12_ftilde_deadline_proportional():
    cfg, data = _collect_obs(T=10)
    obs, _ = data[-1]
    st = Streams(4, "train")
    c = LookupController(cfg, "A12", st, 1.0, cfg.channel.h_min, f_scale=1.0)
    win = list(range(min(len(obs.queue), cfg.task.q_max)))
    ft = c.f_tilde(obs, win)
    k = 0
    tk = obs.queue[win[k]]
    slack = tk.deadline_s - (obs.t - tk.t_arr) * cfg.scenario.slot_s
    exp = np.clip(tk.cycles * tk.complexity / (slack * obs.f_max), 0, 1)
    assert np.allclose(ft[k], exp)
