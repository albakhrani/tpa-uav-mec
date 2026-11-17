"""Tests of the key-shuffled control A15 (write keys k_{pi(t)} of a logged random earlier
slot; merge decided on the substituted key; retrieval query = the TRUE k_t), pi drawn once
per defined slot regardless of the gate, replay == rollout for A15, A0 / A14 unchanged
(stored reference rollouts, arrival-key replay), phase_coverage hand values, state round
trip (pi log, rng_a15)."""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch

from test_memory_driver import Driver, make_cfg
from tpa.arms import ARMS
from tpa.memory import (MemoryFlags, MemoryParams, SegmentStore, TwoLevelMemory, flags_for_arm,
                        phase_coverage, q_dim, reconstruct, x_dim)
from tpa.memory.retrieval import cos_rows
from tpa.rng import STREAMS, Streams

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
A15 = flags_for_arm("A15")


def _setup(flags=None, rng=None, **ov):
    cfg = make_cfg(memory__key_source="arrivals", **ov)
    params = MemoryParams(cfg)
    flags = A15 if flags is None else flags
    mem = TwoLevelMemory(cfg, flags, params, rng_a15=rng)
    store = SegmentStore(cfg, flags)
    mem.attach_store(store)
    return cfg, params, mem, store


def _rec(store, cfg, t, za, zm):
    store.record(t, np.zeros(x_dim(cfg)), np.zeros(q_dim(cfg)), np.asarray(za, float),
                 np.asarray(zm, float))


def _recon(d, cfg, T):
    errs = []
    for lo in range(0, T, 64):
        chunk = list(range(lo, min(T, lo + 64)))
        r = reconstruct(d.params, cfg, d.flags, [(d.store, t) for t in chunk], grad=False)
        r = r.double().cpu().numpy()
        errs += [np.abs(r[i] - d.r[t]).max() for i, t in enumerate(chunk)]
    return max(errs)


# ---------------------------------------------------------------- flags / raises
def test_a15_flags_stream_and_raises():
    assert A15 == MemoryFlags(key_shuffled=True) and ARMS["A15"].key_shuffled
    assert ARMS["A15"].displaced is False and ARMS["A15"].memory
    assert STREAMS["a15"] == 11
    s = Streams(1)
    assert isinstance(s.a15, np.random.Generator) and "a15" in s.get_state()
    cfg = make_cfg(memory__key_source="arrivals")
    params = MemoryParams(cfg)
    with pytest.raises(ValueError):                           # no rng
        TwoLevelMemory(cfg, A15, params)
    cfg_c = make_cfg(memory__key_source="candidate")
    with pytest.raises(ValueError):                           # "candidate": undefined
        TwoLevelMemory(cfg_c, A15, MemoryParams(cfg_c), rng_a15=np.random.default_rng(0))
    with pytest.raises(ValueError):
        SegmentStore(cfg_c, A15)
    with pytest.raises(ValueError):
        Driver(cfg_c, A15)
    # A0 under "arrivals" ignores a passed rng_a15; store pi log stays -1
    m0 = TwoLevelMemory(cfg, MemoryFlags(), params, rng_a15=np.random.default_rng(0))
    assert not m0.shuffled


# ---------------------------------------------------------------- hand values
def test_a15_substitution_hand_values():
    """w = 1 (every slot's key defined, first defined slot = t0 = 0), zm = 1, K_L = 4,
    rho = 0.8. Per slot: write_short -> consolidate -> retrieve. Checks per slot: pi(t)
    equals an independent replay of rng_a15 (integers(0, t); pi(0) = 0, no draw); the M_S
    row's S_key, the event's cand_key and (new / replaced) the slot's L_key equal
    store.arrival_key(pi(t)); the merge / new / replace decision is the one the SUBSTITUTED
    key makes against the L_key rows (and differs from what the true key would do on at
    least one slot); the retrieval equals a hand cos / tau attention with the TRUE k_t over
    the stored (substituted) keys; contents (S_raw) are the raw e_t."""
    seed = 7
    cfg, params, mem, store = _setup(rng=np.random.default_rng(seed), memory__window_w=1,
                                     memory__k_long=4, memory__k_short=6, memory__rho_merge=0.8,
                                     memory__key_mean_horizon_slots=10 ** 7, memory__key_temperature=0.1)
    Z, d = cfg.scenario.n_zones, cfg.memory.d
    rng = np.random.default_rng(11)
    ref = np.random.default_rng(seed)                         # independent replay of the draws
    T = 14
    n_diff = 0
    for t in range(T):
        za = rng.integers(0, 5, Z).astype(float) + 0.5 * rng.random(Z)
        _rec(store, cfg, t, za, np.ones(Z))
        e = torch.as_tensor(rng.standard_normal(d), dtype=torch.float32)
        z = torch.as_tensor(rng.standard_normal(d), dtype=torch.float32)
        qin = rng.standard_normal(q_dim(cfg))
        L_key0, L_filled0, L_use0 = mem.L_key.clone(), mem.L_filled.copy(), mem.L_use.copy()
        mem.write_short(e, t, t)
        pi_t = t if t == 0 else int(ref.integers(0, t))
        assert store.pi_of(t) == pi_t and store.pi[t] == pi_t and 0 <= pi_t <= t
        k_sub = store.arrival_key(pi_t)
        k_true = store.arrival_key(t)
        assert np.array_equal(mem.S_key[t % 6].numpy(), k_sub) and mem.S_keyed[t % 6]
        assert np.array_equal(mem.S_raw[t % 6].numpy(), e.numpy())
        assert np.array_equal(mem.key_for_slot(t).numpy(), k_true)
        assert np.array_equal(mem.write_key_for_slot(t).numpy(), k_sub)  # second call: logged, no re-draw
        ev = mem.consolidate(z, t, t)
        assert np.array_equal(ev["cand_key"], k_sub)
        assert store.pi_of(t) == pi_t                         # consolidate did not re-draw
        # expected decision on the substituted key
        filled = np.flatnonzero(L_filled0)
        want_mode, want_k = "new", -1
        if filled.size:
            c = cos_rows(L_key0[filled], torch.as_tensor(k_sub)).numpy()
            j = int(np.argmax(c))
            if c[j] >= 0.8:
                want_mode, want_k = "merge", int(filled[j])
            c_true = cos_rows(L_key0[filled], torch.as_tensor(k_true)).numpy()
            jt = int(np.argmax(c_true))
            true_k = int(filled[jt]) if c_true[jt] >= 0.8 else -1
            n_diff += int(true_k != want_k)
        if want_mode == "new":
            empty = np.flatnonzero(~L_filled0)
            want_k = int(empty[0]) if empty.size else int(np.argmin(L_use0))
            want_mode = "new" if empty.size else "replace"
        assert (ev["mode"], ev["k"]) == (want_mode, want_k), t
        if want_mode != "merge":
            assert np.array_equal(mem.L_key[want_k].numpy(), k_sub)
        else:
            assert np.array_equal(mem.L_key[want_k].numpy(), L_key0[want_k].numpy())
        # retrieval: TRUE query over the substituted keys
        r, info = mem.retrieve(qin, t, update_lru=False)
        with torch.no_grad():
            kt = torch.as_tensor(k_true)
            MS, ML = mem.M_S(), mem.M_L()
            mS = torch.as_tensor(mem.S_keyed) & (torch.arange(6) < mem.S_fill)
            lS = (cos_rows(mem.S_key, kt) / 0.1).masked_fill(~mS, float("-inf"))
            lL = (cos_rows(mem.L_key, kt) / 0.1).masked_fill(~torch.as_tensor(mem.L_filled), float("-inf"))
            om = params.level_weights(qin)
            want = om[0] * params.W_V(torch.softmax(lS, 0) @ MS) + om[1] * params.W_V(torch.softmax(lL, 0) @ ML)
        assert np.allclose(r, want.double().numpy(), atol=1e-5), t
        assert not info["empty"]
    assert n_diff > 0                                         # the shuffle changed a decision
    assert store.pi[0] == 0 and (store.pi[1:T] < np.arange(1, T)).all() and (store.pi >= 0).all()
    assert set(store.events["mode"]) >= {"new", "merge"}
    # every stored key is one of A0's keys (the arrival keys of the segment)
    seq = store.arrival_keys_upto(T)
    for row in mem.L_key[mem.L_filled].numpy():
        assert any(np.array_equal(row, s) for s in seq)
    # an explicit key argument still bypasses the substitution
    _rec(store, cfg, T, np.ones(Z), np.ones(Z))
    mem.write_short(torch.zeros(d), T, T)
    ev = mem.consolidate(torch.zeros(d), T, T, key=np.eye(Z)[0])
    assert np.array_equal(ev["cand_key"], np.eye(Z)[0].astype(np.float32))


def test_a15_first_defined_slot_and_undefined_slots():
    """w = 3: slots 0, 1 have no key -> no draw, pi = -1, unkeyed M_S rows; slot 2 is the
    first defined slot -> pi(2) = 2 without consuming the stream; slot 3 -> pi(3) = 2."""
    rng = np.random.default_rng(5)
    cfg, params, mem, store = _setup(rng=rng, memory__window_w=3, memory__k_short=4)
    Z, d = cfg.scenario.n_zones, cfg.memory.d
    st0 = copy.deepcopy(rng.bit_generator.state)
    for t in range(3):
        _rec(store, cfg, t, np.arange(1, Z + 1, dtype=float) * (t + 1), np.ones(Z))
        mem.write_short(torch.zeros(d), t, t)
        assert mem.consolidate(torch.zeros(d), t, t) is None or t == 2
    assert store.pi[:3].tolist() == [-1, -1, 2]
    assert rng.bit_generator.state == st0                     # no draw yet
    assert mem.S_keyed.tolist() == [False, False, True, False]
    assert np.array_equal(mem.S_key[2].numpy(), store.arrival_key(2))
    assert np.array_equal(mem.L_key[0].numpy(), store.arrival_key(2))
    _rec(store, cfg, 3, np.ones(Z), np.ones(Z))
    mem.write_short(torch.zeros(d), 3, 3)
    # slot 3: the range {2} has one element: pi(3) = 2 (numpy's integers() consumes no bits
    # for a one-element range, so the state is unchanged here; it advances at slot 4)
    assert store.pi[3] == 2 and rng.bit_generator.state == st0
    assert np.array_equal(mem.S_key[3].numpy(), store.arrival_key(2))
    assert np.array_equal(mem.key_for_slot(3).numpy(), store.arrival_key(3))
    assert not np.array_equal(store.arrival_key(2), store.arrival_key(3))
    _rec(store, cfg, 4, 2 * np.ones(Z), np.ones(Z))
    mem.write_short(torch.zeros(d), 4, 4)
    assert store.pi[4] in (2, 3) and rng.bit_generator.state != st0
    assert store.pi[4] == int(np.random.default_rng(5).integers(2, 4))
    store._pi.a[4] = -1                                       # (undo for the guard checks below)
    store.n = 4
    # write_keys_upto: rows 0, 1 zero; rows 2, 3 = k_2
    wk = store.write_keys_upto(4)
    assert np.all(wk[:2] == 0) and np.array_equal(wk[2], store.arrival_key(2)) and np.array_equal(wk[3], store.arrival_key(2))
    # log_pi guards: not an earlier defined slot / already logged differently
    with pytest.raises(ValueError):
        store.log_pi(3, 1)
    with pytest.raises(ValueError):
        store.log_pi(3, 3)
    store.log_pi(3, 2)                                        # idempotent re-log is fine
    # a defined slot without a logged pi cannot be replayed (no re-draw)
    _rec(store, cfg, 4, np.ones(Z), np.ones(Z))
    with pytest.raises(ValueError):
        store.write_keys_upto(5)


# ---------------------------------------------------------------- pi independent of the gate
def test_a15_pi_drawn_once_per_defined_slot_regardless_of_gate():
    T, w = 260, 100
    cfg_lo = make_cfg(memory__key_source="arrivals", memory__theta_p=float("-inf"))
    cfg_hi = make_cfg(memory__key_source="arrivals", memory__theta_p=1e9)
    assert cfg_lo.memory.theta_p == float("-inf") and cfg_hi.memory.theta_p == 1e9
    d_lo = Driver(cfg_lo, A15, seed=3, period=40).run(T)
    d_hi = Driver(cfg_hi, A15, seed=3, period=40).run(T)
    assert d_lo.store.n_ev > 100 and d_hi.store.n_ev == 0   # gate open / closed
    pi = d_lo.store.pi
    assert np.array_equal(pi, d_hi.store.pi)
    assert (pi[: w - 1] == -1).all() and pi[w - 1] == w - 1
    i = np.arange(w, T)
    assert (pi[i] >= w - 1).all() and (pi[i] < i).all()
    # exactly one integers() draw per defined slot after the first: replay the stream
    ref = np.random.default_rng(3 + 3000)
    want = np.array([int(ref.integers(w - 1, t)) for t in range(w, T)])
    assert np.array_equal(pi[w:], want)
    assert d_lo.rng_a15.bit_generator.state == d_hi.rng_a15.bit_generator.state
    assert d_lo.rng_a15.bit_generator.state == ref.bit_generator.state
    # the stored M_S keys are the substituted keys in both runs
    seq = d_lo.store.arrival_keys_upto(T)
    for p in range(cfg_lo.memory.k_short):
        t = int(d_lo.mem.S_time[p])
        assert np.array_equal(d_lo.mem.S_key[p].numpy(), seq[pi[t]]) and np.array_equal(d_hi.mem.S_key[p].numpy(), seq[pi[t]])
    # pi is a function of the stream only: different driver seed, same rng_a15 -> same pi
    d2 = Driver(cfg_lo, A15, seed=8, period=40)
    d2.rng_a15 = np.random.default_rng(3003)
    d2.mem.rng_a15 = d2.rng_a15
    d2.run(T)
    assert np.array_equal(d2.store.pi, pi)


# ---------------------------------------------------------------- replay == rollout
@pytest.mark.parametrize("case", ["A0-shape", "A0-smallKL", "A3-shape"])
@pytest.mark.parametrize("mode", ["fixed_key", "soft"])
def test_a15_replay_equals_rollout(case, mode):
    ov = dict(memory__key_source="arrivals", memory__candidate_repr="layernorm",
              memory__lt_addressing=mode)
    flags, T, per = A15, 450, 40
    if case == "A0-smallKL":
        ov.update(memory__k_long=8, memory__rho_merge=0.999)
        T, per = 450, 300
    if case == "A3-shape":
        if mode == "soft":
            pytest.skip("soft addressing is an M_L mode")
        ov.update(memory__k_short=20, memory__k_long=30)
        flags = MemoryFlags(flat=True, key_shuffled=True)
    cfg = make_cfg(**ov)
    d = Driver(cfg, flags, seed=3, period=per).run(T)
    w, Z = cfg.memory.window_w, cfg.scenario.n_zones
    assert d.store.n_ev > 0 and d.store.events["key"].shape[1] == Z
    pi_before = d.store.pi.copy()
    rng_before = copy.deepcopy(d.rng_a15.bit_generator.state)
    err = _recon(d, cfg, T)
    assert err < 1e-5, err
    assert np.array_equal(d.store.pi, pi_before)              # replay never re-draws
    assert d.rng_a15.bit_generator.state == rng_before
    assert (pi_before[: w - 1] == -1).all() and (pi_before[w - 1:] >= w - 1).all()
    # the logged keys are the substituted ones
    seq = d.store.arrival_keys_upto(T)
    ev = d.store.events
    assert all(np.array_equal(ev["cand_key"][i], seq[pi_before[ev["t"][i]]]) for i in range(d.store.n_ev))
    if case != "A3-shape":
        modes = set(ev["mode"])
        assert ("soft" in modes) if mode == "soft" else ("merge" in modes or "replace" in modes)
    else:
        assert set(ev["mode"]) == {"fifo"}
        wk = d.store.write_keys_upto(T)
        for p in range(d.mem.F_fill):
            t = int(d.mem.F_time[p])
            if d.mem.F_keyed[p]:
                assert np.array_equal(d.mem.F_key[p].numpy(), wk[t]) and np.array_equal(wk[t], seq[pi_before[t]])
    # the replay depends on the log: a tampered pi changes r(t) on some slot
    d.store._pi.a[w:T] = w - 1
    r2 = reconstruct(d.params, cfg, d.flags, [(d.store, t) for t in range(T - 64, T)], grad=False)
    r2 = r2.double().numpy()
    assert max(np.abs(r2[i] - d.r[t]).max() for i, t in enumerate(range(T - 64, T))) > 1e-5


# ---------------------------------------------------------------- A0 / A14 unchanged
@pytest.mark.parametrize("name,arm,ov,T,per", [
    ("A0", "A0", {}, 300, 40),
    ("A3", "A3", {}, 300, 40),
    ("A14", "A14", dict(workload__period_slots=400, memory__k_long=8, memory__rho_merge=0.999), 700, 300),
])
def test_candidate_source_still_reproduces_reference_rollouts(name, arm, ov, T, per):
    ref = np.load(os.path.join(DATA, f"d4_reference_{name}.npz"))
    pre_d4 = dict(memory__key_source="candidate", memory__candidate_repr="tanh",
                  memory__rho_merge=0.8, memory__theta_p=0.15)       # settings the arrays depend on
    cfg = make_cfg(memory__key_centering=False, memory__candidate_source="pattern",
                   **{**pre_d4, **ov})
    d = Driver(cfg, flags_for_arm(arm), seed=3, period=per).run(T)
    r = np.stack([d.r[t] for t in range(T)])
    assert np.abs(r - ref["r"]).max() <= 1e-6
    ev = d.store.events
    assert list(ev["mode"]) == list(ref["ev_mode"])
    assert np.array_equal(ev["t"], ref["ev_t"]) and np.array_equal(ev["k"], ref["ev_k"])
    assert np.abs(ev["row_after"] - ref["ev_row_after"]).max() <= 1e-6
    assert np.abs(d.mem.L_raw.numpy() - ref["L_raw"]).max() <= 1e-6
    assert np.abs(d.mem.L_key.numpy() - ref["L_key"]).max() <= 1e-6
    assert np.abs(d.mem.F_raw.numpy() - ref["F_raw"]).max() <= 1e-6
    assert int(d.mem.F_fill) == int(ref["F_fill"])
    assert [d.info[t]["top_level"] or "" for t in range(T)] == list(ref["top_level"])
    assert np.array_equal(np.array([d.info[t]["top_age"] for t in range(T)]), ref["top_age"])
    assert (d.store.pi == -1).all() and not d.mem.shuffled


@pytest.mark.parametrize("arm", ["A0", "A14"])
def test_a0_a14_arrivals_replay_and_contents_shared_with_a15(arm):
    ov = dict(memory__key_source="arrivals", memory__candidate_repr="layernorm")
    T, per = 300, 40
    if arm == "A14":
        ov.update(workload__period_slots=400, memory__k_long=8, memory__rho_merge=0.999)
        T, per = 700, 300
    cfg = make_cfg(**ov)
    d = Driver(cfg, flags_for_arm(arm), seed=3, period=per).run(T)
    assert _recon(d, cfg, T) < 1e-5
    assert (d.store.pi == -1).all()
    w = cfg.memory.window_w
    assert all(np.all(d.r[t] == 0) and d.info[t]["empty"] for t in range(w - 1))
    # the arrival keys of the stored entries are the slot's own key (no substitution)
    seq = d.store.arrival_keys_upto(T)
    assert np.array_equal(d.store.write_keys_upto(T), seq)
    for i in range(d.store.n_ev):
        assert np.array_equal(d.store.events["cand_key"][i], seq[d.store.events["t"][i]])
    if arm == "A0":
        # A15 on the same inputs: identical arrival keys, e_t and M_S contents (contents are
        # untouched by the shuffle); only the stored keys differ
        d15 = Driver(cfg, A15, seed=3, period=per).run(T)
        assert np.array_equal(d15.store.arrival_keys_upto(T), seq)
        assert np.array_equal(d15.store.e, d.store.e) and np.array_equal(d15.store.x, d.store.x)
        assert np.array_equal(d15.mem.S_raw.numpy(), d.mem.S_raw.numpy())
        assert not np.array_equal(d15.mem.S_key.numpy(), d.mem.S_key.numpy())
        assert any(not np.array_equal(d15.r[t], d.r[t]) for t in range(w, T))


# ---------------------------------------------------------------- phase_coverage
def test_phase_coverage_hand_values():
    P, t0 = 80, 5
    e = np.eye(6)
    # filled slots: times -> bins ((t - t0) mod P) * 8 // P
    times = [t0 + 0, t0 + 9, t0 + 10, t0 + P + 79, t0 + 2 * P, t0 + 3 * P + 40]
    #        bin 0,  0 (72//80), 1,     7 (wrap: 79*8//80 = 7), 0,   4
    keys = [e[0], e[0] + e[1], e[2], e[3], e[1], e[4]]
    KL = 8
    L_time = np.full(KL, -1, np.int64)
    L_key = np.zeros((KL, 6))
    L_filled = np.zeros(KL, bool)
    for j, (tm, k) in enumerate(zip(times, keys)):
        L_time[j], L_key[j], L_filled[j] = tm, k, True
    L_time[6], L_key[6] = t0 + 20, e[5]                       # unfilled: ignored
    pc = phase_coverage(L_key, L_time, L_filled, P, t0)
    assert pc["bins"] == [3, 1, 0, 0, 1, 0, 0, 1] and pc["covered"] is False
    # within: the three bin-0 pairs: (e0, e0+e1) = 1/sqrt2, (e0, e1) = 0, (e0+e1, e1) = 1/sqrt2
    assert pc["n_within"] == 3 and abs(pc["within"] - (2 / np.sqrt(2)) / 3) < 1e-12
    # between: 15 - 3 = 12 pairs, every one orthogonal (bins 1, 4, 7 hold e2, e4, e3)
    assert pc["n_between"] == 12 and abs(pc["between"] - 0.0) < 1e-12
    assert pc["key_pass"] is True
    # undefined within (every bin holds at most one slot): nan, key_pass False
    L_filled2 = L_filled.copy()
    L_filled2[[1, 4]] = False                                 # bin 0 keeps one slot
    pc2 = phase_coverage(L_key, L_time, L_filled2, P, t0)
    assert pc2["bins"] == [1, 1, 0, 0, 1, 0, 0, 1] and pc2["n_within"] == 0 and np.isnan(pc2["within"])
    assert pc2["n_between"] == 6 and pc2["between"] == 0.0 and pc2["key_pass"] is False
    # full coverage with within <= between -> covered but key_pass False
    tm8 = t0 + np.arange(8) * (P // 8) + P              # one per bin, previous cycle
    tm8 = np.concatenate([tm8, [t0 + P]])               # a second slot in bin 0
    k8 = np.vstack([e[:6], e[0], e[1], e[1]])            # bin-0 pair (e0, e1): within 0
    pc3 = phase_coverage(k8, tm8, np.ones(9, bool), P, t0)
    assert pc3["bins"] == [2, 1, 1, 1, 1, 1, 1, 1] and pc3["covered"] is True
    assert pc3["n_within"] == 1 and pc3["within"] == 0.0 and pc3["n_between"] == 35
    assert pc3["between"] > 0.0 and pc3["key_pass"] is False
    # bin origin: the same t0 shift for every slot leaves the counts a permutation
    pc4 = phase_coverage(k8, tm8, np.ones(9, bool), P, t0 + 10)
    assert sorted(pc4["bins"]) == sorted(pc3["bins"]) and pc4["n_within"] == pc3["n_within"]
    # empty / single slot: no pairs
    pc5 = phase_coverage(L_key, L_time, np.zeros(KL, bool), P, t0)
    assert pc5["bins"] == [0] * 8 and pc5["n_within"] == 0 and pc5["n_between"] == 0 and not pc5["key_pass"]
    assert np.isnan(pc5["within"]) and np.isnan(pc5["between"]) and pc5["covered"] is False
    with pytest.raises(ValueError):
        phase_coverage(L_key, L_time, L_filled, 0, t0)
    # on a driver store (A0, arrivals): bins sum to the filled count
    cfg = make_cfg(memory__key_source="arrivals", memory__k_long=8, memory__rho_merge=0.999)
    d = Driver(cfg, flags_for_arm("A0"), seed=2, period=40).run(200)
    pc6 = phase_coverage(d.mem.L_key.numpy(), d.mem.L_time, d.mem.L_filled, 40, 0)
    assert sum(pc6["bins"]) == int(d.mem.L_filled.sum()) == 8


# ---------------------------------------------------------------- state round trip
def test_a15_state_round_trip():
    cfg = make_cfg(memory__key_source="arrivals", memory__k_long=8, memory__rho_merge=0.999,
                   memory__candidate_repr="layernorm")
    a = Driver(cfg, A15, seed=9, period=300, min_age=150).run(350)
    s = copy.deepcopy(a.get_state())
    assert "rng_a15" in s and "pi" in s["store"]
    assert np.array_equal(s["store"]["pi"], a.store.pi) and (s["store"]["pi"][99:] >= 99).all()
    b = Driver(cfg, A15, seed=123, period=300, params=a.params, min_age=150)
    b.set_state(copy.deepcopy(s))
    assert np.array_equal(b.store.pi, a.store.pi)
    assert b.rng_a15.bit_generator.state == a.rng_a15.bit_generator.state
    assert np.array_equal(b.store.write_keys_upto(350), a.store.write_keys_upto(350))
    for _ in range(150):
        ra, ia = a.step()
        rb, ib = b.step()
        assert np.array_equal(ra, rb) and ia == ib
    assert np.array_equal(a.store.pi, b.store.pi) and (a.store.pi[350:] >= 0).all()
    sa, sb = a.get_state(), b.get_state()
    assert np.array_equal(sa["mem"]["S_key"], sb["mem"]["S_key"]) and np.array_equal(sa["mem"]["L_key"], sb["mem"]["L_key"])
    assert _recon(a, cfg, 500) < 1e-5
    # an older state without rng_a15 / pi still loads (A0 states): pi comes back as -1
    s0 = copy.deepcopy(Driver(cfg, flags_for_arm("A0"), seed=1, period=300).run(120).get_state())
    del s0["rng_a15"]
    del s0["store"]["pi"]
    c = Driver(cfg, flags_for_arm("A0"), seed=2, period=300)
    c.set_state(s0)
    assert (c.store.pi == -1).all() and c.store.n == 120


# ---------------------------------------------------------------- controller path (control.py)
def test_a15_controller_passes_stream_and_logs_pi():
    """tpa/control.py passes streams.a15 under A15; the controller's slot order (record ->
    write_short -> consolidate -> retrieve) draws pi once per defined slot; replay on the
    controller's store equals its rollout r(t) (no learner updates: train=False)."""
    from _sys_helpers import tiny_cfg
    from tpa.control import TPAController
    from tpa.env.simulator import Simulator
    from tpa.env.world import build_world
    with pytest.raises(ValueError):                           # "candidate" (explicit; default "arrivals"): undefined
        st = Streams(1, "train")
        cfg = tiny_cfg(**{"memory.key_source": "candidate"})
        TPAController(cfg, "A15", st, 1.0, 0.1)
    cfg = tiny_cfg(**{"memory.key_source": "arrivals"})
    T, w = 60, cfg.memory.window_w
    st = Streams(1, "train")
    world = build_world(cfg, st.world)
    sim = Simulator(cfg, world, st, arm_key="A15", load=0.55, t0=0, horizon=T)
    c = TPAController(cfg, "A15", st, sim.r_ref, sim.h_min)
    assert c.mem.shuffled and c.mem.rng_a15 is st.a15 and c.mflags.key_shuffled
    c.reset_segment(0)
    c.start_episode()
    for t in range(T):
        obs = sim.begin_slot(t)
        dec, me = c.act(obs, sim.truth_phase_bin(t), train=False)
        out = sim.end_slot(dec, me)
        c.observe(obs, dec, out, train=False)
    pi = c.store.pi
    assert pi.shape == (T,) and (pi[: w - 1] == -1).all() and pi[w - 1] == w - 1
    assert (pi[w:] >= w - 1).all() and (pi[w:] < np.arange(w, T)).all()
    ref = Streams(1, "train").a15
    assert np.array_equal(pi[w:], [int(ref.integers(w - 1, t)) for t in range(w, T)])
    assert st.a15.bit_generator.state == ref.bit_generator.state
    seq = c.store.arrival_keys_upto(T)
    ev = c.store.events
    assert c.store.n_ev > 0
    assert all(np.array_equal(ev["cand_key"][i], seq[pi[ev["t"][i]]]) for i in range(c.store.n_ev))
    r_log = c.r_log[c.seg_id]
    r = reconstruct(c.params, cfg, c.mflags, [(c.store, t) for t in range(T)], grad=False).double().numpy()
    assert max(np.abs(r[t] - r_log[t]).max() for t in range(T)) < 1e-5
    s = c.get_state()
    assert "pi" in s["stores"][c.seg_id] and np.array_equal(s["stores"][c.seg_id]["pi"], pi)
