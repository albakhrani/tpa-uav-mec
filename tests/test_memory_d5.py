"""Tests of the arrival keys (memory.key_source = "arrivals"): hand values of a_t, mu
and the arrival key k_t = a_t - mu_{t-1}; the merge decision on that key (not on the
contents); the "candidate" source reproduces the stored reference rollouts (atol 1e-6);
retrieval logits cos(k_t, K_j) / tau_key broadcast over the heads (value path unchanged;
W_Q / W_K unused); unkeyed M_S rows are not attended; top_old_* with min_age on a
hand-built store and on the A14 snapshot; greedy / isolated key counts and the
age-matched phase share on hand-built inputs; the Driver's cumulative zone mean; state
round-trips carry S_key / S_keyed / F_keyed and the snapshot S_key; reconstruction
equals rollout under "arrivals" (the per-arm sweep is in tests/test_memory_recon.py)."""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch

from test_memory_driver import Driver, make_cfg
from tpa.memory import (EPS, MemoryFlags, MemoryParams, SegmentStore, TwoLevelMemory,
                        age_matched_phase_share, attend, flags_for_arm, greedy_key_groups,
                        isolated_key_count, key_dim, q_dim, reconstruct, x_dim)
from tpa.memory.retrieval import cos_rows

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _setup(flags=None, zero_gate=True, **ov):
    """Memory + attached store under key_source = "arrivals"."""
    cfg = make_cfg(memory__key_source="arrivals", **ov)
    params = MemoryParams(cfg)
    if zero_gate:
        with torch.no_grad():                               # g = sigmoid(0) = 0.5 everywhere
            params.W_g.weight.zero_()
            params.W_g.bias.zero_()
    flags = MemoryFlags() if flags is None else flags
    mem = TwoLevelMemory(cfg, flags, params)
    store = SegmentStore(cfg, flags)
    mem.attach_store(store)
    return cfg, params, mem, store


def _rec(store, cfg, t, za, zm):
    store.record(t, np.zeros(x_dim(cfg)), np.zeros(q_dim(cfg)), np.asarray(za, float),
                 np.asarray(zm, float))


# ---------------------------------------------------------------- hand values: a_t, mu, k_t
def test_arrival_key_hand_values():
    """w = 3, alpha = 1/4. a_t = mean(za[t-2..t]) / zm_t; k_2 = a_2 (first defined, uncentred);
    mu_2 = a_2; k_3 = a_3 - a_2; mu_3 = ((1-a) a_2 + a_3) / (2 - a); k_4 = a_4 - mu_3."""
    cfg, params, mem, store = _setup(memory__window_w=3, memory__key_mean_horizon_slots=4)
    Z = cfg.scenario.n_zones
    assert Z == 6 and key_dim(cfg) == Z and mem.kd == Z and store.kd == Z
    rng = np.random.default_rng(0)
    za = rng.integers(0, 7, size=(6, Z)).astype(float)
    zm = 1.0 + rng.random((6, Z)) * 3.0
    for t in range(6):
        _rec(store, cfg, t, za[t], zm[t])
    assert mem.key_for_slot(0) is None and mem.key_for_slot(1) is None
    a = {t: za[t - 2:t + 1].mean(axis=0) / zm[t] for t in range(2, 6)}
    al = 0.25
    mu2 = a[2]
    mu3 = ((1 - al) * al * a[2] + al * a[3]) / ((1 - al) * al + al)
    assert np.allclose(mu3, ((1 - al) * a[2] + a[3]) / (2 - al))
    mu4 = ((1 - al) ** 2 * al * a[2] + (1 - al) * al * a[3] + al * a[4]) / \
        ((1 - al) ** 2 * al + (1 - al) * al + al)
    want = {2: a[2], 3: a[3] - mu2, 4: a[4] - mu3, 5: a[5] - mu4}
    for t, k in want.items():
        got = mem.key_for_slot(t)
        assert got.dtype == torch.float32 and got.shape == (Z,)
        assert np.allclose(got.numpy(), k.astype(np.float32), atol=1e-6), t
        assert np.allclose(store.arrival_key(t), k, atol=1e-6)
    seq = store.arrival_keys_upto(6)
    assert seq.shape == (6, Z) and seq.dtype == np.float32
    assert np.all(seq[:2] == 0.0) and store.arrival_defined(6).tolist() == [False, False] + [True] * 4
    # the cache is the single source: the same values on every call (no recomputation drift)
    assert np.array_equal(store.arrival_keys_upto(6), seq)
    assert np.array_equal(mem.key_for_slot(4).numpy(), seq[4])
    # EPS floor on the zone mean, as in compute_v_batch
    zm6 = np.array([0.0] + [1.0] * (Z - 1))
    _rec(store, cfg, 6, np.ones(Z), zm6)
    a6 = np.vstack([za[4:6], np.ones(Z)]).mean(axis=0) / np.maximum(zm6, EPS)
    mu5 = store._ak_S / store._ak_c                           # mu after slot 5 (cache state)
    store.arrival_keys_upto(6)
    assert np.allclose(store.arrival_key(6), (a6 - mu5).astype(np.float32), rtol=1e-6)
    assert a6[0] > 1e7                                        # zone-mean floor at EPS
    with pytest.raises(ValueError):
        store.arrival_key(7)                                  # not recorded
    # no store attached under "arrivals" (the default config, then cfg): key_for_slot raises
    cfg_c = make_cfg()
    mem_c = TwoLevelMemory(cfg_c, MemoryFlags(), MemoryParams(cfg_c))
    with pytest.raises(ValueError):
        mem_c.key_for_slot(0)
    with pytest.raises(ValueError):
        TwoLevelMemory(cfg, MemoryFlags(), params).key_for_slot(0)
    with pytest.raises(ValueError):
        TwoLevelMemory(make_cfg(memory__key_source="zone"), MemoryFlags(), params)
    with pytest.raises(ValueError):
        TwoLevelMemory(make_cfg(memory__key_source="arrivals", memory__key_temperature=0.0),
                       MemoryFlags(), params)


# ---------------------------------------------------------------- merge decision on the key
def test_merge_decision_on_arrival_key_not_contents():
    """w = 1 (a_t = za_t / zm_t), zm = 1, alpha tiny (mu = running mean to 1e-6).
    t0: a = e0 -> key e0 (uncentred), slot 0. t1: a = e1, mu_0 = e0 -> key e1 - e0: cos to
    e0 = -0.71 -> new slot 1. t2: a = 1.5 e1 - 0.5 e0, mu_1 = (e0 + e1)/2 -> key e1 - e0 ->
    cos = 1 with slot 1 -> MERGE into slot 1, although the CONTENT of slot 2's write equals
    slot 0's content (content addressing would pick slot 0). Before the key is defined
    (w = 3 variant) consolidate writes nothing."""
    cfg, params, mem, store = _setup(memory__window_w=1, memory__key_mean_horizon_slots=10 ** 7,
                                     memory__k_long=3, memory__rho_merge=0.8)
    Z, d = cfg.scenario.n_zones, cfg.memory.d
    eZ, ed = np.eye(Z), torch.eye(d)
    z0, z1 = ed[0], ed[1]
    _rec(store, cfg, 0, eZ[0], np.ones(Z))
    ev0 = mem.consolidate(z0, 0, 0)
    _rec(store, cfg, 1, eZ[1], np.ones(Z))
    ev1 = mem.consolidate(z1, 1, 1)
    assert (ev0["mode"], ev0["k"], ev1["mode"], ev1["k"]) == ("new", 0, "new", 1)
    assert np.allclose(mem.L_key[0].numpy(), eZ[0]) and np.allclose(mem.L_key[1].numpy(), eZ[1] - eZ[0], atol=1e-6)
    assert ev1["cand_key"].shape == (Z,) and float(ev1["maxcos"]) < 0.0
    _rec(store, cfg, 2, 1.5 * eZ[1] - 0.5 * eZ[0], np.ones(Z))
    k2 = mem.key_for_slot(2).numpy()
    assert np.allclose(k2, eZ[1] - eZ[0], atol=1e-6)
    ev2 = mem.consolidate(z0, 2, 2)                           # content identical to slot 0's
    assert ev2["mode"] == "merge" and ev2["k"] == 1 and float(ev2["maxcos"]) > 0.999
    assert np.allclose(ev2["cand_key"], k2)
    assert np.allclose(mem.L_key[1].numpy(), eZ[1] - eZ[0], atol=1e-6)   # key unchanged by a merge
    want = 0.5 * (0.5 * z1) + 0.5 * z0                        # row * g + z * (1 - g), g = 0.5
    assert np.allclose(mem.L_raw[1].numpy(), want.numpy(), atol=1e-6)
    assert mem.mu_n == 0                                      # candidate mean untouched
    mem.observe_candidate(z0)
    assert mem.mu_n == 0                                      # ... and a no-op under "arrivals"
    # contents vs keys: content addressing would have merged into slot 0
    assert float(cos_rows(mem.L_raw[:2], z0)[0]) > 0.99
    # explicit key argument is honoured (Z-dim)
    ev3 = mem.consolidate(z1, 3, 3, key=eZ[0])
    assert ev3["mode"] == "merge" and ev3["k"] == 0
    # before the key is defined: no write, no event
    cfg3, p3, m3, s3 = _setup(memory__window_w=3)
    _rec(s3, cfg3, 0, eZ[0], np.ones(Z))
    assert m3.consolidate(z0, 0, 0) is None and s3.n_ev == 0 and not m3.L_filled.any()
    _rec(s3, cfg3, 1, eZ[0], np.ones(Z))
    _rec(s3, cfg3, 2, eZ[0], np.ones(Z))
    assert m3.consolidate(z0, 2, 2)["mode"] == "new" and s3.n_ev == 1
    assert s3.events["key"].shape == (1, Z) and s3.events["cand_key"].shape == (1, Z)


# ---------------------------------------------------------------- "candidate" == stored reference
@pytest.mark.parametrize("name,arm,ov,T,per", [
    ("A0", "A0", {}, 300, 40),
    ("A3", "A3", {}, 300, 40),
    ("A14", "A14", dict(workload__period_slots=400, memory__k_long=8, memory__rho_merge=0.999), 700, 300),
])
def test_candidate_source_reproduces_reference_rollouts(name, arm, ov, T, per):
    """key_source = "candidate" (explicit) + the off path == tests/data/d4_reference_*.npz
    (the same comparison as test_memory_d4.test_off_path_reproduces_reference_rollouts)."""
    ref = np.load(os.path.join(DATA, f"d4_reference_{name}.npz"))
    pre_d4 = dict(memory__key_source="candidate", memory__candidate_repr="tanh",
                  memory__rho_merge=0.8, memory__theta_p=0.15)       # settings the arrays depend on
    cfg = make_cfg(memory__key_centering=False, memory__candidate_source="pattern",
                   **{**pre_d4, **ov})
    assert cfg.memory.key_source == "candidate" and make_cfg().memory.key_source == "arrivals"   # the default
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
    assert d.mem.kd == cfg.memory.d and d.store.kd == cfg.memory.d
    assert not d.mem.S_keyed.any() and float(d.mem.S_key.abs().sum()) == 0.0   # S_key unused
    assert "top_old_age" not in d.info[T - 1]                                   # min_age None


def test_candidate_default_path_unchanged_by_key_source_flag():
    """The default config (zone, centred, key_source = "arrivals") gives the same rollout
    with key_source left at its default and with key_source = "arrivals" written explicitly;
    the candidate-key path (mu advances, d-dimensional keys) is selected only by writing
    key_source = "candidate"."""
    a = Driver(make_cfg(), flags_for_arm("A0"), seed=3, period=40).run(200)
    b = Driver(make_cfg(memory__key_source="arrivals"), flags_for_arm("A0"), seed=3, period=40).run(200)
    assert all(np.array_equal(a.r[t], b.r[t]) for t in range(200))
    assert a.mem.mu_n == 0 and a.store.events["key"].shape[1] == a.cfg.scenario.n_zones
    c = Driver(make_cfg(memory__key_source="candidate"), flags_for_arm("A0"), seed=3, period=40).run(200)
    assert c.mem.mu_n > 0 and c.store.events["key"].shape[1] == c.cfg.memory.d


# ---------------------------------------------------------------- retrieval under "arrivals"
def test_arrivals_retrieval_logits_and_value_path():
    """r = sum_l omega_l W_V (sum_j softmax_j(cos(k_t, K_j) / tau) M_j): one head on the key
    side, the value path (W_V, head pooling) and omega unchanged; W_Q / W_K do not enter."""
    cfg, params, mem, store = _setup(memory__window_w=1, memory__k_long=4, memory__k_short=3,
                                     memory__rho_merge=0.999, memory__key_mean_horizon_slots=10 ** 7,
                                     memory__key_temperature=0.1, zero_gate=False)
    Z, d = cfg.scenario.n_zones, cfg.memory.d
    rng = np.random.default_rng(1)
    for t in range(4):
        _rec(store, cfg, t, rng.integers(1, 6, Z).astype(float), 1.0 + rng.random(Z))
        mem.write_short(torch.as_tensor(rng.standard_normal(d), dtype=torch.float32), t, t)
        mem.consolidate(torch.as_tensor(rng.standard_normal(d), dtype=torch.float32), t, t)
    assert mem.L_filled.sum() == 4 and mem.S_fill == 3 and mem.S_keyed.all()
    t = 3
    qin = rng.standard_normal(q_dim(cfg))
    r, info = mem.retrieve(qin, t, update_lru=False)
    k = mem.key_for_slot(t)
    with torch.no_grad():
        MS, ML = mem.M_S(), mem.M_L()
        aS = torch.softmax(cos_rows(mem.S_key, k) / 0.1, dim=0)
        aL = torch.softmax(cos_rows(mem.L_key, k) / 0.1, dim=0)
        om = params.level_weights(qin)
        want = om[0] * params.W_V(aS @ MS) + om[1] * params.W_V(aL @ ML)
    assert np.allclose(r, want.double().numpy(), atol=1e-5)
    assert info["top_level"] in ("S", "L") and not info["empty"]
    # W_Q / W_K are not on this path
    with torch.no_grad():
        params.W_Q.weight.mul_(3.0)
        params.W_K.weight.mul_(-2.0)
    r2, _ = mem.retrieve(qin, t, update_lru=False)
    assert np.array_equal(r, r2)
    with torch.no_grad():
        params.W_V.weight.mul_(2.0)
    r3, _ = mem.retrieve(qin, t, update_lru=False)
    assert np.allclose(r3, 2 * r, atol=1e-5)
    # attend requires keys on every level under "arrivals"
    with pytest.raises(ValueError):
        attend(params, torch.as_tensor(qin, dtype=torch.float32)[None],
               [(MS[None], torch.ones(1, 3, dtype=torch.bool)), None], qkey=k[None], key_tau=0.1)


def test_arrivals_undefined_key_and_unkeyed_short_rows():
    """w = 3: slots 0, 1 have no key -> no LT write, r = 0, info empty; their M_S rows are
    unkeyed and never attended; at t = 2 only row 2 is attended."""
    cfg, params, mem, store = _setup(memory__window_w=3, memory__k_short=4, zero_gate=False)
    Z, d = cfg.scenario.n_zones, cfg.memory.d
    rng = np.random.default_rng(2)
    es = [torch.as_tensor(rng.standard_normal(d), dtype=torch.float32) for _ in range(3)]
    qin = rng.standard_normal(q_dim(cfg))
    for t in range(3):
        _rec(store, cfg, t, rng.integers(1, 6, Z).astype(float), np.ones(Z))
        mem.write_short(es[t], t, t)
        ev = mem.consolidate(es[t], t, t)
        assert (ev is None) == (t < 2)
        r, info = mem.retrieve(qin, t, update_lru=False)
        if t < 2:
            assert np.all(r == 0) and info["empty"] and info["top_level"] is None
            assert store.n_ev == 0
    assert mem.S_keyed.tolist() == [False, False, True, False]
    assert float(mem.S_key[:2].abs().sum()) == 0.0
    r, info = mem.retrieve(qin, 2, update_lru=False)
    assert not info["empty"] and info["top_level"] in ("S", "L")
    with torch.no_grad():
        om = params.level_weights(qin)
        wS = params.W_V(mem.M_S()[2])                          # the only keyed M_S row
        wL = params.W_V(mem.M_L()[0])                          # the single L entry
        want = om[0] * wS + om[1] * wL
    assert np.allclose(r, want.double().numpy(), atol=1e-5)
    # A3: unkeyed e rows likewise excluded; z rows and keyed e rows carry the slot key
    cfg3, p3, m3, s3 = _setup(MemoryFlags(flat=True), memory__window_w=3, memory__k_short=4,
                              memory__k_long=4, zero_gate=False)
    for t in range(3):
        _rec(s3, cfg3, t, rng.integers(1, 6, Z).astype(float), np.ones(Z))
        m3.write_short(es[t], t, t)
        m3.consolidate(es[t], t, t)
    assert m3.F_fill == 4 and m3.F_keyed[:4].tolist() == [False, False, True, True]
    k2 = m3.key_for_slot(2).numpy()
    assert np.array_equal(m3.F_key[2].numpy(), k2) and np.array_equal(m3.F_key[3].numpy(), k2)
    assert m3.F_kind[2:4].tolist() == [0, 1]
    r3, info3 = m3.retrieve(qin, 2, update_lru=False)
    with torch.no_grad():
        MF = m3.M_F()
        want = p3.W_V(0.5 * MF[2] + 0.5 * MF[3])              # equal keys -> equal weights
    assert np.allclose(r3, want.double().numpy(), atol=1e-5) and info3["top_level"] == "F"


# ---------------------------------------------------------------- top_old_* (min_age)
def _top_old_expected(L_key, L_filled, L_time, t, k, min_age):
    ages = t - np.asarray(L_time)
    old = np.asarray(L_filled, bool) & (ages >= min_age)
    if not old.any():
        return None, -1
    idx = np.flatnonzero(old)
    K = np.asarray(L_key, np.float64)[idx]
    c = (K @ k) / (np.maximum(np.linalg.norm(K, axis=1), 1e-12) * max(np.linalg.norm(k), 1e-12))
    j = int(idx[int(np.argmax(c))])
    return j, int(ages[j])


def test_top_old_hand_built_store():
    cfg, params, mem, store = _setup(memory__window_w=1, memory__k_long=4, memory__rho_merge=0.999,
                                     memory__key_mean_horizon_slots=10 ** 7, zero_gate=False)
    Z, d = cfg.scenario.n_zones, cfg.memory.d
    rng = np.random.default_rng(3)
    qin = rng.standard_normal(q_dim(cfg))
    for t in range(4):
        _rec(store, cfg, t, rng.integers(1, 6, Z).astype(float), np.ones(Z))
        mem.write_short(torch.as_tensor(rng.standard_normal(d), dtype=torch.float32), t, t)
        mem.consolidate(torch.as_tensor(rng.standard_normal(d), dtype=torch.float32), t, t)
    assert mem.L_time.tolist() == [0, 1, 2, 3]
    t = 9
    _rec(store, cfg, t, rng.integers(1, 6, Z).astype(float), np.ones(Z))
    k = mem.key_for_slot(t).double().numpy()
    base = copy.deepcopy(mem.get_state())
    r0, i0 = mem.retrieve(qin, t)                             # no min_age: no top_old_* keys
    assert "top_old_age" not in i0
    lru0 = mem.L_use.copy()
    mem.set_state(copy.deepcopy(base))
    for min_age in (6, 7, 8, 9, 10):
        mem.set_state(copy.deepcopy(base))
        r, info = mem.retrieve(qin, t, min_age=min_age)
        j, age = _top_old_expected(mem.L_key.numpy(), mem.L_filled, mem.L_time, t, k, min_age)
        assert (info["top_old_slot"], info["top_old_age"]) == (j, age), min_age
        if j is not None:
            assert age >= min_age and mem.L_filled[j]
        assert np.array_equal(r, r0) and np.array_equal(mem.L_use, lru0)   # r, LRU untouched
        assert (info["top_label"], info["top_age"], info["top_level"]) == (i0["top_label"], i0["top_age"], i0["top_level"])
    assert mem.retrieve(qin, t, min_age=10)[1]["top_old_slot"] is None       # ages 6..9 < 10
    # the oldest-only case: min_age = 9 -> slot 0 is the only candidate
    assert mem.retrieve(qin, t, min_age=9)[1]["top_old_slot"] == 0
    # A1 (no M_L) / A4: None / -1
    cfg1, p1, m1, s1 = _setup(MemoryFlags(use_ml=False), memory__window_w=1)
    _rec(s1, cfg1, 0, np.ones(Z), np.ones(Z))
    m1.write_short(torch.zeros(d), 0, 0)
    assert m1.retrieve(qin, 0, min_age=0)[1]["top_old_age"] == -1


def test_top_old_on_a14_snapshot_and_driver_min_age():
    cfg = make_cfg(memory__key_source="arrivals", workload__period_slots=400, memory__k_long=8,
                   memory__rho_merge=0.999)
    d = Driver(cfg, flags_for_arm("A14"), seed=4, period=300, min_age=150)
    n_checked, n_diff = 0, 0
    for t in range(500):
        d.step()                                              # snapshots are pruned: check live
        info = d.info[t]
        if t < 300 or info["mem_time"] < 0 or info["top_old_age"] < 0:
            continue
        snap = d.mem.snapshots[int(info["mem_time"])]
        assert "S_key" in snap and "S_keyed" in snap and "L_key" in snap
        k = d.store.arrival_key(t).astype(np.float64)        # the LIVE query
        j, age = _top_old_expected(snap["L_key"].numpy(), snap["L_filled"], snap["L_time"], t, k, 150)
        assert (info["top_old_slot"], info["top_old_age"]) == (j, age), t
        assert age >= 150
        # the live memory would give a different answer on some slots
        jl, al = _top_old_expected(d.mem.L_key.numpy(), d.mem.L_filled, d.mem.L_time, t, k, 150)
        n_diff += int((jl, al) != (j, age))
        n_checked += 1
    assert n_checked > 10 and n_diff > 0
    mt = d.store.mem_time
    # before the key is defined / no displaced memory: -1
    assert all(d.info[t]["top_old_age"] == -1 for t in range(cfg.memory.window_w - 1))
    assert all(d.info[t]["top_old_age"] == -1 and d.info[t]["empty"] for t in range(500) if mt[t] < 0)
    # A0 with min_age on the default store (K_L = 500, no replacement): entries written from
    # slot w - 1 on persist, so by t = 300 one is older than min_age = 100; >= min_age whenever
    # defined; None / -1 otherwise. (Under the K_L = 8 config above A0 replaces its LRU entry
    # every few slots, so no entry reaches age 150 there.)
    d0 = Driver(make_cfg(memory__key_source="arrivals"), flags_for_arm("A0"), seed=4, period=300,
                min_age=100).run(400)
    ages = [d0.info[t]["top_old_age"] for t in range(400)]
    assert all(a >= 100 for a in ages[300:]) and all(a == -1 or a >= 100 for a in ages)
    assert all(d0.info[t]["top_old_slot"] is None for t in range(400) if d0.info[t]["top_old_age"] < 0)
    assert all(d0.info[t]["top_old_age"] == -1 for t in range(199))    # first write at 99, age < 100


# ---------------------------------------------------------------- key group counts / age-matched phase share
def test_greedy_groups_and_isolated_counts_hand_values():
    e = np.eye(6)
    rows = np.stack([e[0], e[0] + 0.1 * e[1], e[1], e[0] + e[1]])
    assert greedy_key_groups(rows, 0.95) == 3                 # leaders e0, e1, e0 + e1
    assert isolated_key_count(rows, 0.95) == 2                # e1 and e0 + e1 (max cos 0.71)
    assert greedy_key_groups(np.zeros((0, 6))) == 0 and isolated_key_count(np.zeros((0, 6))) == 0
    assert greedy_key_groups(rows[:1]) == 1 and isolated_key_count(rows[:1]) == 1
    assert greedy_key_groups(np.stack([e[0], e[0]])) == 1 and isolated_key_count(np.stack([e[0], e[0]])) == 0
    # slot order matters for the greedy count: a, b (cos .96), c (cos(a,c) .93, cos(b,c) .996)
    th_b, th_c = np.arccos(0.96), np.arccos(0.93)
    a, b, c = np.array([1.0, 0.0]), np.array([np.cos(th_b), np.sin(th_b)]), np.array([np.cos(th_c), np.sin(th_c)])
    assert greedy_key_groups(np.stack([a, b, c]), 0.95) == 2
    assert greedy_key_groups(np.stack([b, a, c]), 0.95) == 1
    assert isolated_key_count(np.stack([a, b, c]), 0.95) == 0
    # threshold is strict: cosine exactly at cos does not start a group / is not isolated
    assert greedy_key_groups(np.stack([e[0], e[0]]), 1.0) == 1 and isolated_key_count(np.stack([e[0], e[0]]), 1.0) == 0


def test_age_matched_phase_share_hand_built_info():
    P, w = 600, 100
    lo, hi = w + P, w + 3 * P
    info, kinds = {}, []
    for i, t in enumerate(range(lo, hi)):
        kind = ["hit_exact", "hit_edge", "miss_phase", "miss_undefined", "miss_none",
                "miss_missing", "hit_fold", "miss_half"][i % 8]
        kinds.append(kind)
        if kind == "hit_exact":
            info[t] = dict(top_old_slot=3, top_old_age=P, empty=False)
        elif kind == "hit_edge":
            info[t] = dict(top_old_slot=3, top_old_age=P + P // 8, empty=False)   # 75 -> hit
        elif kind == "miss_phase":
            info[t] = dict(top_old_slot=3, top_old_age=P + P // 8 + 1, empty=False)
        elif kind == "miss_undefined":
            info[t] = dict(top_old_slot=None, top_old_age=-1, empty=False)
        elif kind == "miss_none":
            info[t] = None                                                         # A14: no memory
        elif kind == "miss_missing":
            pass                                                                   # slot absent
        elif kind == "hit_fold":
            info[t] = dict(top_old_slot=1, top_old_age=2 * P - P // 16, empty=False)   # fold 37
        else:
            info[t] = dict(top_old_slot=1, top_old_age=P // 2 + 10, empty=True)    # fold 290: miss
    want = np.mean([k.startswith("hit") for k in kinds])
    assert abs(want - 3 / 8) < 1e-12
    share = age_matched_phase_share(info, P, w)
    assert abs(share - want) < 1e-12
    # top_age (the field phase_consistency reads) is ignored; only top_old_age counts
    info2 = {t: dict(top_level="L", top_age=P, top_old_age=-1) for t in range(lo, hi)}
    assert age_matched_phase_share(info2, P, w) == 0.0

    class D:
        pass
    dr = D()
    dr.info = info
    assert age_matched_phase_share(dr, P, w, T=hi) == share
    with pytest.raises(ValueError):
        age_matched_phase_share(info, P, w, T=hi - 1)


# ---------------------------------------------------------------- Driver cumulative mode
def test_driver_cumulative_zone_mean():
    cfg = make_cfg(memory__key_source="arrivals")
    dc = Driver(cfg, flags_for_arm("A0"), seed=0, period=40, zone_mean_mode="cumulative").run(60)
    za = dc.store.zone_arr
    cum = np.cumsum(za, axis=0) / np.arange(1, 61)[:, None]
    assert np.allclose(dc.store.zone_mean, cum) and np.allclose(dc.zone_mean, cum[-1])
    de = Driver(cfg, flags_for_arm("A0"), seed=0, period=40).run(60)
    assert de.zone_mean_mode == "ema" and np.array_equal(de.store.zone_arr, za)   # same draws
    zm = np.ones(cfg.scenario.n_zones)
    for t in range(60):
        zm = 0.99 * zm + 0.01 * za[t]
        assert np.allclose(de.store.zone_mean[t], zm)
    assert not np.allclose(de.store.zone_mean, dc.store.zone_mean)
    with pytest.raises(ValueError):
        Driver(cfg, flags_for_arm("A0"), zone_mean_mode="mean")
    # the cumulative accumulator round-trips
    s = copy.deepcopy(dc.get_state())
    assert "za_sum" in s and np.allclose(s["za_sum"], za.sum(axis=0))
    b = Driver(cfg, flags_for_arm("A0"), seed=5, period=40, zone_mean_mode="cumulative", params=dc.params)
    b.set_state(copy.deepcopy(s))
    for _ in range(30):
        ra, _ = dc.step()
        rb, _ = b.step()
        assert np.array_equal(ra, rb)
    assert np.allclose(b.zone_mean, np.cumsum(dc.store.zone_arr, axis=0)[-1] / 90)


# ---------------------------------------------------------------- state round-trip
@pytest.mark.parametrize("arm", ["A0", "A3", "A14"])
def test_state_round_trip_arrivals(arm):
    cfg = make_cfg(memory__key_source="arrivals", memory__k_long=8, memory__rho_merge=0.999,
                   workload__period_slots=400, memory__candidate_repr="layernorm")
    a = Driver(cfg, flags_for_arm(arm), seed=9, period=300, min_age=150).run(350)
    s = copy.deepcopy(a.get_state())
    for k in ("S_key", "S_keyed", "F_keyed", "L_key", "F_key"):
        assert k in s["mem"], k
    Z = cfg.scenario.n_zones
    assert s["mem"]["S_key"].shape == (cfg.memory.k_short, Z) and s["mem"]["L_key"].shape[1] == Z
    assert s["mem"]["F_key"].shape[1] == Z
    assert s["mem"]["mu_n"] == 0                              # candidate mean never advanced
    assert s["store"]["ev_key"].shape[1] == Z and s["store"]["ev_ckey"].shape[1] == Z and s["store"]["fkey"].shape[1] == Z
    if arm == "A3":
        assert s["mem"]["F_keyed"].sum() > 0
    else:
        assert s["mem"]["S_keyed"].all()
    b = Driver(cfg, flags_for_arm(arm), seed=123, period=300, params=a.params, min_age=150)
    b.set_state(copy.deepcopy(s))
    assert b.store._ak_n == 0                                 # derived cache rebuilt on demand
    assert np.array_equal(b.store.arrival_keys_upto(350), a.store.arrival_keys_upto(350))
    assert copy.deepcopy(b.get_state()).keys() == s.keys()
    for _ in range(150):
        ra, ia = a.step()
        rb, ib = b.step()
        assert np.array_equal(ra, rb) and ia == ib
    sa, sb = a.get_state(), b.get_state()
    assert np.array_equal(sa["mem"]["S_key"], sb["mem"]["S_key"]) and np.array_equal(sa["mem"]["L_key"], sb["mem"]["L_key"])
    if arm == "A14":
        assert all("S_key" in v and "S_keyed" in v for v in a.mem.snapshots.values())
        assert all(np.array_equal(sa["mem"]["snapshots"][t]["S_key"], sb["mem"]["snapshots"][t]["S_key"])
                   for t in sa["mem"]["snapshots"])


# ---------------------------------------------------------------- reconstruction / gradients
@pytest.mark.parametrize("arm", ["A0", "A1", "A2", "A3", "A4", "A14"])
def test_recon_equals_rollout_arrivals_short(arm):
    ov = dict(memory__key_source="arrivals", memory__candidate_repr="layernorm")
    T, per = 300, 40
    if arm in ("A2", "A4", "A14"):
        ov.update(memory__k_long=8, memory__rho_merge=0.999)
        T, per = 400, 300
    if arm == "A14":
        ov.update(workload__period_slots=400)
        T = 700
    if arm == "A3":
        ov.update(memory__k_short=20, memory__k_long=30)
    cfg = make_cfg(**ov)
    d = Driver(cfg, flags_for_arm(arm), seed=3, period=per).run(T)
    errs = []
    for lo in range(0, T, 64):
        chunk = list(range(lo, min(T, lo + 64)))
        r = reconstruct(d.params, cfg, d.flags, [(d.store, t) for t in chunk], grad=False).double().numpy()
        errs += [np.abs(r[i] - d.r[t]).max() for i, t in enumerate(chunk)]
    assert max(errs) < 1e-5, max(errs)
    w = cfg.memory.window_w
    assert all(np.all(d.r[t] == 0) and d.info[t]["empty"] for t in range(w - 1))
    if arm != "A1":
        assert d.store.n_ev > 0 and d.store.events["key"].shape[1] == cfg.scenario.n_zones
        assert d.store._ev_t.a[0] >= w - 1


def test_gradient_under_arrivals_skips_query_and_key_maps():
    cfg = make_cfg(memory__key_source="arrivals", memory__k_long=8, memory__rho_merge=0.999,
                   memory__candidate_repr="layernorm")
    d = Driver(cfg, flags_for_arm("A0"), seed=5, period=300).run(300)
    refs = [(d.store, t) for t in list(range(0, 12)) + list(range(150, 300, 3))]
    d.params.zero_grad()
    r = reconstruct(d.params, cfg, d.flags, refs, grad=True)
    assert torch.isfinite(r).all()
    (r.pow(2).sum() + r.sum()).backward()
    none = {"W_Q.weight", "W_Q.bias", "W_K.weight", "W_z.weight", "W_z.bias"}
    for name, p in d.params.named_parameters():
        g = p.grad
        if name in none:
            assert g is None or float(g.abs().sum()) == 0.0, name
        else:
            assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0.0, name
