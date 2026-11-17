"""Unit tests: Eq. 42 projection, Eq. 44 compute_P (hand values, detached c), Eq. 47
spectral sigma_p + clip, warm-up / P_ref rule, long-term addressing and distinct
LT slots, Eqs. 25/26/46 retrieval, A1/A2/A5 flags, A14 displaced content, state
round-trips."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from test_memory_driver import Driver, make_cfg
from tpa.memory import (MEPR, MemoryFlags, MemoryParams, SegmentStore, TwoLevelMemory, attend,
                        compute_P, flags_for_arm, project_norm, q_dim, x_dim)


# ---------------------------------------------------------------- dims / flags
def test_dims_from_config():
    cfg = make_cfg()
    assert x_dim(cfg) == 147 and q_dim(cfg) == 83
    assert flags_for_arm("A9") is None and flags_for_arm("B_ZHOU") is None
    assert flags_for_arm("A3").flat and not flags_for_arm("A2").use_ms


# ---------------------------------------------------------------- Eq. 42
def test_project_norm_eq42():
    M = torch.ones(10, 4)                          # ||M||_F = sqrt(40)
    P = project_norm(M, 3.0)
    assert abs(float(torch.linalg.vector_norm(P)) - 3.0) < 1e-6
    assert torch.allclose(project_norm(M, 100.0), M)   # inactive when inside the ball


def test_eq42_holds_in_memory_after_writes():
    cfg = make_cfg(memory__key_source="candidate")      # stand-alone memory (no store attached)
    mc = cfg.memory
    params = MemoryParams(cfg)
    mem = TwoLevelMemory(cfg, MemoryFlags(), params)
    for t in range(mc.k_short + 5):
        mem.write_short(torch.ones(mc.d), 0, t)    # raw ||M_S||_F = 80 > 72
    s, _ = mem.fro_norms()
    assert s <= mc.m_max_s + 1e-4 and abs(s - mc.m_max_s) < 1e-3
    mem.L_raw[:] = 1.0                              # raw ||M_L||_F = sqrt(32000) > 161
    mem.L_filled[:] = True
    _, l_ = mem.fro_norms()
    assert l_ <= mc.m_max_l + 1e-3 and abs(l_ - mc.m_max_l) < 1e-2
    # A3 flat bound memory.m_max_flat (= sqrt(Ms^2 + Ml^2) in default.toml)
    memf = TwoLevelMemory(cfg, MemoryFlags(flat=True), params)
    for t in range(mc.k_short + mc.k_long):
        memf.write_short(torch.ones(mc.d), 0, t)
    f, _ = memf.fro_norms()
    assert abs(f - mc.m_max_flat) < 1e-2
    assert abs(mc.m_max_flat - math.sqrt(mc.m_max_s ** 2 + mc.m_max_l ** 2)) < 0.1


# ---------------------------------------------------------------- Eq. 44
def test_compute_P_hand_values_and_detached_c():
    F = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    A = np.array([[1.0], [2.0], [3.0]])
    zm = np.array([2.0])
    P = compute_P(F, A, zm, lag=8)
    # c = (1, 1, 0); a~ = (0.5, 1, 1.5); P = (0.5*[1,0] + 1*[1,0] + 0) / 3
    assert torch.allclose(P, torch.tensor([[0.5, 0.0]]), atol=1e-6)
    P.sum().backward()
    # c detached: dP/dF_tau = c_tau * a~_tau / n (no gradient through the cosine)
    want = torch.tensor([[0.5 / 3] * 2, [1.0 / 3] * 2, [0.0, 0.0]])
    assert torch.allclose(F.grad, want, atol=1e-6)


def test_compute_P_ewm_lag_weights():
    # c of the last row uses only the previous `lag` rows with weights exp(-0.5 j)
    F = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    A = np.ones((3, 1))
    P = compute_P(F, A, np.ones(1), lag=1)          # lag 1: ref of row 2 = row 1
    c2 = 1.0 / math.sqrt(2.0)
    c1 = 0.0                                         # cos([1,0],[0,1]) = 0
    want = (np.array([0.0, 1.0]) * 1 + np.array([1.0, 0.0]) * c1 + np.array([1.0, 1.0]) * c2) / 3
    assert np.allclose(P.numpy()[0], want, atol=1e-6)
    P2 = compute_P(F, A, np.ones(1), lag=8)          # ref of row 2 = (1*[1,0] + e^-0.5*[0,1]) / (1+e^-0.5)
    ref = np.array([1.0, math.exp(-0.5)]) / (1 + math.exp(-0.5))
    c2b = (ref.sum()) / (math.sqrt(2) * np.linalg.norm(ref))
    want2 = (np.array([0.0, 1.0]) + np.array([1.0, 1.0]) * c2b) / 3
    assert np.allclose(P2.numpy()[0], want2, atol=1e-6)


# ---------------------------------------------------------------- Alg. 1 / Eq. 47 / warm-up and P_ref rule
def _const_window(val, d, Z, n=1):
    return np.full((n, d), val, np.float32), np.ones((n, Z))


def test_pref_warmup_rule_and_ewm():
    w = 5
    cfg = make_cfg(memory__window_w=w)
    mc = cfg.memory
    d, Z = mc.d, cfg.scenario.n_zones
    m = MEPR(cfg, MemoryFlags(gate_on=False))       # A5: theta = -inf, but not in warm-up
    Ps = []
    for i in range(w):
        F, A = _const_window(0.1 * (i + 1), d, Z)
        P, sig, cons = m.step(F, A, np.ones(Z))
        assert sig == 0.0 and cons is False
        Ps.append(P)
    assert np.allclose(m.P_ref, np.mean(Ps, axis=0))
    ref0 = m.P_ref.copy()
    F, A = _const_window(0.9, d, Z)
    P, sig, cons = m.step(F, A, np.ones(Z))
    want = np.linalg.norm(P - ref0, 2) / np.linalg.norm(ref0, 2)       # spectral norm
    assert abs(sig - min(want, mc.sigma_max)) < 1e-9
    assert cons is True                                                # A5 after warm-up
    assert np.allclose(m.P_ref, (1 - mc.alpha_ref) * ref0 + mc.alpha_ref * P)
    assert np.allclose(m.P_prev, Ps[-1])


def test_sigma_clip_and_spectral_not_frobenius():
    w = 3
    cfg = make_cfg(memory__window_w=w)
    mc = cfg.memory
    d, Z = mc.d, cfg.scenario.n_zones
    m = MEPR(cfg, MemoryFlags())
    for _ in range(w):
        m.step(*_const_window(0.01, d, Z), np.ones(Z))
    P, sig, cons = m.step(*_const_window(0.9, d, Z), np.ones(Z))
    assert sig == mc.sigma_max and cons is True                         # clipped at 3
    # spectral vs Frobenius differ for a rank-2 difference
    m2 = MEPR(cfg, MemoryFlags())
    rng = np.random.default_rng(0)
    for _ in range(w):
        m2.step(*_const_window(0.5, d, Z), np.ones(Z))
    F = (0.5 + 0.05 * rng.standard_normal((1, d))).astype(np.float32)
    A = np.ones((1, Z)) + 0.2 * np.arange(Z)
    ref = m2.P_ref.copy()
    P, sig, _ = m2.step(F, A, np.ones(Z))
    spec = np.linalg.norm(P - ref, 2) / np.linalg.norm(ref, 2)
    fro = np.linalg.norm(P - ref) / np.linalg.norm(ref)
    assert abs(sig - spec) < 1e-9 and abs(spec - fro) > 1e-6


def test_mepr_flags_A1_A2():
    cfg = make_cfg(memory__window_w=4)
    d, Z = cfg.memory.d, cfg.scenario.n_zones
    m1 = MEPR(cfg, flags_for_arm("A1"))
    m2 = MEPR(cfg, flags_for_arm("A2"))
    m5 = MEPR(cfg, flags_for_arm("A5"))
    for t in range(12):
        e = np.full(d, 0.1 * (t % 3 + 1), np.float32)
        a = np.ones(Z) * (t % 4 + 1)
        _, _, c1 = m1.step_buffered(e, a, np.ones(Z))
        m2.step_buffered(e, a, np.ones(Z))
        _, _, c5 = m5.step_buffered(e, a, np.ones(Z))
        assert c1 is False                              # A1: no consolidation path
        assert c5 is (t >= 4)                           # A5: every slot after warm-up
    assert len(m2.F_buf) == 1                           # A2: window = current slot
    assert len(m1.F_buf) == 5                           # w + 1


# ---------------------------------------------------------------- long-term addressing
def _mem_small(K_L=3, rho=0.8, mode="content"):
    """Stand-alone memory on candidate keys (key_source = "candidate", rho_merge 0.8: the
    addressing hand values; the defaults are "arrivals" / 0.95)."""
    cfg = make_cfg(memory__k_long=K_L, memory__rho_merge=rho, memory__lt_addressing=mode,
                   memory__key_source="candidate")
    params = MemoryParams(cfg)
    with torch.no_grad():                               # g = sigmoid(0) = 0.5 everywhere
        params.W_g.weight.zero_()
        params.W_g.bias.zero_()
    return cfg, TwoLevelMemory(cfg, MemoryFlags(), params)


def test_addressing_new_merge_replace():
    cfg, mem = _mem_small()
    d = cfg.memory.d
    basis = torch.eye(d)
    ev = mem.consolidate(basis[0], 1, 0)
    assert ev["mode"] == "new" and ev["k"] == 0
    assert np.allclose(ev["row_after"], 0.5 * basis[0].numpy())      # z * (1 - g), g = 0.5
    ev = mem.consolidate(basis[1], 2, 1)
    assert ev["mode"] == "new" and ev["k"] == 1
    ev = mem.consolidate(basis[0], 3, 2)                               # cos = 1 >= 0.8
    assert ev["mode"] == "merge" and ev["k"] == 0
    assert np.allclose(ev["row_after"], 0.75 * basis[0].numpy())     # 0.5*0.5 + 1*0.5
    assert mem.L_label[0] == 3 and mem.L_time[0] == 2
    ev = mem.consolidate(basis[2], 4, 3)
    assert ev["mode"] == "new" and ev["k"] == 2
    # full: slot 1 is least recently used (written at t=1)
    ev = mem.consolidate(basis[3], 5, 4)
    assert ev["mode"] == "replace" and ev["k"] == 1
    assert np.allclose(ev["row_before"], 0.5 * basis[1].numpy())
    assert np.allclose(ev["row_after"], 0.25 * basis[1].numpy() + 0.5 * basis[3].numpy())


def test_fixed_key_addressing():
    """Compare with first-write keys; merge updates content only; a replaced slot
    restarts from zero content with key = z."""
    cfg, mem = _mem_small(K_L=2, mode="fixed_key")
    d = cfg.memory.d
    e = torch.eye(d)
    z0 = e[0]
    ev = mem.consolidate(z0, 0, 0)
    assert ev["mode"] == "new" and np.allclose(ev["key"], z0.numpy())
    # cos(z1, key) = 0.9 >= 0.8 -> merge, key unchanged, content gated
    z1 = 0.9 * e[0] + math.sqrt(1 - 0.81) * e[1]
    ev = mem.consolidate(z1, 1, 1)
    assert ev["mode"] == "merge" and ev["k"] == 0
    assert np.allclose(mem.L_key[0].numpy(), z0.numpy())
    assert np.allclose(ev["row_after"], (0.25 * z0 + 0.5 * z1).numpy())    # g = 0.5
    # close to the drifted CONTENT (cos 0.88) but not to the key (cos 0.70)
    z2 = 0.7 * e[0] + 0.714 * e[1]
    ev = mem.consolidate(z2, 2, 2)
    assert ev["mode"] == "new" and ev["k"] == 1
    z3 = e[2]
    ev = mem.consolidate(z3, 3, 3)                       # full -> LRU slot 0, fresh content
    assert ev["mode"] == "replace" and ev["k"] == 0
    assert np.allclose(ev["row_before"], 0.0) and np.allclose(ev["row_after"], 0.5 * z3.numpy())
    assert np.allclose(mem.L_key[0].numpy(), z3.numpy())
    # content mode on the same sequence: z2 merges into the drifted content of slot 0
    cfg_c, mem_c = _mem_small(K_L=2, mode="content")
    for t, z in enumerate((z0, z1)):
        mem_c.consolidate(z, t, t)
    assert mem_c.consolidate(z2, 2, 2)["mode"] == "merge"


def test_occupancy_fixed_key_vs_content():
    """One full period of a slowly drifting periodic input, rho_merge = 0.8:
    content addressing collapses to one LT slot, fixed-key fills distinct slots."""
    occ = {}
    for mode in ("content", "fixed_key"):
        cfg = make_cfg(memory__lt_addressing=mode, memory__rho_merge=0.8)
        per = 600
        d = Driver(cfg, flags_for_arm("A0"), seed=0, period=per).run(cfg.memory.window_w + per)
        rows = d.mem.L_raw[torch.as_tensor(d.mem.L_filled)].numpy()
        occ[mode] = int(d.mem.L_filled.sum())
        assert np.unique(rows.round(6), axis=0).shape[0] == occ[mode]
        assert d.store.n_ev > 0.9 * per
    print("LT occupancy over one period:", occ)
    assert occ["content"] == 1
    assert occ["fixed_key"] >= 5


def test_lru_updated_by_top1_retrieval():
    cfg, mem = _mem_small(K_L=2)
    d = cfg.memory.d
    basis = torch.eye(d)
    mem.consolidate(basis[0], 0, 0)
    mem.consolidate(basis[1], 0, 1)
    flags_noS = MemoryFlags(use_ms=False)
    mem.flags = flags_noS                                   # retrieval over M_L only
    _, info = mem.retrieve(np.zeros(q_dim(cfg)), t=5)
    k = info["used_slot"]
    assert info["top_level"] == "L" and k in (0, 1) and mem.L_use[k] == 5
    ev = mem.consolidate(basis[2], 0, 6)
    assert ev["mode"] == "replace" and ev["k"] == 1 - k      # the other slot is now LRU


def test_distinct_lt_slots_after_distinct_writes():
    cfg = make_cfg(memory__key_source="candidate", memory__rho_merge=0.8)   # candidate keys; cos < 0.8 -> new
    params = MemoryParams(cfg, seed=1)
    mem = TwoLevelMemory(cfg, MemoryFlags(), params)
    rng = np.random.default_rng(0)
    n = 50
    for t in range(n):
        z = torch.tanh(torch.as_tensor(rng.standard_normal(cfg.memory.d) * 2, dtype=torch.float32))
        mem.consolidate(z, t, t)
    rows = mem.L_raw[torch.as_tensor(mem.L_filled)].numpy()
    assert rows.shape[0] == n                                 # random z: cos < 0.8 -> new slots
    assert np.unique(rows.round(6), axis=0).shape[0] == n
    assert not np.allclose(rows[0], rows[1])


# ---------------------------------------------------------------- Eqs. 25/26/46
def test_attend_matches_explicit_multihead():
    cfg = make_cfg()
    p = MemoryParams(cfg, seed=2)
    H, dk, d = p.n_heads, p.d_k, p.d
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, q_dim(cfg), generator=g)
    MS = torch.randn(1, 5, d, generator=g)
    ML = torch.randn(1, 7, d, generator=g)
    mS = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    mL = torch.ones(1, 7, dtype=torch.bool)
    with torch.no_grad():
        r, overall, _ = attend(p, q, [(MS, mS), (ML, mL)])
        Q = p.W_Q(q)[0].view(H, dk)
        om = torch.softmax(p.W_omega(q)[0], -1)
        out = torch.zeros(d)
        for li, (M, m) in enumerate([(MS[0, :3], None), (ML[0], None)]):
            K = p.W_K(M).view(-1, H, dk)
            V = p.W_V(M).view(-1, H, dk)
            heads = []
            for h in range(H):
                a = torch.softmax(K[:, h] @ Q[h] / math.sqrt(dk), 0)
                heads.append(a @ V[:, h])
            out += om[li] * torch.cat(heads)
    assert torch.allclose(r[0], out, atol=1e-5)
    assert abs(float(overall[0].sum() + overall[1].sum()) - 1.0) < 1e-5


def test_attend_single_level_and_empty_and_mean_pool():
    cfg = make_cfg()
    p = MemoryParams(cfg, seed=2)
    d = p.d
    g = torch.Generator().manual_seed(1)
    q = torch.randn(2, q_dim(cfg), generator=g)
    ML = torch.randn(2, 4, d, generator=g)
    empty = torch.zeros(2, 3, dtype=torch.bool)
    with torch.no_grad():
        r_two, _, _ = attend(p, q, [(torch.zeros(2, 3, d), empty), (ML, torch.ones(2, 4, dtype=torch.bool))])
        r_one, _, _ = attend(p, q, [None, (ML, torch.ones(2, 4, dtype=torch.bool))])
        assert torch.allclose(r_two, r_one)                 # empty level dropped, other used alone
        r0, _, pres = attend(p, q, [(torch.zeros(2, 3, d), empty), None])
        assert torch.all(r0 == 0) and not pres.any()
        rm, _, _ = attend(p, q, [None, (ML, torch.ones(2, 4, dtype=torch.bool))], mean_pool=True)
        assert torch.allclose(rm, p.W_V(ML).mean(1), atol=1e-6)    # A4 uniform weights


def test_retrieve_info_fields():
    cfg = make_cfg(memory__key_source="candidate")      # S-level hit at t = 4 (arrivals: key undefined before w)
    d = Driver(cfg, flags_for_arm("A0")).run(5)
    info = d.info[4]
    for k in ("top_label", "top_age", "top_level", "used_slot", "l1", "empty"):
        assert k in info
    assert info["top_level"] == "S" and 0 <= info["top_age"] <= 4 and not info["empty"]
    assert abs(info["l1"] - np.abs(d.r[4]).sum()) < 1e-9
    d4 = Driver(cfg, flags_for_arm("A4")).run(5)
    assert d4.info[4]["top_level"] is None and d4.info[4]["top_label"] == -1


# ---------------------------------------------------------------- A14
def test_a14_content_from_displaced_time():
    cfg = make_cfg(workload__period_slots=400, memory__k_long=8, memory__rho_merge=0.999,
                   memory__key_source="candidate")      # the probe memory has no store attached
    mc = cfg.memory
    P = cfg.workload.period_slots
    lo, hi = int(mc.a14_lo_frac * P), int(mc.a14_hi_frac * P)
    d14 = Driver(cfg, flags_for_arm("A14"), seed=4, period=300)
    # reference A0 run with the same inputs and params; LRU-by-retrieval off so that its
    # writes are identical to A14's (A14 does not update LRU on displaced reads)
    d0 = Driver(cfg, flags_for_arm("A0"), seed=4, period=300, params=d14.params)
    d0.update_lru = False
    states = {}
    T = 700
    for t in range(T):
        d14.step()
        d0.step()
        states[t] = copy.deepcopy(d0.mem.get_state())
    deltas = d14.store.delta
    for b in range(0, T, mc.a14_redraw_slots):                  # one draw per 60-slot block
        blk = deltas[b:b + mc.a14_redraw_slots]
        assert (blk == blk[0]).all() and lo <= blk[0] <= hi
    assert len(set(deltas.tolist())) > 1
    n_checked = 0
    probe = TwoLevelMemory(cfg, MemoryFlags(), d14.params)
    for t in range(T):
        s = int(d14.store.mem_time[t])
        if t < deltas[t]:
            assert s == -1 and np.all(d14.r[t] == 0) and d14.info[t]["empty"]
            continue
        assert s % mc.a14_snapshot_slots == 0 and t - deltas[t] - mc.a14_snapshot_slots < s <= t - deltas[t]
        # identical writes in A0 (same inputs and params): A14's read == retrieval on A0's memory at s
        probe.set_state(states[s])
        qin = d14.store.qin[t]
        r_disp, _ = probe.retrieve(qin, t=t, update_lru=False)
        assert np.abs(r_disp - d14.r[t]).max() < 1e-6
        probe.set_state(states[t])
        r_now, _ = probe.retrieve(qin, t=t, update_lru=False)
        assert np.abs(r_now - d14.r[t]).max() > 1e-4          # not the current memory
        n_checked += 1
    assert n_checked > 100


# ---------------------------------------------------------------- state round-trip
def _eq(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    return a == b


@pytest.mark.parametrize("rep", ["tanh", "layernorm", "novelty"])
@pytest.mark.parametrize("mode", ["fixed_key", "content", "soft"])
@pytest.mark.parametrize("arm", ["A0", "A3", "A14"])
def test_state_round_trip(arm, mode, rep):
    if rep != "tanh" and (arm != "A0" or mode != "fixed_key"):
        pytest.skip("representation x (arm, mode) cross product covered by the A0/fixed_key cell")
    ov = dict(memory__k_long=8, memory__rho_merge=0.999, workload__period_slots=400,
              memory__lt_addressing=mode, memory__candidate_repr=rep)
    cfg = make_cfg(**ov)
    a = Driver(cfg, flags_for_arm(arm), seed=9, period=300).run(350)
    s = copy.deepcopy(a.get_state())
    b = Driver(cfg, flags_for_arm(arm), seed=123, period=300, params=a.params)
    b.set_state(copy.deepcopy(s))
    assert _eq(b.get_state(), s)                                  # exact round-trip
    for _ in range(150):
        ra, _ = a.step()
        rb, _ = b.step()
        assert np.array_equal(ra, rb)
    assert _eq(a.get_state(), b.get_state())


def test_segment_store_capacity_growth():
    cfg = make_cfg()
    st = SegmentStore(cfg, MemoryFlags(), capacity=4)
    for t in range(10, 30):
        st.record(t, np.full(x_dim(cfg), t), np.zeros(q_dim(cfg)), np.zeros(6), np.ones(6))
    assert st.n == 20 and st.t0 == 10 and st.x[-1, 0] == 29


# ---------------------------------------------------------------- soft mode
def test_soft_addressing_hand_values():
    """w = softmax(cos/tau) over keys; g_k = 1 - w_k (1 - s_k); every filled slot touched."""
    from tpa.memory.stores import soft_update
    cfg, mem = _mem_small(K_L=3, mode="soft")             # W_g = 0 -> s = 0.5
    tau = mem.soft_tau
    d = cfg.memory.d
    e = torch.eye(d)
    ev = mem.consolidate(e[0], 0, 0)                      # empty -> new slot, key e0
    assert ev["mode"] == "new" and np.allclose(mem.L_key[0].numpy(), e[0].numpy())
    ev = mem.consolidate(e[1], 1, 1)                      # cos(e1, e0) = 0 < 0.8 -> new
    assert ev["mode"] == "new" and ev["k"] == 1
    c0, c1 = 0.5 * e[0], 0.5 * e[1]                       # contents after fresh gated writes
    z = 0.9 * e[0] + math.sqrt(0.19) * e[1]               # cos with keys: 0.9, 0.436
    ev = mem.consolidate(z, 2, 2)
    assert ev["mode"] == "soft" and ev["k"] == 0 and list(ev["ks"]) == [0, 1]
    w = np.exp(np.array([0.9, math.sqrt(0.19)]) / tau)
    w = w / w.sum()
    assert np.allclose(ev["w"], w, atol=1e-6)
    for j, c in enumerate((c0, c1)):
        g = 1 - w[j] * 0.5
        want = (c * g + z * (1 - g)).numpy()
        assert np.allclose(ev["rows_after"][j], want, atol=1e-6)
        assert np.allclose(mem.L_raw[j].numpy(), want, atol=1e-6)
    assert np.allclose(mem.L_key[0].numpy(), e[0].numpy())           # keys unchanged
    assert mem.L_label[0] == 2 and mem.L_label[1] == 1               # dominant slot only
    # soft_update with one valid slot == the hard gated write
    before = torch.randn(1, d, generator=torch.Generator().manual_seed(0))
    after, w1 = soft_update(before, e[3][None], z, torch.ones(1, dtype=torch.bool), tau, mem.params)
    g = mem.params.gate(before[0], z)
    assert float(w1) == 1.0 and torch.allclose(after[0], before[0] * g + z * (1 - g))


@pytest.mark.parametrize("source", ["pattern", "zone"])
def test_soft_gradient_reaches_W_z_through_address(source):
    """Soft mode: the gradient reaches the candidate map (W_z under "pattern", W_v under
    "zone"; the other map is unused and gets no gradient). The soft address w is
    recomputed from z only without key_centering (with centring it uses the logged
    k_t and is detached), so this test runs with key_centering=false."""
    from tpa.memory import reconstruct
    cfg = make_cfg(memory__lt_addressing="soft", memory__k_long=8, memory__rho_merge=0.8,
                   memory__candidate_source=source, memory__key_centering=False,
                   memory__key_source="candidate")      # candidate keys: W_Q / W_K in the graph
    d = Driver(cfg, flags_for_arm("A0"), seed=5, period=300).run(300)
    assert "soft" in set(d.store.events["mode"])
    refs = [(d.store, t) for t in list(range(0, 12)) + list(range(150, 300, 3))]
    d.params.zero_grad()
    reconstruct(d.params, cfg, d.flags, refs, grad=True).pow(2).sum().backward()
    unused = {"W_v.weight", "W_v.bias"} if source == "pattern" else {"W_z.weight", "W_z.bias"}
    for name, p in d.params.named_parameters():
        if name in unused:
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, name
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0.0, name


def test_diag_agrees_with_driver():
    from tpa.memory import MEPR, SegmentStore, TwoLevelMemory, run_diagnostic
    for mode in ("soft", "fixed_key"):
        cfg = make_cfg(memory__lt_addressing=mode, memory__k_long=8, memory__rho_merge=0.8)
        T = 260
        gen = Driver(cfg, flags_for_arm("A0"), seed=11, period=120)
        raw = [gen.inputs(t) + (gen.occupancy(t),) for t in range(T)]   # same rng sequences
        ref = Driver(cfg, flags_for_arm("A0"), seed=11, period=120)
        zm, inputs = np.ones(cfg.scenario.n_zones), []
        for t, (s_sys, tb, za, uo) in enumerate(raw):
            zm = 0.99 * zm + 0.01 * za
            inputs.append((np.concatenate([s_sys, tb]), za, zm.copy(), (t % 120) * 24 // 120, uo))
        ref.run(T)
        mem = TwoLevelMemory(cfg, ref.flags, ref.params)
        mepr = MEPR(cfg, ref.flags)
        st = SegmentStore(cfg, ref.flags)
        mem.attach_store(st)
        out = run_diagnostic(mem, mepr, ref.params, inputs, store=st)
        assert np.allclose(out["sigma_p"], [ref.sigma[t] for t in range(T)])
        assert out["consolidated"].tolist() == [ref.cons[t] for t in range(T)]
        assert out["top1_age"].tolist() == [ref.info[t]["top_age"] for t in range(T)]
        assert out["top1_level"].tolist() == [ref.info[t]["top_level"] or "" for t in range(T)]
        kinds = [k for k in out["event_kind"] if k]
        assert kinds == ref.store.events["mode"] and out["n_events"] == ref.store.n_ev
        assert out["lt_occupancy"][-1] == ref.mem.L_filled.sum()
        assert np.array_equal(st.x, ref.store.x) and np.array_equal(st.flat_after, ref.store.flat_after)
        assert 1 <= out["n_distinct"][-1] <= out["lt_occupancy"][-1]
        assert out["mode"] == mode and out["content_cos"].shape[0] == out["lt_occupancy"][-1]


# ---------------------------------------------------------------- candidate representation
def test_candidate_representations_hand_values():
    cfg = make_cfg(memory__candidate_repr="tanh")       # hand values start from tanh (default: layernorm)
    p = MemoryParams(cfg, seed=2)
    Z, d = p.Z, p.d
    g = torch.Generator().manual_seed(0)
    P = torch.randn(Z, d, generator=g)
    P_ref = torch.randn(Z, d, generator=g)
    h = p.W_z(P.reshape(-1))
    eps = cfg.memory.eps
    # tanh
    assert torch.allclose(p.candidate(P), torch.tanh(h))
    # layernorm: no affine, eps from cfg; unit variance, zero mean
    p.repr = "layernorm"
    z = p.candidate(P)
    want = (h - h.mean()) / torch.sqrt(h.var(unbiased=False) + eps)
    assert torch.allclose(z, want, atol=1e-6)
    zd = z.detach()
    assert abs(float(zd.mean())) < 1e-6 and abs(float(zd.var(unbiased=False)) - 1.0) < 1e-4
    # novelty: layernorm of W_z vec(P - P_ref); P_ref detached; P_ref required
    p.repr = "novelty"
    P_req = P_ref.clone().requires_grad_(True)
    z = p.candidate(P, P_req)
    h2 = p.W_z((P - P_ref).reshape(-1))
    want = (h2 - h2.mean()) / torch.sqrt(h2.var(unbiased=False) + eps)
    assert torch.allclose(z, want, atol=1e-6)
    z.sum().backward()
    assert P_req.grad is None                                # detached reference
    with pytest.raises(ValueError):
        p.candidate(P)
    with pytest.raises(ValueError):
        MemoryParams(make_cfg(memory__candidate_repr="relu"))


@pytest.mark.parametrize("source", ["pattern", "zone"])
def test_novelty_p_ref_logged_and_used_in_recon(source):
    """The event log carries the P_ref (source "pattern") / v_ref ("zone") that
    produced z; recon uses it (corrupting it changes r)."""
    from tpa.memory import reconstruct
    cfg = make_cfg(memory__candidate_repr="novelty", memory__k_long=8,
                   memory__candidate_source=source)
    d = Driver(cfg, flags_for_arm("A0"), seed=5, period=300).run(300)
    ev = d.store.events
    Z, M = cfg.scenario.n_zones, cfg.scenario.n_uav
    assert ev["p_ref"].shape == (d.store.n_ev, Z, cfg.memory.d)
    assert ev["v_ref"].shape == (d.store.n_ev, Z + M)
    assert np.abs(ev["p_ref"]).sum() > 0 and np.abs(ev["v_ref"]).sum() > 0
    assert np.isfinite(ev["maxcos"]).all()
    refs = [(d.store, t) for t in range(250, 300)]
    r0 = reconstruct(d.params, cfg, d.flags, refs, grad=False).numpy()
    assert np.abs(r0 - np.stack([d.r[t] for _, t in refs])).max() < 1e-5
    log = d.store._ev_pref if source == "pattern" else d.store._ev_vref
    log.a[: d.store.n_ev] *= 0.0                              # corrupt the logged reference
    r1 = reconstruct(d.params, cfg, d.flags, refs, grad=False).numpy()
    assert np.abs(r1 - r0).max() > 1e-4


def test_candidate_representations_synthetic_driver():
    """Driver numbers per candidate representation (LT occupancy, distinct entries, max-cosine
    quantiles): synthetic input, rho_merge 0.8, fixed_key, w + 2 periods."""
    from tpa.memory import MEPR, TwoLevelMemory, run_diagnostic
    res = {}
    for rep in ("tanh", "layernorm", "novelty"):
        cfg = make_cfg(memory__candidate_repr=rep, memory__lt_addressing="fixed_key",
                       memory__rho_merge=0.8, memory__key_source="candidate")   # stand-alone memory
        per = 600
        gen = Driver(cfg, flags_for_arm("A0"), seed=0, period=per)
        T = cfg.memory.window_w + 2 * per
        raw = [gen.inputs(t) + (gen.occupancy(t),) for t in range(T)]
        zm, inputs = np.ones(cfg.scenario.n_zones), []
        for t, (s_sys, tb, za, uo) in enumerate(raw):
            zm = 0.99 * zm + 0.01 * za
            inputs.append((np.concatenate([s_sys, tb]), za, zm.copy(), (t % per) * 24 // per, uo))
        mem = TwoLevelMemory(cfg, gen.flags, gen.params)
        out = run_diagnostic(mem, MEPR(cfg, gen.flags), gen.params, inputs)
        mc = out["maxcos"][np.isfinite(out["maxcos"]) & (out["maxcos"] >= 0)]
        res[rep] = dict(occupancy=int(out["lt_occupancy"][-1]), n_distinct=int(out["n_distinct"][-1]),
                        maxcos_q10=float(np.percentile(mc, 10)), maxcos_q50=float(np.percentile(mc, 50)),
                        maxcos_q90=float(np.percentile(mc, 90)), events=int(out["n_events"]))
        assert out["candidate_repr"] == rep and res[rep]["events"] > 0.9 * 2 * per
    print("Candidate representations (synthetic, rho 0.8, fixed_key, w + 2 periods):")
    for rep, v in res.items():
        print(f"  {rep:9s} occupancy={v['occupancy']:3d} n_distinct={v['n_distinct']:3d} "
              f"maxcos q10/50/90 = {v['maxcos_q10']:.3f}/{v['maxcos_q50']:.3f}/{v['maxcos_q90']:.3f}")
    assert all(v["occupancy"] >= 1 for v in res.values())
