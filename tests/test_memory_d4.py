"""Tests of the offset-free (centred) candidate keys, the zone-resolved candidate and the
phase consistency of retrieval: hand values of the centred key and the merge decision on it;
the bias-corrected mean mu and the mu_{t-1} rule; key_centering=false + candidate_source=
"pattern" reproduces the original rollout (reference npz recorded from the implementation
without key centring and the zone candidate, tests/data/d4_reference_*.npz); MemoryParams
init order; compute_v hand values; A14 snapshots carry keys; state round-trips; the
phase-consistency statistic on a hand-built info dict."""
from __future__ import annotations

import copy
import math
import os

import numpy as np
import pytest
import torch

from test_memory_driver import Driver, make_cfg
from tpa.memory import (MEPR, MemoryFlags, MemoryParams, TwoLevelMemory, compute_v,
                        compute_v_batch, flags_for_arm, phase_consistency, phase_hit, q_dim)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _mem_small(K_L=3, rho=0.8, mode="fixed_key", **ov):
    """Memory on candidate keys (key_source = "candidate"; the default is "arrivals")."""
    cfg = make_cfg(memory__k_long=K_L, memory__rho_merge=rho, memory__lt_addressing=mode,
                   memory__key_source="candidate", **ov)
    params = MemoryParams(cfg)
    with torch.no_grad():                               # g = sigmoid(0) = 0.5 everywhere
        params.W_g.weight.zero_()
        params.W_g.bias.zero_()
    return cfg, TwoLevelMemory(cfg, MemoryFlags(), params)


# ---------------------------------------------------------------- hand values
def test_centred_key_merge_hand_values():
    """Keys e0, e1 in slots 0, 1. z = e0 + c with mu = e0 + c - e1 gives k = z - mu = e1:
    the merge goes to slot 1 (the centred key) under key_centering, to slot 0 without it.
    Contents hold the full z (gated with g = 0.5)."""
    d = make_cfg().memory.d
    e = torch.eye(d)
    c = 0.3 * e[2]
    z = e[0] + c                                         # cos(z, e0) = 0.958 >= 0.8
    out = {}
    for kc in (True, False):
        cfg, mem = _mem_small(K_L=3, memory__key_centering=kc)
        ev0 = mem.consolidate(e[0], 0, 0)
        ev1 = mem.consolidate(e[1], 1, 1)
        assert (ev0["mode"], ev0["k"], ev1["mode"], ev1["k"]) == ("new", 0, "new", 1)
        assert np.allclose(mem.L_key[0].numpy(), e[0].numpy())
        assert np.allclose(mem.L_key[1].numpy(), e[1].numpy())
        mu = (e[0] + c - e[1]).double()
        mem.mu_S, mem.mu_c, mem.mu_n = mu.clone(), 1.0, 1   # mu_{t-1} = e0 + c - e1
        k = mem.key_of(z)
        if kc:
            assert torch.allclose(k, e[1], atol=1e-6)
        else:
            assert torch.allclose(k, z)
        ev = mem.consolidate(z, 2, 2)
        assert ev["mode"] == "merge"
        assert np.allclose(ev["cand_key"], k.numpy(), atol=1e-6)
        out[kc] = ev["k"]
        # content = 0.5 * e_k * 0.5 + z * 0.5  (row * g + z * (1 - g), g = 0.5), the FULL z
        want = 0.25 * e[ev["k"]] + 0.5 * z
        assert np.allclose(ev["row_after"], want.numpy(), atol=1e-6)
        assert np.allclose(mem.L_raw[ev["k"]].numpy(), want.numpy(), atol=1e-6)
        assert np.allclose(mem.L_key[ev["k"]].numpy(), e[ev["k"]].numpy())    # key unchanged
        assert float(ev["maxcos"]) > 0.95
    assert out[True] == 1 and out[False] == 0


def test_centred_key_new_slot_stores_key_not_z():
    cfg, mem = _mem_small(K_L=2, memory__key_centering=True)
    d = cfg.memory.d
    e = torch.eye(d)
    mem.consolidate(e[0], 0, 0)                          # first candidate: key = z
    mem.observe_candidate(e[0])
    z = e[1] + e[0]                                      # k = z - mu = e1
    ev = mem.consolidate(z, 1, 1)
    assert ev["mode"] == "new" and ev["k"] == 1
    assert np.allclose(mem.L_key[1].numpy(), e[1].numpy(), atol=1e-6)
    assert np.allclose(ev["key"], e[1].numpy(), atol=1e-6)
    assert np.allclose(mem.L_raw[1].numpy(), 0.5 * z.numpy(), atol=1e-6)   # content: full z
    # flat store (A3): z rows get key k_t, e rows key e_t
    cfg3 = make_cfg(memory__k_short=4, memory__k_long=4, memory__key_centering=True,
                    memory__key_source="candidate")
    m3 = TwoLevelMemory(cfg3, flags_for_arm("A3"), MemoryParams(cfg3))
    m3.write_short(e[3], 0, 0)
    m3.consolidate(e[0], 0, 0)
    m3.observe_candidate(e[0])
    m3.consolidate(z, 1, 1)
    assert np.allclose(m3.F_key[0].numpy(), e[3].numpy())
    assert np.allclose(m3.F_key[1].numpy(), e[0].numpy())
    assert np.allclose(m3.F_key[2].numpy(), e[1].numpy(), atol=1e-6)
    assert np.allclose(m3.F_raw[2].numpy(), z.numpy())


# ---------------------------------------------------------------- mu
def test_mu_bias_corrected_ema_and_previous_slot_rule():
    H = 7200
    cfg, mem = _mem_small(K_L=2, memory__key_centering=True, memory__key_mean_horizon_slots=H)
    d = cfg.memory.d
    a = 1.0 / H
    rng = np.random.default_rng(0)
    zs = [torch.as_tensor(rng.standard_normal(d), dtype=torch.float32) for _ in range(25)]
    assert mem.mu is None and mem.mu_n == 0
    assert torch.equal(mem.key_of(zs[0]), zs[0])                 # no previous candidate
    for n, z in enumerate(zs, start=1):
        # key at slot n uses mu_{n-1} (the mean of z_1..z_{n-1}); then mu is updated with z_n
        k = mem.key_of(z)
        if n == 1:
            assert torch.equal(k, z)
        else:
            prev = torch.stack(zs[: n - 1]).double()
            wts = torch.tensor([(1 - a) ** (n - 1 - i) for i in range(1, n)], dtype=torch.float64)
            mu_prev = (wts[:, None] * prev).sum(0) / wts.sum()      # closed form of the EMA
            assert torch.allclose(k.double(), z.double() - mu_prev, atol=1e-6)
            # for n << H the bias-corrected EMA is the exact mean up to O(n * alpha)
            assert torch.allclose(mu_prev, prev.mean(0), atol=2 * (n - 1) * a * float(prev.abs().max()))
        mem.observe_candidate(z)
        assert mem.mu_n == n
        wts = torch.tensor([(1 - a) ** (n - i) for i in range(1, n + 1)], dtype=torch.float64)
        want = (wts[:, None] * torch.stack(zs[:n]).double()).sum(0) / wts.sum()
        assert torch.allclose(mem.mu, want, atol=1e-6)
        assert abs(mem.mu_c - (1 - (1 - a) ** n)) < 1e-12
    assert torch.allclose(mem.mu, torch.stack(zs).double().mean(0), atol=25 * a * 4)
    # key_centering off: observe_candidate is a no-op and key_of is the identity
    cfg0, mem0 = _mem_small(K_L=2, memory__key_centering=False)
    mem0.observe_candidate(zs[0])
    assert mem0.mu_n == 0 and torch.equal(mem0.key_of(zs[1]), zs[1])


def test_consolidate_uses_mu_before_the_slot_update():
    """Within a slot: key = z - mu_{t-1} (consolidate first), mu_t includes z after."""
    cfg, mem = _mem_small(K_L=4, memory__key_centering=True, memory__key_mean_horizon_slots=10)
    d = cfg.memory.d
    e = torch.eye(d)
    for t, z in enumerate((e[0], e[1] + e[0], 2 * e[2] + e[0])):
        mu_before = None if mem.mu is None else mem.mu.clone()
        ev = mem.consolidate(z, t, t)
        want = z if mu_before is None else z - mu_before.float()
        assert np.allclose(ev["cand_key"], want.numpy(), atol=1e-6)
        mem.observe_candidate(z)
        assert mem.mu_n == t + 1


# ---------------------------------------------------------------- reference rollouts before key centring
@pytest.mark.parametrize("name,arm,ov,T,per", [
    ("A0", "A0", {}, 300, 40),
    ("A3", "A3", {}, 300, 40),
    ("A14", "A14", dict(workload__period_slots=400, memory__k_long=8, memory__rho_merge=0.999), 700, 300),
])
def test_off_path_reproduces_reference_rollouts(name, arm, ov, T, per):
    """key_centering=false + candidate_source="pattern" == the code before key centring: reference
    recorded from the modules without key centring and the zone candidate, with
    Driver(make_cfg(**ov), flags_for_arm(arm), seed=3, period=per).run(T). The original
    settings the arrays depend on (key_source "candidate", tanh, rho_merge 0.8, theta_p 0.15)
    are pinned explicitly (the defaults are arrivals / layernorm / 0.95 / 0.35)."""
    ref = np.load(os.path.join(DATA, f"d4_reference_{name}.npz"))
    pre_d4 = dict(memory__key_source="candidate", memory__candidate_repr="tanh",
                  memory__rho_merge=0.8, memory__theta_p=0.15)
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
    assert d.mem.mu_n == 0                               # mu never touched on the off path


def test_memory_params_init_order_unchanged():
    """W_v is created last in the seeded block: every other layer keeps the init values
    recorded before key centring (tests/data/d4_reference_params_init.npz)."""
    ref = np.load(os.path.join(DATA, "d4_reference_params_init.npz"))
    for seed in (0, 3):
        p = MemoryParams(make_cfg(), seed=seed)
        sd = {k: v.numpy() for k, v in p.state_dict().items()}
        assert "W_v.weight" in sd and sd["W_v.weight"].shape == (p.d, p.Z + p.M)
        for k in ("W_S", "W_z", "W_g", "W_Q", "W_K", "W_V", "W_omega"):
            for leaf in ("weight", "bias"):
                key = f"s{seed}__{k}.{leaf}"
                if key in ref:
                    # The reference arrays were recorded on one machine. torch's vectorised
                    # uniform_ kernel can round the last float32 bit differently on another CPU
                    # (measured on a second CPU: max |diff| 7.5e-9, i.e. 1 ulp). An init-ORDER
                    # change would shift the RNG stream and change every value by O(1e-2), so a
                    # 1e-7 tolerance still detects what this test exists to detect.
                    assert np.allclose(sd[f"{k}.{leaf}"], ref[key], rtol=0, atol=1e-7), key
    with pytest.raises(ValueError):
        MemoryParams(make_cfg(memory__candidate_source="grid"))


def test_default_config_centred_keys_zone_candidate():
    cfg = make_cfg()
    assert cfg.memory.key_centering is True and cfg.memory.candidate_source == "zone"
    assert cfg.memory.key_mean_horizon_slots == 7200 == 2 * cfg.workload.period_slots
    mem = TwoLevelMemory(cfg, MemoryFlags(), MemoryParams(cfg))
    assert mem.key_centering and abs(mem.mu_alpha - 1 / 7200) < 1e-15


# ---------------------------------------------------------------- zone-resolved candidate input v_t
def test_compute_v_hand_values_and_mepr_window():
    A = np.array([[1.0, 0.0], [2.0, 4.0], [3.0, 2.0]])
    zm = np.array([2.0, 4.0])
    uo = np.array([0.5, 0.25, 1.0])
    v = compute_v(A, zm, uo).numpy()
    assert np.allclose(v, [1.0, 0.5, 0.5, 0.25, 1.0])        # mean of A / zm, then o_t
    mask = torch.tensor([[False, True, True]])
    vb = compute_v_batch(torch.as_tensor(A, dtype=torch.float32)[None],
                         torch.as_tensor(zm, dtype=torch.float32)[None], mask,
                         torch.as_tensor(uo, dtype=torch.float32)[None])[0].numpy()
    assert np.allclose(vb, [1.25, 0.75, 0.5, 0.25, 1.0])     # padded row excluded
    # MEPR: v from the same window rows as P, zone_mean of the current slot; v_ref like P_ref
    cfg = make_cfg(memory__window_w=3)
    Z, M, d = cfg.scenario.n_zones, cfg.scenario.n_uav, cfg.memory.d
    m = MEPR(cfg, MemoryFlags())
    vs = []
    for t in range(6):
        e = np.full(d, 0.1 * (t + 1), np.float32)
        a = np.ones(Z) * (t + 1)
        zmean = np.ones(Z) * 2.0
        uo_t = np.full(M, 0.1 * t)
        m.step_buffered(e, a, zmean, uav_occ=uo_t)
        A_win = np.stack(m.A_buf)
        want = np.concatenate([(A_win / 2.0).mean(0), uo_t])
        assert np.allclose(m.v, want, atol=1e-6) and m.v_torch.dtype == torch.float32
        vs.append(m.v.copy())
        if t < 3:
            assert m.v_ref_used is None and m.P_ref_used is None
        elif t == 3:
            assert np.allclose(m.v_ref_used, np.mean(vs[:3], axis=0))
    assert np.allclose(m.v_ref, (1 - cfg.memory.alpha_ref) * m.v_ref_used + cfg.memory.alpha_ref * vs[-1])
    m2 = MEPR(cfg, MemoryFlags())
    m2.step_buffered(np.ones(d, np.float32), np.ones(Z), np.ones(Z))        # uav_occ None -> zeros
    assert np.allclose(m2.v[Z:], 0.0)


def test_candidate_zone_hand_values_and_dispatcher():
    cfg = make_cfg(memory__candidate_source="zone", memory__candidate_repr="tanh")   # hand values start from tanh
    p = MemoryParams(cfg, seed=2)
    Z, M, d = p.Z, p.M, p.d
    g = torch.Generator().manual_seed(0)
    v = torch.randn(Z + M, generator=g)
    v_ref = torch.randn(Z + M, generator=g)
    P = torch.randn(Z, d, generator=g)
    h = p.W_v(v)
    assert torch.allclose(p.candidate_zone(v), torch.tanh(h))
    assert torch.allclose(p.candidate_from(P, v), torch.tanh(h))
    p.repr = "layernorm"
    want = (h - h.mean()) / torch.sqrt(h.var(unbiased=False) + cfg.memory.eps)
    assert torch.allclose(p.candidate_zone(v), want, atol=1e-6)
    p.repr = "novelty"
    h2 = p.W_v(v - v_ref)
    want = (h2 - h2.mean()) / torch.sqrt(h2.var(unbiased=False) + cfg.memory.eps)
    assert torch.allclose(p.candidate_zone(v, v_ref), want, atol=1e-6)
    with pytest.raises(ValueError):
        p.candidate_zone(v)
    p.source = "pattern"
    p.repr = "tanh"
    assert torch.allclose(p.candidate_from(P, v), p.candidate(P))


# ---------------------------------------------------------------- A14 snapshot keys
def test_a14_displaced_read_uses_snapshot_keys():
    cfg = make_cfg(workload__period_slots=400, memory__k_long=8, memory__rho_merge=0.999,
                   memory__key_centering=True)
    d = Driver(cfg, flags_for_arm("A14"), seed=4, period=300).run(500)
    mt = d.store.mem_time
    ts = [t for t in range(300, 500) if mt[t] >= 0 and d.info[t]["top_level"] == "L"]
    assert len(ts) > 10
    t = ts[-1]
    qin = d.store.qin[t]
    r0, info0 = d.mem.retrieve(qin, t, update_lru=False)
    assert np.abs(r0 - d.r[t]).max() < 1e-6
    live = d.mem.L_key.clone()
    d.mem.L_key.zero_()                                  # live keys gone: displaced read unchanged
    r1, _ = d.mem.retrieve(qin, t, update_lru=False)
    assert np.array_equal(r0, r1)
    d.mem.L_key.copy_(live)
    s = int(mt[t])
    snap = d.mem.snapshots[s]
    assert "L_key" in snap and torch.isfinite(snap["L_key"]).all()
    keep = snap["L_key"].clone()
    snap["L_key"].zero_()                                # the snapshot's keys ARE used
    r2, _ = d.mem.retrieve(qin, t, update_lru=False)
    assert np.abs(r2 - r0).max() > 1e-6
    snap["L_key"].copy_(keep)
    r3, _ = d.mem.retrieve(qin, t, update_lru=False)
    assert np.array_equal(r3, r0)


def test_retrieval_keys_are_L_key_not_contents():
    """Attention logits use L_key (unprojected) and values the projected contents."""
    from tpa.memory import attend
    cfg, mem = _mem_small(K_L=3, memory__key_centering=True)
    mem.flags = MemoryFlags(use_ms=False)
    d = cfg.memory.d
    e = torch.eye(d)
    mem.consolidate(e[0], 0, 0)
    mem.observe_candidate(e[0])
    mem.consolidate(e[1] + e[0], 1, 1)                  # key e1, content 0.5 (e0 + e1)
    mem.observe_candidate(e[1] + e[0])
    q = np.ones(q_dim(cfg))
    r, info = mem.retrieve(q, 2, update_lru=False)
    with torch.no_grad():
        ML = mem.M_L()[None]
        mask = torch.as_tensor(mem.L_filled)[None]
        want, _, _ = attend(mem.params, torch.as_tensor(q, dtype=torch.float32)[None],
                            [None, (ML, mask, mem.L_key[None])])
        other, _, _ = attend(mem.params, torch.as_tensor(q, dtype=torch.float32)[None],
                             [None, (ML, mask)])
    assert np.allclose(r, want[0].double().numpy(), atol=1e-6)
    assert np.abs(r - other[0].double().numpy()).max() > 1e-6
    mem.key_centering = False
    r_off, _ = mem.retrieve(q, 2, update_lru=False)
    assert np.allclose(r_off, other[0].double().numpy(), atol=1e-6)


# ---------------------------------------------------------------- state round-trip
def _eq(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    return a == b


@pytest.mark.parametrize("arm", ["A0", "A3", "A14"])
def test_state_round_trip_includes_mu_keys_vref(arm):
    cfg = make_cfg(memory__k_long=8, memory__rho_merge=0.999, workload__period_slots=400,
                   memory__candidate_repr="novelty",
                   memory__key_source="candidate")      # zone + centring (defaults); candidate keys (mu advances)
    a = Driver(cfg, flags_for_arm(arm), seed=9, period=300).run(350)
    s = copy.deepcopy(a.get_state())
    for k in ("mu_S", "mu_c", "mu_n", "F_key"):
        assert k in s["mem"], k
    assert s["mem"]["mu_n"] > 0 and np.abs(s["mem"]["mu_S"]).sum() > 0
    for k in ("v_ref", "v_ref_used", "v", "v_sum", "v_torch"):
        assert k in s["mepr"], k
    assert s["mepr"]["v_ref"] is not None and s["mepr"]["v_ref_used"] is not None
    for k in ("uav_occ", "ev_vref", "ev_ckey"):
        assert k in s["store"], k
    assert np.abs(s["store"]["uav_occ"]).sum() > 0 and np.abs(s["store"]["ev_vref"]).sum() > 0
    ev = a.store.events
    assert ev["v_ref"].shape[1] == cfg.scenario.n_zones + cfg.scenario.n_uav
    assert ev["cand_key"].shape == (a.store.n_ev, cfg.memory.d)
    if arm != "A3":
        assert np.abs(ev["cand_key"] - ev["key"]).max() > 1e-6        # merge events keep slot key
    b = Driver(cfg, flags_for_arm(arm), seed=123, period=300, params=a.params)
    b.set_state(copy.deepcopy(s))
    assert _eq(b.get_state(), s)
    for _ in range(150):
        ra, _ = a.step()
        rb, _ = b.step()
        assert np.array_equal(ra, rb)
    assert _eq(a.get_state(), b.get_state())
    if arm == "A14":
        assert all("L_key" in v for v in a.mem.snapshots.values())


# ---------------------------------------------------------------- phase consistency of retrieval
def test_phase_consistency_hand_built_info():
    P, w = 600, 100
    lo, hi = w + P, w + 3 * P
    info = {}
    kinds = []
    for i, t in enumerate(range(lo, hi)):
        kind = ["hit_exact", "hit_edge", "miss_phase", "miss_recent", "miss_S", "miss_empty",
                "miss_none", "hit_two_cycles"][i % 8]
        kinds.append(kind)
        if kind == "hit_exact":
            info[t] = dict(top_level="L", top_age=P, empty=False)            # age mod P = 0
        elif kind == "hit_edge":
            info[t] = dict(top_level="L", top_age=P + P // 8, empty=False)   # exactly P/8 -> hit
        elif kind == "miss_phase":
            info[t] = dict(top_level="L", top_age=P + P // 8 + 1, empty=False)
        elif kind == "miss_recent":
            info[t] = dict(top_level="L", top_age=P // 2 - 1, empty=False)   # age < P/2 (same cycle)
        elif kind == "miss_S":
            info[t] = dict(top_level="S", top_age=P, empty=False)
        elif kind == "miss_empty":
            info[t] = dict(top_level=None, top_age=-1, empty=True)
        elif kind == "miss_none":
            info[t] = None                                                   # A14: no displaced memory
        else:
            info[t] = dict(top_level="L", top_age=2 * P - P // 16, empty=False)   # P - 37 mod -> hit
    want = np.mean([k.startswith("hit") for k in kinds])
    pc = phase_consistency(dict(info), P, w)
    assert abs(pc - want) < 1e-12 and abs(want - 3 / 8) < 1e-12
    # age = P/2 exactly passes the earlier-cycle rule but its phase distance is P/2 > P/8
    assert not phase_hit(dict(top_level="L", top_age=P // 2, empty=False), P)
    assert phase_hit(dict(top_level="L", top_age=P - P // 8, empty=False), P)   # distance P/8

    class D:                                                                   # driver-like object
        pass
    dr = D()
    dr.info = info
    assert phase_consistency(dr, P, w) == pc
    with pytest.raises(ValueError):
        phase_consistency({t: info[t] for t in range(lo, hi - 1)}, P, w)       # window not covered
    assert phase_consistency(dr, P, w, T=hi) == pc                             # run length argument
    with pytest.raises(ValueError):
        phase_consistency(dr, P, w, T=hi - 1)
