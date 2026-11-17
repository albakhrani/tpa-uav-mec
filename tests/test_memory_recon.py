"""Main replay test: replay reconstruction == rollout r(t) when the parameters
are unchanged (atol 1e-5), for A0, A1, A2, A3, A4, A14; gradients reach every
MemoryParams parameter used by the arm and the candidate source, through reconstruct().

The cases run over candidate_source x key_centering. The default combo (zone, True) runs
every case x mode x repr; the other three combos run A0, A0-smallKL, A3 and A14 under
fixed_key x every repr plus soft for A0 (runtime).

test_recon_equals_rollout_arrivals runs the same COMBO_CASES under memory.key_source =
"arrivals" (zone contents, every repr, fixed_key, plus soft on A0); the other arms are
covered by tests/test_memory_d5.py."""
from __future__ import annotations

import collections

import numpy as np
import pytest
import torch

from test_memory_driver import Driver, make_cfg
from tpa.memory import flags_for_arm, reconstruct

# (case id, arm, config overrides, slots, driver period)
CASES = [
    ("A0", "A0", {}, 450, 40),
    ("A0-smallKL", "A0", dict(memory__k_long=8, memory__rho_merge=0.999), 450, 300),
    ("A0-saturated", "A0", {}, 300, 40),
    ("A1", "A1", {}, 300, 40),
    ("A2", "A2", dict(memory__k_long=8, memory__rho_merge=0.999), 400, 300),
    ("A3", "A3", dict(memory__k_short=20, memory__k_long=30), 450, 40),
    ("A4", "A4", dict(memory__k_long=8, memory__rho_merge=0.999), 400, 300),
    ("A14", "A14", dict(workload__period_slots=400, memory__k_long=8, memory__rho_merge=0.999), 700, 300),
]


MODES = ["fixed_key", "content", "soft"]
REPRS = ["tanh", "layernorm", "novelty"]
SOURCES = ["pattern", "zone"]
CENTERING = [True, False]
DEFAULT_COMBO = ("zone", True)                       # defaults (config/default.toml)
COMBO_CASES = ("A0", "A0-smallKL", "A3", "A14")      # run under every combo


def _skip_combo(case, mode, rep, source, centering):
    if (source, centering) == DEFAULT_COMBO:
        return
    if case[0] not in COMBO_CASES:
        pytest.skip("non-default (source, key_centering) combos run A0/A0-smallKL/A3/A14 only")
    if mode == "content" or (mode == "soft" and case[0] != "A0"):
        pytest.skip("non-default combos: fixed_key (all reprs) and soft on A0 only")


def _run(case, mode="fixed_key", rep="tanh", source="zone", centering=True, key_source="candidate"):
    cid, arm, ov, T, per = case
    cfg = make_cfg(memory__lt_addressing=mode, memory__candidate_repr=rep,
                   memory__candidate_source=source, memory__key_centering=centering,
                   memory__key_source=key_source, **ov)
    d = Driver(cfg, flags_for_arm(arm), seed=3, period=per)
    if cid.endswith("saturated"):
        with torch.no_grad():
            d.params.W_S.weight.mul_(60.0)       # saturate tanh so Eq. 42 binds on M_S
    return cfg, d.run(T)


@pytest.mark.parametrize("centering", CENTERING, ids=["kc", "nokc"])
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("rep", REPRS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_recon_equals_rollout(case, mode, rep, source, centering):
    _skip_combo(case, mode, rep, source, centering)
    cfg, d = _run(case, mode, rep, source, centering)
    T = case[3]
    ts = list(range(T))
    errs = []
    for lo in range(0, T, 64):
        chunk = ts[lo:lo + 64]
        r = reconstruct(d.params, cfg, d.flags, [(d.store, t) for t in chunk], grad=False)
        r = r.double().cpu().numpy()
        errs += [np.abs(r[i] - d.r[t]).max() for i, t in enumerate(chunk)]
    assert max(errs) < 1e-5, max(errs)
    modes = collections.Counter(d.store.events["mode"])
    cid = case[0]
    if cid == "A1":
        assert d.store.n_ev == 0
    else:
        assert d.store.n_ev > 0
    if cid in ("A0-smallKL", "A2", "A4", "A14"):
        assert modes["new"] == cfg.memory.k_long and modes["replace"] > 0, modes
    if mode == "soft":
        assert "merge" not in modes, modes
        if cid == "A0":                                  # rho_merge = 0.95 (default): soft events occur
            assert modes["soft"] > 0, modes
    if cid == "A3":
        assert set(modes) == {"fifo"} and d.mem.F_fill == cfg.memory.k_short + cfg.memory.k_long
    if cid == "A0-saturated":
        assert abs(d.mem.fro_norms()[0] - cfg.memory.m_max_s) < 1e-3   # projection binds
    if cid == "A14":
        mt = d.store.mem_time
        assert (mt == -1).any() and (mt >= 0).any()
        assert any(np.abs(d.r[t]).sum() > 0 for t in range(T) if mt[t] >= 0)


@pytest.mark.parametrize("rep", REPRS)
@pytest.mark.parametrize("mode", ["fixed_key", "soft"])
@pytest.mark.parametrize("case", [c for c in CASES if c[0] in COMBO_CASES], ids=list(COMBO_CASES))
def test_recon_equals_rollout_arrivals(case, mode, rep):
    """Replay == rollout (atol 1e-5) under key_source = "arrivals" for A0, A0-smallKL,
    A3, A14 (fixed_key x every repr; soft on A0)."""
    if mode == "soft" and case[0] != "A0":
        pytest.skip("arrivals: soft on A0 only")
    cfg, d = _run(case, mode, rep, key_source="arrivals")
    T = case[3]
    errs = []
    for lo in range(0, T, 64):
        chunk = list(range(lo, min(T, lo + 64)))
        r = reconstruct(d.params, cfg, d.flags, [(d.store, t) for t in chunk], grad=False)
        r = r.double().cpu().numpy()
        errs += [np.abs(r[i] - d.r[t]).max() for i, t in enumerate(chunk)]
    assert max(errs) < 1e-5, max(errs)
    Z, w = cfg.scenario.n_zones, cfg.memory.window_w
    assert d.store.n_ev > 0 and d.store.events["key"].shape[1] == Z and d.mem.L_key.shape[1] == Z
    assert all(d.info[t]["empty"] and np.all(d.r[t] == 0) for t in range(w - 1))
    modes = collections.Counter(d.store.events["mode"])
    if case[0] in ("A0-smallKL", "A14"):
        assert modes["new"] == cfg.memory.k_long and modes["replace"] > 0, modes
    if mode == "soft":
        assert "merge" not in modes and modes["soft"] > 0, modes
    if case[0] == "A3":
        assert set(modes) == {"fifo"}
    if case[0] == "A14":
        mt = d.store.mem_time
        assert (mt == -1).any() and (mt >= 0).any()
        assert all("S_key" in s for s in d.mem.snapshots.values())


def test_recon_batch_mixes_segments_and_arbitrary_order():
    cfg, d1 = _run(CASES[1])
    d2b = Driver(cfg, d1.flags, seed=7, period=200, params=d1.params).run(300)
    refs = [(d1.store, 400), (d2b.store, 250), (d1.store, 120), (d2b.store, 0), (d1.store, 449)]
    r = reconstruct(d1.params, cfg, d1.flags, refs, grad=False).double().numpy()
    want = [d1.r[400], d2b.r[250], d1.r[120], d2b.r[0], d1.r[449]]
    assert max(np.abs(r[i] - w).max() for i, w in enumerate(want)) < 1e-5


@pytest.mark.parametrize("centering", CENTERING, ids=["kc", "nokc"])
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("rep", REPRS)
@pytest.mark.parametrize("arm,expect_none", [
    ("A0", set()),
    ("A3", {"W_g.weight", "W_g.bias", "W_omega.weight", "W_omega.bias"}),
])
def test_gradient_reaches_params(arm, expect_none, rep, source, centering):
    """The gradient reaches W_v (not W_z) under "zone" and W_z (not W_v) under "pattern";
    the keys carry no gradient, so centring changes nothing here. Pinned to
    key_source = "candidate" (the candidate-key path, where W_Q / W_K receive a
    gradient); the "arrivals" case (the default; W_Q / W_K unused) is
    tests/test_memory_d5.py::test_gradient_under_arrivals_skips_query_and_key_maps."""
    expect_none = set(expect_none) | ({"W_z.weight", "W_z.bias"} if source == "zone"
                                      else {"W_v.weight", "W_v.bias"})
    ov = dict(memory__k_long=8, memory__rho_merge=0.999, memory__candidate_repr=rep,
              memory__candidate_source=source, memory__key_centering=centering,
              memory__key_source="candidate")
    if arm == "A3":
        ov = dict(memory__k_short=20, memory__k_long=30, memory__candidate_repr=rep,
                  memory__candidate_source=source, memory__key_centering=centering,
                  memory__key_source="candidate")
    cfg = make_cfg(**ov)
    d = Driver(cfg, flags_for_arm(arm), seed=5, period=300).run(300)
    # include slots BEFORE the first long-term event (no event / empty M_L) and the first
    # slots of the segment: these produced NaN gradients once (all-masked softmax)
    refs = [(d.store, t) for t in list(range(0, 12)) + list(range(150, 300, 3))]
    d.params.zero_grad()
    r = reconstruct(d.params, cfg, d.flags, refs, grad=True)
    assert torch.isfinite(r).all()
    (r.pow(2).sum() + r.sum()).backward()
    for name, p in d.params.named_parameters():
        g = p.grad
        if name in expect_none:
            assert g is None or float(g.abs().sum()) == 0.0, name   # unused by this arm
        else:
            assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0.0, name


def test_no_grad_mode_has_no_graph():
    cfg, d = _run(CASES[1])
    r = reconstruct(d.params, cfg, d.flags, [(d.store, 300)], grad=False)
    assert not r.requires_grad


@pytest.mark.parametrize("mode", MODES)
def test_recon_cuda_matches_cpu(mode):
    """CUDA path: reconstruct() with params on CUDA gives the CPU r (atol 1e-5)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from tpa.memory import MemoryParams
    for case in (CASES[1], CASES[5], CASES[7]):           # A0-smallKL, A3, A14
        cfg, d = _run(case, mode)
        pc = MemoryParams(cfg, device="cuda")
        pc.load_state_dict(d.params.state_dict())
        refs = [(d.store, t) for t in range(0, case[3], max(1, case[3] // 64))][:64]
        r_cpu = reconstruct(d.params, cfg, d.flags, refs, grad=False).numpy()
        r_gpu = reconstruct(pc, cfg, d.flags, refs, grad=True)
        assert r_gpu.device.type == "cuda" and r_gpu.requires_grad
        assert np.abs(r_gpu.detach().cpu().numpy() - r_cpu).max() < 1e-5
        want = np.stack([d.r[t] for _, t in refs])
        assert np.abs(r_gpu.detach().cpu().double().numpy() - want).max() < 1e-5
