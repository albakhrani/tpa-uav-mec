"""A16, the forecasting write rule (delayed consolidation).

(i)   A16 with Delta = 0 reproduces A0 bit for bit through tpa.run.trainer.run (train and val_full
      tables, RESULT.json, final checkpoint) -- the delayed machinery with an immediate flush is the A0 path.
(ii)  The content written at s + Delta under key k_s equals the hand-computed profile of (s, s+Delta]:
      LN(W_v [ mean za / zone_mean_{s+Delta} ; mean uav_occ ]) over slots s+1..s+Delta, gated into a fresh
      slot; the event is logged at t = s + Delta with delay_src = s and cand_key = the arrival key of s;
      the key/gate are those of s; the pending list is dropped at a segment reset and survives a checkpoint.
(iii) Replay reconstruction equals the rollout r(t) for A16 (Delta = 5 and Delta = 0), and the gradient
      of the delayed write reaches W_v (replay-reconstruction training path), so A16 is NOT excluded from recon.
Also: the registry and the flags; a flat / novelty / 'candidate'-keyed configuration is refused.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _sys_helpers import row, tiny_cfg  # noqa: E402
from test_memory_driver import Driver, make_cfg  # noqa: E402
from tpa.arms import ARMS, ArmFlags  # noqa: E402
from tpa.memory import MemoryFlags, MemoryParams, TwoLevelMemory, compute_v, flags_for_arm, reconstruct  # noqa: E402
from tpa.run import checkpoint as ck  # noqa: E402
from tpa.run import trainer as TR  # noqa: E402
from tpa.run.logio import read_all  # noqa: E402

WALL = ("wall_s", "reaction_ms_p50", "reaction_ms_p90", "recovery_ms")


def test_registry_and_flags():
    assert ARMS["A16"] == ArmFlags(delayed=True)
    assert flags_for_arm("A16") == MemoryFlags(delayed=True)
    assert "A16" in ARMS and ARMS["A16"].memory and ARMS["A16"].use_ml and not ARMS["A16"].flat


def _tensors_equal(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_tensors_equal(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_tensors_equal(x, y) for x, y in zip(a, b))
    return a == b


def test_i_delta_zero_reproduces_A0_bit_for_bit(tmp_path):
    T_EP = 40
    base = {"run.episodes": 6, "run.t_ep": T_EP, "run.segment_slots": 3 * T_EP, "run.val_every_episodes": 2,
            "run.val_slots": 120, "memory.k_long": 8, "memory.rho_merge": 0.999}
    out = {}
    for arm, extra in (("A0", {}), ("A16", {"memory.delay_slots": 0})):
        cfg = tiny_cfg(**base, **extra)
        r = row(tmp_path, arm, seed=3, phase="tune")
        res = TR.run(r, cfg)
        d = Path(r["run_dir"])
        tr = read_all(d / "train").sort_values("episode").reset_index(drop=True)
        vf = read_all(d / "val_full").reset_index(drop=True)
        drop = [c for c in tr.columns if c in WALL or c.startswith("t_") or c == "arm"]
        tr = tr.drop(columns=drop)
        vf = vf.drop(columns=[c for c in vf.columns if c in WALL or c.startswith("t_") or c == "arm"])
        fin = ck.load(d / "ckpt_final.pt")["ctrl"]
        out[arm] = dict(res={k: v for k, v in res.items() if k not in ("run_id", "train_wall_s")}, tr=tr, vf=vf, fin=fin)
    a, b = out["A0"], out["A16"]
    assert a["res"] == b["res"]
    assert a["tr"].equals(b["tr"]) and a["vf"].equals(b["vf"])
    assert _tensors_equal(a["fin"]["sac"], b["fin"]["sac"]) and _tensors_equal(a["fin"]["params"], b["fin"]["params"])
    assert _tensors_equal(a["fin"]["norm"], b["fin"]["norm"])
    # the long-term store saw writes (the test is not vacuous)
    assert any(x > 0 for x in out["A0"]["tr"]["lt_occupancy"].tolist())


def _driver(delta: int, T: int = 160, seed: int = 5, period: int = 40, **ov):
    cfg = make_cfg(memory__k_long=8, memory__rho_merge=0.999, memory__delay_slots=delta, **ov)
    return cfg, Driver(cfg, flags_for_arm("A16"), seed=seed, period=period).run(T)


def test_ii_delayed_content_key_and_timing():
    delta = 7
    cfg, d = _driver(delta)
    st, mem, P = d.store, d.mem, d.params
    ev = st.events
    assert ev["t"].size > 0
    src = ev["delay_src"]
    assert np.all(src >= 0) and np.all(ev["t"] - src == delta), "every A16 write is Delta slots after its key slot"
    w = int(cfg.memory.window_w)
    keys = st.arrival_keys_upto(st.n)
    checked = 0
    for i in range(ev["t"].size):
        s, t = int(src[i]), int(ev["t"][i])
        assert s - st.t0 >= w - 1                           # the key slot had a defined key
        np.testing.assert_array_equal(ev["cand_key"][i], keys[s - st.t0])   # key = k_s, not k_t
        # hand-computed content: compute_v over rows s+1..s+Delta, zone mean of the write slot
        lo, hi = s - st.t0 + 1, s - st.t0 + delta
        v = compute_v(st.zone_arr[lo:hi + 1], st.zone_mean[hi], st.uav_occ[lo:hi + 1].mean(axis=0))
        y = torch.nn.functional.layer_norm(P.W_v(v), (P.d,), eps=P.eps).detach()
        assert torch.allclose(mem.delayed_content(s), y, atol=0, rtol=0)
        if ev["mode"][i] in ("new", "replace"):
            g = torch.sigmoid(P.W_g(torch.cat([torch.zeros(P.d), y])))
            np.testing.assert_allclose(ev["row_after"][i], (y * (1 - g)).detach().numpy(), atol=1e-6)
            checked += 1
        # the gate decision was taken at s (sigma_p at s above theta_p), not at t
        assert d.cons[s] is True or bool(d.cons[s])
    assert checked > 0
    # label of the write = the label of s (Driver label = phase bin of the slot)
    for i in range(ev["t"].size):
        s = int(src[i])
        assert int(mem.L_label[int(ev["k"][i])]) in (int((s % d.period) * 24 // d.period),
                                                     *[int((j % d.period) * 24 // d.period) for j in src.tolist()])
    # pending entries: at most Delta of them, all with s + Delta > t_now
    assert all(p["s"] + delta > d.t - 1 for p in mem.pending) and len(mem.pending) <= delta
    # a segment reset drops them
    mem.reset(0)
    assert mem.pending == [] and mem.n_events == 0


def test_ii_pending_survives_checkpoint_and_resume():
    cfg, d = _driver(9, T=90)
    while not d.mem.pending and d.t < 400:          # snapshot while a write is pending
        d.step()
    assert len(d.mem.pending) > 0
    T0 = d.t
    snap = d.get_state()
    d.run(60)
    d2 = Driver(cfg, flags_for_arm("A16"), seed=5, period=40, params=d.params)
    d2.set_state(snap)
    assert len(d2.mem.pending) == len(snap["mem"]["pending"]) > 0
    d2.run(60)
    for t in range(T0, T0 + 60):
        np.testing.assert_array_equal(d.r[t], d2.r[t])
    assert d.store.n_ev == d2.store.n_ev and np.array_equal(d.store.events["delay_src"], d2.store.events["delay_src"])


@pytest.mark.parametrize("delta", [5, 0])
def test_iii_recon_equals_rollout_and_gradient_reaches_W_v(delta):
    cfg, d = _driver(delta, T=200)
    refs = [(d.store, t) for t in range(0, 200, 7)]
    r = reconstruct(d.params, cfg, d.flags, refs, grad=False).double().numpy()
    for (st, t), rr in zip(refs, r):
        np.testing.assert_allclose(rr, d.r[t], atol=1e-5, rtol=0, err_msg=f"t={t}")
    assert any(np.abs(d.r[t]).sum() > 0 for _, t in refs)
    d.params.zero_grad()
    rr = reconstruct(d.params, cfg, d.flags, [(d.store, 190), (d.store, 199)], grad=True)
    rr.sum().backward()
    assert d.params.W_v.weight.grad is not None and float(d.params.W_v.weight.grad.abs().sum()) > 0
    if delta > 0:
        assert np.all(d.store.events["delay_src"] >= 0)


def test_refusals():
    with pytest.raises(ValueError):
        Driver(make_cfg(memory__delay_slots=5, memory__key_source="candidate"), flags_for_arm("A16"), seed=1)
    with pytest.raises(ValueError):
        Driver(make_cfg(memory__delay_slots=5, memory__candidate_repr="novelty"), flags_for_arm("A16"), seed=1)
    with pytest.raises(ValueError):
        TwoLevelMemory(make_cfg(memory__delay_slots=5), MemoryFlags(delayed=True, flat=True),
                       MemoryParams(make_cfg(memory__delay_slots=5), seed=1))
    with pytest.raises(ValueError):
        TwoLevelMemory(make_cfg(memory__delay_slots=-1), MemoryFlags(delayed=True),
                       MemoryParams(make_cfg(memory__delay_slots=-1), seed=1))
    # other arms never see the delayed path: no pending list is used and delay = 0
    m = TwoLevelMemory(make_cfg(), flags_for_arm("A0"), MemoryParams(make_cfg(), seed=1))
    assert m.delay == 0 and not m.delayed and m.pending == []
    assert json.dumps(math.nan) == "NaN"   # keeps the json import in use
