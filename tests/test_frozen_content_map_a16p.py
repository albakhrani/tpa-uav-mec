"""A16' = arm key "A16P", A16 with the long-term content map W_v FROZEN at its
seeded initialisation.

(a) Registry and flags; A16P refuses Delta = 0 (the A0 identity belongs to A16,
    tests/test_delayed_consolidation_a16.py (i)).
(b) W_v receives no gradient for A16P: after a reconstruction + backward on the driver, W_v.weight.grad and
    W_v.bias.grad are None while W_V (retrieval values), W_K, W_Q, W_S and W_g do receive gradients; the same
    driver with the A16 flags gives W_v a non-zero gradient (the only difference between the two arms).
(c) Through tpa.run.trainer.run (A16P, tiny config, several SAC updates): W_v in ckpt_final equals the seeded
    initialisation of MemoryParams for the run's torch seed, bit for bit, while W_V and the SAC actor changed;
    both parameter copies of the controller (rollout and learner) carry the frozen flags.
(d) Replay reconstruction equals the rollout r(t) for A16P (the A16 path with a frozen map), and the other arms
    see no change: an A0 TwoLevelMemory leaves every W_v parameter trainable.
"""
from __future__ import annotations

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
from tpa.memory import MemoryFlags, MemoryParams, TwoLevelMemory, flags_for_arm, reconstruct  # noqa: E402
from tpa.run import checkpoint as ck  # noqa: E402
from tpa.run import trainer as TR  # noqa: E402
from tpa.rng import Streams  # noqa: E402


def test_a_registry_flags_and_refusal():
    assert ARMS["A16P"] == ArmFlags(delayed=True, frozen_content=True)
    assert flags_for_arm("A16P") == MemoryFlags(delayed=True, frozen_content=True)
    assert ARMS["A16"] == ArmFlags(delayed=True) and not ARMS["A16"].frozen_content
    for a, f in ARMS.items():
        if a != "A16P":
            assert not f.frozen_content, a
    with pytest.raises(ValueError):
        Driver(make_cfg(memory__delay_slots=0), flags_for_arm("A16P"), seed=1)
    with pytest.raises(ValueError):                      # frozen content without the delayed write rule
        TwoLevelMemory(make_cfg(memory__delay_slots=5), MemoryFlags(frozen_content=True),
                       MemoryParams(make_cfg(memory__delay_slots=5), seed=1))


def _driver(arm: str, delta: int = 5, T: int = 200, seed: int = 5, period: int = 40):
    cfg = make_cfg(memory__k_long=8, memory__rho_merge=0.999, memory__delay_slots=delta)
    return cfg, Driver(cfg, flags_for_arm(arm), seed=seed, period=period).run(T)


def _grads(d, cfg):
    d.params.zero_grad()
    rr = reconstruct(d.params, cfg, d.flags, [(d.store, 190), (d.store, 199)], grad=True)
    rr.sum().backward()
    g = {}
    for name in ("W_v", "W_V", "W_K", "W_Q", "W_S", "W_g"):
        p = getattr(d.params, name).weight.grad
        g[name] = None if p is None else float(p.abs().sum())
    return g


def test_b_W_v_gets_no_gradient_for_A16P_but_does_for_A16():
    cfg, d = _driver("A16P")
    assert not d.params.W_v.weight.requires_grad and not d.params.W_v.bias.requires_grad
    for name in ("W_V", "W_K", "W_Q", "W_S", "W_g", "W_z", "W_omega"):
        assert all(p.requires_grad for p in getattr(d.params, name).parameters()), name
    g = _grads(d, cfg)
    assert g["W_v"] is None and d.params.W_v.bias.grad is None
    assert g["W_V"] is not None and g["W_V"] > 0
    assert g["W_S"] is not None and g["W_S"] > 0
    cfg16, d16 = _driver("A16")
    g16 = _grads(d16, cfg16)
    assert g16["W_v"] is not None and g16["W_v"] > 0            # A16 trains W_v: test_delayed_consolidation_a16.py (iii)


def test_c_trainer_keeps_W_v_at_its_seeded_init(tmp_path):
    T_EP = 40
    cfg = tiny_cfg(**{"run.episodes": 6, "run.t_ep": T_EP, "run.segment_slots": 3 * T_EP, "run.val_every_episodes": 2,
                      "run.val_slots": 120, "memory.k_long": 8, "memory.rho_merge": 0.999, "memory.delay_slots": 5})
    r = row(tmp_path, "A16P", seed=3, phase="tune")
    TR.run(r, cfg)
    d = Path(r["run_dir"])
    fin = ck.load(d / "ckpt_final.pt")["ctrl"]
    init = MemoryParams(cfg, seed=Streams(3, "train").torch_seed()).state_dict()
    for k in ("W_v.weight", "W_v.bias"):
        assert torch.equal(fin["params"][k].cpu(), init[k]), k
    # training happened: the retrieval value map and the actor moved away from their initialisation
    assert not torch.equal(fin["params"]["W_V.weight"].cpu(), init["W_V.weight"])
    # both controller copies carry the frozen flags
    from tpa.env.simulator import Simulator
    from tpa.env.world import build_world
    from tpa.run.timing import ModuleTimer
    streams = Streams(3, "train")
    world = build_world(cfg, streams.world)
    _, s0, load0 = TR._segment_of(0, cfg)
    sim = Simulator(cfg, world, streams, arm_key="A16P", workload_kind="sinusoidal", load=load0, t0=s0,
                    horizon=int(cfg.run.segment_slots))
    ctrl = TR.make_controller(cfg, "A16P", streams, sim.r_ref, sim.h_min, device="cpu", timer=ModuleTimer())
    for P in {id(ctrl.params): ctrl.params, id(ctrl.params_learn): ctrl.params_learn}.values():
        assert all(not p.requires_grad for p in P.W_v.parameters())
        assert all(p.requires_grad for p in P.W_V.parameters())


def test_d_recon_equals_rollout_for_A16P_and_other_arms_untouched():
    cfg, d = _driver("A16P", T=200)
    refs = [(d.store, t) for t in range(0, 200, 7)]
    r = reconstruct(d.params, cfg, d.flags, refs, grad=False).double().numpy()
    for (st, t), rr in zip(refs, r):
        np.testing.assert_allclose(rr, d.r[t], atol=1e-5, rtol=0, err_msg=f"t={t}")
    assert any(np.abs(d.r[t]).sum() > 0 for _, t in refs)
    assert np.all(d.store.events["delay_src"] >= 0)
    m = TwoLevelMemory(make_cfg(), flags_for_arm("A0"), MemoryParams(make_cfg(), seed=1))
    assert not m.frozen_content and all(p.requires_grad for p in m.params.W_v.parameters())
    m16 = TwoLevelMemory(make_cfg(memory__delay_slots=5), flags_for_arm("A16"), MemoryParams(make_cfg(), seed=1))
    assert not m16.frozen_content and all(p.requires_grad for p in m16.params.W_v.parameters())
