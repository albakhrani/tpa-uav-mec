"""Integration: TPAController (tpa/control.py) + Simulator on the tiny config, 60 slots with
learner updates, must keep every MemoryParams parameter finite after every update
(regression: NaN gradients from an all-masked softmax / zero-norm rows in the replay
reconstruction when a sampled transition had no long-term event yet)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from _sys_helpers import tiny_cfg  # noqa: E402

try:
    from tpa.control import TPAController
    from tpa.env.simulator import Simulator
    from tpa.env.world import build_world
    from tpa.rng import Streams
    HAVE_CONTROL = True
except ImportError:                                       # controller modules not available
    HAVE_CONTROL = False


def _run(cfg, arm="A0", T=60, seed=1):
    st = Streams(seed, "train")
    world = build_world(cfg, st.world)
    sim = Simulator(cfg, world, st, arm_key=arm, load=0.55, t0=0, horizon=T)
    c = TPAController(cfg, arm, st, sim.r_ref, sim.h_min)
    c.reset_segment(0)
    c.start_episode()
    n_updates = 0
    for t in range(T):
        obs = sim.begin_slot(t)
        dec, me = c.act(obs, sim.truth_phase_bin(t), train=True)
        out = sim.end_slot(dec, me)
        info = c.observe(obs, dec, out, train=True)
        if isinstance(info, dict) and info.get("update") is not None:
            n_updates += 1
        for name, p in c.params_learn.named_parameters():
            assert torch.isfinite(p).all(), f"slot {t}: non-finite {name}"
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"slot {t}: non-finite grad {name}"
        if c.has_mem and c.last.get("P") is not None:
            assert np.isfinite(c.last["P"]).all(), f"slot {t}: non-finite P"
    return c, n_updates


@pytest.mark.skipif(not HAVE_CONTROL, reason="tpa.control not available")
@pytest.mark.parametrize("ov", [
    {},
    {"memory.lt_addressing": "soft"},
    {"memory.lt_addressing": "content"},
    {"memory.candidate_repr": "layernorm"},
    {"memory.candidate_repr": "novelty", "memory.lt_addressing": "soft"},
    {"memory.key_source": "arrivals", "memory.candidate_repr": "layernorm"},          # arrival keys
    {"memory.key_source": "arrivals", "memory.lt_addressing": "soft"},
], ids=["default", "soft", "content", "layernorm", "novelty+soft", "arrivals", "arrivals+soft"])
def test_controller_updates_keep_memory_params_finite(ov):
    try:
        cfg = tiny_cfg(**ov)
    except KeyError as e:                                  # key not in default.toml
        pytest.skip(f"config key missing: {e}")
    c, n_updates = _run(cfg)
    assert n_updates > 0, "no learner update happened in 60 slots"


@pytest.mark.skipif(not HAVE_CONTROL, reason="tpa.control not available")
@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A4", "A14"])
def test_controller_updates_finite_other_arms(arm):
    c, n_updates = _run(tiny_cfg(), arm=arm)
    assert n_updates > 0


@pytest.mark.skipif(not HAVE_CONTROL, reason="tpa.control not available")
@pytest.mark.parametrize("arm", ["A0", "A1", "A3", "A14"])
def test_controller_updates_finite_arrivals(arm):
    """The controller path under key_source = "arrivals" (slot keys from the store's
    record of za / zone_mean; Z-dimensional event log) with learner updates."""
    cfg = tiny_cfg(**{"memory.key_source": "arrivals"})
    c, n_updates = _run(cfg, arm=arm)
    assert n_updates > 0
    Z = cfg.scenario.n_zones
    assert c.mem.kd == Z and c.store.kd == Z
    if arm != "A1":
        assert c.store.n_ev > 0 and c.store.events["key"].shape[1] == Z
    s = c.get_state()
    assert s["mem"]["S_key"].shape[1] == Z
