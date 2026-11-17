"""The best and final checkpoints hold every controller's learned state.

Every controller declares learned_state_keys() (what its eval_clone takes from the trained
controller) and tpa.run.trainer._net_state saves exactly those. A fixed key list ("sac", "params",
"norm") would miss B_HUANG (DDPG learner, key "agent") and B_SONG (ES parameters, key "theta"):
their checkpoints would hold only the normaliser, reloading ckpt_best.pt would evaluate an untrained
policy, and the trainer's end-of-run evaluation (trainer.py, after the training loop) would score
the FINAL learned state with the best episode's normaliser.

test_checkpoint_round_trip_reproduces_evaluation fails with such a fixed key list for B_HUANG
and B_SONG: it perturbs every learning container the controller has ("sac", "params", "agent",
"theta"; chosen independently of the declarations, so a missing declaration is caught), evaluates
the live controller, saves _net_state through the checkpoint writer, reloads it into a fresh
controller and requires the identical evaluation. A12 passes either way: it learns nothing (its
lookup table starts empty at every segment and evaluation run, tpa/baselines/lookup.py docstring).
"""
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
from tpa.arms import ARMS  # noqa: E402
from tpa.env.simulator import Simulator  # noqa: E402
from tpa.env.world import build_world  # noqa: E402
from tpa.rng import Streams  # noqa: E402
from tpa.run import checkpoint as ck  # noqa: E402
from tpa.run import trainer as TR  # noqa: E402
from tpa.run.evaluator import evaluate, validate  # noqa: E402
from tpa.run.timing import ModuleTimer  # noqa: E402

ALL_ARMS = sorted(ARMS)                                   # every arm key: A0-A16, A16P and the four baselines
LEARNING = ("sac", "params", "agent", "theta")            # every learning container in the code
WALL = ("wall_s", "reaction_ms_p50", "reaction_ms_p90", "recovery_ms")   # wall-clock, not results
EXPECTED = {**{a: ("sac", "params", "norm") for a in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7",
                                                      "A8", "A13", "A14", "A15")},
            "A9": ("sac", "norm"), "A10": ("sac", "norm"), "A11": ("sac", "norm"), "A12": ("norm",),
            "A16": ("sac", "params", "norm"),                       # A0's controller with memory
            "A16P": ("sac", "params", "norm"),                      # same containers, W_v frozen
            "B_ZHOU": ("norm",), "B_QIN": ("norm",), "B_HUANG": ("agent", "norm"),
            "B_SONG": ("theta", "norm")}
SEED = 3


def _ctrl(cfg, arm, seed=SEED):
    """A controller built the way trainer.run builds it (world and r_ref/h_min of the seed)."""
    streams = Streams(seed, "train")
    world = build_world(cfg, streams.world)
    _, s0, load0 = TR._segment_of(0, cfg)
    sim = Simulator(cfg, world, streams, arm_key=arm, workload_kind="sinusoidal", load=load0, t0=s0,
                    horizon=int(cfg.run.segment_slots))
    return TR.make_controller(cfg, arm, streams, sim.r_ref, sim.h_min, device="cpu", timer=ModuleTimer())


def _perturb(x, rng):
    """Out-of-place Gaussian perturbation of every float tensor / array inside a state."""
    if isinstance(x, torch.Tensor):
        return x + torch.as_tensor(rng.standard_normal(tuple(x.shape)), dtype=x.dtype) if x.is_floating_point() else x
    if isinstance(x, np.ndarray):
        return x + rng.standard_normal(x.shape) if np.issubdtype(x.dtype, np.floating) else x
    if isinstance(x, dict):
        return type(x)((k, _perturb(v, rng)) for k, v in x.items()) if not isinstance(x, type({}.keys())) else x
    if isinstance(x, list):
        return [_perturb(v, rng) for v in x]
    return x


def _eval_row(ctrl, cfg, arm):
    rows = evaluate(ctrl, cfg, SEED, arm, loads=[list(cfg.workload.loads)[1]], block="val")
    run = [r for r in rows if r["row_kind"] == "run"][0]
    return {k: v for k, v in run.items() if isinstance(v, (int, float)) and k not in WALL
            and not k.startswith("t_")}


def test_every_arm_declares_its_learned_state_and_the_checkpoint_holds_it():
    """For every arm key: the declared learned-state keys, present in _net_state."""
    cfg = tiny_cfg()
    for arm in ALL_ARMS:
        c = _ctrl(cfg, arm)
        keys = tuple(c.learned_state_keys())
        assert keys == EXPECTED[arm], (arm, keys)
        net = TR._net_state(c)
        assert tuple(net) == keys, (arm, tuple(net))
        st = c.get_state()
        # every learning container the controller has is declared (nothing learned is left out)
        assert {k for k in LEARNING if k in st} <= set(keys), (arm, sorted(st))


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_checkpoint_round_trip_reproduces_evaluation(arm, tmp_path):
    cfg = tiny_cfg(**{"run.val_slots": 200})
    rng = np.random.default_rng(12)
    live = _ctrl(cfg, arm)
    st = live.get_state()
    changed = [k for k in LEARNING if k in st]
    if changed:
        live.set_state({**st, **{k: _perturb(st[k], rng) for k in changed}})
    want = _eval_row(live, cfg, arm)
    ck.save(tmp_path / "ckpt_best.pt", dict(ctrl=TR._net_state(live), ep=1, val=float(want["completion"])))
    fresh = _ctrl(cfg, arm)
    best = ck.load(tmp_path / "ckpt_best.pt")
    fresh.set_state(TR._merge_net_state(fresh.get_state(), best["ctrl"]))
    got = _eval_row(fresh, cfg, arm)
    assert got.keys() == want.keys()
    bad = {k: (want[k], got[k]) for k in want
           if not (want[k] == got[k] or (isinstance(want[k], float) and math.isnan(want[k]) and math.isnan(got[k])))}
    assert not bad, (arm, bad)
    if changed:
        # the perturbation matters: the untouched fresh controller evaluates differently
        untouched = _eval_row(_ctrl(cfg, arm), cfg, arm)
        assert untouched != want, arm


@pytest.mark.parametrize("arm", ["B_HUANG", "B_SONG", "A12", "A0"])
def test_trainer_best_checkpoint_reload_reproduces_logged_best_validation(arm, tmp_path):
    """End to end through trainer.run on the stage-1 path: reloading ckpt_best.pt into a fresh
    controller and validating reproduces the best validation score the run logged."""
    T_EP = 40
    cfg = tiny_cfg(**{"run.episodes": 8, "run.t_ep": T_EP, "run.segment_slots": 12 * T_EP,
                      "run.val_every_episodes": 2, "run.val_slots": 200, "baselines.es_pairs": 1})
    r = row(tmp_path, arm, seed=SEED, phase="tune")
    res = TR.run(r, cfg)
    best = ck.load(Path(r["run_dir"]) / "ckpt_best.pt")
    assert tuple(best["ctrl"]) == EXPECTED[arm]
    fresh = _ctrl(cfg, arm)
    fresh.set_state(TR._merge_net_state(fresh.get_state(), best["ctrl"]))
    v = validate(fresh, cfg, SEED, arm)
    assert v == res["best_val"] == best["val"], (arm, v, res["best_val"])
    assert json.loads((Path(r["run_dir"]) / "RESULT.json").read_text())["best_ep"] == best["ep"]
