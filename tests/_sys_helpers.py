"""Shared tiny configuration for system tests (determinism, resume, CRN, arms)."""
from __future__ import annotations

from tpa.config import load_config

TINY = {
    "run.episodes": 4, "run.t_ep": 40, "run.segment_slots": 80, "run.val_every_episodes": 2,
    "run.val_slots": 40, "run.ckpt_every_episodes": 2, "run.test_runs": 1, "run.test_slots": 50,
    "learner.warmup_slots": 30, "learner.batch": 8, "learner.update_every": 4,
    "learner.replay": 2000, "learner.norm_freeze_slots": 60,
    "memory.window_w": 10, "memory.b_max": 11, "memory.k_short": 20, "memory.k_long": 30,
    "workload.period_slots": 60, "scenario.n_devices": 60,
}


def tiny_cfg(**extra):
    ov = dict(TINY)
    ov.update(extra)
    return load_config(overrides=ov)


def row(tmp_path, arm="A0", seed=1, **kw):
    r = dict(run_id=f"{arm}-s{seed}", arm=arm, seed=seed, workload="sinusoidal",
             run_dir=str(tmp_path / f"{arm}-s{seed}"), device="cpu")
    r.update(kw)
    return r
