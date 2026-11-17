"""The mechanism-check arm A15 is excluded from the tuning and campaign arm sets; every
tuning row carries the adopted scenario (tuning.ADOPTED_FILE); theta_p and rho_merge
are never in a search space (fixed, mechanism-selected)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tuning  # noqa: E402


def test_a15_excluded_from_tuning_sets():
    assert "A15" not in tuning.ARMS
    assert "A15" not in tuning.SAC_ARMS and "A15" not in tuning.GATED and "A15" not in tuning.BURST
    assert tuning.MECHANISM_ONLY == {"A15"}
    from tpa.arms import ARMS as all_arms
    assert set(all_arms) - set(tuning.ARMS) == {"A15"}


def test_no_space_tunes_theta_p_or_rho_merge():
    for arm in tuning.ARMS:
        sp = tuning.space_of(arm)
        assert "memory.theta_p" not in sp and "memory.rho_merge" not in sp
        # the consolidation delay of the delayed arms (A16, A16P) is the one memory key that IS tuned
        allowed = {"memory.delay_slots"} if getattr(tuning.ARMS[arm], "delayed", False) else set()
        assert not any(k.startswith("memory.") and k not in allowed for k in sp)


def test_row_overrides_carry_adopted_scenario_and_defaults():
    ad = json.loads(tuning.ADOPTED_FILE.read_text())["overrides"]
    ov = tuning.row_overrides({"learner.lr": 1e-3}, 300)
    for k, v in ad.items():
        assert ov[k] == v
    assert ov["run.episodes"] == 300 and ov["learner.lr"] == 1e-3
    from tpa.config import load_config
    cfg = load_config(overrides=ov)
    assert cfg.scenario.n_uav == 25 and cfg.workload.loads[1] == 0.374
    assert cfg.memory.theta_p == 0.35 and cfg.memory.rho_merge == 0.95 and cfg.memory.key_source == "arrivals"


def test_sample_is_deterministic_and_equal_budget():
    for arm in ("A0", "A9", "B_SONG"):
        a, b = tuning.sample(arm), tuning.sample(arm)
        assert a == b and len(a) == tuning.N_CONFIGS


def test_enqueue_refuses_a15(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "enqueue.py"), "--phase", "main",
                        "--arms", "A0,A15", "--seeds", "1", "--db", str(tmp_path / "q.db")],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode != 0 and "A15" in (r.stdout + r.stderr)
    assert not (tmp_path / "q.db").exists()          # refused before the queue is opened


def test_enqueue_merges_adopted_scenario(tmp_path):
    from tpa.run.queue import RunQueue
    db = tmp_path / "q.db"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "enqueue.py"), "--phase", "main",
                        "--arms", "A9", "--seeds", "1", "--db", str(db)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    q = RunQueue(db)
    rows = q.rows(phase="main")
    q.close()
    assert len(rows) == 1
    ov = rows[0]["overrides"]
    ov = json.loads(ov) if isinstance(ov, str) else ov
    assert ov["scenario.n_uav"] == 25 and ov["workload.loads"] == [0.17, 0.374, 0.578]


def test_reduced_tuning_budget_constants():
    """Reduced tuning budget: 6 of the 12 sampled configurations, stage 1 on seed 9004 only, stage 2
    top-2 on seed 9005 only; configs.json holds the sampler's 12 per tunable arm (stage 1 uses the first 6)."""
    assert tuning.N_CONFIGS == 12 and tuning.N_STAGE1 == 6
    assert tuning.STAGE1_SEEDS == (9004,) and tuning.STAGE1_EPISODES == 300
    assert tuning.STAGE2_SEEDS == (9005,) and tuning.STAGE2_EPISODES == 1000 and tuning.STAGE2_TOP == 2
    cfgs = json.loads((tuning.TUNE / "configs.json").read_text())
    for arm in tuning.ARMS:
        if tuning.space_of(arm):
            assert cfgs[arm] == tuning.sample(arm) and len(cfgs[arm]) == 12
    n_tunable = sum(1 for a in tuning.ARMS if tuning.space_of(a))
    n_a16 = sum(1 for a in tuning.ARMS if getattr(tuning.ARMS[a], "delayed", False))   # the delayed arms A16 and A16P
    assert (n_tunable - n_a16) * tuning.N_STAGE1 * len(tuning.STAGE1_SEEDS) == 102     # the other 17 tunable arms
    assert n_a16 == 2 and n_tunable * tuning.N_STAGE1 * len(tuning.STAGE1_SEEDS) == 114


def test_tuning_queue_holds_exactly_the_reduced_set():
    """The live queue with the reduced tuning budget: 102 stage-1 rows, c00-c05 x seed 9004, nothing else."""
    import re
    import sqlite3
    db = ROOT / "runs" / "runs.db"
    if not db.exists():
        pytest.skip("no live queue")
    c = sqlite3.connect(db)
    rows = c.execute("SELECT run_id, arm, seed FROM runs WHERE phase='tune'").fetchall()
    c.close()
    s1 = [r for r in rows if r[0].startswith("tune1-")]
    for rid, arm, seed in s1:
        assert seed in tuning.STAGE1_SEEDS and int(re.search(r"-c(\d\d)-", rid).group(1)) < tuning.N_STAGE1
        assert arm in tuning.ARMS
    assert len(s1) == 102
