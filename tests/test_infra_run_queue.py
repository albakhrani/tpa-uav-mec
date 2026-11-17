"""End-to-end test of scripts/run_queue.py in --dry-run mode (stub, no training)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tpa.config import load_config  # noqa: E402
from tpa.run.manifest import code_hash  # noqa: E402
from tpa.run.queue import RunQueue  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_queue.py"


def _topology(tmp_path):
    import cpu_topology
    if os.name == "nt":
        return cpu_topology.build(tmp_path / "topo.json")
    # non-Windows fixture: a minimal topology in the schema of scripts/cpu_topology.py, including
    # p_core_all_logical (needed by the launcher's default --cores p-ht)
    fake = {"p_core_logical": [0, 1], "p_core_all_logical": [0, 1], "e_core_logical": [],
            "e_core_one_per_physical": [], "n_physical": 2, "n_logical": 2,
            "cores": [{"logical": [0]}, {"logical": [1]}]}
    (tmp_path / "topo.json").write_text(json.dumps(fake))
    return fake


def _enqueue(db, n, cfg, ch):
    q = RunQueue(db)
    for i in range(n):
        q.enqueue(f"dry{i}", "A0", i + 1, "sinusoidal", "pilot", cfg.hash(), ch)
    q.close()


def _run(tmp_path, *extra, env=None):
    cmd = [sys.executable, str(SCRIPT), "--db", str(tmp_path / "runs.db"),
           "--runs-root", str(tmp_path / "runs"), "--topology", str(tmp_path / "topo.json"),
           "--no-keep-awake", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env, cwd=str(ROOT))


def test_dry_run_two_workers(tmp_path):
    topo = _topology(tmp_path)
    cfg = load_config()
    _enqueue(tmp_path / "runs.db", 5, cfg, code_hash())
    r = _run(tmp_path, "--workers", "2", "--dry-run", "--phase", "pilot", "--cores", "p", "--skip-cpus", "")
    assert r.returncode == 0, r.stdout + r.stderr
    q = RunQueue(tmp_path / "runs.db")
    assert q.counts()["done"] == 5, q.status_table()
    pins = set()
    for row in q.rows():
        marker = json.loads((tmp_path / "runs" / "pilot" / row["run_id"] /
                             "DRY_RUN_STUB_DONE.json").read_text())
        assert "DRY RUN" in marker["note"]
        assert marker["thread_env"]["OMP_NUM_THREADS"] == "1"
        pins.add(tuple(marker["affinity"]))
        assert row["run_bytes"] > 0
    allowed = {(c,) for c in topo["p_core_logical"][:2]}
    assert pins <= allowed, (pins, allowed)


def test_dry_run_staggered_starts(tmp_path):
    """--stagger-s: worker i starts stagger_s seconds after worker i-1."""
    import time
    _topology(tmp_path)
    cfg = load_config()
    _enqueue(tmp_path / "runs.db", 4, cfg, code_hash())
    t0 = time.perf_counter()
    r = _run(tmp_path, "--workers", "2", "--dry-run", "--phase", "pilot", "--cores", "p", "--skip-cpus", "",
             "--stagger-s", "4")
    wall = time.perf_counter() - t0
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("spawned ") == 2 and wall >= 4.0, (wall, r.stdout)
    q = RunQueue(tmp_path / "runs.db")
    assert q.counts()["done"] == 4, q.status_table()


def test_numa_alternate_order(tmp_path):
    """--numa-alternate: node0, node1, node0, ... in each node's own order."""
    import run_queue
    for n, cl in (("node0", "0-31"), ("node1", "32-63")):
        (tmp_path / n).mkdir()
        (tmp_path / n / "cpulist").write_text(cl + "\n")
    order = list(range(1, 64))                       # core 0 skipped
    alt = run_queue.numa_alternate(order, tmp_path)
    assert alt[:6] == [1, 32, 2, 33, 3, 34]
    assert sorted(alt) == order and len(alt) == len(order)
    assert sorted(alt[:20]) == list(range(1, 11)) + list(range(32, 42))
    (tmp_path / "node1" / "cpulist").write_text("")  # one populated node -> unchanged
    assert run_queue.numa_alternate(order, tmp_path) == order
    assert run_queue.numa_alternate(order, tmp_path / "absent") == order


def test_without_dry_run_missing_trainer_claims_nothing(tmp_path):
    try:
        import tpa.run.trainer  # noqa: F401
        pytest.skip("the trainer exists; this test covers the missing-trainer path")
    except ImportError:
        pass
    _topology(tmp_path)
    cfg = load_config()
    _enqueue(tmp_path / "runs.db", 2, cfg, code_hash())
    r = _run(tmp_path, "--workers", "1")
    assert r.returncode != 0
    assert RunQueue(tmp_path / "runs.db").counts()["pending"] == 2


def test_disk_guard_stops_workers(tmp_path):
    _topology(tmp_path)
    cfg = load_config()
    _enqueue(tmp_path / "runs.db", 2, cfg, code_hash())
    env = dict(os.environ, TPA_FAKE_FREE_GB="10")
    r = _run(tmp_path, "--workers", "1", "--dry-run", env=env)
    assert r.returncode == 3
    assert "disk guard refused" in r.stderr
    assert RunQueue(tmp_path / "runs.db").counts()["pending"] == 2


def test_hash_mismatch_fails_row(tmp_path):
    _topology(tmp_path)
    cfg = load_config()
    _enqueue(tmp_path / "runs.db", 1, cfg, "not-the-code-hash")
    r = _run(tmp_path, "--workers", "1", "--dry-run")
    assert r.returncode == 0
    row = RunQueue(tmp_path / "runs.db").get("dry0")
    assert row["state"] == "failed" and "code_hash mismatch" in row["error"]
