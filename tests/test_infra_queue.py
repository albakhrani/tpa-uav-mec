import io
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpa.run import checkpoint as ck  # noqa: E402
from tpa.run.queue import RunQueue, disk_guard, run_dir_for, utc_now  # noqa: E402

GB = 1024 ** 3


def _fill(q, n, phase="pilot"):
    for i in range(n):
        assert q.enqueue(f"r{i:03d}", "A0", i, "sinusoidal", phase, "cfg", "code")


def test_enqueue_idempotent_and_wal(tmp_path):
    q = RunQueue(tmp_path / "runs.db")
    assert q.enqueue("a", "A0", 1, "sinusoidal", "main", "c", "k", loads=[0.25, 0.85])
    assert not q.enqueue("a", "A0", 1, "sinusoidal", "main", "c", "k")
    assert q.con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert q.get("a")["loads"] == [0.25, 0.85]
    assert q.counts()["pending"] == 1


def test_claim_lifecycle(tmp_path):
    q = RunQueue(tmp_path / "runs.db")
    _fill(q, 2)
    r = q.claim("w0")
    assert r["run_id"] == "r000" and r["state"] == "running" and r["pid"] == os.getpid()
    assert q.heartbeat("r000", episodes_done=10, last_ckpt="x.pt")
    assert q.get("r000")["episodes_done"] == 10
    assert q.complete("r000", wall_s=1.5, run_bytes=123)
    r = q.claim("w0")
    assert r["run_id"] == "r001"
    assert q.fail("r001", "boom")
    assert q.claim("w0") is None
    c = q.counts()
    assert c["done"] == 1 and c["failed"] == 1
    assert "pilot" in q.status_table()
    assert q.requeue_failed() == 1 and q.counts()["pending"] == 1


_CLAIMER = r"""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
from tpa.run.queue import RunQueue
db, go, out = sys.argv[2], sys.argv[3], sys.argv[4]
q = RunQueue(db)
open(out + ".ready", "w").close()
while not os.path.exists(go):
    time.sleep(0.005)
got = []
while True:
    r = q.claim(os.path.basename(out))
    if r is None:
        break
    got.append(r["run_id"])
    time.sleep(0.003)          # "work" outside the transaction, as real workers do
    q.complete(r["run_id"])
json.dump(got, open(out, "w"))
"""


def test_claim_exclusive_two_processes(tmp_path):
    db = tmp_path / "runs.db"
    q = RunQueue(db)
    n = 200
    _fill(q, n)
    script = tmp_path / "claimer.py"
    script.write_text(_CLAIMER)
    go = tmp_path / "go"
    outs = [tmp_path / f"w{i}.json" for i in range(2)]
    procs = [subprocess.Popen([sys.executable, str(script), str(ROOT), str(db), str(go), str(o)])
             for o in outs]
    deadline = time.time() + 120
    while not all(Path(str(o) + ".ready").exists() for o in outs):   # both connected, waiting
        assert time.time() < deadline, "claimer processes did not start"
        time.sleep(0.01)
    go.write_text("go")
    for p in procs:
        assert p.wait(timeout=120) == 0
    a, b = (set(json.loads(o.read_text())) for o in outs)
    assert not (a & b), f"double-claimed: {sorted(a & b)[:5]}"
    assert len(a | b) == n
    assert q.counts()["done"] == n
    # both processes did take part (not strictly required, but expected for 200 rows)
    assert len(a) > 0 and len(b) > 0


def test_stale_requeue(tmp_path):
    q = RunQueue(tmp_path / "runs.db")
    _fill(q, 2)
    t0 = utc_now()
    q.claim("w0", now=t0)
    q.claim("w1", now=t0)
    q.heartbeat("r001", now=t0 + timedelta(minutes=9))
    stale = q.requeue_stale(minutes=10, now=t0 + timedelta(minutes=11))
    assert stale == ["r000"]
    assert q.get("r000")["state"] == "pending" and q.get("r000")["pid"] is None
    assert q.get("r001")["state"] == "running"
    # the original holder's heartbeat now reports the claim as lost
    assert not q.heartbeat("r000")
    # it can be claimed again; attempts counted
    r = q.claim("w2", now=t0 + timedelta(minutes=12))
    assert r["run_id"] == "r000" and r["attempts"] == 2


def test_release_worker(tmp_path):
    q = RunQueue(tmp_path / "runs.db")
    _fill(q, 1)
    q.claim("wX")
    assert q.release_worker("wX") == ["r000"]
    assert q.get("r000")["state"] == "pending"


def test_disk_guard_free_space_refusal(tmp_path):
    q = RunQueue(tmp_path / "runs.db")
    _fill(q, 3)
    err = io.StringIO()
    g = disk_guard(q, tmp_path, budget_gb=100, min_free_gb=40,
                   free_bytes_fn=lambda p: int(39.5 * GB), stream=err)
    assert not g.ok and "disk_min_free_gb" in g.reason
    assert "REFUSING NEW RUNS" in err.getvalue()
    ok = disk_guard(q, tmp_path, budget_gb=100, min_free_gb=40,
                    free_bytes_fn=lambda p: 200 * GB, stream=io.StringIO())
    assert ok.ok and not ok.warnings
    warn = io.StringIO()
    w = disk_guard(q, tmp_path, budget_gb=100, min_free_gb=40,
                   free_bytes_fn=lambda p: 42 * GB, stream=warn)
    assert w.ok and w.warnings and "WARNING" in warn.getvalue()


def test_disk_guard_projection_from_measured_sizes(tmp_path):
    q = RunQueue(tmp_path / "runs.db")
    _fill(q, 10)
    # two finished runs measured at 15 GB each -> 10 runs project to 150 GB > 100 GB
    for _ in range(2):
        r = q.claim("w0")
        q.complete(r["run_id"], run_bytes=15 * GB)
    err = io.StringIO()
    g = disk_guard(q, tmp_path, budget_gb=100, min_free_gb=40,
                   free_bytes_fn=lambda p: 500 * GB, prior_rolling_gb=0.0, stream=err)
    assert g.measured_runs == 2 and abs(g.per_run_gb - 15) < 1e-9
    assert abs(g.projected_gb - 150) < 1e-6
    assert not g.ok and "disk_budget_gb" in g.reason and "REFUSING" in err.getvalue()


def test_disk_guard_counts_rolling_checkpoint(tmp_path):
    q = RunQueue(tmp_path / "runs.db")
    _fill(q, 1)
    r = q.claim("w0")
    p = ck.rolling_path(run_dir_for(tmp_path, r))
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\0" * (2 * 1024 ** 2))
    g = disk_guard(q, tmp_path, budget_gb=100, min_free_gb=40,
                   free_bytes_fn=lambda _: 500 * GB, prior_run_mb=1.0, stream=None)
    assert abs(g.projected_gb * 1024 - 3.0) < 1e-6   # 1 MB per run + 2 MB rolling


def test_claim_arm_filter(tmp_path):
    """Workers can be restricted to a set of arms (GPU workers take only memory arms)."""
    from tpa.run.queue import RunQueue
    q = RunQueue(tmp_path / "r.db")
    for arm in ("A0", "A9", "A3", "A12"):
        q.enqueue(f"{arm}-1", arm, 1, "sinusoidal", "pilot", "h", "c")
    got = [q.claim("g", phase="pilot", arms=["A0", "A3"])["arm"] for _ in range(2)]
    assert sorted(got) == ["A0", "A3"]
    assert q.claim("g", phase="pilot", arms=["A0", "A3"]) is None
    assert sorted(q.claim("c", phase="pilot")["arm"] for _ in range(1)) in (["A9"], ["A12"])
    q.close()
