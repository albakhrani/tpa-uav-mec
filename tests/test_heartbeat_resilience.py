"""Worker heartbeat thread (scripts/run_queue.py::_Heartbeat):
a locked database must make the thread retry, never die; only heartbeat() returning False
(rowcount 0: the row is no longer running under this pid) sets ``lost``."""
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_queue  # noqa: E402
from tpa.run.queue import RunQueue  # noqa: E402


def _db_with_running_row(tmp_path, pid):
    db = tmp_path / "runs.db"
    q = RunQueue(db)
    q.enqueue("r1", "A0", 9004, "sinusoidal", "tune", "cfg", "code")
    row = q.claim("w-test", phase="tune", pid=pid)
    assert row["run_id"] == "r1"
    q.close()
    return db


def _hb_utc(db):
    c = sqlite3.connect(db)
    try:
        return c.execute("SELECT heartbeat_utc FROM runs WHERE run_id='r1'").fetchone()[0]
    finally:
        c.close()


def _wait(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_locked_db_makes_heartbeat_retry_not_die(tmp_path, capsys):
    db = _db_with_running_row(tmp_path, os.getpid())
    before = _hb_utc(db)
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")                 # another writer holds the lock
    hb = run_queue._Heartbeat(db, "r1", every_s=0.05, busy_timeout_s=0.1)
    hb.start()
    try:
        assert _wait(lambda: hb.n_errors >= 3), "thread should have hit the lock several times"
        assert hb.is_alive(), "thread died on 'database is locked'"
        assert not hb.lost.is_set()
        assert _hb_utc(db) == before
        holder.execute("COMMIT")                     # lock released: the next beat succeeds
        assert _wait(lambda: hb.n_beats >= 1), "thread did not recover after the lock was released"
        assert _wait(lambda: _hb_utc(db) != before)
        assert hb.is_alive() and not hb.lost.is_set()
    finally:
        hb.stop.set()
        hb.join(timeout=10)
        holder.close()
    assert not hb.is_alive()
    out = capsys.readouterr().out
    assert "database is locked" in out and "retrying on the next beat" in out


def test_rowcount_zero_sets_lost(tmp_path):
    db = _db_with_running_row(tmp_path, os.getpid() + 1)   # row held by another pid
    hb = run_queue._Heartbeat(db, "r1", every_s=0.05, busy_timeout_s=1.0)
    hb.start()
    hb.join(timeout=10)
    assert not hb.is_alive(), "thread should exit once the claim is lost"
    assert hb.lost.is_set()
    assert hb.n_errors == 0


def test_other_exception_is_retried(tmp_path, monkeypatch):
    db = _db_with_running_row(tmp_path, os.getpid())
    calls = {"n": 0}
    real = RunQueue.heartbeat

    def flaky(self, run_id, *a, **k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("transient")
        return real(self, run_id, *a, **k)

    monkeypatch.setattr(RunQueue, "heartbeat", flaky)
    hb = run_queue._Heartbeat(db, "r1", every_s=0.05, busy_timeout_s=1.0)
    hb.start()
    try:
        assert _wait(lambda: hb.n_beats >= 1)
        assert hb.n_errors == 2 and hb.is_alive() and not hb.lost.is_set()
    finally:
        hb.stop.set()
        hb.join(timeout=10)


def test_default_busy_timeout_is_300_s(tmp_path):
    hb = run_queue._Heartbeat(tmp_path / "x.db", "r1", every_s=60.0)
    assert hb.busy_timeout_s == run_queue.HEARTBEAT_BUSY_TIMEOUT_S == 300.0
