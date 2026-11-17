"""SQLite run queue in WAL mode.

One row per (arm, seed, workload, phase) run. Workers claim rows atomically
(``BEGIN IMMEDIATE`` takes the database write lock before the SELECT, so two
workers can never claim the same row), send heartbeats, and complete or fail
them. :meth:`RunQueue.requeue_stale` moves a row whose heartbeat is older than its
``minutes`` argument (default 10) back to ``pending``.

The disk guard (:func:`disk_guard`) is evaluated before every claim by
``scripts/run_queue.py``: it refuses new runs when free disk is below
``cfg.run.disk_min_free_gb`` or the projected campaign total (measured
per-run sizes of finished runs extrapolated to the remaining rows, plus the
transient rolling checkpoints of active runs) exceeds ``cfg.run.disk_budget_gb``.

Timestamps are ISO-8601 UTC strings with fixed microsecond precision, so
lexical comparison equals chronological comparison.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

PHASES = ("tune", "pilot", "main", "robust")
STATES = ("pending", "running", "done", "failed")

_GB = 1024 ** 3
_MB = 1024 ** 2

# Core run columns, then these extras:
#   loads      JSON list of loads evaluated (the load set)
#   overrides  JSON dict of dotted config overrides applied on top of default.toml
#   priority   higher first; ties by created_utc, run_id
#   worker     worker id string that holds the claim
#   attempts   number of claims so far
#   run_bytes  measured on-disk size of the run directory at completion (disk guard)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    arm           TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    workload      TEXT NOT NULL,
    phase         TEXT NOT NULL CHECK (phase IN ('tune','pilot','main','robust')),
    config_hash   TEXT NOT NULL,
    code_hash     TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'pending'
                  CHECK (state IN ('pending','running','done','failed')),
    pid           INTEGER,
    host          TEXT,
    heartbeat_utc TEXT,
    episodes_done INTEGER NOT NULL DEFAULT 0,
    last_ckpt     TEXT,
    created_utc   TEXT NOT NULL,
    started_utc   TEXT,
    finished_utc  TEXT,
    wall_s        REAL,
    error         TEXT,
    loads         TEXT,
    overrides     TEXT,
    priority      INTEGER NOT NULL DEFAULT 0,
    worker        TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    run_bytes     INTEGER
);
CREATE INDEX IF NOT EXISTS runs_state_idx ON runs(state, priority, created_utc);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def run_dir_for(runs_root: str | Path, row: dict) -> Path:
    """Convention for a run's directory: ``<runs_root>/<phase>/<run_id>``."""
    return Path(runs_root) / str(row["phase"]) / str(row["run_id"])


def dir_size(path: str | Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class RunQueue:
    """Thin wrapper around one SQLite connection (one per process/thread)."""

    def __init__(self, db_path: str | Path, timeout_s: float = 60.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: autocommit; transactions are explicit BEGIN IMMEDIATE.
        self.con = sqlite3.connect(str(self.db_path), timeout=timeout_s,
                                   isolation_level=None, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute(f"PRAGMA busy_timeout={int(timeout_s * 1000)}")
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.executescript(_SCHEMA)

    # ---------------------------------------------------------------- helpers
    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _write(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run ``fn`` inside BEGIN IMMEDIATE ... COMMIT (rollback on error)."""
        self.con.execute("BEGIN IMMEDIATE")
        try:
            out = fn(self.con)
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        self.con.execute("COMMIT")
        return out

    @staticmethod
    def _row(r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k in ("loads", "overrides"):
            if d.get(k):
                d[k] = json.loads(d[k])
        return d

    # ---------------------------------------------------------------- enqueue
    def enqueue(self, run_id: str, arm: str, seed: int, workload: str, phase: str,
                config_hash: str, code_hash: str, loads: Iterable[float] | None = None,
                overrides: dict | None = None, priority: int = 0,
                now: datetime | None = None) -> bool:
        """Insert a pending row. Returns False (and changes nothing) if run_id exists."""
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
        ts = iso(now or utc_now())
        params = (run_id, arm, int(seed), workload, phase, config_hash, code_hash, ts,
                  json.dumps(list(loads)) if loads is not None else None,
                  json.dumps(overrides, sort_keys=True) if overrides else None, int(priority))

        def tx(c):
            cur = c.execute(
                "INSERT OR IGNORE INTO runs (run_id, arm, seed, workload, phase, config_hash,"
                " code_hash, created_utc, loads, overrides, priority)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)", params)
            return cur.rowcount == 1
        return self._write(tx)

    # ---------------------------------------------------------------- claim
    def claim(self, worker_id: str, phase: str | None = None, pid: int | None = None,
              host: str | None = None, now: datetime | None = None,
              arms: Iterable[str] | None = None) -> dict | None:
        """Atomically move the highest-priority pending row to ``running``.

        ``BEGIN IMMEDIATE`` acquires the write lock before the SELECT, so the
        select-then-update pair is exclusive across processes.
        """
        ts = iso(now or utc_now())
        pid = os.getpid() if pid is None else pid
        host = host or socket.gethostname()

        def tx(c):
            q = "SELECT run_id FROM runs WHERE state='pending'"
            args: list = []
            if phase is not None:
                q += " AND phase=?"
                args.append(phase)
            if arms:                           # e.g. GPU workers take only memory arms
                al = list(arms)
                q += " AND arm IN (" + ",".join("?" * len(al)) + ")"
                args.extend(al)
            q += " ORDER BY priority DESC, created_utc, run_id LIMIT 1"
            r = c.execute(q, args).fetchone()
            if r is None:
                return None
            cur = c.execute(
                "UPDATE runs SET state='running', pid=?, host=?, worker=?, heartbeat_utc=?,"
                " started_utc=COALESCE(started_utc, ?), attempts=attempts+1, error=NULL"
                " WHERE run_id=? AND state='pending'",
                (pid, host, worker_id, ts, ts, r["run_id"]))
            if cur.rowcount != 1:              # cannot happen under the write lock
                return None
            return c.execute("SELECT * FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()
        return self._row(self._write(tx))

    # ---------------------------------------------------------------- progress
    def heartbeat(self, run_id: str, pid: int | None = None, episodes_done: int | None = None,
                  last_ckpt: str | None = None, now: datetime | None = None) -> bool:
        """Refresh the heartbeat. Returns False if this process no longer holds the
        claim (e.g. the row was requeued as stale) - the worker should then stop."""
        pid = os.getpid() if pid is None else pid
        ts = iso(now or utc_now())

        def tx(c):
            cur = c.execute(
                "UPDATE runs SET heartbeat_utc=?,"
                " episodes_done=COALESCE(?, episodes_done), last_ckpt=COALESCE(?, last_ckpt)"
                " WHERE run_id=? AND state='running' AND pid=?",
                (ts, episodes_done, last_ckpt, run_id, pid))
            return cur.rowcount == 1
        return self._write(tx)

    def complete(self, run_id: str, wall_s: float | None = None, run_bytes: int | None = None,
                 pid: int | None = None, now: datetime | None = None) -> bool:
        pid = os.getpid() if pid is None else pid
        ts = iso(now or utc_now())

        def tx(c):
            cur = c.execute(
                "UPDATE runs SET state='done', finished_utc=?, wall_s=?, run_bytes=?,"
                " heartbeat_utc=? WHERE run_id=? AND state='running' AND pid=?",
                (ts, wall_s, run_bytes, ts, run_id, pid))
            return cur.rowcount == 1
        return self._write(tx)

    def fail(self, run_id: str, error: str, wall_s: float | None = None,
             pid: int | None = None, now: datetime | None = None) -> bool:
        pid = os.getpid() if pid is None else pid
        ts = iso(now or utc_now())

        def tx(c):
            cur = c.execute(
                "UPDATE runs SET state='failed', finished_utc=?, wall_s=?, error=?"
                " WHERE run_id=? AND state='running' AND pid=?",
                (ts, wall_s, str(error)[-20000:], run_id, pid))
            return cur.rowcount == 1
        return self._write(tx)

    # ---------------------------------------------------------------- recovery
    def requeue_stale(self, minutes: float = 10.0, now: datetime | None = None) -> list[str]:
        """Return ``running`` rows with heartbeat older than ``minutes`` to ``pending``."""
        cutoff = iso((now or utc_now()) - timedelta(minutes=minutes))

        def tx(c):
            ids = [r["run_id"] for r in c.execute(
                "SELECT run_id FROM runs WHERE state='running' AND"
                " (heartbeat_utc IS NULL OR heartbeat_utc < ?)", (cutoff,))]
            for rid in ids:
                c.execute("UPDATE runs SET state='pending', pid=NULL, worker=NULL,"
                          " error='requeued: stale heartbeat' WHERE run_id=?", (rid,))
            return ids
        return self._write(tx)

    def release_worker(self, worker_id: str) -> list[str]:
        """Return the rows a (shut-down) worker holds to ``pending`` immediately."""
        def tx(c):
            ids = [r["run_id"] for r in c.execute(
                "SELECT run_id FROM runs WHERE state='running' AND worker=?", (worker_id,))]
            for rid in ids:
                c.execute("UPDATE runs SET state='pending', pid=NULL, worker=NULL,"
                          " error='requeued: worker released' WHERE run_id=?", (rid,))
            return ids
        return self._write(tx)

    def requeue_failed(self, phase: str | None = None) -> int:
        def tx(c):
            q = "UPDATE runs SET state='pending', pid=NULL, worker=NULL WHERE state='failed'"
            args: list = []
            if phase is not None:
                q += " AND phase=?"
                args.append(phase)
            return c.execute(q, args).rowcount
        return self._write(tx)

    # ---------------------------------------------------------------- queries
    def get(self, run_id: str) -> dict | None:
        return self._row(self.con.execute("SELECT * FROM runs WHERE run_id=?",
                                          (run_id,)).fetchone())

    def rows(self, state: str | None = None, phase: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM runs WHERE 1=1", []
        if state is not None:
            q += " AND state=?"
            args.append(state)
        if phase is not None:
            q += " AND phase=?"
            args.append(phase)
        q += " ORDER BY created_utc, run_id"
        return [self._row(r) for r in self.con.execute(q, args)]

    def counts(self, phase: str | None = None) -> dict[str, int]:
        q, args = "SELECT state, COUNT(*) n FROM runs", []
        if phase is not None:
            q += " WHERE phase=?"
            args.append(phase)
        q += " GROUP BY state"
        out = {s: 0 for s in STATES}
        for r in self.con.execute(q, args):
            out[r["state"]] = r["n"]
        return out

    def status_table(self) -> str:
        """Plain-text table: phase x state counts, then the running rows."""
        grid: dict[str, dict[str, int]] = {}
        for r in self.con.execute("SELECT phase, state, COUNT(*) n FROM runs GROUP BY phase, state"):
            grid.setdefault(r["phase"], {s: 0 for s in STATES})[r["state"]] = r["n"]
        lines = [f"{'phase':<8}" + "".join(f"{s:>9}" for s in STATES) + f"{'total':>9}"]
        for ph in PHASES:
            if ph in grid:
                g = grid[ph]
                lines.append(f"{ph:<8}" + "".join(f"{g[s]:>9}" for s in STATES)
                             + f"{sum(g.values()):>9}")
        running = self.rows(state="running")
        if running:
            lines.append("")
            lines.append(f"{'running run_id':<40}{'worker':>10}{'pid':>8}{'ep':>6}  heartbeat_utc")
            for r in running:
                lines.append(f"{r['run_id']:<40}{str(r['worker']):>10}{str(r['pid']):>8}"
                             f"{r['episodes_done']:>6}  {r['heartbeat_utc']}")
        return "\n".join(lines)


# ===================================================================== disk guard
@dataclass
class GuardResult:
    ok: bool
    free_gb: float
    projected_gb: float
    per_run_gb: float
    measured_runs: int
    warnings: list[str] = field(default_factory=list)
    reason: str = ""


def _free_bytes_default(path: str | Path) -> int:
    return shutil.disk_usage(path).free


def disk_guard(queue: RunQueue, runs_root: str | Path, budget_gb: float, min_free_gb: float,
               free_bytes_fn: Callable[[str | Path], int] | None = None,
               prior_run_mb: float = 20.0, prior_rolling_gb: float = 0.6,
               warn_frac: float = 0.9, stream=sys.stderr) -> GuardResult:
    """Decide whether a new run may be claimed.

    projected = sum(measured run_bytes of done runs)
              + (pending + running) x per-run size
              + running x rolling-checkpoint size (transient)

    per-run size is the mean measured ``run_bytes`` of done runs (``prior_run_mb``,
    an estimate of ~2x8 MB + logs, until a run has finished); the rolling size is the
    largest rolling checkpoint currently on disk (``prior_rolling_gb`` if none).
    Refuses if free < ``min_free_gb`` or projected > ``budget_gb``; warns from
    ``warn_frac`` of either limit. Warnings/refusals are printed to ``stream``.
    """
    from . import checkpoint  # local: keep module import light

    free_fn = free_bytes_fn or _free_bytes_default
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    free_gb = free_fn(root) / _GB

    done_sizes = [r[0] for r in queue.con.execute(
        "SELECT run_bytes FROM runs WHERE state='done' AND run_bytes IS NOT NULL")]
    n_done_unmeasured = queue.con.execute(
        "SELECT COUNT(*) FROM runs WHERE state='done' AND run_bytes IS NULL").fetchone()[0]
    c = queue.counts()
    per_run = (sum(done_sizes) / len(done_sizes)) if done_sizes else prior_run_mb * _MB

    rolling = 0
    for r in queue.rows(state="running"):
        p = checkpoint.rolling_path(run_dir_for(root, r))
        if p.is_file():
            rolling = max(rolling, p.stat().st_size)
    if rolling == 0:
        rolling = prior_rolling_gb * _GB

    projected = (sum(done_sizes) + n_done_unmeasured * per_run
                 + (c["pending"] + c["running"]) * per_run + c["running"] * rolling)
    projected_gb = projected / _GB
    res = GuardResult(ok=True, free_gb=free_gb, projected_gb=projected_gb,
                      per_run_gb=per_run / _GB, measured_runs=len(done_sizes))

    if free_gb < min_free_gb:
        res.ok = False
        res.reason = (f"free disk {free_gb:.1f} GB < disk_min_free_gb {min_free_gb:.1f} GB")
    elif projected_gb > budget_gb:
        res.ok = False
        res.reason = (f"projected campaign use {projected_gb:.1f} GB > disk_budget_gb "
                      f"{budget_gb:.1f} GB (per-run {per_run / _MB:.1f} MB from "
                      f"{len(done_sizes)} measured runs)")
    else:
        if free_gb < min_free_gb / warn_frac:
            res.warnings.append(f"free disk {free_gb:.1f} GB is approaching the "
                                f"{min_free_gb:.1f} GB floor")
        if projected_gb > warn_frac * budget_gb:
            res.warnings.append(f"projected campaign use {projected_gb:.1f} GB is within "
                                f"{(1 - warn_frac) * 100:.0f}% of the {budget_gb:.1f} GB budget")
    if stream is not None:
        for w in res.warnings:
            print(f"[disk-guard] WARNING: {w}", file=stream, flush=True)
        if not res.ok:
            print(f"[disk-guard] REFUSING NEW RUNS: {res.reason}", file=stream, flush=True)
    return res
