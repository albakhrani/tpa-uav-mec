"""The evaluation-only sensitivity rows (file prefix b1): frozen A0 policies re-evaluated with one setting changed.
No training, no tuning, no change to tpa/ or config/ (this file is outside the code hash).

    python scripts/b1_eval.py --enqueue --dry-run    # plan the 450 rows from the main queue; enqueue nothing
    python scripts/b1_eval.py --enqueue              # write them (pending) into runs/b1/runs_b1.db
    python scripts/b1_eval.py --status               # queue counts, the main-queue gate, live workers (no result)
    python scripts/b1_eval.py --launch [--dry-run]   # the launcher of the sensitivity workers
    python scripts/b1_eval.py --release-dead         # rows 'running' whose worker process is gone -> pending

Rows. 15 per test seed 1-30: the 14 settings of SETTINGS and one default re-evaluation (tag 'default', no
setting changed); 450 rows in an own queue file (runs/b1/runs_b1.db; schema of tpa/run/queue.py; phase main; queue arm
A0; run ids b1-A0-s<seed>-<tag>; seed-major claim order, the default row first). Overrides = the main A0 row's queue
overrides + run.test_runs = 1 + the setting.

Evaluation (eval_only). The controller is built as tpa/run/trainer.py run() builds it before training (train streams,
build_world, the segment-0 simulator for r_ref / h_min, make_controller); the learned state of the source row's
ckpt_best.pt is merged in as run() does before its test evaluation (_merge_net_state); then the unchanged
tpa.run.evaluator.evaluate(block="test", loads=[heavy]) runs. With run.test_runs = 1 that is the main test's run k = 0:
Streams(seed, "test", 0), t0 = run.test_seed_offset_slots, 10,800 slots = 3P. The evaluation clone takes only the
learned state from the controller (tpa/control.py:292-305) and deterministic actions draw no random numbers
(tpa/learner/sac.py:103-115), so a default row reproduces the main test's first heavy run.

Claim rule, "behind every main row". A sensitivity row is claimable only when no main row is pending in runs/runs.db
(every main row has been claimed) and its source row main-A0-s<seed>-sinusoidal is done with ckpt_best.pt and
RESULT.json present.
Launcher. Every --poll-s seconds it adds sensitivity workers, each pinned to one CPU of TPA_CPUS that no live worker
process uses, while the worker processes in total stay at or below --max-total (default and upper limit MAX_TOTAL),
MemAvailable stays above MEM_FLOOR_GB + WORKER_GB per new worker and the disk keeps DISK_FLOOR_GB free. Worker
processes are the main and sensitivity workers whose cwd is one of the TPA_TREES directories under --home-tpa (see
classify); the launcher starts its own workers in this checkout, so it counts them only if the checkout is one of
those directories. It reads other processes' /proc entries only to count worker processes; it signals nothing but
its own children. It exits when every sensitivity row is done or failed, or (exit 2) when pending sensitivity rows
can never run because no main row is open and their source rows are not done.
Each launcher pass returns to pending the rows of workers that have exited and the stale rows of the sensitivity
queue; a failing pass is retried (30 in a row end the launcher); at the time limit no worker is added and the
launcher waits for its workers; every exit path terminates its workers and returns their rows; workers run in their
own process session. Before each claim a worker stops if more than MAX_TOTAL worker processes run or another worker
process shares its CPU (e.g. after a main-side restart). Rows whose error note starts with 'stopped by' are never
claimed. The tool refuses any --db other than a runs_b1.db that is not the main queue, and any queue file that holds
a row whose run id does not start with b1-A0-s; --dry-run goes with --enqueue or --launch only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "scripts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_queue  # noqa: E402  (no numpy / torch at module level)

HEAVY = 0.578
SEEDS = tuple(range(1, 31))
SOURCE_RUN = "main-A0-s{seed}-sinusoidal"
DEFAULT_TAG = "default"
# (run-id tag, config key, value); the values are the registered decimal literals, never computed products
SETTINGS = (
    ("theta_p-0.30", "memory.theta_p", 0.30),
    ("theta_p-0.40", "memory.theta_p", 0.40),
    ("rho_merge-0.90", "memory.rho_merge", 0.90),
    ("rho_merge-0.98", "memory.rho_merge", 0.98),
    ("theta_b-x0.75", "burst.theta_b", 0.675),
    ("theta_b-x1.25", "burst.theta_b", 1.125),
    ("alpha_t-x0.75", "drm.alpha_t", 0.3),
    ("alpha_t-x1.25", "drm.alpha_t", 0.5),
    ("alpha_m-x0.75", "drm.alpha_m", 0.225),
    ("alpha_m-x1.25", "drm.alpha_m", 0.375),
    ("alpha_d-x0.75", "drm.alpha_d", 0.225),
    ("alpha_d-x1.25", "drm.alpha_d", 0.375),
    ("omega_l-x0.75", "drm.omega_l", 0.45),
    ("omega_l-x1.25", "drm.omega_l", 0.75),
)
# the value of each key in the main A0 configuration (burst.theta_b = A0's tuned value; the rest default.toml)
MAIN_VALUES = {"memory.theta_p": 0.35, "memory.rho_merge": 0.95, "burst.theta_b": 0.9, "drm.alpha_t": 0.4,
               "drm.alpha_m": 0.3, "drm.alpha_d": 0.3, "drm.omega_l": 0.6}
TAGS = (DEFAULT_TAG,) + tuple(t for t, _, _ in SETTINGS)
EXPECTED_ROWS = len(SEEDS) * len(TAGS)                       # 450
TPA_CPUS = tuple(range(1, 21)) + tuple(range(32, 52))        # the CPUs the launcher may pin workers to
MAX_TOTAL = 30                                               # upper limit of worker processes (main + sensitivity)
TPA_TREES = ("tpa_v2", "tpa_v2_a13", "tpa_v2_a15")           # code trees under --home-tpa whose worker processes count
WORKER_GB, MEM_FLOOR_GB, DISK_FLOOR_GB = 2.0, 15.0, 60.0
WALL_COLS = ("wall_s", "reaction_ms_p50", "reaction_ms_p90", "recovery_ms")   # + every t_*_ms timer
NOT_STOPPED = "COALESCE(error, '') NOT LIKE 'stopped by%'"   # a row marked 'stopped by ...' is never claimed
B1_DB_NAME = "runs_b1.db"


# ---------------------------------------------------------------- rows
def run_id(seed: int, tag: str) -> str:
    return f"b1-A0-s{int(seed)}-{tag}"


def parse_run_id(rid: str) -> tuple[int, str]:
    if not rid.startswith("b1-A0-s"):
        raise ValueError(f"not a B1 run id: {rid!r}")
    s, tag = rid[len("b1-A0-s"):].split("-", 1)
    if tag not in TAGS:
        raise ValueError(f"unknown B1 tag {tag!r} in {rid!r}")
    return int(s), tag


def setting_of(tag: str) -> tuple[str | None, float | None]:
    if tag == DEFAULT_TAG:
        return None, None
    for t, k, v in SETTINGS:
        if t == tag:
            return k, v
    raise KeyError(tag)


def b1_overrides(base: dict, key: str | None, value) -> dict:
    """The sensitivity row's queue overrides: the main A0 row's + run.test_runs = 1 + the setting."""
    ov = dict(base)
    ov["run.test_runs"] = 1
    if key is not None:
        ov[key] = value
    return ov


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True, timeout=60)


def source_rows(main_db: Path) -> dict[int, dict]:
    """main-A0-s<seed>-sinusoidal rows of the main queue (state, overrides, hashes); read-only, no result."""
    con = _ro(main_db)
    try:
        out = {}
        for rid, seed, st, ov, code, cfgh in con.execute(
                "SELECT run_id, seed, state, overrides, code_hash, config_hash FROM runs WHERE phase='main' AND arm='A0'"):
            if rid == SOURCE_RUN.format(seed=seed):
                out[int(seed)] = dict(run_id=rid, seed=int(seed), state=st, overrides=json.loads(ov or "{}"),
                                      code_hash=code, config_hash=cfgh)
        return out
    finally:
        con.close()


def plan_rows(main_db: Path, cur_code: str) -> tuple[list[dict], dict]:
    """The 450 sensitivity rows, from the 30 main A0 rows; SystemExit on any inconsistency."""
    from tpa.config import load_config
    src = source_rows(main_db)
    missing = [s for s in SEEDS if s not in src]
    if missing:
        raise SystemExit(f"main A0 rows missing for seeds {missing}: nothing planned")
    ovs = {json.dumps(src[s]["overrides"], sort_keys=True) for s in SEEDS}
    if len(ovs) != 1:
        raise SystemExit(f"the 30 main A0 rows do not share one set of overrides ({len(ovs)} distinct): nothing planned")
    codes = {src[s]["code_hash"] for s in SEEDS}
    if codes != {cur_code}:
        raise SystemExit(f"main A0 code hashes {sorted(c[:12] for c in codes)} != this tree {cur_code[:12]}: nothing planned")
    base = src[SEEDS[0]]["overrides"]
    base_cfg = load_config(overrides=base)
    bad = [s for s in SEEDS if src[s]["config_hash"] != base_cfg.hash()]
    if bad:
        raise SystemExit(f"config hash of the main A0 rows of seeds {bad} differs from their overrides: nothing planned")
    if abs(float(base_cfg.workload.loads[-1]) - HEAVY) > 1e-12:
        raise SystemExit(f"heavy load {base_cfg.workload.loads[-1]} != {HEAVY}: nothing planned")
    for k, v in MAIN_VALUES.items():
        sec, leaf = k.split(".")
        if abs(float(getattr(getattr(base_cfg, sec), leaf)) - v) > 1e-12:
            raise SystemExit(f"main A0 value of {k} is {getattr(getattr(base_cfg, sec), leaf)}, registered {v}: nothing planned")
    rows = []
    for seed in SEEDS:
        for tag in TAGS:
            key, val = setting_of(tag)
            ov = b1_overrides(base, key, val)
            rows.append(dict(run_id=run_id(seed, tag), arm="A0", seed=seed, workload="sinusoidal", phase="main",
                             config_hash=load_config(overrides=ov).hash(), code_hash=cur_code, loads=[HEAVY],
                             overrides=ov, tag=tag, key=key, value=val))
    if len(rows) != EXPECTED_ROWS or len({r["run_id"] for r in rows}) != EXPECTED_ROWS:
        raise SystemExit(f"{len(rows)} rows planned, expected {EXPECTED_ROWS} distinct: nothing planned")
    return rows, base


def enqueue(rows: list[dict], b1_db: Path) -> tuple[int, bool]:
    """Insert the planned rows (pending, seed-major); returns (rows inserted, claim order == plan order)."""
    from tpa.run.queue import RunQueue
    q = RunQueue(b1_db)
    try:
        n = 0
        for r in rows:
            n += bool(q.enqueue(r["run_id"], r["arm"], r["seed"], r["workload"], r["phase"], r["config_hash"],
                                r["code_hash"], loads=r["loads"], overrides=r["overrides"], priority=0))
            time.sleep(0.002)                     # distinct created_utc: the claim order is the plan order
        order = [x[0] for x in q.con.execute("SELECT run_id FROM runs WHERE phase='main' "
                                             "ORDER BY priority DESC, created_utc, run_id")]
        return n, order == [r["run_id"] for r in rows]
    finally:
        q.close()


# ---------------------------------------------------------------- the gate "behind every main row"
def main_gate(main_db: Path, main_runs_root: Path) -> tuple[int, int, set[int]]:
    """(main rows pending, main rows running, seeds whose source row is final). Queue states and file presence only."""
    con = _ro(main_db)
    try:
        n_pend = con.execute("SELECT COUNT(*) FROM runs WHERE phase='main' AND state='pending' "
                             "AND COALESCE(error, '') NOT LIKE 'stopped by%'").fetchone()[0]
        n_run = con.execute("SELECT COUNT(*) FROM runs WHERE phase='main' AND state='running'").fetchone()[0]
        done = [(rid, int(s)) for rid, s in con.execute(
            "SELECT run_id, seed FROM runs WHERE phase='main' AND arm='A0' AND state='done'")]
    finally:
        con.close()
    ready = set()
    for rid, s in done:
        d = Path(main_runs_root) / "main" / rid
        if rid == SOURCE_RUN.format(seed=s) and (d / "ckpt_best.pt").is_file() and (d / "RESULT.json").is_file():
            ready.add(s)
    return int(n_pend), int(n_run), ready


def claim_ready(q, worker_id: str, seeds) -> dict | None:
    """RunQueue.claim (tpa/run/queue.py) restricted to rows of the given seeds; same UPDATE, same order."""
    from tpa.run.queue import iso, utc_now
    sl = sorted({int(s) for s in seeds})
    if not sl:
        return None
    ts, pid, host = iso(utc_now()), os.getpid(), socket.gethostname()

    sql = ("SELECT run_id FROM runs WHERE state='pending' AND phase='main' AND " + NOT_STOPPED + " AND seed IN ("
           + ",".join("?" * len(sl)) + ") ORDER BY priority DESC, created_utc, run_id LIMIT 1")

    def tx(c):
        r = c.execute(sql, sl).fetchone()
        if r is None:
            return None
        cur = c.execute("UPDATE runs SET state='running', pid=?, host=?, worker=?, heartbeat_utc=?,"
                        " started_utc=COALESCE(started_utc, ?), attempts=attempts+1, error=NULL"
                        " WHERE run_id=? AND state='pending'", (pid, host, worker_id, ts, ts, r["run_id"]))
        if cur.rowcount != 1:
            return None
        return c.execute("SELECT * FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()
    return q._row(q._write(tx))


def n_ready_pending(b1_db: Path, seeds) -> int:
    sl = sorted({int(s) for s in seeds})
    if not sl:
        return 0
    con = _ro(b1_db)
    try:
        sql = ("SELECT COUNT(*) FROM runs WHERE state='pending' AND phase='main' AND " + NOT_STOPPED + " AND seed IN ("
               + ",".join("?" * len(sl)) + ")")
        return int(con.execute(sql, sl).fetchone()[0])
    finally:
        con.close()


def b1_counts(b1_db: Path) -> dict[str, int]:
    """Rows by state; a pending row marked as stopped (error 'stopped by ...') counts as 'stopped', not pending."""
    con = _ro(b1_db)
    try:
        out = {s: 0 for s in ("pending", "running", "done", "failed", "stopped")}
        for st, stopped, n in con.execute("SELECT state, (state='pending' AND NOT (" + NOT_STOPPED + ")), COUNT(*) "
                                          "FROM runs GROUP BY 1, 2"):
            out["stopped" if stopped else st] += int(n)
        return out
    finally:
        con.close()


def housekeeping(b1_db: Path, exited_workers, stale_min: float) -> list[str]:
    """Launcher, once per pass: rows still 'running' under a worker id that has exited go back to pending (an
    exited worker holds no valid claim), then stale rows of the sensitivity queue (a worker that died unseen).
    Returns the requeued run ids. Sensitivity queue only."""
    from tpa.run.queue import RunQueue
    q = RunQueue(b1_db)
    try:
        out = []
        for wid in exited_workers:
            out += q.release_worker(wid)
        return out + q.requeue_stale(stale_min)
    finally:
        q.close()


def overbooked(live: dict, my_pid: int, my_cpus: set, max_total: int = MAX_TOTAL) -> str | None:
    """Worker, before each claim: a reason to stop claiming, or None. More worker processes than max_total, or
    another worker process on this worker's CPU (e.g. after a main-side restart), makes the sensitivity worker
    yield."""
    if len(live) > max_total:
        return f"{len(live)} TPA workers > {max_total}"
    for pid, w in live.items():
        if pid != my_pid and w["cpus"] & set(my_cpus):
            return f"vCPU {sorted(w['cpus'] & set(my_cpus))} shared with TPA worker pid {pid} ({w['kind']})"
    return None


# ---------------------------------------------------------------- evaluation only
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve_source(main_db: Path, main_runs_root: Path, seed: int) -> dict:
    s = source_rows(main_db).get(int(seed))
    if s is None:
        raise RuntimeError(f"source row {SOURCE_RUN.format(seed=seed)} not in {main_db}")
    s["run_dir"] = str(Path(main_runs_root) / "main" / s["run_id"])
    return s


def build_controller(cfg, arm_key: str, seed: int, workload: str = "sinusoidal", env_kw: dict | None = None):
    """The controller of tpa/run/trainer.py run() before training (lines 117-124), on the CPU."""
    from tpa.env.simulator import Simulator
    from tpa.env.world import build_world
    from tpa.rng import Streams
    from tpa.run import trainer as T
    from tpa.run.timing import ModuleTimer
    env_kw = dict(env_kw or {})
    streams = Streams(int(seed), "train")
    world = build_world(cfg, streams.world)
    timer = ModuleTimer()
    _, s0, load0 = T._segment_of(0, cfg)
    sim = Simulator(cfg, world, streams, arm_key=arm_key, workload_kind=workload, load=load0, t0=s0,
                    horizon=int(cfg.run.segment_slots), **env_kw)
    return T.make_controller(cfg, arm_key, streams, sim.r_ref, sim.h_min, device="cpu", timer=timer)


def eval_only(row: dict, cfg, src: dict, block: str = "test") -> dict:
    """One sensitivity row: the source row's selected checkpoint, evaluated once at heavy load on test run k = 0 of
    `block` (always "test" for sensitivity rows; the validation-data check in tests/test_sensitivity_eval.py passes
    "val_full"). Writes manifest.json, <block>/part-*.parquet (the trainer's format) and RESULT.json into
    row['run_dir']."""
    import pyarrow as pa
    import torch
    from tpa.config import load_config
    from tpa.run import checkpoint as ck
    from tpa.run import trainer as T
    from tpa.run.evaluator import evaluate
    from tpa.run.logio import ParquetLog
    from tpa.run.manifest import write_manifest

    seed, tag = parse_run_id(row["run_id"])
    if seed != int(row["seed"]) or row["arm"] != "A0":
        raise RuntimeError(f"{row['run_id']}: seed / arm do not match the run id")
    key, value = setting_of(tag)
    if src.get("state") != "done":
        raise RuntimeError(f"source row {src.get('run_id')} is {src.get('state')!r}, not done")
    expect = b1_overrides(src["overrides"], key, value)
    if json.dumps(expect, sort_keys=True) != json.dumps(row["overrides"], sort_keys=True):
        raise RuntimeError(f"{row['run_id']}: overrides != source overrides + run.test_runs = 1 + the setting")
    if src["code_hash"] != row["code_hash"]:
        raise RuntimeError(f"{row['run_id']}: code hash {row['code_hash'][:12]} != source {src['code_hash'][:12]}")
    if cfg.hash() != load_config(overrides=row["overrides"]).hash():
        raise RuntimeError(f"{row['run_id']}: config does not match the row's overrides")
    loads = [float(x) for x in (row.get("loads") or [HEAVY])]
    if len(loads) != 1 or abs(loads[0] - float(cfg.workload.loads[-1])) > 1e-12 or int(cfg.run.test_runs) != 1:
        raise RuntimeError(f"{row['run_id']}: B1 evaluates the heavy load once (loads {loads}, test_runs {cfg.run.test_runs})")
    workload, env_kw = T.parse_workload(row.get("workload", "sinusoidal"))
    if workload != "sinusoidal" or env_kw:
        raise RuntimeError(f"{row['run_id']}: B1 rows are sinusoidal rows")
    best_p = ck.best_path(src["run_dir"])
    if not Path(best_p).is_file():
        raise RuntimeError(f"{best_p} not found")
    sha = sha256_file(Path(best_p))

    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    b1 = dict(tag=tag, setting=key, value=value, source_run=src["run_id"], source_ckpt=str(best_p),
              source_ckpt_sha256=sha, block=block, test_run=0, load=loads[0])
    write_manifest(run_dir, cfg, extra=dict(row={k: v for k, v in row.items()
                                                 if isinstance(v, (str, int, float, bool, list, dict))}, b1=b1))
    ctrl = build_controller(cfg, "A0", seed, workload, env_kw)
    best = ck.load(best_p)
    ctrl.set_state(T._merge_net_state(ctrl.get_state(), best["ctrl"]))          # trainer.py:227-228
    w0 = time.perf_counter()
    rows = evaluate(ctrl, cfg, seed, "A0", workloads=(workload,), loads=loads, env_kw=env_kw, block=block)
    keys = sorted({k for r in rows for k in r})
    rows = [{k: r.get(k, float("nan")) for k in keys} for r in rows]             # trainer.py:233-237
    tl = ParquetLog(run_dir / block, schema=pa.Table.from_pylist(rows).schema)
    tl.extend(rows)
    tl.flush()
    result = dict(run_id=row["run_id"], b1=b1, source_best_ep=best.get("ep"), test_rows=len(rows),
                  eval_wall_s=time.perf_counter() - w0)
    (run_dir / "RESULT.json").write_text(json.dumps(result, default=float, indent=1))
    return result


# ---------------------------------------------------------------- worker
def worker_main(a) -> int:
    wid = a.worker_id
    import psutil
    cpus = [int(x) for x in a.cpus.split(",")] if a.cpus else None
    if cpus:
        psutil.Process().cpu_affinity(cpus)
    for v in run_queue.THREAD_VARS:
        os.environ.setdefault(v, "1")
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from tpa.config import load_config
    from tpa.run import queue as Q
    from tpa.run.manifest import code_hash
    base_cfg, cur_code = load_config(), code_hash()
    q = Q.RunQueue(a.db)
    runs_root = Path(a.runs_root)
    n_done = 0
    while a.max_runs is None or n_done < a.max_runs:
        stale = q.requeue_stale(a.stale_min)                       # the sensitivity queue file only
        if stale:
            print(f"[b1 worker {wid}] requeued stale: {stale}", flush=True)
        g = Q.disk_guard(q, runs_root, base_cfg.run.disk_budget_gb, base_cfg.run.disk_min_free_gb)
        if not g.ok:
            q.close()
            return run_queue.EXIT_DISK
        if os.path.isdir("/proc"):                                 # yield to an overbooked CPU set or worker limit
            why = overbooked(live_tpa_workers(Path(a.home_tpa)), os.getpid(), set(cpus or ()), MAX_TOTAL) if cpus else None
            if why:
                print(f"[b1 worker {wid}] not claiming: {why}; worker exits", flush=True)
                break
        n_pend, _n_run, ready = main_gate(Path(a.main_db), Path(a.main_runs_root))
        row = claim_ready(q, wid, ready if n_pend == 0 else ())
        if row is None:
            break
        rid, t0 = row["run_id"], time.perf_counter()
        hb = run_queue._Heartbeat(Path(a.db), rid, a.heartbeat_s)
        hb.start()
        try:
            cfg = base_cfg.with_overrides(row["overrides"]) if row.get("overrides") else base_cfg
            if cfg.hash() != row["config_hash"]:
                raise RuntimeError(f"config_hash mismatch: row {row['config_hash']} != effective {cfg.hash()}")
            if row["code_hash"] != cur_code:
                raise RuntimeError(f"code_hash mismatch: row {row['code_hash'][:12]} != tree {cur_code[:12]}")
            row = dict(row)
            row.update(run_dir=str(Q.run_dir_for(runs_root, row)), db_path=str(a.db), worker_id=wid, device="cpu",
                       affinity=cpus)
            print(f"[b1 worker {wid}] start {rid} on cpus={cpus}", flush=True)
            eval_only(row, cfg, resolve_source(Path(a.main_db), Path(a.main_runs_root), int(row["seed"])))
            hb.stop.set()
            hb.join()
            if hb.lost.is_set():
                print(f"[b1 worker {wid}] {rid} finished after losing its claim; not completing", flush=True)
            else:
                q.complete(rid, wall_s=time.perf_counter() - t0, run_bytes=Q.dir_size(row["run_dir"]))
                print(f"[b1 worker {wid}] done {rid} in {time.perf_counter() - t0:.1f}s", flush=True)
        except BaseException as e:  # noqa: BLE001 - record any failure on the row
            hb.stop.set()
            hb.join()
            if isinstance(e, KeyboardInterrupt):
                q.release_worker(wid)
                print(f"[b1 worker {wid}] interrupted; released {rid}", file=sys.stderr, flush=True)
                break
            q.fail(rid, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", wall_s=time.perf_counter() - t0)
            print(f"[b1 worker {wid}] FAILED {rid}: {e}", file=sys.stderr, flush=True)
        n_done += 1
    q.close()
    return run_queue.EXIT_OK


# ---------------------------------------------------------------- launcher
def parse_cpulist(s: str) -> set[int]:
    out = set()
    for part in s.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _proc_info(pid: int) -> dict | None:
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
        cwd = os.readlink(f"/proc/{pid}/cwd")
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            argv = [x.decode("utf-8", "replace") for x in fh.read().split(b"\0") if x]
        with open(f"/proc/{pid}/status") as fh:
            status = fh.read()
    except OSError:
        return None
    m = re.search(r"^Cpus_allowed_list:\s*(\S+)", status, re.M)
    return dict(exe=exe, cwd=cwd, argv=argv, cpus=parse_cpulist(m.group(1)) if m else set())


def classify(info: dict | None, trees: set[str]) -> str | None:
    """'main' / 'b1' for a worker process (python, cwd in a TPA_TREES tree, --_worker, our script), else None."""
    if not info or "python" not in os.path.basename(info["exe"]) or info["cwd"] not in trees or "--_worker" not in info["argv"]:
        return None
    if any(x.endswith("scripts/run_queue.py") for x in info["argv"]):
        return "main"
    if any(x.endswith("scripts/b1_eval.py") for x in info["argv"]):
        return "b1"
    return None


def tpa_trees(home_tpa: Path) -> set[str]:
    return {os.path.realpath(home_tpa / t) for t in TPA_TREES}


def live_tpa_workers(home_tpa: Path) -> dict[int, dict]:
    trees, out = tpa_trees(home_tpa), {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            info = _proc_info(int(d))
            kind = classify(info, trees)
            if kind:
                out[int(d)] = dict(kind=kind, cpus=info["cpus"])
    return out


def spawn_plan(live: dict, n_ready: int, mem_avail_gb: float, disk_free_gb: float, max_total: int = MAX_TOTAL) -> list[int]:
    """CPUs of TPA_CPUS for new sensitivity workers (one each): free CPUs only, total worker processes <= max_total."""
    if disk_free_gb < DISK_FLOOR_GB:
        return []
    used = set().union(*(w["cpus"] for w in live.values())) if live else set()
    free = [c for c in TPA_CPUS if c not in used]
    mem_room = int(max(0.0, mem_avail_gb - MEM_FLOOR_GB) // WORKER_GB)
    n = max(0, min(max_total - len(live), len(free), int(n_ready), mem_room))
    return free[:n]


def _mem_available_gb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    return 0.0


def _disk_free_gb(p: Path) -> float:
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 1024 ** 3


def launcher_main(a) -> int:
    from tpa.run.queue import RunQueue
    log = lambda m: print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {m}", flush=True)  # noqa: E731
    home_tpa = Path(a.home_tpa)
    log_dir = Path(a.runs_root) / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    procs: dict[str, tuple] = {}
    k, t_start, last_note = 0, time.time(), ""

    def _raise(_sig, _frm):
        raise KeyboardInterrupt
    old_handlers = {}
    for name in ("SIGTERM", "SIGHUP"):                          # SIGHUP: controlling terminal closed (not on Windows)
        sg = getattr(signal, name, None)
        if sg is not None:
            old_handlers[sg] = signal.signal(sg, _raise)
    log(f"B1 launcher pid {os.getpid()}: queue {a.db}, main queue {a.main_db}, max {a.max_total} TPA workers in total, "
        f"TPA vCPUs 1-20 and 32-51, poll {a.poll_s} s{' (DRY RUN: spawns nothing)' if a.dry_run else ''}")
    rc, draining, n_err = 1, False, 0
    to_release: set[str] = set()                     # exited workers whose rows are not yet returned
    try:
        while True:
            try:
                for wid in [w for w, (p, _fh, _c) in procs.items() if p.poll() is not None]:
                    p, fh, c = procs.pop(wid)
                    fh.close()
                    to_release.add(wid)
                    log(f"worker {wid} (cpu {c}) exited with code {p.returncode}")
                back = [] if a.dry_run else housekeeping(Path(a.db), sorted(to_release), a.stale_min)
                to_release.clear()                                    # only once housekeeping has succeeded
                if back:
                    log(f"returned to pending (worker gone or stale): {back}")
                cnt = b1_counts(Path(a.db))
                if cnt["pending"] == 0 and cnt["running"] == 0 and not procs:
                    log(f"every B1 row is done, failed or stopped: {cnt}; launcher exits")
                    rc = 0
                    return rc
                if not draining and time.time() - t_start > a.max_hours * 3600:
                    draining = True                                   # stop adding, let workers finish
                    log(f"time limit {a.max_hours} h reached with {cnt}: no new workers; waiting for {len(procs)} to finish")
                if draining:
                    if not procs:
                        log(f"time limit: every worker has finished; launcher exits with {cnt}")
                        rc = 3
                        return rc
                    n_err = 0                                         # a successful pass
                    time.sleep(a.poll_s)
                    continue
                n_pend, n_run, ready = main_gate(Path(a.main_db), Path(a.main_runs_root))
                nr = n_ready_pending(Path(a.db), ready) if n_pend == 0 else 0
                if n_pend > 0:
                    note = f"waiting: {n_pend} main rows pending (B1 runs behind every main row); B1 {cnt}"
                elif nr == 0:
                    if n_run == 0 and cnt["running"] == 0 and not procs and cnt["pending"] > 0:
                        log(f"{cnt['pending']} B1 rows pending, none can run (no main row open; their source rows are not "
                            f"done; ready seeds {sorted(ready)}); launcher exits, needs the user")
                        rc = 2
                        return rc
                    note = f"waiting: no claimable B1 row (ready seeds {len(ready)}); main running {n_run}; B1 {cnt}"
                else:
                    live = live_tpa_workers(home_tpa)
                    mem, disk = _mem_available_gb(), _disk_free_gb(Path(a.runs_root))
                    cpus = spawn_plan(live, nr, mem, disk, a.max_total)
                    note = (f"claimable B1 rows {nr}; live TPA workers {len(live)} (b1 {sum(w['kind'] == 'b1' for w in live.values())}); "
                            f"MemAvailable {mem:.0f} GB; disk free {disk:.0f} GB; start {len(cpus)} on cpus {cpus}")
                    for c in ([] if a.dry_run else cpus):
                        wid = f"b1w{k:03d}-{stamp}"
                        k += 1
                        env = dict(os.environ)
                        for v in run_queue.THREAD_VARS:
                            env[v] = "1"
                        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                        env["CUDA_VISIBLE_DEVICES"] = ""
                        cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", "--worker-id", wid, "--cpus", str(c),
                               "--db", str(a.db), "--runs-root", str(a.runs_root), "--main-db", str(a.main_db),
                               "--main-runs-root", str(a.main_runs_root), "--home-tpa", str(a.home_tpa),
                               "--stale-min", str(a.stale_min), "--heartbeat-s", str(a.heartbeat_s)]
                        fh = open(log_dir / f"{wid}.log", "w", encoding="utf-8")
                        # own session: a worker is not hung up when the launcher's terminal closes; every exit path
                        # of this launcher terminates and releases its workers itself (finally below)
                        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                             start_new_session=True)
                        procs[wid] = (p, fh, c)
                        log(f"spawned {wid} pid={p.pid} cpu={c} log={fh.name}")
                if note != last_note:
                    log(note)
                    last_note = note
                n_err = 0
                if a.dry_run:
                    log("dry run: one pass done, nothing started")
                    rc = 0
                    return rc
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # noqa: BLE001 - e.g. 'database is locked' under disk stalls: retry next pass
                n_err += 1
                log(f"pass failed ({type(e).__name__}: {e}); {n_err} in a row\n{traceback.format_exc()}")
                if a.dry_run:                                         # a dry run reports, never retries
                    rc = 4
                    return rc
                if n_err >= 30:
                    log("30 failed passes in a row: launcher exits (workers terminated and released below)")
                    rc = 4
                    return rc
            time.sleep(a.poll_s)
    except KeyboardInterrupt:
        log(f"interrupted: terminating {len(procs)} B1 worker(s) and returning their rows to pending")
        rc = 1
        return rc
    finally:
        if procs or to_release:                                 # any exit with live or unreleased workers
            log(f"terminating {len(procs)} B1 worker(s); returning their rows and those of {sorted(to_release)} to pending")
            for p, _fh, _c in procs.values():
                if p.poll() is None:
                    p.terminate()
            q = RunQueue(a.db)
            try:
                for wid, (p, fh, _c) in procs.items():
                    try:
                        p.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                    fh.close()
                    log(f"released {q.release_worker(wid)} of {wid}")
                for wid in sorted(to_release):
                    log(f"released {q.release_worker(wid)} of {wid}")
            finally:
                q.close()
        for sg, h in old_handlers.items():
            signal.signal(sg, h)


def release_dead(a) -> int:
    """Sensitivity rows 'running' whose worker process is gone (or is now another process) -> pending.
    Sensitivity queue only."""
    from tpa.run.queue import RunQueue
    trees = tpa_trees(Path(a.home_tpa))
    q = RunQueue(a.db)
    try:
        def tx(c):
            out = []
            for r in c.execute("SELECT run_id, pid, worker FROM runs WHERE state='running'").fetchall():
                info = _proc_info(int(r["pid"])) if r["pid"] is not None else None
                live = classify(info, trees) == "b1" and r["worker"] in (info or {}).get("argv", [])
                if r["pid"] is not None and not live:
                    c.execute("UPDATE runs SET state='pending', pid=NULL, worker=NULL, error='requeued: worker process "
                              "gone (b1_eval.py --release-dead)' WHERE run_id=? AND state='running'", (r["run_id"],))
                    out.append(r["run_id"])
            return out
        ids = q._write(tx)
    finally:
        q.close()
    print(f"requeued {len(ids)} B1 row(s): {ids}")
    return 0


def status(a) -> int:
    cnt = b1_counts(Path(a.db)) if Path(a.db).exists() else None
    n_pend, n_run, ready = main_gate(Path(a.main_db), Path(a.main_runs_root))
    print(f"B1 queue {a.db}: {cnt}")
    print(f"main queue: {n_pend} pending, {n_run} running; source rows final for {len(ready)} seeds {sorted(ready)}")
    print(f"B1 rows claimable now: {n_ready_pending(Path(a.db), ready) if (cnt and n_pend == 0) else 0}"
          f"{'' if n_pend == 0 else ' (main rows pending: B1 waits)'}")
    if os.path.isdir("/proc"):
        live = live_tpa_workers(Path(a.home_tpa))
        print(f"live TPA workers: {len(live)} (main {sum(w['kind'] == 'main' for w in live.values())}, "
              f"b1 {sum(w['kind'] == 'b1' for w in live.values())})")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Amendment 16 Part B1: evaluation-only rows")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--enqueue", action="store_true")
    m.add_argument("--status", action="store_true")
    m.add_argument("--launch", action="store_true")
    m.add_argument("--release-dead", dest="release_dead", action="store_true")
    m.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=ROOT / "runs" / "b1" / "runs_b1.db")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs" / "b1")
    ap.add_argument("--main-db", type=Path, default=ROOT / "runs" / "runs.db")
    ap.add_argument("--main-runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--home-tpa", type=Path, default=ROOT.parent)
    ap.add_argument("--max-total", type=int, default=MAX_TOTAL)
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--max-hours", type=float, default=240.0)
    ap.add_argument("--stale-min", type=float, default=10.0)
    ap.add_argument("--heartbeat-s", type=float, default=60.0)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--worker-id", default="b1w-manual", help=argparse.SUPPRESS)
    ap.add_argument("--cpus", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.max_total > MAX_TOTAL or a.max_total < 1:
        ap.error(f"--max-total must be 1-{MAX_TOTAL}")
    if a.dry_run and not (a.enqueue or a.launch):                  # the other modes would ignore --dry-run
        ap.error("--dry-run goes with --enqueue or --launch only")
    # this tool writes only the sensitivity queue file, never the main queue
    if a.db.name != B1_DB_NAME or os.path.realpath(a.db) == os.path.realpath(a.main_db):
        ap.error(f"--db must be the B1 queue file ({B1_DB_NAME}), not {a.db}")
    return a


def check_b1_only(db: Path) -> None:
    """Refuse a queue file that holds any row that is not a sensitivity row (run id b1-A0-s...)."""
    if not Path(db).exists():
        return
    con = _ro(db)
    try:
        bad = [r[0] for r in con.execute("SELECT run_id FROM runs WHERE run_id NOT LIKE 'b1-A0-s%' LIMIT 3")]
    finally:
        con.close()
    if bad:
        raise SystemExit(f"{db} holds rows that are not B1 rows (e.g. {bad}): refused")


def main(argv=None) -> int:
    a = parse_args(argv)
    if not a.status:
        check_b1_only(a.db)
    if (a.launch or a._worker or a.release_dead) and not a.db.exists():         # never create it
        raise SystemExit(f"no B1 queue at {a.db}: run scripts/b1_eval.py --enqueue first")
    if a.launch and sum(b1_counts(a.db).values()) != EXPECTED_ROWS:
        raise SystemExit(f"{a.db} holds {sum(b1_counts(a.db).values())} rows, not {EXPECTED_ROWS}: launcher refused")
    if a._worker:
        return worker_main(a)
    if a.launch:
        return launcher_main(a)
    if a.release_dead:
        return release_dead(a)
    if a.status:
        return status(a)
    from tpa.run.manifest import code_hash
    rows, base = plan_rows(a.main_db, code_hash())
    print(f"main A0 overrides (all 30 identical): {json.dumps(base, sort_keys=True)}")
    for tag in TAGS:
        r = next(x for x in rows if x["tag"] == tag)
        print(f"  {tag:15} {str(r['key']):17} {str(r['value']):6} config {r['config_hash']}")
    print(f"{len(rows)} rows = {len(SEEDS)} seeds x {len(TAGS)} (14 settings + default); code {rows[0]['code_hash'][:12]}; "
          f"queue {a.db}")
    if a.dry_run:
        print("dry run: nothing enqueued")
        return 0
    if a.db.exists() and sum(b1_counts(a.db).values()):
        raise SystemExit(f"{a.db} already holds rows: nothing enqueued")
    n, same = enqueue(rows, a.db)
    print(f"enqueued {n} rows; claim order equals the plan order: {same}")
    return 0 if (n == EXPECTED_ROWS and same) else 1


if __name__ == "__main__":
    sys.exit(main())
