"""Launcher of worker processes for the SQLite run queue.

    python scripts/run_queue.py --workers 8 --phase pilot [--gpu-workers 1] [--cores p-then-e]
    python scripts/run_queue.py --status
    python scripts/run_queue.py --requeue-stale
    python scripts/run_queue.py --workers 2 --dry-run          # infrastructure smoke test

The launcher spawns N worker *processes* (``subprocess``, one interpreter each).
Each worker is pinned (psutil) to one logical CPU (with --pin core: to all logical
CPUs of its physical core) - by default the P-core logical CPUs, hyper-threads
included (--cores p-ht), except those in --skip-cpus, from ``runs/cpu_topology.json``
(built by scripts/cpu_topology.py if missing; that script reads a Windows API, so
on other systems provide the file in the same schema) - and runs with
OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 and torch.set_num_threads(1).
With ``--gpu-workers k`` the first k workers see CUDA_VISIBLE_DEVICES=0 (plus
CUBLAS_WORKSPACE_CONFIG=:4096:8 and deterministic algorithms); all
others get CUDA_VISIBLE_DEVICES="" (no GPU).

Worker loop: requeue stale rows -> disk guard -> claim -> run -> complete/fail.
The run entry point is ``tpa.run.trainer.run(row, cfg)``, imported
lazily. ``--dry-run`` never trains: it uses a clearly-labelled stub that writes
``DRY_RUN_STUB_DONE.json`` into the run directory. Without ``--dry-run`` a
missing trainer is an error and the worker exits *before* claiming anything.

While workers run, the launcher asks Windows not to sleep
(SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED), released on exit;
this is a per-process request, not a settings change). Disable with --no-keep-awake.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# NOTE: no numpy/torch import at module level - thread env vars must be set first.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
EXIT_OK, EXIT_ERROR, EXIT_DISK = 0, 1, 3
STUB_MARKER = "DRY_RUN_STUB_DONE.json"


# ============================================================ helpers
def _keep_awake(on: bool) -> None:
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))


def core_order(topo: dict, mode: str) -> list[int]:
    """Logical CPUs to pin workers to: one per physical core, except mode p-ht (every P-core logical CPU)."""
    p = list(topo["p_core_logical"])
    e = list(topo.get("e_core_one_per_physical", topo["e_core_logical"]))
    if mode == "p":
        return p
    if mode == "p-ht":                          # every logical CPU of the P-cores (hyper-threads included)
        return list(topo["p_core_all_logical"])
    if mode == "e":
        return e
    if mode == "p-then-e":
        return p + e
    raise ValueError(mode)


def numa_alternate(order: list[int], sysfs: str | Path = "/sys/devices/system/node") -> list[int]:
    """Reorder ``order`` so consecutive workers alternate NUMA nodes
    (node0 cpu, node1 cpu, node0 cpu, ...), keeping each node's own order. Nodes come from
    ``<sysfs>/node*/cpulist``; with one node (or no sysfs, e.g. Windows) ``order`` is unchanged."""
    nodes = []
    for d in sorted(Path(sysfs).glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
        cpus = set()
        for part in (d / "cpulist").read_text().strip().split(","):
            if part:
                a, _, b = part.partition("-")
                cpus.update(range(int(a), int(b or a) + 1))
        nodes.append([c for c in order if c in cpus])
    nodes = [n for n in nodes if n]
    if len(nodes) < 2:
        return list(order)
    out = []
    for i in range(max(len(n) for n in nodes)):
        out += [n[i] for n in nodes if i < len(n)]
    return out + [c for c in order if c not in out]


def core_siblings(topo: dict, cpu: int) -> list[int]:
    for c in topo["cores"]:
        if cpu in c["logical"]:
            return list(c["logical"])
    return [cpu]


def dry_run_stub(row: dict, cfg) -> dict:
    """DRY-RUN STUB - NOT A TRAINER. Writes a done marker and returns at once."""
    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = {"stub": "scripts/run_queue.py:dry_run_stub", "note": "DRY RUN - no training was done",
              "run_id": row["run_id"], "pid": os.getpid(), "config_hash": cfg.hash(),
              "affinity": row.get("affinity"), "device": row.get("device"),
              "thread_env": {k: os.environ.get(k) for k in THREAD_VARS},
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    tmp = run_dir / (STUB_MARKER + ".tmp")
    tmp.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / STUB_MARKER)
    return marker


# Busy timeout of the heartbeat connection. When many workers write ~1.3 GB rolling
# checkpoints at the same episode boundaries, the disk can stall and a writer can hold the SQLite
# lock past the default 60 s; with the default, beats then fail on "database is locked" and the
# rows' heartbeats go stale. The longer timeout keeps the heartbeat alive through such stalls.
HEARTBEAT_BUSY_TIMEOUT_S = 300.0


class _Heartbeat(threading.Thread):
    """Background heartbeat on its own SQLite connection.

    Never dies on a database error: any exception (sqlite3.OperationalError "database is
    locked" included) is logged, the connection is reopened and the beat is retried on the
    next tick. ``lost`` is set only when ``heartbeat()`` returns False, i.e. the row is no
    longer running under this pid (requeued as stale or released).
    """

    def __init__(self, db: Path, run_id: str, every_s: float,
                 busy_timeout_s: float | None = None):
        super().__init__(daemon=True)
        self.db, self.run_id, self.every_s = db, run_id, every_s
        self.busy_timeout_s = HEARTBEAT_BUSY_TIMEOUT_S if busy_timeout_s is None else busy_timeout_s
        self.stop = threading.Event()
        self.lost = threading.Event()
        self.n_errors = 0
        self.n_beats = 0

    def run(self):
        from tpa.run.queue import RunQueue
        q = None
        try:
            while not self.stop.wait(self.every_s):
                try:
                    if q is None:
                        q = RunQueue(self.db, timeout_s=self.busy_timeout_s)
                    ok = q.heartbeat(self.run_id)
                except Exception as e:  # noqa: BLE001 - bookkeeping must not kill the thread
                    self.n_errors += 1
                    print(f"[worker {os.getpid()}] heartbeat {self.run_id} failed "
                          f"({type(e).__name__}: {e}); retrying on the next beat", flush=True)
                    try:
                        if q is not None:
                            q.close()
                    except Exception:  # noqa: BLE001
                        pass
                    q = None
                    continue
                self.n_beats += 1
                if not ok:
                    self.lost.set()
                    print(f"[worker {os.getpid()}] lost claim on {self.run_id}", flush=True)
                    return
        finally:
            if q is not None:
                q.close()


# ============================================================ worker process
def worker_main(args) -> int:
    pid = os.getpid()
    wid = args.worker_id
    # 1. pin + threads (env vars were set by the launcher before this interpreter started)
    import psutil
    cpus = [int(x) for x in args.cpus.split(",")] if args.cpus else None
    if cpus:
        psutil.Process().cpu_affinity(cpus)
    for v in THREAD_VARS:
        os.environ.setdefault(v, "1")
    device = "cpu"
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if args.gpu:
            torch.use_deterministic_algorithms(True)
            if torch.cuda.is_available():
                device = "cuda"
    except ImportError:
        if not args.dry_run:
            print(f"[worker {wid}] torch is not importable", file=sys.stderr, flush=True)
            return EXIT_ERROR

    from tpa.config import load_config
    from tpa.run import queue as Q
    from tpa.run.manifest import code_hash

    # 2. resolve the run function before claiming anything
    if args.dry_run:
        run_fn = dry_run_stub
        print(f"[worker {wid}] DRY RUN: using dry_run_stub (no training)", flush=True)
    else:
        try:
            from tpa.run.trainer import run as run_fn  # the training entry point
        except ImportError as e:
            print(f"[worker {wid}] cannot import tpa.run.trainer.run ({e}); "
                  f"refusing to claim. Use --dry-run for an infrastructure test.",
                  file=sys.stderr, flush=True)
            return EXIT_ERROR

    base_cfg = load_config()
    cur_code = code_hash()
    q = Q.RunQueue(args.db)
    runs_root = Path(args.runs_root)
    n_done = 0
    while args.max_runs is None or n_done < args.max_runs:
        stale = q.requeue_stale(args.stale_min)
        if stale:
            print(f"[worker {wid}] requeued stale: {stale}", flush=True)
        g = Q.disk_guard(q, runs_root, base_cfg.run.disk_budget_gb, base_cfg.run.disk_min_free_gb,
                         free_bytes_fn=_free_fn_from_env())
        if not g.ok:
            q.close()
            return EXIT_DISK
        row = q.claim(wid, phase=args.phase,
                      arms=[a for a in args.arms.split(",") if a] if args.arms else None)
        if row is None:
            break
        rid = row["run_id"]
        t0 = time.perf_counter()
        hb = _Heartbeat(Path(args.db), rid, args.heartbeat_s)
        hb.start()
        try:
            cfg = base_cfg.with_overrides(row["overrides"]) if row.get("overrides") else base_cfg
            if cfg.hash() != row["config_hash"]:
                raise RuntimeError(f"config_hash mismatch: row {row['config_hash']} "
                                   f"!= effective {cfg.hash()}")
            if row["code_hash"] != cur_code and not args.allow_code_change:
                raise RuntimeError(f"code_hash mismatch: row {row['code_hash'][:12]} != tree "
                                   f"{cur_code[:12]} (code changed since enqueue; "
                                   f"re-enqueue or pass --allow-code-change)")
            row = dict(row)
            row["run_dir"] = str(Q.run_dir_for(runs_root, row))
            row["db_path"] = str(args.db)
            row["worker_id"] = wid
            row["device"] = device
            row["affinity"] = cpus
            row["gpu_env"] = {k: os.environ.get(k) for k in (
                "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "DISABLE_ADDMM_CUDA_LT",
                "CUDA_LAUNCH_BLOCKING")}                    # recorded in the run manifest
            Path(row["run_dir"]).mkdir(parents=True, exist_ok=True)
            print(f"[worker {wid}] start {rid} on cpus={cpus} device={device}", flush=True)
            run_fn(row, cfg)
            hb.stop.set()
            hb.join()
            if hb.lost.is_set():
                print(f"[worker {wid}] {rid} finished after losing its claim; not completing",
                      flush=True)
            else:
                q.complete(rid, wall_s=time.perf_counter() - t0,
                           run_bytes=Q.dir_size(row["run_dir"]))
                print(f"[worker {wid}] done {rid} in {time.perf_counter() - t0:.1f}s", flush=True)
        except BaseException as e:  # noqa: BLE001 - record any failure on the row
            hb.stop.set()
            hb.join()
            if isinstance(e, KeyboardInterrupt):
                q.release_worker(wid)          # back to pending; resumes from its checkpoint
                print(f"[worker {wid}] interrupted; released {rid}", file=sys.stderr, flush=True)
                break
            q.fail(rid, f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                   wall_s=time.perf_counter() - t0)
            print(f"[worker {wid}] FAILED {rid}: {e}", file=sys.stderr, flush=True)
            if "CUDA error" in str(e) or type(e).__name__ == "AcceleratorError":
                # a CUDA fault leaves this process's context unusable: stop claiming
                print(f"[worker {wid}] CUDA fault: worker exits (requeue with --requeue-failed)",
                      file=sys.stderr, flush=True)
                break
        n_done += 1
    q.close()
    return EXIT_OK


def _free_fn_from_env():
    """Test hook: TPA_FAKE_FREE_GB makes the disk guard see that much free space."""
    v = os.environ.get("TPA_FAKE_FREE_GB")
    if v is None:
        return None
    return lambda _p: int(float(v) * 1024 ** 3)


# ============================================================ launcher
def launcher_main(args) -> int:
    from tpa.run import queue as Q
    sys.path.insert(0, str(ROOT / "scripts"))
    import cpu_topology

    if args.status or args.requeue_stale_only or args.requeue_failed:
        q = Q.RunQueue(args.db)
        if args.requeue_failed:
            print("requeued failed:", q.requeue_failed(args.phase))
        if args.requeue_stale_only:
            print("requeued:", q.requeue_stale(args.stale_min))
        print(q.status_table())
        q.close()
        return EXIT_OK

    topo = cpu_topology.load_or_build(Path(args.topology))
    order = core_order(topo, args.cores)
    if args.skip_cpus:
        skip = {int(c) for c in args.skip_cpus.split(",") if c}
        order = [c for c in order if c not in skip]
    if args.numa_alternate:
        order = numa_alternate(order)
    if args.workers > len(order):
        print(f"--workers {args.workers} > {len(order)} physical cores available in mode "
              f"{args.cores!r}", file=sys.stderr)
        return EXIT_ERROR
    if args.gpu_workers > args.workers:
        print("--gpu-workers cannot exceed --workers", file=sys.stderr)
        return EXIT_ERROR

    log_dir = Path(args.runs_root) / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    procs = []
    if not args.no_keep_awake:
        _keep_awake(True)
    try:
        for i in range(args.workers):
            if i > 0 and args.stagger_s > 0:
                # resuming an old-format checkpoint peaks at up to 9.4 GB per
                # worker; staggered starts keep those peaks from coinciding
                time.sleep(args.stagger_s)
            cpu = order[i]
            pin = core_siblings(topo, cpu) if args.pin == "core" else [cpu]
            gpu = i < args.gpu_workers
            env = dict(os.environ)
            for v in THREAD_VARS:
                env[v] = "1"
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            if gpu:
                env["CUDA_VISIBLE_DEVICES"] = "0"
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                # pilot finding: intermittent CUBLAS_STATUS_EXECUTION_FAILED in cublasLtMatmul
                # (torch 2.14.1+cu126) followed by an illegal address; route
                # addmm through plain cuBLAS instead of the fused cuBLASLt path
                env.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            wid = f"w{i:02d}-{stamp}"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker",
                   "--worker-id", wid, "--cpus", ",".join(map(str, pin)),
                   "--db", str(args.db), "--runs-root", str(args.runs_root),
                   "--stale-min", str(args.stale_min), "--heartbeat-s", str(args.heartbeat_s)]
            if args.phase:
                cmd += ["--phase", args.phase]
            arms_w = (args.gpu_arms if gpu else args.cpu_arms) or args.arms
            if arms_w:
                cmd += ["--arms", arms_w]
            if gpu:
                cmd.append("--gpu")
            if args.dry_run:
                cmd.append("--dry-run")
            if args.allow_code_change:
                cmd.append("--allow-code-change")
            if args.max_runs is not None:
                cmd += ["--max-runs", str(args.max_runs)]
            fh = open(log_dir / f"{wid}.log", "w", encoding="utf-8")
            p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
            procs.append((wid, p, fh, pin, gpu))
            print(f"spawned {wid} pid={p.pid} cpus={pin} gpu={gpu} log={fh.name}", flush=True)
        rc_all = EXIT_OK
        for wid, p, fh, _pin, _gpu in procs:
            rc = p.wait()
            fh.close()
            if rc == EXIT_DISK:
                print(f"[disk-guard] {wid} stopped: disk guard refused new runs "
                      f"(see {log_dir / (wid + '.log')})", file=sys.stderr, flush=True)
            elif rc != EXIT_OK:
                print(f"{wid} exited with code {rc}", file=sys.stderr, flush=True)
            rc_all = max(rc_all, rc)
    except KeyboardInterrupt:
        print("interrupted: terminating workers and releasing their rows", file=sys.stderr)
        for _wid, p, _fh, _pin, _gpu in procs:
            p.terminate()
        q = Q.RunQueue(args.db)
        for wid, p, fh, _pin, _gpu in procs:
            p.wait()
            fh.close()
            q.release_worker(wid)
        q.close()
        rc_all = EXIT_ERROR
    finally:
        if not args.no_keep_awake:
            _keep_awake(False)
    q = Q.RunQueue(args.db)
    print(q.status_table())
    q.close()
    return rc_all


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="TPA run-queue worker launcher")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--phase", choices=["tune", "pilot", "main", "robust"], default=None,
                    help="only claim rows of this phase (default: any)")
    ap.add_argument("--gpu-workers", type=int, default=0)
    ap.add_argument("--arms", default="", help="comma list: claim only these arms (e.g. memory arms on GPU workers)")
    ap.add_argument("--skip-cpus", default="2", help="comma list of logical CPUs not to pin to. Default "
                    "'2': pilot measurement showed logical CPU 2 spending ~37%% DPC + ~17%% interrupt "
                    "time while GPU workers run (GPU driver DPC core), starving a worker pinned there")
    ap.add_argument("--requeue-failed", dest="requeue_failed", action="store_true",
                    help="move failed rows of --phase back to pending, then exit")
    ap.add_argument("--gpu-arms", default="", help="arms GPU workers may claim (overrides --arms for them)")
    ap.add_argument("--cpu-arms", default="", help="arms CPU workers may claim (overrides --arms for them)")
    ap.add_argument("--cores", choices=["p-ht", "p", "p-then-e", "e"], default="p-ht",
                    help="p-ht (default): all 16 P-core threads; E-cores are ~15x slower under the "
                         "Balanced plan and ~50x slower when P-cores are busy (measured 2026-10-27)")
    ap.add_argument("--pin", choices=["logical", "core"], default="logical",
                    help="pin to one logical CPU of the core, or to all its HT siblings")
    ap.add_argument("--db", type=Path, default=ROOT / "runs" / "runs.db")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--topology", type=Path, default=ROOT / "runs" / "cpu_topology.json")
    ap.add_argument("--stale-min", type=float, default=10.0)
    ap.add_argument("--heartbeat-s", type=float, default=60.0)
    ap.add_argument("--stagger-s", type=float, default=0.0,
                    help="seconds between worker starts (default 0: all at once)")
    ap.add_argument("--numa-alternate", action="store_true",
                    help="pin consecutive workers to alternating NUMA nodes (Linux sysfs; default off)")
    ap.add_argument("--max-runs", type=int, default=None, help="per worker")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the labelled stub instead of tpa.run.trainer.run (no training)")
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--no-keep-awake", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--requeue-stale", dest="requeue_stale_only", action="store_true")
    # internal (worker process)
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-id", default="w-manual", help=argparse.SUPPRESS)
    ap.add_argument("--cpus", default="", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return worker_main(args) if args._worker else launcher_main(args)


if __name__ == "__main__":
    sys.exit(main())
