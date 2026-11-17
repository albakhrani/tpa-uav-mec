"""Enqueue runs into runs/runs.db (one row per arm x seed x workload).

  python scripts/enqueue.py --phase pilot --arms A0,A3,A9,A12 --seeds 8001-8003 \
      --overrides run.episodes=100

Each row carries the effective config hash and the code hash at enqueue time; workers
refuse a row whose hashes no longer match (scripts/run_queue.py), so code must be frozen
between enqueue and completion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpa.config import load_config          # noqa: E402
from tpa.run.manifest import code_hash      # noqa: E402
from tpa.run.queue import RunQueue
from tuning import ADOPTED_FILE, MECHANISM_ONLY   # noqa: E402  (scripts/ on sys.path)          # noqa: E402


def _seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workload", default="sinusoidal")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--priority", type=int, default=0)
    ap.add_argument("--db", type=Path, default=ROOT / "runs" / "runs.db")
    ap.add_argument("--no-adopted", action="store_true",
                    help="do NOT merge the adopted D3 scenario (runs/d3/adopted.json); diagnostics only")
    a = ap.parse_args()
    arms = a.arms.split(",")
    bad = sorted(set(arms) & MECHANISM_ONLY)
    if bad:
        raise SystemExit(f"{bad} are mechanism-check arms (D7-3a) and cannot be enqueued")
    ov = dict(kv.split("=", 1) for kv in a.overrides)
    ov = {k: _value(v) for k, v in ov.items()}
    if not a.no_adopted:                                   # adopted scenario on every row
        ov = dict(json.loads(ADOPTED_FILE.read_text())["overrides"], **ov)
    cfg = load_config(overrides=ov) if ov else load_config()
    ch, cc = cfg.hash(), code_hash()
    q = RunQueue(a.db)
    n = 0
    for arm in arms:
        for s in _seeds(a.seeds):
            rid = f"{a.phase}-{arm}-s{s}-{a.workload.replace('|', '_')}"
            n += q.enqueue(rid, arm, s, a.workload, a.phase, ch, cc, overrides=ov or None,
                           priority=a.priority)
    print(f"enqueued {n} rows (config {ch}, code {cc[:12]})")
    print(q.status_table())
    q.close()


if __name__ == "__main__":
    main()
