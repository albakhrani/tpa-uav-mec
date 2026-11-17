"""Main-campaign enqueue (tiers A/B/C, training length E_main; 260 rows).

    python scripts/enqueue_main.py --episodes E_MAIN --dry-run       # prints the rows, touches nothing
    python scripts/enqueue_main.py --episodes E_MAIN                 # only once tuning and E_main are final

`scripts/enqueue.py` takes one override set per call and does not read the tuning result, so
this wrapper enqueues per arm: adopted scenario + the arm's selected hyperparameters
(`runs/tune/selected.json`, written by `tuning.py select`) + `run.episodes = E_MAIN`, exactly as
`tuning.row_overrides`. Tiers and test seeds: A n = 30, B n = 8, C n = 6; seeds 1..n;
workload sinusoidal; phase main; 5 x 30 + 13 x 8 + 1 x 6 = 260 rows. Run ids follow
`scripts/enqueue.py`: main-<arm>-s<seed>-sinusoidal. It refuses unless the total is 260.

Rows are enqueued seed by seed (seed 1 of every arm, then seed 2, ...), and workers claim in
enqueue order (priority, created_utc, run_id). The order changes no row and no result (runs are
independent and deterministic per seed); it makes the first runs cover every arm (so an early
check of failures and throughput sees all of them) and keeps the paired seeds of all arms
advancing together if the campaign is ever cut short.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TIERS = {
    "A": (30, ["A0", "A3", "A9", "A10", "A12"]),
    "B": (8, ["A1", "A2", "A4", "A5", "A6", "A7", "A8", "A11", "A14", "B_ZHOU", "B_QIN", "B_HUANG", "B_SONG"]),
    "C": (6, ["A13"]),
}
EXPECTED_ROWS = 260
E_SET = (500, 600, 700, 800, 900, 1000)
WORKLOAD, PHASE = "sinusoidal", "main"


def plan_rows(selected: dict, episodes: int) -> list[dict]:
    """One dict per arm: tier, seeds, overrides, config hash (no queue access)."""
    import tuning
    from tpa.config import load_config
    out = []
    for tier, (n, arms) in TIERS.items():
        for arm in arms:
            if arm not in selected:
                raise SystemExit(f"{arm}: no entry in the selected configurations")
            ov = tuning.row_overrides(selected[arm], episodes)
            out.append(dict(arm=arm, tier=tier, seeds=list(range(1, n + 1)), n=n, overrides=ov,
                            tuned=selected[arm], config_hash=load_config(overrides=ov).hash()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, required=True, help="E_main (A10.3)")
    ap.add_argument("--selected", type=Path, default=ROOT / "runs" / "tune" / "selected.json")
    ap.add_argument("--db", type=Path, default=ROOT / "runs" / "runs.db")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.episodes not in E_SET:
        raise SystemExit(f"--episodes {a.episodes} is not in {E_SET}")
    plan = plan_rows(json.loads(a.selected.read_text()), a.episodes)
    total = sum(p["n"] for p in plan)
    for p in plan:
        print(f"tier {p['tier']} {p['arm']:8} seeds 1-{p['n']:<2} rows {p['n']:3} config {p['config_hash']} tuned {json.dumps(p['tuned'])}")
    print(f"total rows {total} (expected {EXPECTED_ROWS}); run.episodes = {a.episodes}; workload {WORKLOAD}; phase {PHASE}")
    if total != EXPECTED_ROWS:
        raise SystemExit("row count is not 260: nothing enqueued")
    if a.dry_run:
        print("dry run: nothing enqueued")
        return
    from tpa.run.manifest import code_hash
    from tpa.run.queue import RunQueue
    q, cc, n = RunQueue(a.db), code_hash(), 0
    for s in range(1, max(p["n"] for p in plan) + 1):          # seed-major order
        for p in plan:
            if s <= p["n"]:
                n += q.enqueue(f"{PHASE}-{p['arm']}-s{s}-{WORKLOAD}", p["arm"], s, WORKLOAD, PHASE,
                               p["config_hash"], cc, overrides=p["overrides"], priority=0)
    print(f"enqueued {n} rows (code {cc[:12]})")
    print(q.status_table())
    q.close()


if __name__ == "__main__":
    main()
