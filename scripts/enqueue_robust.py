"""Robustness family enqueue (other workloads; A0, A3, A9, A10; test seeds 1..6), into its own queue file.

    python scripts/enqueue_robust.py --episodes E_MAIN --selected PATH --db PATH --dry-run
    python scripts/enqueue_robust.py --episodes E_MAIN --selected PATH --db PATH
    python scripts/enqueue_robust.py --family pareto --episodes E_MAIN --selected PATH --db PATH [--dry-run]

Family mmpp_gm (default): workloads MMPP ("mmpp") and Gauss-Markov user mobility
("sinusoidal|mobility=gauss_markov"); arms A0, A3, A9, A10; test seeds 1..6; 2 x 4 x 6 = 48 rows.
Family pareto: Pareto on-off ("pareto", config defaults) x A0, A3, A9, A10 x test seeds 1..6 = 24 rows;
checkpoints are selected on the sinusoidal validation block as for every row (disclosed in the paper).
Phase robust; each row = adopted scenario + the arm's selected hyperparameters + run.episodes = E_MAIN (as
tuning.row_overrides). The queue is a separate file (--db), never the main campaign's; nothing from it is
pooled with main-campaign runs. Run ids follow scripts/enqueue.py. Rows are enqueued seed by seed. It refuses
unless the total is the family's (48 or 24).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FAMILIES = {"mmpp_gm": (("mmpp", "sinusoidal|mobility=gauss_markov"), 48), "pareto": (("pareto",), 24)}
WORKLOADS = FAMILIES["mmpp_gm"][0]
ARMS = ("A0", "A3", "A9", "A10")
SEEDS = tuple(range(1, 7))
PHASE, EXPECTED = "robust", 48
E_SET = (500, 600, 700, 800, 900, 1000)


def main():
    global WORKLOADS, EXPECTED
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), default="mmpp_gm")
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    WORKLOADS, EXPECTED = FAMILIES[a.family]
    if a.episodes not in E_SET:
        raise SystemExit(f"--episodes {a.episodes} is not in {E_SET}")
    import tuning
    from tpa.config import load_config
    sel = json.loads(a.selected.read_text())
    plan = []
    for arm in ARMS:
        ov = tuning.row_overrides(sel[arm], a.episodes)
        plan.append(dict(arm=arm, overrides=ov, tuned=sel[arm], config_hash=load_config(overrides=ov).hash()))
    rows = [(s, wl, p) for s in SEEDS for wl in WORKLOADS for p in plan]          # seed-major order
    for p in plan:
        print(f"{p['arm']:4} config {p['config_hash']} tuned {json.dumps(p['tuned'])}")
    print(f"workloads {WORKLOADS}; seeds {SEEDS[0]}-{SEEDS[-1]}; total rows {len(rows)} (expected {EXPECTED}); "
          f"run.episodes = {a.episodes}; phase {PHASE}; queue {a.db}")
    if len(rows) != EXPECTED:
        raise SystemExit(f"row count is not {EXPECTED}: nothing enqueued")
    if a.dry_run:
        print("dry run: nothing enqueued")
        return
    from tpa.run.manifest import code_hash
    from tpa.run.queue import RunQueue
    a.db.parent.mkdir(parents=True, exist_ok=True)
    q, cc, n = RunQueue(a.db), code_hash(), 0
    import time
    ids = []
    for s, wl, p in rows:
        rid = f"{PHASE}-{p['arm']}-s{s}-{wl.replace('|', '_')}"
        n += q.enqueue(rid, p["arm"], s, wl, PHASE, p["config_hash"], cc, overrides=p["overrides"], priority=0)
        ids.append(rid)
        time.sleep(0.02)        # distinct created_utc on Windows too, so the claim order is the enqueue order
    order = [r[0] for r in q.con.execute("SELECT run_id FROM runs WHERE phase=? ORDER BY priority DESC, created_utc, run_id",
                                         (PHASE,))]
    print("claim order equals the seed-by-seed enqueue order:", order == ids)
    print(f"enqueued {n} rows (code {cc[:12]})")
    print(q.status_table())
    q.close()


if __name__ == "__main__":
    main()
