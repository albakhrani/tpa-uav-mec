"""Extra main-campaign rows: 56 main-phase rows ADDED to the running campaign.

    python scripts/enqueue_main_extra.py --episodes 500 --dry-run
    python scripts/enqueue_main_extra.py --episodes 500

  - A5, A7 -> tier A: seeds 9..30 (22 each, 44 rows), the arm's selected configuration.
  - A15 at tier C: seeds 1..6, A0's selected configuration (A15 = A0 with shuffled write keys; never tuned).
  - A0 with t_mem_ms.A0 = 25.0 (T_mem sensitivity): seeds 1..6, A0's selected configuration,
    run ids main-A0-tmem25-s<seed>-sinusoidal (queue arm A0; the config hash differs by design).
Overrides = tuning.row_overrides(selected, E) (adopted scenario + selected hyperparameters + run.episodes),
plus the T_mem override for the T_mem rows. Phase main, workload sinusoidal, priority 0 (the main rows' priority),
enqueued seed by seed, so they are claimed after the pending main rows. Refuses unless exactly 56 new rows result,
none of the run ids exists, and the live code hash is the campaign's (EXPECT_CODE).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

EXPECT_CODE = "2892281e5ba9"
EXPECTED = 56
E_SET = (500, 600, 700, 800, 900, 1000)
WORKLOAD, PHASE = "sinusoidal", "main"
TMEM_OVERRIDE = {"t_mem_ms.A0": 25.0}


def plan(selected: dict, E: int) -> list[dict]:
    import tuning
    from tpa.config import load_config
    out = []
    for arm in ("A5", "A7"):
        ov = tuning.row_overrides(selected[arm], E)
        out.append(dict(arm=arm, label=arm, seeds=list(range(9, 31)), overrides=ov, config_hash=load_config(overrides=ov).hash(),
                        rid=lambda s, arm=arm: f"{PHASE}-{arm}-s{s}-{WORKLOAD}", note="A13.1 tier A, seeds 9-30"))
    ov = tuning.row_overrides(selected["A0"], E)
    out.append(dict(arm="A15", label="A15", seeds=list(range(1, 7)), overrides=ov, config_hash=load_config(overrides=ov).hash(),
                    rid=lambda s: f"{PHASE}-A15-s{s}-{WORKLOAD}", note="A13.2 tier C, A0's selected configuration"))
    ov2 = dict(ov); ov2.update(TMEM_OVERRIDE)
    out.append(dict(arm="A0", label="A0-tmem25", seeds=list(range(1, 7)), overrides=ov2, config_hash=load_config(overrides=ov2).hash(),
                    rid=lambda s: f"{PHASE}-A0-tmem25-s{s}-{WORKLOAD}", note="A13.3 T_mem sensitivity, t_mem_ms.A0 = 25.0"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--selected", type=Path, default=ROOT / "runs" / "tune" / "selected.json")
    ap.add_argument("--db", type=Path, default=ROOT / "runs" / "runs.db")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.episodes not in E_SET:
        raise SystemExit(f"--episodes {a.episodes} not in {E_SET}")
    from tpa.run.manifest import code_hash
    cc = code_hash()
    if not cc.startswith(EXPECT_CODE):
        raise SystemExit(f"live code hash {cc[:12]} != {EXPECT_CODE}")
    selected = json.loads(a.selected.read_text())
    groups = plan(selected, a.episodes)
    rows = []
    for g in groups:
        for s in g["seeds"]:
            rows.append((s, g))
    rows.sort(key=lambda x: (x[0], x[1]["label"]))                                # seed-major
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    existing = {r[0] for r in con.execute("SELECT run_id FROM runs")}
    a0 = con.execute("SELECT config_hash FROM runs WHERE run_id='main-A0-s1-sinusoidal'").fetchone()
    con.close()
    for g in groups:
        print(f"{g['label']:10} arm {g['arm']:4} seeds {g['seeds'][0]}-{g['seeds'][-1]} ({len(g['seeds'])}) config {g['config_hash']}  {g['note']}")
    print(f"A0 main rows carry config {a0[0] if a0 else None}; A0-tmem25 differs: {a0 is None or a0[0] != groups[3]['config_hash']}")
    dup = [g["rid"](s) for s, g in rows if g["rid"](s) in existing]
    print(f"total new rows {len(rows)} (expected {EXPECTED}); run.episodes = {a.episodes}; phase {PHASE}; already present: {dup}")
    if len(rows) != EXPECTED or dup:
        raise SystemExit("refused: row count or duplicates")
    if a.dry_run:
        print("dry run: nothing enqueued")
        return
    from tpa.run.queue import RunQueue
    q, n, ids = RunQueue(a.db), 0, []
    for s, g in rows:
        rid = g["rid"](s)
        n += q.enqueue(rid, g["arm"], s, WORKLOAD, PHASE, g["config_hash"], cc, overrides=g["overrides"], priority=0)
        ids.append(rid)
        time.sleep(0.02)
    print(f"enqueued {n} rows (code {cc[:12]})")
    print(q.status_table())
    q.close()


if __name__ == "__main__":
    main()
