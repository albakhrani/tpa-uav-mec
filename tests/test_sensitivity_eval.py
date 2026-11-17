"""Tests of scripts/b1_eval.py (the evaluation-only sensitivity rows) on SYNTHETIC queues and VALIDATION data only
(validation seed 9001, validation block); no test-block result of the campaign is read.

- the settings table is the registered one and each setting reaches the evaluation clone, and nothing else
  changes there;
- the trace does not depend on any setting (world, streams; the simulator reads none of the keys);
- the 450 rows planned from a synthetic main queue, the refusals, the enqueue order;
- the claim gate "behind every main row" and the worker budget of the launcher (CPUs of TPA_CPUS only, at most
  MAX_TOTAL = 30 worker processes in total);
- end to end on validation data: a short A0 training run (tpa.run.trainer.run, phase tune, so its own evaluation is on
  the validation block val_full, 3 loads x 3 runs, heavy run k = 0 evaluated after light and medium), then
  eval_only() of its selected checkpoint on val_full run k = 0 at heavy load must give the SAME run row and minute rows
  (every column but the wall-clock ones), while a controller without the checkpoint does not.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import b1_eval as B  # noqa: E402
from tpa.config import load_config  # noqa: E402
from tpa.run.manifest import code_hash  # noqa: E402
from tpa.run.queue import RunQueue  # noqa: E402

MAIN_A0_OV = {"burst.theta_b": 0.9, "learner.lr": 0.00239, "learner.target_entropy_scale": 0.25,
              "learner.update_every": 8, "run.episodes": 500, "scenario.n_uav": 25, "task.deadline_s": [3.0, 10.0],
              "task.size_mb": [0.5, 2.5], "workload.loads": [0.17, 0.374, 0.578]}
VAL_SEED = 9001


def _get(cfg, key):
    sec, leaf = key.split(".")
    return getattr(getattr(cfg, sec), leaf)


# ---------------------------------------------------------------- settings
def test_settings_are_the_registered_ones():
    assert len(B.SETTINGS) == 14 and len(B.TAGS) == 15 and len(set(B.TAGS)) == 15
    assert B.EXPECTED_ROWS == 450 and B.SEEDS == tuple(range(1, 31))
    keys = {k for _, k, _ in B.SETTINGS}
    assert keys == set(B.MAIN_VALUES) and len(keys) == 7
    absolute = {"theta_p-0.30": 0.30, "theta_p-0.40": 0.40, "rho_merge-0.90": 0.90, "rho_merge-0.98": 0.98}
    for tag, key, val in B.SETTINGS:
        assert tag.split("-")[0] == key.split(".")[1]
        assert B.setting_of(tag) == (key, val)
        if tag in absolute:
            assert val == absolute[tag]
        else:
            f = {"x0.75": 0.75, "x1.25": 1.25}[tag.split("-")[1]]
            want = B.MAIN_VALUES[key] * f
            if key == "drm.omega_l":
                want = min(1.0, want)
            assert val == pytest.approx(want, abs=1e-12)
            assert repr(val) == repr(round(want, 6))              # a decimal literal, not a float product
    # A0's main value of theta_b is its tuned 0.9; every other key keeps default.toml
    cfg = load_config(overrides=MAIN_A0_OV)
    for key, v in B.MAIN_VALUES.items():
        assert _get(cfg, key) == pytest.approx(v)
    sel = json.loads((ROOT.parent / "gate2" / "selected.json").read_text()) if (ROOT.parent / "gate2" / "selected.json").exists() else None
    if sel is not None:
        assert sel["A0"]["burst.theta_b"] == 0.9


def test_run_ids_and_overrides():
    assert B.run_id(7, "default") == "b1-A0-s7-default"
    assert B.parse_run_id("b1-A0-s12-theta_p-0.30") == (12, "theta_p-0.30")
    with pytest.raises(ValueError):
        B.parse_run_id("main-A0-s1-sinusoidal")
    with pytest.raises(ValueError):
        B.parse_run_id("b1-A0-s1-theta_p-0.5")
    ov = B.b1_overrides(MAIN_A0_OV, "burst.theta_b", 0.675)
    assert ov["burst.theta_b"] == 0.675 and ov["run.test_runs"] == 1 and MAIN_A0_OV["burst.theta_b"] == 0.9
    assert B.b1_overrides(MAIN_A0_OV, None, None) == {**MAIN_A0_OV, "run.test_runs": 1}


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def test_each_b1_config_differs_from_the_main_config_in_the_setting_and_test_runs_only():
    base = _flat(load_config(overrides=MAIN_A0_OV).to_dict())
    for tag in B.TAGS:
        key, val = B.setting_of(tag)
        cfg = _flat(load_config(overrides=B.b1_overrides(MAIN_A0_OV, key, val)).to_dict())
        diff = {k for k in base if base[k] != cfg[k]}
        assert diff == ({"run.test_runs"} | ({key} if key else set())), tag
        assert cfg["run.test_runs"] == 1 and (key is None or cfg[key] == val)


# ---------------------------------------------------------------- the setting reaches the evaluation clone
def _clone_view(c):
    return dict(theta_p=c.mepr.theta, rho_merge=c.mem.rho, theta_b=c.drm.burst.theta_b, alpha_t=c.drm.a_T,
                alpha_m=c.drm.a_M, alpha_d=c.drm.a_D, omega_l=c.drm.omega_l, gamma_p=c.drm.gamma_p,
                beta_c=c.drm.beta_c, burst_on=c.drm.burst.on, has_mem=c.has_mem)


def test_each_setting_reaches_the_evaluation_clone_and_nothing_else_changes():
    from tpa.rng import Streams
    cfg0 = load_config(overrides=B.b1_overrides(MAIN_A0_OV, None, None))
    v0 = _clone_view(B.build_controller(cfg0, "A0", VAL_SEED).eval_clone(Streams(VAL_SEED, "val", 1)))
    assert v0["has_mem"] and v0["burst_on"]
    assert (v0["theta_p"], v0["rho_merge"], v0["theta_b"], v0["alpha_t"], v0["alpha_m"], v0["alpha_d"], v0["omega_l"]) \
        == (0.35, 0.95, 0.9, 0.4, 0.3, 0.3, 0.6)
    for tag, key, val in B.SETTINGS:
        cfg = load_config(overrides=B.b1_overrides(MAIN_A0_OV, key, val))
        v = _clone_view(B.build_controller(cfg, "A0", VAL_SEED).eval_clone(Streams(VAL_SEED, "val", 1)))
        name = key.split(".")[1]
        assert v[name] == val, tag
        assert {k: x for k, x in v.items() if k != name} == {k: x for k, x in v0.items() if k != name}, tag


def test_the_trace_does_not_depend_on_any_setting():
    from tpa.env.world import build_world
    from tpa.rng import Streams
    names = ("theta_p", "rho_merge", "theta_b", "alpha_t", "alpha_m", "alpha_d", "omega_l")
    for p in list((ROOT / "tpa" / "env").glob("*.py")) + [ROOT / "tpa" / "rng.py"]:
        txt = p.read_text(encoding="utf-8")
        assert not any(n in txt for n in names), p
    ref = None
    for tag in B.TAGS:
        key, val = B.setting_of(tag)
        w = build_world(load_config(overrides=B.b1_overrides(MAIN_A0_OV, key, val)), Streams(VAL_SEED, "train").world)
        view = (np.asarray(w.zone_phase).tobytes(), np.asarray(w.f_max).tobytes(), np.asarray(w.eta).tobytes())
        ref = ref or view
        assert view == ref, tag


# ---------------------------------------------------------------- planning and enqueue (synthetic main queue)
def _main_queue(tmp_path, n_seeds=30, ov=None, code=None, pending_other=True):
    db = tmp_path / "runs" / "runs.db"
    q = RunQueue(db)
    cc = code or code_hash()
    for s in range(1, n_seeds + 1):
        o = dict(ov or MAIN_A0_OV)
        q.enqueue(B.SOURCE_RUN.format(seed=s), "A0", s, "sinusoidal", "main", load_config(overrides=o).hash(), cc,
                  overrides=o)
        q.enqueue(f"main-A0-tmem25-s{s}-sinusoidal", "A0", s, "sinusoidal", "main", "x", cc, overrides={"x": 1})
    if pending_other:
        q.enqueue("main-A3-s1-sinusoidal", "A3", 1, "sinusoidal", "main", "y", cc, overrides={})
    q.close()
    return db


def test_plan_rows_from_the_main_queue(tmp_path):
    db = _main_queue(tmp_path)
    rows, base = B.plan_rows(db, code_hash())
    assert base == MAIN_A0_OV and len(rows) == 450 and len({r["run_id"] for r in rows}) == 450
    assert [r["run_id"] for r in rows[:16]] == [B.run_id(1, t) for t in B.TAGS] + [B.run_id(2, "default")]
    for r in rows:
        assert r["config_hash"] == load_config(overrides=r["overrides"]).hash()
        assert r["loads"] == [B.HEAVY] and r["arm"] == "A0" and r["phase"] == "main" and r["workload"] == "sinusoidal"
        assert r["overrides"] == B.b1_overrides(MAIN_A0_OV, r["key"], r["value"])
    assert len({r["config_hash"] for r in rows}) == 15


def test_plan_refuses_inconsistent_main_rows(tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        B.plan_rows(_main_queue(tmp_path / "a", n_seeds=29), code_hash())
    with pytest.raises(SystemExit, match="code hashes"):
        B.plan_rows(_main_queue(tmp_path / "b", code="0" * 64), code_hash())
    db = _main_queue(tmp_path / "c")
    con = sqlite3.connect(db)
    con.execute("UPDATE runs SET overrides=? WHERE run_id='main-A0-s5-sinusoidal'",
                (json.dumps({**MAIN_A0_OV, "learner.lr": 0.001}, sort_keys=True),))
    con.commit(); con.close()
    with pytest.raises(SystemExit, match="one set of overrides"):
        B.plan_rows(db, code_hash())
    with pytest.raises(SystemExit, match="registered"):
        B.plan_rows(_main_queue(tmp_path / "d", ov={**MAIN_A0_OV, "burst.theta_b": 0.75}), code_hash())


def test_enqueue_and_the_gate_behind_every_main_row(tmp_path):
    main_db = _main_queue(tmp_path)
    runs = main_db.parent
    rows, _ = B.plan_rows(main_db, code_hash())
    b1_db = tmp_path / "runs" / "b1" / "runs_b1.db"
    n, same = B.enqueue(rows, b1_db)
    assert n == 450 and same
    assert B.enqueue(rows[:3], b1_db)[0] == 0                      # idempotent: existing run ids are not re-inserted
    # a main row is pending: nothing is claimable, whatever the A0 rows' states
    con = sqlite3.connect(main_db)
    con.execute("UPDATE runs SET state='done' WHERE run_id='main-A0-s1-sinusoidal'")
    con.execute("UPDATE runs SET state='running' WHERE run_id<>'main-A3-s1-sinusoidal' AND state='pending'")
    con.commit()
    d = runs / "main" / "main-A0-s1-sinusoidal"
    d.mkdir(parents=True)
    (d / "ckpt_best.pt").write_bytes(b"x")
    (d / "RESULT.json").write_text("{}")
    n_pend, n_run, ready = B.main_gate(main_db, runs)
    assert n_pend == 1 and ready == {1}
    q = RunQueue(b1_db)
    assert B.claim_ready(q, "w", ready if n_pend == 0 else ()) is None
    (d / "RESULT.json").unlink()
    # every main row claimed, but seed 1's source has RESULT.json and no ckpt_best.pt: not final
    con.execute("UPDATE runs SET state='running' WHERE run_id='main-A3-s1-sinusoidal'")
    con.commit()
    (d / "ckpt_best.pt").unlink()
    (d / "RESULT.json").write_text("{}")
    assert B.main_gate(main_db, runs) == (0, 60, set())
    (d / "ckpt_best.pt").write_bytes(b"x")
    n_pend, n_run, ready = B.main_gate(main_db, runs)
    assert (n_pend, ready) == (0, {1}) and B.n_ready_pending(b1_db, ready) == 15
    got = [B.claim_ready(q, "w", ready)["run_id"] for _ in range(15)]
    assert got == [B.run_id(1, t) for t in B.TAGS]
    assert B.claim_ready(q, "w", ready) is None                    # seed 2's source is not done
    r = q.get(B.run_id(1, "default"))
    assert r["state"] == "running" and r["pid"] == os.getpid() and r["worker"] == "w" and r["attempts"] == 1
    # a 'stopped by' pending main row does not hold the sensitivity rows back; the tmem25 rows are never sources
    con.execute("UPDATE runs SET state='pending', error='stopped by test' WHERE run_id='main-A3-s1-sinusoidal'")
    con.execute("UPDATE runs SET state='done' WHERE run_id LIKE 'main-A0-tmem25-%'")
    con.commit(); con.close()
    assert B.main_gate(main_db, runs)[0] == 0 and B.main_gate(main_db, runs)[2] == {1}
    q.close()
    assert B.b1_counts(b1_db) == dict(pending=435, running=15, done=0, failed=0, stopped=0)


def test_spawn_plan_keeps_to_tpa_cpus():
    live30 = {i: dict(kind="main", cpus={c}) for i, c in enumerate(list(range(1, 16)) + list(range(32, 47)))}
    assert B.spawn_plan(live30, 300, 200.0, 500.0) == []                       # 30 worker processes: none added
    live25 = dict(list(live30.items())[:25])
    got = B.spawn_plan(live25, 300, 200.0, 500.0)
    assert len(got) == 5 and all(c in B.TPA_CPUS for c in got) and not set(got) & set().union(*(w["cpus"] for w in live25.values()))
    assert B.spawn_plan(live25, 2, 200.0, 500.0) == got[:2]                    # no more than the claimable rows
    assert len(B.spawn_plan(live25, 300, B.MEM_FLOOR_GB + 2 * B.WORKER_GB + 0.1, 500.0)) == 2   # memory floor
    assert B.spawn_plan(live25, 300, 200.0, B.DISK_FLOOR_GB - 1) == []        # disk floor
    unpinned = {**live25, 99: dict(kind="main", cpus=set(range(64)))}
    assert B.spawn_plan(unpinned, 300, 200.0, 500.0) == []                    # an unpinned worker blocks every CPU
    got = B.spawn_plan({}, 300, 400.0, 500.0)
    assert len(got) == 30 and set(got) <= set(B.TPA_CPUS) and not set(got) & (set(range(21, 32)) | set(range(52, 64)) | {0})
    assert B.parse_cpulist("1-3,7,32-33") == {1, 2, 3, 7, 32, 33}


def test_classify_counts_only_tpa_workers():
    trees = {"/home/u/tpa/tpa_v2"}
    w = dict(exe="/usr/bin/python3.12", cwd="/home/u/tpa/tpa_v2", cpus={3},
             argv=["/home/u/tpa/tpa_v2/.venv/bin/python", "/home/u/tpa/tpa_v2/scripts/run_queue.py", "--_worker"])
    assert B.classify(w, trees) == "main"
    assert B.classify({**w, "argv": ["python", "/home/u/tpa/tpa_v2/scripts/b1_eval.py", "--_worker"]}, trees) == "b1"
    assert B.classify({**w, "cwd": "/home/u/ccp"}, trees) is None             # a process outside the code trees
    assert B.classify({**w, "argv": w["argv"][:2] + ["--workers", "20"]}, trees) is None   # a launcher
    assert B.classify({**w, "exe": "/usr/bin/bash"}, trees) is None
    assert B.classify(None, trees) is None


def test_arguments():
    assert B.parse_args(["--status"]).status
    with pytest.raises(SystemExit):
        B.parse_args(["--launch", "--max-total", "31"])
    with pytest.raises(SystemExit):
        B.parse_args(["--launch", "--status"])
    with pytest.raises(SystemExit):
        B.parse_args([])
    assert B.parse_args(["--enqueue", "--dry-run"]).dry_run and B.parse_args(["--launch", "--dry-run"]).dry_run
    for bad in (["--release-dead", "--dry-run"], ["--status", "--dry-run"]):              # --dry-run elsewhere: refused
        with pytest.raises(SystemExit):
            B.parse_args(bad)
    for bad in (["--release-dead", "--db", str(ROOT / "runs" / "runs.db")],               # never the main queue
                ["--launch", "--db", "x/runs_robust.db"],
                ["--release-dead", "--db", "q/runs_b1.db", "--main-db", "q/runs_b1.db"]):
        with pytest.raises(SystemExit):
            B.parse_args(bad)


def test_main_refuses_a_missing_or_short_queue(tmp_path):
    """--launch / --release-dead never create the queue; --launch needs all 450 rows."""
    db = tmp_path / "runs_b1.db"
    for argv in (["--launch"], ["--release-dead"]):
        with pytest.raises(SystemExit, match="no B1 queue"):
            B.main(argv + ["--db", str(db)])
    assert not db.exists()
    q = RunQueue(db)
    q.enqueue(B.run_id(1, "default"), "A0", 1, "sinusoidal", "main", "c", "k")
    q.close()
    with pytest.raises(SystemExit, match="not 450"):
        B.main(["--launch", "--db", str(db)])


def test_queue_file_with_non_b1_rows_is_refused(tmp_path):
    db = tmp_path / "runs_b1.db"
    q = RunQueue(db)
    q.enqueue("main-A0-s1-sinusoidal", "A0", 1, "sinusoidal", "main", "x", "y")
    q.close()
    with pytest.raises(SystemExit, match="not B1 rows"):
        B.check_b1_only(db)
    with pytest.raises(SystemExit, match="not B1 rows"):
        B.main(["--release-dead", "--db", str(db)])
    B.check_b1_only(tmp_path / "absent" / "runs_b1.db")                          # nothing to check yet


def test_housekeeping_returns_rows_of_exited_workers_and_stale_rows(tmp_path):
    db = tmp_path / "runs_b1.db"
    q = RunQueue(db)
    for s in (1, 2, 3):
        q.enqueue(B.run_id(s, "default"), "A0", s, "sinusoidal", "main", "c", "k")
    for s, w in ((1, "b1w000-x"), (2, "b1w001-x"), (3, "b1w002-x")):
        assert B.claim_ready(q, w, {s})["run_id"] == B.run_id(s, "default")
    con = sqlite3.connect(db)                                                     # seed 3's heartbeat is stale
    con.execute("UPDATE runs SET heartbeat_utc='2000-01-01T00:00:00+00:00' WHERE seed=3")
    con.commit(); con.close()
    back = B.housekeeping(db, ["b1w000-x"], stale_min=10)                         # b1w000 exited; b1w001 is alive
    assert sorted(back) == [B.run_id(1, "default"), B.run_id(3, "default")]
    st = {r["run_id"]: r["state"] for r in q.rows()}
    assert st == {B.run_id(1, "default"): "pending", B.run_id(2, "default"): "running", B.run_id(3, "default"): "pending"}
    q.close()


def test_stopped_rows_are_never_claimed_and_counted_apart(tmp_path):
    db = tmp_path / "runs_b1.db"
    q = RunQueue(db)
    for s in (1, 2):
        q.enqueue(B.run_id(s, "default"), "A0", s, "sinusoidal", "main", "c", "k")
    con = sqlite3.connect(db)
    con.execute("UPDATE runs SET error='stopped by a ruling (test)' WHERE seed=1")
    con.commit(); con.close()
    assert B.n_ready_pending(db, {1, 2}) == 1
    assert B.claim_ready(q, "w", {1, 2})["run_id"] == B.run_id(2, "default")
    assert B.claim_ready(q, "w", {1, 2}) is None
    assert q.get(B.run_id(1, "default"))["error"].startswith("stopped by")      # the note is kept
    assert B.b1_counts(db) == dict(pending=0, running=1, done=0, failed=0, stopped=1)
    q.close()


def test_overbooked_makes_a_b1_worker_yield():
    live = {i: dict(kind="main", cpus={c}) for i, c in enumerate(range(1, 21))}
    me = {100: dict(kind="b1", cpus={32})}
    assert B.overbooked({**live, **me}, 100, {32}) is None                       # 21 workers, own CPU
    assert "shared" in B.overbooked({**live, **me, 7: dict(kind="main", cpus={32})}, 100, {32})
    many = {i: dict(kind="main", cpus={i}) for i in range(1000, 1030)}
    assert "> 30" in B.overbooked({**many, **me}, 100, {32})                      # 31 worker processes
    assert "shared" in B.overbooked({**me, 5: dict(kind="main", cpus=set(range(64)))}, 100, {32})   # unpinned worker


# ---------------------------------------------------------------- end to end on validation data
def _runs(d: Path, block: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    df = pa.concat_tables([pq.read_table(f) for f in sorted((d / block).glob("*.parquet"))],
                          promote_options="default").to_pandas()
    return df


def _comparable(df):
    drop = [c for c in df.columns if c in B.WALL_COLS or (c.startswith("t_") and c.endswith("_ms"))]
    return df.drop(columns=drop)


def _heavy0(df):
    sel = df[np.isclose(df["load"].astype(float), B.HEAVY) & (df["test_run"] == 0)]
    return sel.sort_values(["row_kind", "t_end"]).reset_index(drop=True)


def test_eval_only_reproduces_the_trainers_own_evaluation_on_validation_data(tmp_path):
    from tpa.run import trainer as T
    from tpa.run.evaluator import evaluate
    short = {"run.episodes": 2, "run.val_every_episodes": 1, "run.val_slots": 300, "run.test_slots": 600,
             "run.ckpt_every_episodes": 1, "learner.warmup_slots": 200}
    src_ov = {**MAIN_A0_OV, **short}
    cfg = load_config(overrides=src_ov)
    src_dir = tmp_path / "src"
    t0 = time.perf_counter()
    T.run(dict(run_id="tune-b1check-A0-s9001", arm="A0", seed=VAL_SEED, workload="sinusoidal", run_dir=str(src_dir),
               phase="tune"), cfg)
    t_train = time.perf_counter() - t0
    ref = _heavy0(_runs(src_dir, "val_full"))
    assert len(ref) == 1 + 600 // 60 and (ref["row_kind"] == "run").sum() == 1
    cc = code_hash()
    src = dict(run_id="tune-b1check-A0-s9001", state="done", overrides=src_ov, code_hash=cc, run_dir=str(src_dir))
    run_r = ref[ref["row_kind"] == "run"].iloc[0]
    print(f"\n[sensitivity check] reference heavy run: completion {run_r['completion']:.4f}, cons_rate {run_r['cons_rate']:.4f}, "
          f"sigma_mean {run_r.get('sigma_mean', float('nan')):.4f}")
    changed = {}
    for tag in ("default", "theta_b-x0.75", "alpha_t-x1.25", "omega_l-x0.75"):
        key, val = B.setting_of(tag)
        ov = B.b1_overrides(src_ov, key, val)
        row = dict(run_id=B.run_id(VAL_SEED, tag), arm="A0", seed=VAL_SEED, workload="sinusoidal", phase="main",
                   overrides=ov, code_hash=cc, loads=[B.HEAVY], run_dir=str(tmp_path / tag))
        t1 = time.perf_counter()
        res = B.eval_only(row, load_config(overrides=ov), src, block="val_full")
        got = _heavy0(_runs(tmp_path / tag, "val_full"))
        man = json.loads((tmp_path / tag / "manifest.json").read_text())
        assert man["extra"]["b1"]["source_ckpt_sha256"] == B.sha256_file(src_dir / "ckpt_best.pt") \
            == res["b1"]["source_ckpt_sha256"]
        assert man["config_hash"] == load_config(overrides=ov).hash() and man["code_hash"] == cc
        assert res["test_rows"] == len(got)
        print(f"\n[sensitivity check] train+val_full {t_train:.0f} s; eval_only {tag}: {time.perf_counter() - t1:.1f} s for 600 slots")
        a, b = _comparable(ref), _comparable(got)
        assert list(a.columns) == list(b.columns)
        if tag == "default":
            for col in a.columns:                                   # bit-identical, NaN == NaN
                x, y = a[col].to_numpy(), b[col].to_numpy()
                same = [(u == v) or (isinstance(u, float) and isinstance(v, float) and np.isnan(u) and np.isnan(v))
                        for u, v in zip(x, y)]
                assert all(same), (col, x[:3], y[:3])
        else:                                                       # a setting acting on every slot changes the run
            run_a, run_b = a[a["row_kind"] == "run"].iloc[0], b[b["row_kind"] == "run"].iloc[0]
            diff = [c for c in a.columns if isinstance(run_a[c], (float, np.floating)) and np.isfinite(run_a[c])
                    and run_a[c] != run_b[c]]
            print(f"[sensitivity check] {tag}: run-row columns that differ from the default: {diff[:8]}")
            changed[tag] = bool(diff)
    # the row's config reaches the evaluation (not the source's): a setting acting on every slot changes the run. A
    # setting can leave a barely trained policy's run unchanged (e.g. no load pressure above either burst threshold),
    # so at least one of the three must; that each setting reaches the clone is the attribute test above.
    assert any(changed.values()), changed
    # negative control: the same evaluation WITHOUT the checkpoint's learned state differs
    cfg_d = load_config(overrides=B.b1_overrides(src_ov, None, None))
    fresh = B.build_controller(cfg_d, "A0", VAL_SEED)
    rows = evaluate(fresh, cfg_d, VAL_SEED, "A0", loads=[B.HEAVY], block="val_full")
    run_f = [r for r in rows if r["row_kind"] == "run"][0]
    diff_f = [c for c in ("completion", "wait_s", "cpu_util", "cons_rate", "sigma_mean")
              if c in run_f and np.isfinite(run_f[c]) and run_f[c] != run_r[c]]
    print(f"[sensitivity check] without the checkpoint, differing: {diff_f}")
    assert diff_f


# ---------------------------------------------------------------- the launcher loop, simulated (no process, no /proc)
class _FakeProc:
    n = 0

    def __init__(self, life):
        _FakeProc.n += 1
        self.pid, self.life, self.returncode, self.terminated = 90000 + _FakeProc.n, life, None, False

    def poll(self):
        self.life -= 1
        if self.life < 0 and self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated, self.returncode = True, -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _sim_launcher(tmp_path, monkeypatch, fails, max_hours=0.5, n_spawn=3, life=40, max_passes=5000):
    """Run launcher_main with a fake clock and fake workers. fails(i) says whether the i-th housekeeping call (1-based)
    raises 'database is locked'. The time limit (max_hours) falls while the workers still run, so the launcher drains.
    max_passes bounds the simulation (a test failure, not a hang, if the launcher never exits)."""
    import argparse
    import sqlite3 as _sq
    clock = {"t": 1_000_000.0, "passes": 0}
    calls, procs, released = [], [], []

    def sleep(s):
        clock["t"] += s
        clock["passes"] += 1
        if clock["passes"] > max_passes:
            raise RuntimeError("simulated launcher did not exit")
    monkeypatch.setattr(B.time, "time", lambda: clock["t"])
    monkeypatch.setattr(B.time, "sleep", sleep)

    def hk(db, exited, stale_min):
        calls.append(list(exited))
        if fails(len(calls)):
            raise _sq.OperationalError("database is locked")
        return []
    monkeypatch.setattr(B, "housekeeping", hk)
    alive = lambda: any(p.returncode is None for p in procs)                      # noqa: E731
    monkeypatch.setattr(B, "b1_counts", lambda db: dict(pending=5, running=3 if alive() else 0, done=0, failed=0, stopped=0))
    monkeypatch.setattr(B, "main_gate", lambda db, root: (0, 0, {1}))
    monkeypatch.setattr(B, "n_ready_pending", lambda db, seeds: n_spawn if not procs else 0)
    monkeypatch.setattr(B, "live_tpa_workers", lambda home: {})
    monkeypatch.setattr(B, "_mem_available_gb", lambda: 200.0)
    monkeypatch.setattr(B, "_disk_free_gb", lambda p: 500.0)

    def popen(cmd, **kw):
        assert kw.get("start_new_session") is True and "--_worker" in cmd
        procs.append(_FakeProc(life))
        return procs[-1]
    monkeypatch.setattr(B.subprocess, "Popen", popen)

    class _Q:
        def __init__(self, db, *a, **k):
            pass

        def release_worker(self, wid):
            released.append(wid)
            return []

        def close(self):
            pass
    import tpa.run.queue as Qmod
    monkeypatch.setattr(Qmod, "RunQueue", _Q)
    a = argparse.Namespace(home_tpa=tmp_path, runs_root=tmp_path / "b1", db=tmp_path / "runs_b1.db", main_db=tmp_path / "runs.db",
                           main_runs_root=tmp_path, max_total=30, poll_s=120.0, max_hours=max_hours, stale_min=10.0,
                           heartbeat_s=60.0, dry_run=False)
    rc = B.launcher_main(a)
    return rc, calls, procs, released


def test_launcher_drain_survives_intermittent_failures(tmp_path, monkeypatch):
    """During a long drain (time limit at pass 15, workers finish after 200 passes) more than 30 housekeeping calls
    fail in all, never 30 in a row: the launcher waits for its workers and exits with 3, it does not end with 4 and
    terminate them (every successful pass of the drain resets the failure counter)."""
    for k in (3, 2):                                        # every 3rd / every 2nd housekeeping call fails
        rc, calls, procs, released = _sim_launcher(tmp_path / str(k), monkeypatch, fails=lambda i, k=k: i % k == 0, life=200)
        n_fail = sum(1 for i in range(1, len(calls) + 1) if i % k == 0)
        assert n_fail > 30, (k, n_fail)                     # more failures in all than the limit of 30 in a row
        assert rc == 3, (k, rc)
        assert len(procs) == 3 and not any(p.terminated for p in procs)


def test_launcher_passes_a_worker_again_when_its_exit_pass_fails(tmp_path, monkeypatch):
    """The workers exit in a pass whose housekeeping fails; their ids reach the next successful call (they stay in
    the launcher's to_release set until a housekeeping call succeeds)."""
    hit = 0
    for life in range(195, 201):
        rc, calls, procs, released = _sim_launcher(tmp_path / str(life), monkeypatch, fails=lambda i: i % 2 == 0, life=life)
        assert rc == 3
        failed_exit = [c for i, c in enumerate(calls, 1) if i % 2 == 0 and c]
        if failed_exit:
            hit += 1
            later_ok = {w for i, c in enumerate(calls, 1) if i % 2 == 1 and c for w in c}
            assert set(failed_exit[0]) <= later_ok, (life, calls[-6:])
    assert hit > 0                                          # the scenario did occur


def test_launcher_ends_after_30_failures_in_a_row_and_releases_its_workers(tmp_path, monkeypatch):
    rc, calls, procs, released = _sim_launcher(tmp_path, monkeypatch, fails=lambda i: i > 2, life=10 ** 6, max_hours=10 ** 6)
    assert rc == 4 and len(calls) == 32                     # 2 good passes (3 workers started), then 30 failures in a row
    assert len(procs) == 3 and all(p.terminated for p in procs)
    stamp = released[0].split("-", 1)[1]
    assert sorted(released) == [f"b1w{i:03d}-{stamp}" for i in range(3)]   # all three returned to pending
