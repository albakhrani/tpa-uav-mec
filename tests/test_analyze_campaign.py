"""Unit tests of scripts/a16_partA.py (per-cycle statistics, P10 on MMPP) and of scripts/analyze_campaign.py, on
SYNTHETIC inputs only (no run of the campaign is read).

Known answers: the frozen P10 clustering ratio on a hand-made gate / state sequence; the MMPP base share against a
simulated two-state chain; the phase-bin geometry of the 60-slot minute rows; per-cycle completion, open slots, open
bins, interval counts and LT-occupancy bounds on a synthetic run with a known occupancy path; kappa_phi of seed 9003
(0.4026, the value given for Fig. 3(b)); and an end-to-end run of analyze_campaign.py on a synthetic queue and
synthetic evaluation tables (main and robust phases), including the guards, the A0-tmem25 label, the tier-C and
exploratory families (outside Holm; no p-values except the paired t of A0 - A0-tmem25 at heavy load) and S1
without A13.
"""
from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import a16_partA as PA  # noqa: E402
from tpa.arms import ARMS  # noqa: E402


# ---------------------------------------------------------------- P10 definitions
def test_p10_clustering_ratio_known_answer():
    n = 1000
    ch = np.zeros(n, bool); ch[100] = True                 # one level change at slot 100: slots 101..200 are "near"
    op = np.zeros(n, bool); op[[150, 160, 500, 600]] = True
    assert PA.p10_clustering_ratio(op, ch, 100) == pytest.approx(0.5 / 0.1)
    assert PA.p10_clustering_ratio(np.ones(n, bool), ch, 100) == pytest.approx(1.0)
    assert math.isnan(PA.p10_clustering_ratio(np.zeros(n, bool), ch, 100))
    near = PA.within_after(ch, 100)
    assert not near[100] and near[101] and near[200] and not near[201]      # 0 < t - c <= w
    assert math.isnan(PA.p10_clustering_ratio(op, np.zeros(n, bool), 100))


def test_level_change_mask_counts_simultaneous_changes_once():
    s = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [1, 0]])
    assert PA.level_change_mask(s).tolist() == [False, False, True, False, True]


def test_mmpp_base_share_formula_and_campaign_value():
    rng = np.random.default_rng(12345)
    p = (0.98, 0.95)
    n, w = 200_000, 10
    st, x = np.empty(n, int), 0
    for t in range(n):
        if rng.random() > p[x]:
            x = 1 - x
        st[t] = x
    emp = PA.within_after(PA.level_change_mask(st), w)[1000:].mean()
    assert emp == pytest.approx(PA.mmpp_base_share(p, w, 1), abs=0.01)
    v = PA.mmpp_base_share(p, 100, 6)                       # the campaign's MMPP: the frozen ratio cannot discriminate
    assert 1 - 1e-6 < v < 1.0


def test_post_warmup_share():
    assert PA.post_warmup_share(0.1, 10_800) == pytest.approx(1080 / 10_700)


# ---------------------------------------------------------------- phase-bin geometry
def test_minute_bins_straddlers():
    straddle = {j: PA.minute_bins(j) for j in range(60) if PA.minute_bins(j)[0] != PA.minute_bins(j)[1]}
    assert set(straddle) == {7, 22, 37, 52}
    assert all(v[2] == 0.5 for v in straddle.values())
    assert [PA.minute_bins(j)[0] for j in (0, 6, 8, 14, 15, 59)] == [0, 0, 1, 1, 2, 7]


def test_short_term_fill_mean():
    assert PA.short_term_fill_mean(0) == pytest.approx(30.5)
    assert PA.short_term_fill_mean(1) == pytest.approx(87.0)
    assert PA.short_term_fill_mean(2) == pytest.approx(100.0)


def test_count_circular_intervals():
    assert PA.count_circular_intervals([1, 1, 0, 0, 0, 0, 1, 0]) == 2
    assert PA.count_circular_intervals([1, 0, 0, 0, 0, 0, 0, 1]) == 1
    assert PA.count_circular_intervals([1] * 8) == 1
    assert PA.count_circular_intervals([0] * 8) == 0
    assert PA.count_circular_intervals([1, 0, 1, 0, 1, 0, 0, 0]) == 3


def _synthetic_run(t0=10_000_000, flags=ARMS["A0"]):
    """180 minute rows with known completion (0.5 / 0.6 / 0.7 per cycle), openings and occupancy path."""
    import pandas as pd
    L_path = np.array([0] * 10 + [3] * 50 + [5] * 60 + [8] * 60, float)     # minute-mean LT occupancy
    opens = np.zeros(180)
    opens[60 + 15:60 + 30] = 30            # cycle 2: minutes 15..29 of the cycle (bins 2, 3; minute 22 straddles 2|3)
    opens[60 + 52] = 60                    # cycle 2: minute 52 straddles bins 6 | 7
    rows = []
    for j in range(180):
        c = j // 60
        s = PA.short_term_fill_mean(j) if flags.use_ms else 0.0
        mu = (s + (L_path[j] if flags.use_ml and not flags.flat else 0.0)) / (PA.K_S + PA.K_L)
        rows.append(dict(t_end=t0 + 60 * j + 59, arrivals=100.0, completed=50.0 + 10 * c, cons_rate=opens[j] / 60, mem_util=mu))
    return pd.DataFrame(rows), dict(lt_occupancy=8.0)


def test_cycle_rows_known_answers():
    m, run = _synthetic_run()
    out = PA.cycle_rows(m, run, ARMS["A0"])
    assert [r["completion"] for r in out] == pytest.approx([0.5, 0.6, 0.7])
    c2 = out[1]
    assert c2["open_slots"] == 15 * 30 + 60
    assert c2["open_share"] == pytest.approx((15 * 30 + 60) / 3600)
    assert out[0]["open_share_postwarmup"] == 0.0
    # bins: minutes 15..21 -> bin 2 (7 x 30), minute 22 split 15 | 15, minutes 23..29 -> bin 3 (7 x 30), minute 52 split 30 | 30
    assert c2["bin2_open"] == pytest.approx(7 * 30 + 15) and c2["bin3_open"] == pytest.approx(7 * 30 + 15)
    assert c2["bin2_lo"] == pytest.approx(7 * 30) and c2["bin2_hi"] == pytest.approx(8 * 30)     # minute 22: 30 open of 60
    assert c2["bin6_open"] == pytest.approx(30) and c2["bin7_open"] == pytest.approx(30)
    # sharp straddler bounds: each half of minute 52 holds at most 30 slots, and all 60 were open
    assert c2["bin6_lo"] == pytest.approx(30) and c2["bin6_hi"] == pytest.approx(30)
    assert c2["n_intervals_certain"] == 2 and c2["n_intervals_5pct"] == 2      # {2, 3} and {6, 7}
    assert out[0]["no_opening"] and not c2["no_opening"] and not c2["all_bins_open_5pct"]
    assert PA.interval_category(2, False, False) == "2" and PA.interval_category(5, False, False) == ">=3"
    assert PA.interval_category(1, True, False) == "none" and PA.interval_category(1, False, True) == "all8"
    # occupancy: bounds for cycles 1-2 from the minute means around the boundary; cycle 3 exact from the run row
    assert (out[0]["lt_end_lo"], out[0]["lt_end_hi"]) == pytest.approx((3.0, 5.0))
    assert (out[1]["lt_end_lo"], out[1]["lt_end_hi"]) == pytest.approx((5.0, 8.0))
    assert out[2]["lt_end_lo"] == out[2]["lt_end_hi"] == 8.0 and out[2]["lt_check_ok"]
    assert (out[2]["lt_opened_c23_lo"], out[2]["lt_opened_c23_hi"]) == pytest.approx((3.0, 5.0))
    assert PA.pooled_completion(out) == pytest.approx(0.65)


def test_cycle_rows_arm_variants_and_bad_input():
    m, run = _synthetic_run(flags=ARMS["A2"])                     # no short-term buffer: L = 600 * mem_util
    assert PA.cycle_rows(m, run, ARMS["A2"])[0]["lt_end_hi"] == pytest.approx(5.0)
    m, run = _synthetic_run(flags=ARMS["A3"])                     # flat store: no M_L, even if the run row says 0
    out3 = PA.cycle_rows(m, dict(lt_occupancy=0.0), ARMS["A3"])
    assert all(math.isnan(r["lt_end_lo"]) for r in out3) and math.isnan(out3[2]["lt_opened_c23_lo"])
    m, run = _synthetic_run(flags=ARMS["A1"])                     # no M_L: occupancy 0
    assert PA.cycle_rows(m, dict(lt_occupancy=0.0), ARMS["A1"])[0]["lt_end_lo"] == 0.0
    with pytest.raises(ValueError):
        PA.cycle_rows(m.iloc[:-1], run, ARMS["A0"])


def test_lt_snaps_to_the_minute_grid_exactly_19_opened():
    """Float noise of mem_util must not turn exactly 19 rows opened in cycles 2-3 into '> 19'."""
    import pandas as pd
    L_slot = np.full(10_800, 10.0)
    for k in range(19):                                  # 19 new rows in cycles 2-3, the first after minute 60
        L_slot[3700 + 300 * k:] += 1
    rows = []
    for j in range(180):
        acc = 0.0
        for i in range(60 * j, 60 * j + 60):             # the evaluator's arithmetic: 60 float additions, / 60, / 600
            acc += float(min(i + 1, PA.K_S) + L_slot[i])
        rows.append(dict(t_end=10_000_000 + 60 * j + 59, arrivals=100.0, completed=50.0, cons_rate=0.0,
                         mem_util=(acc / 60) / (PA.K_S + PA.K_L)))
    out = PA.cycle_rows(pd.DataFrame(rows), dict(lt_occupancy=29.0), ARMS["A0"])
    assert out[0]["lt_end_lo"] == 10.0 and out[0]["lt_end_hi"] == 10.0
    assert out[2]["lt_opened_c23_lo"] == 19.0 and out[2]["lt_opened_c23_hi"] == 19.0


def test_kappa_phi_seed_9003_matches_the_fig3b_value():
    import tuning
    from tpa.config import load_config
    cfg = load_config(overrides=tuning.adopted_overrides())
    assert round(PA.kappa_phi_of_seed(9003, cfg), 4) == 0.4026
    # kappa_phi_of_seed uses the zone phases of build_world from the world stream SeedSequence([seed, 1, 0])
    from tpa.env.world import build_world
    for seed in (1, 17, 9003):
        g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1, 0])))
        assert PA.kappa_phi_of_seed(seed, cfg) == PA.kappa_phi(build_world(cfg, g).zone_phase)
    assert PA.kappa_phi(np.array([0.0, math.pi / 2])) == pytest.approx(0.0)
    assert PA.kappa_phi(np.zeros(6)) == pytest.approx(1.0)


def test_correlation_and_mean_ci():
    x = np.arange(10.0)
    r = PA.correlation(x, 2 * x + 1)
    assert r["pearson_r"] == pytest.approx(1.0) and r["spearman_rho"] == pytest.approx(1.0)
    assert math.isnan(PA.correlation([1, 2], [3, 4])["pearson_r"])
    mc = PA.mean_ci([1.0, 2.0, 3.0])
    assert mc["mean"] == pytest.approx(2.0) and mc["ci95"] == pytest.approx(4.302653 * 1.0 / math.sqrt(3), rel=1e-5)


# ---------------------------------------------------------------- end to end on a synthetic campaign
LOADS = (0.17, 0.374, 0.578)


def _write_run(runs_root: Path, phase: str, run_id: str, arm: str, seed: int, wl_label: str, base: float, rng,
               n_minutes: int = 180, block: str = "test"):
    import pandas as pd
    flags = ARMS[arm]
    rows = []
    for L in LOADS:
        for k in range(3):
            t0 = (10_000_000 if block == "test" else 5_003_600) + 10_800 * k
            lt = 0.0
            tot_a = tot_c = tot_o = 0.0
            for j in range(n_minutes):
                arr = float(rng.integers(80, 120))
                comp = float(np.clip(round(arr * (base - 0.2 * L + rng.normal(0, 0.02))), 0, arr))
                o = float(rng.integers(0, 10)) if flags.memory and flags.use_ml else 0.0
                if flags.memory and flags.use_ml and not flags.flat and j % 40 == 39:
                    lt += 1
                s = PA.short_term_fill_mean(j) if flags.memory and flags.use_ms else 0.0
                mu = (s + lt) / 600 if flags.memory else 0.0
                rows.append(dict(row_kind="minute", t_end=t0 + 60 * j + 59, arm=arm, seed=seed, workload=wl_label, load=L,
                                 test_run=k, block=block, slots=60, arrivals=arr, completed=comp, completion=comp / arr,
                                 cons_rate=o / 60, mem_util=mu, lt_occupancy=math.nan))
                tot_a, tot_c, tot_o = tot_a + arr, tot_c + comp, tot_o + o
            rows.append(dict(row_kind="run", t_end=-1, arm=arm, seed=seed, workload=wl_label, load=L, test_run=k, block=block,
                             slots=10_800, arrivals=tot_a, completed=tot_c, completion=tot_c / tot_a, cons_rate=tot_o / 10_800,
                             mem_util=math.nan, lt_occupancy=lt if flags.memory else math.nan,
                             e_burst_j=40.0 * (1000 + 37 * k + 11 * seed)))
    d = runs_root / phase / run_id / block
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "part-00001.parquet", index=False)


def _queue(db: Path, rows):
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, arm TEXT, seed INTEGER, workload TEXT, phase TEXT, state TEXT, "
                "episodes_done INTEGER, attempts INTEGER, code_hash TEXT, error TEXT)")
    con.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)", [tuple(r) + (None,) * (10 - len(r)) for r in rows])
    con.commit()
    con.close()


def _run(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_campaign.py"), *map(str, args)],
                          capture_output=True, text=True, timeout=600)


def test_end_to_end_main_phase(tmp_path):
    rng = np.random.default_rng(7)
    runs, db, out = tmp_path / "runs", tmp_path / "runs.db", tmp_path / "out"
    q = []
    plan = {"A0": 6, "A3": 6, "A9": 6, "A8": 4, "A13": 3, "A15": 3, **{b: 4 for b in ("B_ZHOU", "B_QIN", "B_HUANG", "B_SONG")}}
    for arm, n in plan.items():
        for s in range(1, n + 1):
            rid = f"main-{arm}-s{s}-sinusoidal"
            _write_run(runs, "main", rid, arm, s, "sinusoidal", 0.8 if arm.startswith("A") else 0.7, rng)
            q.append((rid, arm, s, "sinusoidal", "main", "done", 500, 1, "2892281e5ba9" * 2))
    for s in (1, 2, 3):                                           # T_mem sensitivity rows: queue arm A0
        rid = f"main-A0-tmem25-s{s}-sinusoidal"
        _write_run(runs, "main", rid, "A0", s, "sinusoidal", 0.75, rng)
        q.append((rid, "A0", s, "sinusoidal", "main", "done", 500, 1, "2892281e5ba9" * 2))
    _queue(db, q)
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs, "--no-b1")
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())
    h1 = s["contrasts"][0]
    assert h1["family"] == "H1" and h1["n"] == 6                # the tmem25 rows do not pool with A0 (own label)
    s1 = [c for c in s["contrasts"] if c["family"] == "S1"]
    assert len(s1) == 9 and not any("A13" in c["label"] for c in s1)
    assert not any(c["family"] in ("C", "X") for c in s["contrasts"])
    tc = {c["label"]: c for c in s["tier_c"]}
    assert tc["A0 - A0-tmem25, heavy"]["n"] == 3 and tc["A0 - A13, heavy"]["n"] == 3
    assert "p_paired_t" in tc["A0 - A0-tmem25, heavy"]                         # the only paired t, at heavy load
    assert not any(k in tc[lab] for lab in tc if lab != "A0 - A0-tmem25, heavy" for k in ("p", "p_paired_t", "p_holm"))
    assert len(s["exploratory"]) == 8 and all(c["n"] == 4 for c in s["exploratory"])
    assert not any(k in c for c in s["exploratory"] for k in ("p", "p_paired_t", "p_holm"))
    import pandas as pd
    ex, tcsv = pd.read_csv(out / "exploratory.csv"), pd.read_csv(out / "tierc.csv")
    assert not {"p", "p_holm", "p_paired_t"} & set(ex.columns) and not {"p", "p_holm"} & set(tcsv.columns)
    assert tcsv["p_paired_t"].notna().sum() == 1
    pc = pd.read_csv(out / "per_cycle_contrasts.csv")
    assert set(pc["cycle"].astype(str)) == {"1", "2", "3", "2-3", "whole run (run rows)"}
    whole = pc[(pc["cycle"] == "whole run (run rows)") & np.isclose(pc["load"], 0.578)].iloc[0]
    assert whole["mean_diff"] == pytest.approx(h1["mean_diff"])
    runs_cyc = pd.read_csv(out / "per_cycle_runs.csv")
    assert set(runs_cyc["arm"]) == {"A0", "A0-tmem25", "A3", "A8", "A13", "A15"}        # memory variants only
    p69 = json.loads((out / "p6_p9.json").read_text())
    a0h = [x for x in p69["p6"] if x["arm"] == "A0" and x["load_name"] == "heavy"][0]
    assert a0h["n_runs"] == 18 and a0h["lt_end3_max"] == 4                            # 4 rows opened per run (j % 40 == 39)
    assert a0h["lt_end3_median"] == 4 and a0h["runs_lt_end3_gt19"] == 0              # P6: median, runs above 19
    a9 = [x for x in p69["p9"] if x["arm"] == "A0" and x["load_name"] == "heavy"][0]
    d = a9["intervals_c23_5pct"]
    assert d["n"] == 36 and sum(d[c] for c in ("none", "all8", "1", "2", ">=3")) == pytest.approx(1.0)
    a3 = [x for x in p69["p6"] if x["arm"] == "A3" and x["load_name"] == "heavy"][0]
    assert a3["has_lt"] is False and a3["runs_lt_end3_gt19"] is None                 # A3: no long-term store, n/a
    assert "9003" not in p69["kappa_phi"] and set(p69["kappa_phi"]) == {str(i) for i in range(1, 7)}
    tables = (out / "TABLES.md").read_text(encoding="utf-8")
    for sec in ("## 2a. Tier C", "## 2b. EXPLORATORY, NOT REGISTERED", "## 5. Per cycle", "### 5.3 P6", "### 5.4 P9"):
        assert sec in tables
    _check_markdown_tables(tables)
    assert "## 7. Amendment 16, Part B1: NOT READ (--no-b1 given" in tables and s["b1_not_read"] is True and s["b1"] is None
    r = _run("--phase", "main", "--out", tmp_path / "out_nob1", "--db", db, "--runs-root", runs)      # fail closed
    assert r.returncode != 0 and "no B1 queue file" in (r.stderr + r.stdout)


def test_end_to_end_guards_and_robust_phase(tmp_path):
    rng = np.random.default_rng(11)
    runs, db, out = tmp_path / "runs", tmp_path / "runs.db", tmp_path / "out"
    q = []
    for wl, label in (("mmpp", "mmpp"), ("sinusoidal|mobility=gauss_markov", "sinusoidal")):
        for arm in ("A0", "A3", "A9", "A10"):
            for s in range(1, 5):
                rid = f"robust-{arm}-s{s}-{wl.replace('|', '_')}"
                _write_run(runs, "robust", rid, arm, s, label, 0.8, rng)
                q.append((rid, arm, s, wl, "robust", "done", 500, 1, "2892281e5ba9" * 2))
    q.append(("robust-A0-s9-mmpp", "A0", 9, "mmpp", "robust", "running", 10, 1, "2892281e5ba9" * 2))
    q.append(("robust-A0-s8-mmpp", "A0", 8, "mmpp", "robust", "pending", 300, 1, "2892281e5ba9" * 2,
              "stopped by a ruling (synthetic test)"))
    _queue(db, q)
    r = _run("--phase", "robust", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode != 0 and "robust rows are not finished" in (r.stderr + r.stdout)
    con = sqlite3.connect(db); con.execute("DELETE FROM runs WHERE run_id='robust-A0-s9-mmpp'"); con.commit(); con.close()
    r = _run("--phase", "robust", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())
    fams = {c["family"] for c in s["contrasts"]}
    assert fams == {"R:mmpp", "R:sinusoidal|mobility=gauss_markov"}                 # queue workload, GM kept apart
    assert s["stopped"] == ["robust-A0-s8-mmpp"]                                     # a 'stopped by' row is reported, not blocking
    _check_markdown_tables((out / "TABLES.md").read_text(encoding="utf-8"))
    assert not any("A12" in c["label"] for c in s["contrasts"])
    p10 = json.loads((out / "p10.json").read_text())
    assert p10["p10_computable"] is False and p10["b_computable"] is False and 1 - 1e-6 < p10["base_share_within_w"] < 1
    assert "load_dependence" not in p10["by_arm"]["A0"]                             # descriptive only, no P10 test
    assert set(p10["by_arm"]) == {"A0", "A3"}                                        # memory variants on MMPP
    assert p10["by_arm"]["A0"]["by_load"]["heavy"]["open_fraction"]["n"] == 4
    assert "## 6. P10 on MMPP" in (out / "TABLES.md").read_text(encoding="utf-8")
    # main-phase guard: a pending main row refuses
    db2 = tmp_path / "main.db"
    _queue(db2, [("main-A0-s1-sinusoidal", "A0", 1, "sinusoidal", "main", "pending", 0, 0, "x" * 24)])
    r = _run("--phase", "main", "--out", tmp_path / "out2", "--db", db2, "--runs-root", runs)
    assert r.returncode != 0 and "main rows are not finished" in (r.stderr + r.stdout)


def _check_markdown_tables(text: str):
    """Every row of every Markdown table has as many unescaped '|' as its header."""
    import re
    block = []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append(len(re.findall(r"(?<!\\)\|", line)))
        elif block:
            assert len(set(block)) == 1, f"ragged Markdown table: cell counts {block}"
            block = []


def test_end_to_end_all_cycle_splits_fail(tmp_path):
    """Every memory run has 179 minute rows: the per-cycle section lists them as errors and the run completes."""
    rng = np.random.default_rng(3)
    runs, db, out = tmp_path / "runs", tmp_path / "runs.db", tmp_path / "out"
    q = []
    for arm in ("A0", "A3", "A9"):
        for s in range(1, 4):
            rid = f"main-{arm}-s{s}-sinusoidal"
            _write_run(runs, "main", rid, arm, s, "sinusoidal", 0.8, rng, n_minutes=179)
            q.append((rid, arm, s, "sinusoidal", "main", "done", 500, 1, "2892281e5ba9" * 2))
    _queue(db, q)
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs, "--no-b1")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "could not be split into 3 cycles" in (out / "TABLES.md").read_text(encoding="utf-8")


def test_end_to_end_rehearsal_tags_and_excludes_pareto(tmp_path):
    """--rehearsal: validation block, selected configuration, the REHEARSAL tag in every table; Pareto tuning
    rows are never read."""
    import pandas as pd
    rng = np.random.default_rng(5)
    runs, db, out = tmp_path / "runs", tmp_path / "runs.db", tmp_path / "out"
    (runs / "tune").mkdir(parents=True)
    (runs / "tune" / "gate2_report.json").write_text(json.dumps({"arms": {"A0": {"selected": 0}, "A3": {"selected": 0}}}))
    q = []
    for arm in ("A0", "A3"):
        for st, seed in ((1, 9004), (2, 9005)):
            rid = f"tune{st}-{arm}-c00-s{seed}"
            _write_run(runs, "tune", rid, arm, seed, "sinusoidal", 0.8, rng, block="val_full")
            q.append((rid, arm, seed, "sinusoidal", "tune", "done", 300, 1, "2892281e5ba9" * 2))
    q.append(("tune1-A0-c00-s9004-pareto", "A0", 9004, "pareto", "tune", "done", 300, 1, "790391a95b5d" * 2))
    _queue(db, q)
    r = _run("--phase", "tune", "--rehearsal", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())
    assert s["done"] == 4 and "pareto" not in s["workloads"]
    for f in ("trials.csv", "descriptive.csv", "contrasts.csv", "per_cycle_runs.csv", "per_cycle_summary.csv",
              "per_cycle_contrasts.csv"):
        assert pd.read_csv(out / f)["tag"].str.startswith("REHEARSAL").all(), f
    assert json.loads((out / "p6_p9.json").read_text())["tag"].startswith("REHEARSAL")


# ---------------------------------------------------------------- the evaluation-only sensitivity rows (file prefix b1)
def test_b1_constants_match_the_runner():
    import analyze_campaign as AC
    import b1_eval as B
    assert AC.B1_SETTINGS == B.SETTINGS and AC.B1_MAIN_VALUES == B.MAIN_VALUES and AC.B1_TAGS == B.TAGS
    assert AC.B1_SEEDS == B.SEEDS and AC.B1_WALL == B.WALL_COLS and AC.b1_parse("b1-A0-s3-omega_l-x1.25") == (3, "omega_l-x1.25")


def _b1_db(db: Path, rows):
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, arm TEXT, seed INTEGER, workload TEXT, phase TEXT, state TEXT, "
                "overrides TEXT, code_hash TEXT, error TEXT)")
    con.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def _b1_fixture(tmp_path, n_seeds=6, stopped_row=None, bad_hash_seed=None):
    """Synthetic main campaign (A0, A3 x n seeds) + sensitivity rows with known answers.
    Seeds 1..n-1: the default row copies the main test's first heavy run row exactly; seed n's default completion is
    +0.001. Setting j (index in B1_TAGS) adds to the default: completion off_j + 0.0005 s, cons_rate off_j + 0.0001 s,
    lt_occupancy j + s. Wall-clock columns and t_*_ms timers differ between the sensitivity row and the main row (and
    one timer is in the sensitivity row only): they must never count as differences. Each sensitivity row gets
    RESULT.json with the SHA-256 of a fake main ckpt_best.pt of its seed; bad_hash_seed gets a wrong hash on one row."""
    import hashlib
    import pandas as pd
    import pyarrow.parquet as pq
    import analyze_campaign as AC
    rng = np.random.default_rng(3)
    runs, db = tmp_path / "runs", tmp_path / "runs.db"
    q = []
    for arm in ("A0", "A3"):
        for s in range(1, n_seeds + 1):
            rid = f"main-{arm}-s{s}-sinusoidal"
            _write_run(runs, "main", rid, arm, s, "sinusoidal", 0.8, rng)
            q.append((rid, arm, s, "sinusoidal", "main", "done", 500, 1, "2892281e5ba9" * 2))
    _queue(db, q)
    off = {tag: 0.002 * j for j, tag in enumerate(AC.B1_TAGS)}
    b1, sha = [], {}
    for s in range(1, n_seeds + 1):
        ckp = runs / "main" / f"main-A0-s{s}-sinusoidal" / "ckpt_best.pt"
        ckp.write_bytes(f"fake checkpoint of seed {s}".encode())
        sha[s] = hashlib.sha256(ckp.read_bytes()).hexdigest()
        main_run = pq.read_table(runs / "main" / f"main-A0-s{s}-sinusoidal" / "test").to_pandas()
        base = main_run[(main_run["row_kind"] == "run") & np.isclose(main_run["load"], 0.578)
                        & (main_run["test_run"] == 0)].iloc[0].to_dict()
        if s == n_seeds:
            base["completion"] += 0.001
        for tag in AC.B1_TAGS:
            key, val = next(((k, v) for t, k, v in AC.B1_SETTINGS if t == tag), (None, None))
            ov = {"run.test_runs": 1, **({key: val} if key else {})}
            rid = f"b1-A0-s{s}-{tag}"
            state, err = "done", None
            if tag == "alpha_m-x0.75" and s == 2:
                state = "failed"
            if stopped_row == (s, tag):
                state, err = "pending", "stopped by a ruling (synthetic test)"
            b1.append((rid, "A0", s, "sinusoidal", "main", state, json.dumps(ov), "2892281e5ba9" * 2, err))
            r = dict(base)
            if tag != AC.B1_DEFAULT:
                j = AC.B1_TAGS.index(tag)
                r["completion"] = base["completion"] + off[tag] + 0.0005 * s
                r["cons_rate"] = base["cons_rate"] + off[tag] + 0.0001 * s
                r["lt_occupancy"] = base["lt_occupancy"] + j + s
                r["e_burst_j"] = base["e_burst_j"] + 40.0 * (100 * j + s)
            r.update(wall_s=123.0 + s, reaction_ms_p50=1.0 + s, reaction_ms_p90=2.0 + s, recovery_ms=3.0 + s,
                     t_actor_ms=0.5 + s, t_drm_ms=0.25 + s, t_b1only_ms=9.0)
            d = runs / "b1" / "main" / rid / "test"
            d.mkdir(parents=True)
            pd.DataFrame([r]).to_parquet(d / "part-00001.parquet", index=False)
            h = "0" * 64 if (bad_hash_seed == s and tag == "theta_p-0.40") else sha[s]
            (d.parent / "RESULT.json").write_text(json.dumps({"run_id": rid, "b1": {"source_ckpt_sha256": h}}))
    return runs, db, b1, off


def test_end_to_end_b1_sensitivity_table_known_answers(tmp_path):
    """Paired means and t-intervals of completion, gate-open share and LT occupancy; the levels beside them; the
    reproduction check (wall-clock columns and timers excluded); the missing row of a failed setting; the guards."""
    import pandas as pd
    import pyarrow.parquet as pq
    import analyze_campaign as AC
    from scipy import stats as st
    runs, db, b1, off = _b1_fixture(tmp_path)
    out = tmp_path / "out"
    # a pending sensitivity row blocks the analysis
    _b1_db(runs / "b1" / "runs_b1.db", b1[:-1] + [b1[-1][:5] + ("pending",) + b1[-1][6:]])
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode != 0 and "B1 rows are not finished" in (r.stderr + r.stdout)
    (runs / "b1" / "runs_b1.db").unlink()
    _b1_db(runs / "b1" / "runs_b1.db", b1)
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode == 0, r.stderr[-3000:]
    t = pd.read_csv(out / "b1_sensitivity.csv").set_index("tag")
    assert list(t.index) == [x for x, _, _ in AC.B1_SETTINGS]
    dflt = {s: pq.read_table(runs / "b1" / "main" / f"b1-A0-s{s}-default" / "test").to_pandas().iloc[0] for s in range(1, 7)}

    def hw(v):
        return st.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / math.sqrt(len(v))
    for tag, key, val in AC.B1_SETTINGS:
        j = AC.B1_TAGS.index(tag)
        seeds = [s for s in range(1, 7) if not (tag == "alpha_m-x0.75" and s == 2)]
        dc = np.array([off[tag] + 0.0005 * s for s in seeds])
        dg = np.array([off[tag] + 0.0001 * s for s in seeds])
        dl = np.array([j + s for s in seeds], float)
        assert t.loc[tag, "n"] == len(seeds) and t.loc[tag, "key"] == key and t.loc[tag, "value"] == val
        assert t.loc[tag, "main_value"] == AC.B1_MAIN_VALUES[key]
        for name, d in (("completion", dc), ("gate_open_share", dg), ("lt_occupancy", dl)):
            assert t.loc[tag, f"{name}_diff_mean"] == pytest.approx(d.mean(), abs=1e-12), (tag, name)
            assert t.loc[tag, f"{name}_diff_ci95"] == pytest.approx(hw(d), rel=1e-9), (tag, name)
        assert t.loc[tag, "completion_diff_ci_lo"] == pytest.approx(dc.mean() - hw(dc), abs=1e-12)
        assert t.loc[tag, "completion_diff_ci_hi"] == pytest.approx(dc.mean() + hw(dc), abs=1e-12)
        # levels: the setting's mean and the default's mean over the same seeds
        lv_d = np.array([dflt[s]["completion"] for s in seeds])
        assert t.loc[tag, "completion_default_mean"] == pytest.approx(lv_d.mean(), abs=1e-12)
        assert t.loc[tag, "completion_mean"] == pytest.approx((lv_d + dc).mean(), abs=1e-12)
        assert t.loc[tag, "completion_ci95"] == pytest.approx(hw(lv_d + dc), rel=1e-9)
        g_d = np.array([dflt[s]["cons_rate"] for s in seeds])
        assert t.loc[tag, "gate_open_share_default_mean"] == pytest.approx(g_d.mean(), abs=1e-12)
        assert t.loc[tag, "gate_open_share_mean"] == pytest.approx((g_d + dg).mean(), abs=1e-12)
        l_d = np.array([dflt[s]["lt_occupancy"] for s in seeds])
        assert t.loc[tag, "lt_occupancy_default_mean"] == pytest.approx(l_d.mean(), abs=1e-12)
        assert t.loc[tag, "lt_occupancy_mean"] == pytest.approx((l_d + dl).mean(), abs=1e-12)
    rep = pd.read_csv(out / "b1_reproduction.csv").set_index("seed")
    assert rep["identical"].tolist() == [True] * 5 + [False] and rep.loc[6, "differing"] == "completion"
    assert rep.loc[6, "completion_absdiff"] == pytest.approx(0.001, abs=1e-12)
    s = json.loads((out / "summary.json").read_text())["b1"]["summary"]
    assert s["n_rows"] == 90 and s["n_done"] == 89 and s["missing"] == ["b1-A0-s2-alpha_m-x0.75"]
    assert s["repro_seeds"] == 6 and s["repro_identical"] == 5 and s["errors"] == [] and s["excluded_seeds"] == []
    assert s["ckpt_compared_seeds"] == [1, 2, 3, 4, 5, 6]
    assert s["repro_completion_differs"] == [6]                     # seed 6's default completion is +0.001
    from tpa.config import load_config
    c0 = load_config()
    den = float(c0.burst.delta_e_w) * float(c0.scenario.slot_s) * int(c0.scenario.n_uav) * 10_800
    for tag, key, val in AC.B1_SETTINGS:
        j = AC.B1_TAGS.index(tag)
        seeds = [s_ for s_ in range(1, 7) if not (tag == "alpha_m-x0.75" and s_ == 2)]
        db_ = np.array([40.0 * (100 * j + s_) / den for s_ in seeds])
        lv = np.array([dflt[s_]["e_burst_j"] / den for s_ in seeds])
        assert t.loc[tag, "burst_on_share_diff_mean"] == pytest.approx(db_.mean(), abs=1e-15), tag
        assert t.loc[tag, "burst_on_share_diff_ci95"] == pytest.approx(hw(db_), rel=1e-9), tag
        assert t.loc[tag, "burst_on_share_default_mean"] == pytest.approx(lv.mean(), abs=1e-15), tag
        assert t.loc[tag, "burst_on_share_mean"] == pytest.approx((lv + db_).mean(), abs=1e-15), tag
    assert "completion bit for bit (A16B1.7): differs for seeds [6]." in (out / "TABLES.md").read_text(encoding="utf-8")
    assert "burst_on_share" in pd.read_csv(out / "b1_runs.csv").columns
    runs_csv = pd.read_csv(out / "b1_runs.csv")
    assert len(runs_csv) == 89 and runs_csv["source_ckpt_sha256"].notna().all() and not runs_csv["excluded"].any()
    tables = (out / "TABLES.md").read_text(encoding="utf-8")
    assert "## 7. Sensitivity of the frozen A0 policies" in tables and "p_holm" not in t.columns and "p" not in t.columns
    _check_markdown_tables(tables)
    # a row whose overrides do not carry its setting is reported, not used
    con = sqlite3.connect(runs / "b1" / "runs_b1.db")
    con.execute("UPDATE runs SET overrides=? WHERE run_id='b1-A0-s1-theta_b-x1.25'", (json.dumps({"run.test_runs": 1}),))
    con.commit(); con.close()
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())["b1"]["summary"]
    assert len(s["errors"]) == 1 and "b1-A0-s1-theta_b-x1.25" in s["errors"][0]
    assert pd.read_csv(out / "b1_sensitivity.csv").set_index("tag").loc["theta_b-x1.25", "n"] == 5


def test_b1_checkpoint_hash_mismatch_drops_the_seed_and_stopped_rows_do_not_block(tmp_path):
    import pandas as pd
    runs, db, b1, _off = _b1_fixture(tmp_path, stopped_row=(4, "omega_l-x1.25"), bad_hash_seed=3)
    _b1_db(runs / "b1" / "runs_b1.db", b1)
    out = tmp_path / "out"
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode == 0, r.stderr[-3000:]                     # the 'stopped by' pending row does not block
    s = json.loads((out / "summary.json").read_text())["b1"]["summary"]
    assert "b1-A0-s4-omega_l-x1.25" in s["missing"] and s["excluded_seeds"] == [3]
    assert any("seed 3" in e and "different or missing checkpoint hashes" in e for e in s["errors"])
    t = pd.read_csv(out / "b1_sensitivity.csv").set_index("tag")
    assert t.loc["theta_p-0.30", "n"] == 5 and t.loc["omega_l-x1.25", "n"] == 4 and t.loc["alpha_m-x0.75", "n"] == 4
    assert "3" not in str(t.loc["theta_p-0.30", "seeds"]).split()
    rep = pd.read_csv(out / "b1_reproduction.csv")
    assert 3 not in set(rep["seed"]) and len(rep) == 5
    # the main row's checkpoint changed after the sensitivity rows ran: every hash of seed 5 differs from it -> seed 5
    # left out too
    (runs / "main" / "main-A0-s5-sinusoidal" / "ckpt_best.pt").write_bytes(b"another checkpoint")
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())["b1"]["summary"]
    assert s["excluded_seeds"] == [3, 5] and any("seed 5" in e and "main row's ckpt_best.pt" in e for e in s["errors"])


def test_b1_open_rows_of_a_failed_source_are_missing_not_blocking(tmp_path):
    """Main A0 row of seed 4 failed: its 15 sensitivity rows can never run and stay pending; they are reported
    missing."""
    runs, db, b1, _off = _b1_fixture(tmp_path, n_seeds=4)
    con = sqlite3.connect(db)
    con.execute("UPDATE runs SET state='failed' WHERE run_id='main-A0-s4-sinusoidal'")
    con.commit(); con.close()
    b1 = [r if r[2] != 4 else r[:5] + ("pending",) + r[6:] for r in b1]
    _b1_db(runs / "b1" / "runs_b1.db", b1)
    out = tmp_path / "out"
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())["b1"]["summary"]
    assert sorted(m for m in s["missing"] if "-s4-" in m) == sorted(f"b1-A0-s4-{t}" for t in __import__("analyze_campaign").B1_TAGS)
    assert s["repro_seeds"] == 3


def test_b1_missing_checkpoint_hashes_exclude_the_seed(tmp_path):
    """RESULT.json lost (pandas would turn the missing hash into NaN): the seed is left out with the 'missing' reason,
    with or without the main checkpoint; one lost file is enough."""
    import pandas as pd
    runs, db, b1, _off = _b1_fixture(tmp_path)
    _b1_db(runs / "b1" / "runs_b1.db", b1)
    for p in (runs / "b1" / "main").glob("b1-A0-s3-*/RESULT.json"):
        p.unlink()
    (runs / "b1" / "main" / "b1-A0-s4-alpha_t-x0.75" / "RESULT.json").unlink()
    out = tmp_path / "out"
    for with_ckpt in (True, False):
        if not with_ckpt:
            (runs / "main" / "main-A0-s3-sinusoidal" / "ckpt_best.pt").unlink()
        r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
        assert r.returncode == 0, r.stderr[-3000:]
        s = json.loads((out / "summary.json").read_text())["b1"]["summary"]
        assert s["excluded_seeds"] == [3, 4], (with_ckpt, s["excluded_seeds"])
        assert all("different or missing checkpoint hashes" in e for e in s["errors"] if e.startswith(("seed 3", "seed 4")))
        assert s["ckpt_compared_seeds"] == [1, 2, 5, 6]            # 3 and 4 are excluded before the comparison
        rc = pd.read_csv(out / "b1_runs.csv")
        assert rc.loc[rc["seed"].isin([3, 4]), "excluded"].all() and not rc.loc[~rc["seed"].isin([3, 4]), "excluded"].any()
        assert pd.read_csv(out / "b1_sensitivity.csv").set_index("tag").loc["theta_p-0.30", "n"] == 4


def test_b1_queue_must_hold_exactly_the_registered_rows(tmp_path):
    runs, db, b1, _off = _b1_fixture(tmp_path)
    out = tmp_path / "out"
    _b1_db(runs / "b1" / "runs_b1.db", b1[1:])                                  # one registered row absent
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode != 0 and "lacks 1 registered rows" in (r.stderr + r.stdout)
    (runs / "b1" / "runs_b1.db").unlink()
    extra = ("b1-A0-s7-default", "A0", 7, "sinusoidal", "main", "done", json.dumps({"run.test_runs": 1}), "2892281e5ba9" * 2, None)
    _b1_db(runs / "b1" / "runs_b1.db", b1 + [extra])                            # seed 7 has no main A0 row
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode != 0 and "not registered" in (r.stderr + r.stdout)
    (runs / "b1" / "runs_b1.db").unlink()
    _b1_db(runs / "b1" / "runs_b1.db", [])                                      # an empty queue file
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs)
    assert r.returncode != 0 and "lacks 90 registered rows" in (r.stderr + r.stdout)


def test_b1_root_follows_b1_db(tmp_path):
    """--b1-db alone: the run directories of the sensitivity rows are looked up next to it."""
    import shutil
    runs, db, b1, _off = _b1_fixture(tmp_path)
    _b1_db(runs / "b1" / "runs_b1.db", b1)
    shutil.move(str(runs / "b1"), str(tmp_path / "elsewhere"))
    out = tmp_path / "out"
    r = _run("--phase", "main", "--out", out, "--db", db, "--runs-root", runs, "--b1-db", tmp_path / "elsewhere" / "runs_b1.db")
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())["b1"]["summary"]
    assert s["n_done"] == 89 and s["missing"] == ["b1-A0-s2-alpha_m-x0.75"] and s["repro_identical"] == 5
