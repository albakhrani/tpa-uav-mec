"""Campaign analysis (the registered analysis plan, tier B n = 8; statistics in tpa/stats/analysis.py).

    python scripts/analyze_campaign.py --phase main   --out DIR                 # the registered analysis: once, at the end
    python scripts/analyze_campaign.py --phase robust --out DIR [--db ... --runs-root ...]
    python scripts/analyze_campaign.py --phase tune   --rehearsal --out DIR     # rehearsal in advance, on TUNING rows only

What it does: builds the trial table (one row per (arm, seed, workload, load) = mean of the evaluation runs of that
block, tpa.stats.analysis.trial_table), runs the registered tests and writes tables and figures:
  H1  A0 vs A3, heavy-load completion, two-sided paired t (alpha 0.05), 95 % t-CI, Cohen's d (pooled) and d_z,
      TOST equivalence against +-1.0 pp (the registered SESOI), Shapiro-Wilk on the differences and, if p < 0.01,
      Wilcoxon as a sensitivity analysis;
  S1-S4 the secondary families with Holm step-down within each family, never pooled; S4 adds the interaction
      one-sample t of (difference at heavy - difference at light);
  MDE per contrast from the campaign's own SD_d (paired t, power 0.8; formula in mde());
  descriptive outcomes with 95 % CIs, wall-clock columns excluded (they belong to exclusive timing runs);
  robustness (phase robust): A0 vs each other arm at heavy load per workload family, Holm within family.
Outputs in --out: TABLES.md, contrasts.csv, trials.csv, descriptive.csv, summary.json, fig_*.png.

Further analyses, specified before any main or robustness result was read (statistics in scripts/a16_partA.py):
  P10 on MMPP (phase robust): open fraction per trial by load for each memory variant (run rows' cons_rate),
      descriptive only; the clustering ratio is reported as NOT computable from the records (no per-slot gate decision,
      no MMPP state), with the frozen definition's base share for the configured MMPP;
  per-cycle statistics on the sinusoid (phase main / rehearsal), every memory variant: completion, gate-open share,
      open phase bins (8 x P/8 from the run start), open intervals, long-term occupancy at the end of each cycle (exact
      for cycle 3, bounds for cycles 1-2); A0 - A3 per cycle (paired by seed, 95 % CI) at every load; P6 and P9 on
      cycles 2-3 with whole-run values beside them; the primary test (H1) is unchanged;
  exploratory comparisons, NOT registered: A8 - B_x and A9 - B_x at heavy load, descriptive (no p-values, no Holm).
  Outputs: per_cycle_runs.csv, per_cycle_summary.csv, per_cycle_contrasts.csv, p6_p9.json, p10.json, exploratory.csv,
  tierc.csv and the TABLES.md sections 2a, 2b, 5, 6.
Arm sets and labels:
  A13 is tier C (effect size and CI only), not a member of S1; A0 - A13, A0 - A15 and A0 - A0-tmem25 are reported as
  tier-C effect sizes with 95 % CIs at every load, outside every Holm family, without p-values except the paired t of
  A0 - A0-tmem25 at heavy load; the T_mem sensitivity rows (queue arm A0, run ids main-A0-tmem25-*) are labelled
  A0-tmem25 so they never pool with A0 (pooled, H1 would raise a length mismatch); the robustness arms are A3, A9, A10;
  the family key is the QUEUE workload, so 'sinusoidal|mobility=gauss_markov' stays apart from 'sinusoidal';
  --phase robust refuses unless every robust row is done, failed or stopped deliberately.

Evaluation-only sensitivity rows (file prefix b1), specified before any of them existed and before any main or
robustness result was read: with --phase main the rows of their own queue (default <runs-root>/b1/runs_b1.db, rows
b1-A0-s<seed>-<tag>, tables <runs-root>/b1/main/<run id>/test) are read with the main campaign. Per setting
(B1_SETTINGS below, one setting changed at a time): the paired difference to the default row of the same seed in
heavy-load completion, gate-open share (cons_rate), end-of-run long-term occupancy and the share of UAV-slots with
burst on (from the run row's e_burst_j), mean and 95 % t-interval, the levels beside them; the reproduction check of
the default rows against the main test's first heavy run, with one line naming the seeds whose default-row completion
differs bit for bit from it. Descriptive only: no test, no Holm.
Outputs: b1_sensitivity.csv, b1_runs.csv, b1_reproduction.csv, TABLES.md section 7.

Guards. --phase main refuses unless every main row is done, failed or stopped deliberately (the registered plan: the
script runs once, after all main-campaign runs are done); failed and stopped rows are reported as missing and their
contrasts use the complete pairs. It also refuses while a sensitivity row is open (one whose source main row failed, or
one stopped deliberately, is reported missing). --rehearsal is the only way to read tuning rows: it requires --phase
tune, uses the validation block (val_full) of the SELECTED configuration of each tuned arm (its index is
arms.<arm>.selected in <runs-root>/tune/gate2_report.json, a tuning selection report that is not part of this
repository; read by load_rows), refuses any run id that does not start
with "tune" or any seed below 9000, writes REHEARSAL into every output, and never touches main or robust rows. Test
seeds (1..n) are read only with --phase main or robust. Nothing here changes a run or the queue.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from tpa.stats import analysis as A  # noqa: E402
import a16_partA as PA  # noqa: E402

ALPHA, SESOI = 0.05, 0.010                      # two-sided alpha; equivalence margin 1.0 pp (the registered SESOI)
WALL = ("wall_s", "reaction_ms_p50", "reaction_ms_p90", "recovery_ms", "detect_delay_slots",
        "service_recovery_slots", "t_end", "slots")
DESCRIPTIVE = ["completion", "resource_eff", "resource_eff_cb", "cpu_util", "energy_eff", "hit_rate", "wait_s",
               "trans_s", "comp_s", "pred_bias", "e_uplink_j", "e_downlink_j", "e_hover_j", "e_comp_j", "e_mem_j",
               "lt_occupancy", "cons_rate", "mem_util", "sigma_mean"]
S1 = ["A1", "A2", "A4", "A5", "A6", "A7", "A8", "A9", "A14"]   # A13 is tier C (effect size and CI only): not a member of S1
S2 = [("A10", "A6"), ("A10", "A0"), ("A11", "A9"), ("A12", "A0")]
S3_B = ["B_ZHOU", "B_QIN", "B_HUANG", "B_SONG"]
ROBUST_ARMS = ["A3", "A9", "A10"]                               # robustness family: A0 vs each, per workload
TMEM25 = "A0-tmem25"                                            # T_mem sensitivity rows, never pooled with A0
TIER_C = [("A0", "A13"), ("A0", "A15"), ("A0", TMEM25)]         # tier C: effect size, 95 % CI; p only for TMEM25, heavy
EXPLORATORY = [(x, b) for x in ("A8", "A9") for b in S3_B]       # exploratory comparisons: descriptive, NOT registered


def arm_label(r: dict) -> str:
    """The T_mem sensitivity rows (queue arm A0, run ids main-A0-tmem25-s<seed>-...) get their own label."""
    return TMEM25 if r["arm"] == "A0" and "-tmem25-" in r["run_id"] else r["arm"]


def stopped_by_ruling(r: dict) -> bool:
    """A row stopped deliberately stays 'pending' (the queue's states are pending/running/done/failed) with an error
    note starting 'stopped by' (the row is not to be resumed); it is reported as missing like a failed row."""
    return r["state"] == "pending" and str(r.get("error") or "").startswith("stopped by")


def md(x) -> str:
    """A Markdown table cell: '|' (e.g. in 'sinusoidal|mobility=gauss_markov') escaped."""
    return str(x).replace("|", "\\|")


def arm_flags(label: str):
    """tpa.arms flags of an arm label (A0-tmem25 is A0 with a config override); None for unknown labels."""
    from tpa.arms import ARMS
    return ARMS.get("A0" if label == TMEM25 else label)


def q(db: Path, sql: str, args=()):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args)]
    finally:
        con.close()


def load_rows(a) -> tuple[list[dict], list[dict]]:
    rows = q(a.db, "SELECT run_id, arm, seed, workload, phase, state, episodes_done, attempts, code_hash, error FROM runs "
                   "WHERE phase=? ORDER BY run_id", (a.phase,))
    if a.phase == "tune":
        g = json.loads((a.runs_root / "tune" / "gate2_report.json").read_text())
        keep = {}
        for arm, d in g["arms"].items():
            keep[arm] = int(d["selected"])
        sel = []
        for r in rows:
            if r["workload"] != "sinusoidal":
                continue        # sinusoidal rows only: tuning rows on other workloads (Pareto) never enter the rehearsal
            arm, rest = r["run_id"].split("-", 1)[1].rsplit("-c", 1)
            idx = int(rest.split("-s")[0])
            if keep.get(arm) == idx:
                sel.append(r)
        rows = sel
    for r in rows:
        r["label"] = arm_label(r)
    done = [r for r in rows if r["state"] == "done"]
    if a.rehearsal:
        bad = [r["run_id"] for r in rows if not r["run_id"].startswith("tune") or int(r["seed"]) < 9000]
        if bad:
            sys.exit(f"REHEARSAL refuses non-tuning rows: {bad[:5]}")
    elif a.phase == "main":
        open_ = [r for r in rows if r["state"] not in ("done", "failed") and not stopped_by_ruling(r)]
        if open_:
            sys.exit(f"{len(open_)} main rows are not finished: the analysis runs once, after all main-campaign runs "
                     f"are done (ANALYSIS_PLAN section 6)")
    elif a.phase == "robust":
        open_ = [r for r in rows if r["state"] not in ("done", "failed") and not stopped_by_ruling(r)]
        if open_:
            sys.exit(f"{len(open_)} robust rows are not finished: the robustness analysis runs once, after all its "
                     f"runs are done (Amendment 16, Part A guard)")
    return rows, done


def build_trials(a, done: list[dict]):
    import pandas as pd
    parts, missing = [], []
    for r in done:
        d = a.runs_root / a.phase / r["run_id"] / a.block
        if not d.is_dir():
            missing.append(r["run_id"])
            continue
        t = A.trial_table(d)
        t["workload_kind"] = t["workload"]          # the evaluation's label (the parsed kind)
        t["workload"] = r["workload"]               # family key = the queue workload ('sinusoidal|mobility=...' kept apart)
        t["arm"] = r["label"]                       # the label: A0-tmem25 never pools with A0
        t["run_id"] = r["run_id"]
        t["code_hash"] = r["code_hash"][:12]
        parts.append(t)
    if not parts:
        sys.exit("no evaluation tables found")
    trials = pd.concat(parts, ignore_index=True)
    return trials, missing


def minute_series(a, done: list[dict], arms: tuple[str, ...], load: float):
    """Per-minute completion of the evaluation runs (row_kind == 'minute'), per arm: minute index = minutes since
    the start of each evaluation run (t_end minus the run's first t_end, per (run_id, test_run)), then the mean over
    seeds and evaluation runs at that minute (the warm-up series of the descriptive outcomes)."""
    import pandas as pd
    import pyarrow.parquet as pq
    out = {}
    for arm in arms:
        frames = []
        for r in done:
            if r["label"] != arm:
                continue
            for f in sorted((a.runs_root / a.phase / r["run_id"] / a.block).rglob("*.parquet")):
                df = pq.read_table(f).to_pandas()
                if "row_kind" in df.columns and "test_run" in df.columns:
                    df = df[(df["row_kind"] == "minute") & np.isclose(df["load"].astype(float), load)].copy()
                    if df.empty:
                        continue
                    df["minute"] = (df["t_end"] - df.groupby("test_run")["t_end"].transform("min")) / 60.0
                    frames.append(df[["minute", "completion"]])
        if frames:
            out[arm] = pd.concat(frames).groupby("minute")["completion"].mean().sort_index()
    return out


def tost(d: np.ndarray, margin: float, alpha: float) -> dict:
    """Two one-sided paired t-tests against +-margin; equivalence iff both p < alpha (p_tost = max)."""
    d = d[~np.isnan(d)]
    n = d.size
    if n < 2:
        return dict(n=n, p_lower=math.nan, p_upper=math.nan, p_tost=math.nan, equivalent=None)
    m, se = float(d.mean()), float(d.std(ddof=1)) / math.sqrt(n)
    if se == 0:
        p_lo = 0.0 if m > -margin else 1.0
        p_hi = 0.0 if m < margin else 1.0
    else:
        from scipy import stats as st
        p_lo = float(st.t.sf((m + margin) / se, n - 1))      # H0: mean <= -margin
        p_hi = float(st.t.cdf((m - margin) / se, n - 1))     # H0: mean >= +margin
    p = max(p_lo, p_hi)
    return dict(n=n, p_lower=p_lo, p_upper=p_hi, p_tost=p, equivalent=bool(p < alpha))


def normality(d: np.ndarray) -> dict:
    from scipy import stats as st
    d = d[~np.isnan(d)]
    out = dict(shapiro_p=math.nan, wilcoxon_p=math.nan, note="")
    if d.size < 3:
        out["note"] = "n < 3: Shapiro-Wilk not computed"
        return out
    try:
        out["shapiro_p"] = float(st.shapiro(d).pvalue)
    except ValueError as e:
        out["note"] = f"Shapiro-Wilk: {e}"
        return out
    if out["shapiro_p"] < 0.01:
        try:
            out["wilcoxon_p"] = float(st.wilcoxon(d).pvalue)
            out["note"] = "Shapiro p < 0.01: Wilcoxon reported as sensitivity analysis (primary test unchanged)"
        except ValueError as e:
            out["note"] = f"Wilcoxon: {e}"
    return out


def mde(sd_d: float, n: int, alpha: float = ALPHA, power: float = 0.8) -> float:
    """Minimum detectable effect of a paired t-test: MDE = (t_{1-alpha/2, n-1} + t_{power, n-1}) * SD_d / sqrt(n)."""
    if n < 2 or not np.isfinite(sd_d):
        return math.nan
    from scipy import stats as st
    return float((st.t.ppf(1 - alpha / 2, n - 1) + st.t.ppf(power, n - 1)) * sd_d / math.sqrt(n))


def run_contrast(trials, family, label, arm_a, arm_b, workload, load, metric="completion"):
    r = A.contrast(trials, metric, arm_a, arm_b, workload, load, alpha=ALPHA)
    sub = trials[(trials["workload"] == workload) & np.isclose(trials["load"].astype(float), load)]
    Aa = sub[sub["arm"] == arm_a].set_index("seed")[metric]
    Bb = sub[sub["arm"] == arm_b].set_index("seed")[metric]
    seeds = r["seeds"]
    d = (Aa.loc[seeds] - Bb.loc[seeds]).to_numpy(dtype=float) if seeds else np.array([])
    r.update(family=family, label=label, tost=tost(d, SESOI, ALPHA), normality=normality(d),
             mde_pp=100 * mde(r["sd_diff"], r["n"]) if r["n"] >= 2 else math.nan, diffs=d.tolist())
    return r


def fmt(x, pct=True, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}g}"


def forest(contrasts, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = [c for c in contrasts if c["n"] >= 2]
    fig, ax = plt.subplots(figsize=(7, 0.45 * max(len(cs), 3) + 1.2))
    for i, c in enumerate(cs):
        lo, hi = c["ci"]
        ax.errorbar(100 * c["mean_diff"], i, xerr=[[100 * (c["mean_diff"] - lo)], [100 * (hi - c["mean_diff"])]],
                    fmt="o", color="k", capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.axvspan(-100 * SESOI, 100 * SESOI, color="0.92", zorder=0)
    ax.set_yticks(range(len(cs)))
    ax.set_yticklabels([f"{c['label']} (n={c['n']})" for c in cs], fontsize=8)
    ax.set_xlabel("mean paired difference, completion (pp), 95 % t-CI; shaded = +-1 pp SESOI", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def descriptive_contrast(trials, family, label, arm_a, arm_b, workload, load, metric="completion", with_p=False):
    """Tier C / exploratory comparisons: paired mean difference with 95 % t-CI, d (pooled) and d_z, outside every
    Holm family. No p-value, except with_p=True (A0 vs A0-tmem25 paired at heavy load): two-sided paired t."""
    r = A.contrast(trials, metric, arm_a, arm_b, workload, load, alpha=ALPHA)
    out = dict(family=family, label=label, arm_a=arm_a, arm_b=arm_b, workload=workload, load=float(load), n=r["n"],
               seeds=r["seeds"], unpaired_seeds=r["unpaired_seeds"], mean_a=r["mean_a"], mean_b=r["mean_b"],
               mean_diff=r["mean_diff"], ci=list(r["ci"]), sd_diff=r["sd_diff"], cohen_d_pooled=r["cohen_d_pooled"],
               d_z=r["d_z"])
    if with_p:
        out.update(t=r["t"], p_paired_t=r["p"])
    return out


def read_tables(a, r):
    """All rows (run and minute) of one run's evaluation block; duplicate parts (a re-evaluation after a crash) dropped."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted((a.runs_root / a.phase / r["run_id"] / a.block).rglob("*.parquet"))
    if not files:
        return None
    df = pa.concat_tables([pq.read_table(f) for f in files], promote_options="default").to_pandas()
    keys = [c for c in ("row_kind", "load", "test_run", "t_end") if c in df.columns]
    return df.drop_duplicates(subset=keys) if keys else df


def partA_cycles(a, done):
    """Per-cycle statistics: one row per (run, load, test run, cycle), every memory variant (scripts/a16_partA.py)."""
    import pandas as pd
    recs = []
    for r in done:
        fl = arm_flags(r["label"])
        if fl is None or not fl.memory:
            continue
        df = read_tables(a, r)
        if df is None or "row_kind" not in df.columns:
            continue
        for (L, k), g in df.groupby(["load", "test_run"]):
            base = dict(run_id=r["run_id"], arm=r["label"], seed=int(r["seed"]), load=round(float(L), 6), test_run=int(k))
            runs = g[g["row_kind"] == "run"]
            try:
                rows = PA.cycle_rows(g[g["row_kind"] == "minute"], runs.iloc[0].to_dict() if len(runs) else None, fl)
            except ValueError as e:
                recs.append(dict(base, cycle=0, error=str(e)))
                continue
            recs += [dict(base, **x) for x in rows]
    return pd.DataFrame(recs)


CATEGORIES = ("none", "all8", "1", "2", ">=3")


def interval_distribution(rows, rule: str) -> dict:
    """Share of (run, cycle) records per open-interval category (scripts/a16_partA.interval_category): 'none' (no
    opening), 'all8' (all 8 bins open, not resolvable at P/8), '1', '2', '>=3' (the P9 falsifier side); n records."""
    if rows is None or len(rows) == 0:
        return dict(n=0, **{c: math.nan for c in CATEGORIES})
    cats = [PA.interval_category(int(r[f"n_intervals_{rule}"]), bool(r["no_opening"]), bool(r[f"all_bins_open_{rule}"]))
            for _, r in rows.iterrows()]
    return dict(n=len(cats), **{c: cats.count(c) / len(cats) for c in CATEGORIES})


def _by_seed(df, arm, load, col):
    s = df[(df["arm"] == arm) & np.isclose(df["load"].astype(float), load)].groupby("seed")[col].mean()
    return s


def _paired_series(x, y, alpha=ALPHA):
    seeds = sorted(set(x.index) & set(y.index))
    r = A.paired(x.loc[seeds].to_numpy(dtype=float), y.loc[seeds].to_numpy(dtype=float), alpha=alpha) if seeds else \
        dict(n=0, mean_diff=math.nan, ci=(math.nan, math.nan), d_z=math.nan, mean_a=math.nan, mean_b=math.nan)
    return dict(n=r["n"], seeds=seeds, mean_a=r["mean_a"], mean_b=r["mean_b"], mean_diff=r["mean_diff"],
                ci=list(r["ci"]), d_z=r.get("d_z", math.nan))


def partA_summaries(cyc, trials, wl0, loads):
    """Per (arm, load, cycle) means, A0 - A3 per cycle at every load, and the P6 / P9 tables (cycles 2-3 and whole run)."""
    import pandas as pd
    import tuning
    from tpa.config import load_config
    names = dict(zip(loads, ("light", "medium", "heavy")))
    ok = cyc[cyc["cycle"] > 0].copy() if len(cyc) and "cycle" in cyc.columns else pd.DataFrame()
    errors = cyc[cyc["cycle"] == 0][["run_id", "load", "test_run", "error"]].to_dict("records") if len(cyc) and "error" in cyc.columns else []
    if ok.empty:
        return dict(summary=pd.DataFrame(), contrasts=pd.DataFrame(), p6=[], p9=[], load_dependence=[], kappa_phi={},
                    errors=errors, per_run=pd.DataFrame())
    for b in range(PA.PHASE_BINS):
        ok[f"bin{b}_open_share"] = ok[f"bin{b}_open"] / PA.BIN_SLOTS
    # trial level (mean over the 3 test runs of a seed), per cycle and for cycles 2-3 pooled
    per_run = []
    for (run_id, arm, seed, L, k), g in ok.groupby(["run_id", "arm", "seed", "load", "test_run"]):
        rows = g.sort_values("cycle").to_dict("records")
        per_run.append(dict(arm=arm, seed=seed, load=L, test_run=k, cycle="2-3",
                            completion=PA.pooled_completion(rows), open_share=PA.pooled_open_share(rows),
                            open_share_postwarmup=PA.pooled_open_share(rows),
                            whole_open_share=sum(x["open_slots"] for x in rows) / (PA.P_SLOTS * PA.N_CYCLES),
                            lt_end3=rows[-1]["lt_end_lo"] if rows[-1]["cycle"] == 3 else math.nan,
                            lt_opened_c23_lo=rows[-1].get("lt_opened_c23_lo", math.nan),
                            lt_opened_c23_hi=rows[-1].get("lt_opened_c23_hi", math.nan),
                            n_int_c23_certain_max=max(x["n_intervals_certain"] for x in rows if x["cycle"] in (2, 3)),
                            n_int_c23_5pct_max=max(x["n_intervals_5pct"] for x in rows if x["cycle"] in (2, 3))))
    pr = pd.DataFrame(per_run)
    cols = ["completion", "open_share", "open_share_postwarmup", "lt_end_lo", "lt_end_hi"] + \
           [f"bin{b}_open_share" for b in range(PA.PHASE_BINS)]
    trial_c = ok.groupby(["arm", "seed", "load", "cycle"])[cols].mean().reset_index()
    trial_p = pr.groupby(["arm", "seed", "load"])[["completion", "open_share", "whole_open_share", "lt_end3"]].mean().reset_index()
    srows = []
    for (arm, L, c), g in trial_c.groupby(["arm", "load", "cycle"]):
        rec = dict(arm=arm, load=float(L), load_name=names.get(L, ""), cycle=str(int(c)), n=int(g["seed"].nunique()))
        for m in cols:
            mc = PA.mean_ci(g[m])
            rec[m], rec[m + "_ci95"] = mc["mean"], mc["ci95"]
        srows.append(rec)
    for (arm, L), g in trial_p.groupby(["arm", "load"]):
        rec = dict(arm=arm, load=float(L), load_name=names.get(L, ""), cycle="2-3", n=int(g["seed"].nunique()))
        for m in ("completion", "open_share"):
            mc = PA.mean_ci(g[m])
            rec[m], rec[m + "_ci95"] = mc["mean"], mc["ci95"]
        srows.append(rec)
    summary = pd.DataFrame(srows).sort_values(["arm", "load", "cycle"])
    # A0 - A3 per cycle (paired by seed), every load; the whole-run row repeats the run-row contrast for comparison
    crows = []
    for L in loads:
        for c in ("1", "2", "3", "2-3"):
            src = trial_c[trial_c["cycle"] == int(c)] if c != "2-3" else trial_p
            r = _paired_series(_by_seed(src, "A0", L, "completion"), _by_seed(src, "A3", L, "completion"))
            crows.append(dict(load=float(L), load_name=names[L], cycle=c, **r))
        w = A.contrast(trials, "completion", "A0", "A3", wl0, L, alpha=ALPHA)
        crows.append(dict(load=float(L), load_name=names[L], cycle="whole run (run rows)", n=w["n"], seeds=w["seeds"],
                          mean_a=w["mean_a"], mean_b=w["mean_b"], mean_diff=w["mean_diff"], ci=list(w["ci"]), d_z=w["d_z"]))
    contrasts = pd.DataFrame(crows)
    # P6 and P9, every memory variant (P6 and P9 are stated for A0; the others are reported beside it)
    cfg = load_config(overrides=tuning.adopted_overrides())
    kap = {int(s): PA.kappa_phi_of_seed(int(s), cfg) for s in sorted(set(pr["seed"]))}
    p6, p9 = [], []
    for (arm, L), g in pr.groupby(["arm", "load"]):
        lt3 = g["lt_end3"].to_numpy(dtype=float)
        has = bool(np.isfinite(lt3).any())          # False for A3 (flat store, no M_L) and arms without the column
        olo, ohi = g["lt_opened_c23_lo"].to_numpy(dtype=float), g["lt_opened_c23_hi"].to_numpy(dtype=float)
        p6.append(dict(arm=arm, load=float(L), load_name=names.get(L, ""), n_runs=int(len(g)), has_lt=has,
                       lt_end3_max=float(np.nanmax(lt3)) if has else math.nan,
                       lt_end3_median=float(np.nanmedian(lt3)) if has else math.nan,      # P6, reported descriptively
                       lt_end3_mean=PA.mean_ci(trial_p[(trial_p["arm"] == arm) & (trial_p["load"] == L)]["lt_end3"]),
                       runs_lt_end3_gt19=int(np.sum(lt3[np.isfinite(lt3)] > PA.P6_BOUND)) if has else None,
                       opened_c23_lo_max=float(np.nanmax(olo)) if np.isfinite(olo).any() else math.nan,
                       runs_opened_c23_certainly_gt19=int(np.sum(olo[np.isfinite(olo)] > PA.P6_BOUND)) if has else None,
                       runs_opened_c23_possibly_gt19=int(np.sum(ohi[np.isfinite(ohi)] > PA.P6_BOUND)) if has else None))
        tp = trial_p[(trial_p["arm"] == arm) & (trial_p["load"] == L)].set_index("seed")
        cyc23 = ok[(ok["arm"] == arm) & (ok["load"] == L) & ok["cycle"].isin([2, 3])]
        cyc_all = ok[(ok["arm"] == arm) & (ok["load"] == L)]
        p9.append(dict(arm=arm, load=float(L), load_name=names.get(L, ""),
                       open_share_c23=PA.mean_ci(tp["open_share"]), open_share_whole=PA.mean_ci(tp["whole_open_share"]),
                       open_share_whole_postwarmup=PA.mean_ci(tp["whole_open_share"] * PA.P_SLOTS * PA.N_CYCLES
                                                              / (PA.P_SLOTS * PA.N_CYCLES - PA.WARMUP_SLOTS)),
                       intervals_c23_certain=interval_distribution(cyc23, "certain"),
                       intervals_c23_5pct=interval_distribution(cyc23, "5pct"),
                       intervals_whole_certain=interval_distribution(cyc_all, "certain"),
                       intervals_whole_5pct=interval_distribution(cyc_all, "5pct"),
                       kappa_corr_c23=PA.correlation([kap[int(s)] for s in tp.index], tp["open_share"]),
                       kappa_corr_whole=PA.correlation([kap[int(s)] for s in tp.index], tp["whole_open_share"])))
    load_dep = []
    for arm in sorted(set(pr["arm"])):
        for lo_, hi_ in ((loads[0], loads[2]), (loads[1], loads[2])):
            for col, tag in (("open_share", "cycles 2-3"), ("whole_open_share", "whole run")):
                r = _paired_series(_by_seed(trial_p, arm, lo_, col), _by_seed(trial_p, arm, hi_, col))
                load_dep.append(dict(arm=arm, contrast=f"{names[lo_]} - {names[hi_]}", span=tag, **r))
    return dict(summary=summary, contrasts=contrasts, p6=p6, p9=p9, load_dependence=load_dep, kappa_phi=kap,
                errors=errors, per_run=pr)


def partA_p10(trials, loads):
    """P10 on MMPP: (a) open fraction per trial by load for each memory variant, over all and over post-warm-up
    slots (mean and 95 % t-CI over seeds; descriptive only); (b) the clustering ratio: not computable from the
    records, with the frozen statistic's base share."""
    import tuning
    from tpa.config import load_config
    cfg = load_config(overrides=tuning.adopted_overrides())
    names = dict(zip(loads, ("light", "medium", "heavy")))
    sub = trials[trials["workload"] == "mmpp"]
    out = dict(
        p10_computable=False,
        p10_statement=("P10 cannot be computed from the records (no per-slot gate decision and no MMPP state are written); "
                       "it is not evaluated (decision 2026-11-10, step 3). The open fraction by load below is descriptive only."),
        b_computable=False,
        b_reason=("the test records hold no per-slot gate decision (only the per-minute share cons_rate, tpa/run/metrics.py) "
                  "and no MMPP state (tpa/run/evaluator.py writes run and minute rows only); the MMPP state is policy-"
                  "independent and could be regenerated from (seed, block, test run), but it is not in the records"),
        base_share_within_w=PA.mmpp_base_share(cfg.workload.mmpp_p_stay, PA.P10_W, int(cfg.scenario.n_zones)),
        base_share_note=("stationary share of slots within w = 100 slots after a level change of any zone for "
                         f"mmpp_p_stay = {list(cfg.workload.mmpp_p_stay)}, Z = {int(cfg.scenario.n_zones)}: the frozen ratio "
                         "is ~1 whatever the gate does, so it could not discriminate even with per-slot records; on a finite "
                         "run it is about N / (N - c1 - 1) ~ 1.0007 > 1 for ANY gate (the slots before the first level change lie "
                         "in the warm-up, where the gate cannot open), i.e. biased toward confirming P10"),
        by_arm={})
    n_slots = float(sub["slots"].iloc[0]) if "slots" in sub.columns and len(sub) else float(PA.P_SLOTS * PA.N_CYCLES)
    for arm in sorted(set(sub["arm"])):
        fl = arm_flags(arm)
        if fl is None or not fl.memory:
            continue
        per = {}
        for L in loads:
            v = _by_seed(sub, arm, L, "cons_rate")
            per[names[L]] = dict(open_fraction=PA.mean_ci(v),
                                 open_fraction_postwarmup=PA.mean_ci([PA.post_warmup_share(x, int(n_slots)) for x in v]))
        out["by_arm"][arm] = dict(by_load=per)
    return out


def _ci_txt(mc: dict, nd: int = 4) -> str:
    if not mc or not np.isfinite(mc.get("mean", math.nan)):
        return "n/a"
    return f"{mc['mean']:.{nd}f} +- {mc['ci95']:.{nd}f}" if np.isfinite(mc.get("ci95", math.nan)) else f"{mc['mean']:.{nd}f}"


def dist_txt(d: dict) -> str:
    if not d or not d.get("n"):
        return "n/a"
    return " / ".join(f"{d[c]:.2f}" for c in CATEGORIES) + f" ({d['n']})"


def partA_tables(P: dict) -> list[str]:
    L = ["", "## 5. Per cycle on the sinusoid (Amendment 16, Part A, A2)", "",
         "Cycle c = minutes 60(c-1)..60c-1 of each 10,800-slot test run (run-start origin, P = 3,600 slots); trial = mean of "
         "the 3 test runs of a seed; 95 % t-CIs over seeds. The primary test H1 (section 1) is unchanged.", "",
         "### 5.1 A0 - A3 completion per cycle (paired by seed)", "",
         "| load | cycle | n | A0 (%) | A3 (%) | diff (pp) | 95 % CI (pp) | d_z |", "|---|---|---|---|---|---|---|---|"]
    for _, r in P["contrasts"].iterrows():
        L.append(f"| {r['load_name']} | {r['cycle']} | {r['n']} | {fmt(r['mean_a'])} | {fmt(r['mean_b'])} | {fmt(r['mean_diff'])} | "
                 f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}] | {fmt(r['d_z'], False, 3)} |")
    L += ["", "### 5.2 Every memory variant, per cycle", "",
          "| arm | load | cycle | n | completion (%) | gate-open share | LT occupancy at cycle end [lower, upper] | open share per phase bin 0-7 |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in P["summary"].iterrows():
        per_cycle = r["cycle"] != "2-3"
        bins = " / ".join(f"{float(r.get(f'bin{b}_open_share', math.nan)):.3f}" for b in range(PA.PHASE_BINS)) if per_cycle else "-"
        lo, hi = float(r.get("lt_end_lo", math.nan)), float(r.get("lt_end_hi", math.nan))
        lt = f"[{lo:.2f}, {hi:.2f}]" if per_cycle and np.isfinite(lo) else "-"
        L.append(f"| {r['arm']} | {r['load_name']} | {r['cycle']} | {r['n']} | {fmt(r['completion'])} | {float(r['open_share']):.4f} | {lt} | {bins} |")
    L += ["", "### 5.3 P6 (LT occupancy, sinusoid, A0: <= 19 always, about 5-10 typical; falsifier: > 19 after cycle 1), judged on cycles 2-3", "",
          "| arm | load | runs | LT at end of run (= max over cycles 2-3): max / median / mean +- CI | runs > 19 | rows opened in cycles 2-3: max (lower bound) | runs certainly > 19 | runs possibly > 19 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in P["p6"]:
        if not r["has_lt"]:
            L.append(f"| {md(r['arm'])} | {r['load_name']} | {r['n_runs']} | n/a (no long-term store) | n/a | n/a | n/a | n/a |")
            continue
        L.append(f"| {md(r['arm'])} | {r['load_name']} | {r['n_runs']} | {r['lt_end3_max']:.0f} / {r['lt_end3_median']:.1f} / {_ci_txt(r['lt_end3_mean'], 2)} | "
                 f"{r['runs_lt_end3_gt19']} | {r['opened_c23_lo_max']:.0f} | {r['runs_opened_c23_certainly_gt19']} | {r['runs_opened_c23_possibly_gt19']} |")
    L += ["", "LT occupancy never decreases within a run, so its maximum over cycles 2-3 equals the exact end-of-run value "
          "(run row lt_occupancy) and equals the whole-run maximum; 'rows opened in cycles 2-3' = end of cycle 3 minus end of "
          "cycle 1 (bounds: the end of cycle 1 is bracketed by the minute means around it). Which of the two judges P6 on "
          "cycles 2-3 is the supervisor's reading; both are given.", "",
          "### 5.4 P9 (gate-open fraction, sinusoid: 0.10-0.14, two intervals per cycle; refined 2026-11-09: correlates with "
          "kappa_phi across seeds and weakly decreases with load), judged on cycles 2-3", "",
          "| arm | load | open share, cycles 2-3 | whole run (all slots) | whole run (post-warm-up) | intervals, cycles 2-3, rule (ii): none / all8 / 1 / 2 / >=3 (n) | rule (i) | whole run, rule (ii) | r(kappa_phi, share) cycles 2-3 [95 % CI]; rho | whole run |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in P["p9"]:
        k23, kw = r["kappa_corr_c23"], r["kappa_corr_whole"]
        kt = lambda k: (f"{k['pearson_r']:.2f} [{k['ci95'][0]:.2f}, {k['ci95'][1]:.2f}]; {k['spearman_rho']:.2f} (n {k['n']})"
                        if np.isfinite(k["pearson_r"]) else f"n/a (n {k['n']})")
        L.append(f"| {r['arm']} | {r['load_name']} | {_ci_txt(r['open_share_c23'])} | {_ci_txt(r['open_share_whole'])} | "
                 f"{_ci_txt(r['open_share_whole_postwarmup'])} | {dist_txt(r['intervals_c23_5pct'])} | {dist_txt(r['intervals_c23_certain'])} | "
                 f"{dist_txt(r['intervals_whole_5pct'])} | {kt(k23)} | {kt(kw)} |")
    L += ["", "Load dependence of the open share (paired by seed):", "",
          "| arm | contrast | span | n | difference | 95 % CI |", "|---|---|---|---|---|---|"]
    for r in P["load_dependence"]:
        L.append(f"| {r['arm']} | {r['contrast']} | {r['span']} | {r['n']} | {r['mean_diff']:.4f} | [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] |")
    L += ["", "Open phase bins: 8 bins of 450 slots from the run start (the plan's origin t0); they are NOT phase-aligned across "
          "seeds (each seed's zone phases differ), so the bin means of 5.2 smear the per-run profiles, whose counts are in "
          "per_cycle_runs.csv. A minute row straddling a bin edge (minutes 7, 22, 37, 52 of each cycle, 30 | 30) is split "
          "half-half for the point value, with sharp bounds (each half holds at most 30 openings). Open intervals per cycle "
          "= circular runs of open bins, (i) bins certainly open (lower bound > 0), (ii) bins holding >= 5 % of the cycle's "
          "openings; 'none' = no opening and 'all8' = all 8 bins open are not resolvable at P/8 and are kept apart from 1 "
          "and 2 intervals; '>=3' is the P9 falsifier side. "
          "kappa_phi = |sum_z exp(2 i phi_z)| / Z of each seed's world (unweighted, as registered). Post-warm-up share = "
          "openings / (slots - 100): no opening is possible in the first w = 100 slots."]
    if P["errors"]:
        L += ["", f"Runs whose minute rows could not be split into 3 cycles (excluded): {P['errors']}"]
    return L


def p10_tables(p: dict) -> list[str]:
    L = ["", "## 6. P10 on MMPP (Amendment 16, Part A, A1; registered: openings cluster after level changes, fraction lower at light load; falsifier: uniform openings)", "",
         f"**{p['p10_statement']}** Clustering ratio: {p['b_reason']}. Base share of the frozen statistic: "
         f"{p['base_share_within_w']:.7f} ({p['base_share_note']}).", "",
         "Descriptive only, NOT a P10 test: open fraction per run (= per trial: the mean of the run's 3 test runs; seeds as "
         "units; share of slots with the gate open; mean +- 95 % t-CI):", "",
         "| arm | load | n | all slots | post-warm-up slots |", "|---|---|---|---|---|"]
    for arm, d in p["by_arm"].items():
        for name, v in d["by_load"].items():
            L.append(f"| {md(arm)} | {name} | {v['open_fraction']['n']} | {_ci_txt(v['open_fraction'])} | {_ci_txt(v['open_fraction_postwarmup'])} |")
    return L


# ---------------------------------------------------------------- evaluation-only sensitivity rows
# Evaluation-only sensitivity of the frozen A0 policies: own queue file runs/b1/runs_b1.db, run ids b1-A0-s<seed>-<tag>,
# one run row per sensitivity row (heavy load, test run 0). The table below equals scripts/b1_eval.py SETTINGS (a test
# checks it).
B1_DEFAULT = "default"
B1_SETTINGS = (
    ("theta_p-0.30", "memory.theta_p", 0.30), ("theta_p-0.40", "memory.theta_p", 0.40),
    ("rho_merge-0.90", "memory.rho_merge", 0.90), ("rho_merge-0.98", "memory.rho_merge", 0.98),
    ("theta_b-x0.75", "burst.theta_b", 0.675), ("theta_b-x1.25", "burst.theta_b", 1.125),
    ("alpha_t-x0.75", "drm.alpha_t", 0.3), ("alpha_t-x1.25", "drm.alpha_t", 0.5),
    ("alpha_m-x0.75", "drm.alpha_m", 0.225), ("alpha_m-x1.25", "drm.alpha_m", 0.375),
    ("alpha_d-x0.75", "drm.alpha_d", 0.225), ("alpha_d-x1.25", "drm.alpha_d", 0.375),
    ("omega_l-x0.75", "drm.omega_l", 0.45), ("omega_l-x1.25", "drm.omega_l", 0.75),
)
B1_MAIN_VALUES = {"memory.theta_p": 0.35, "memory.rho_merge": 0.95, "burst.theta_b": 0.9, "drm.alpha_t": 0.4,
                  "drm.alpha_m": 0.3, "drm.alpha_d": 0.3, "drm.omega_l": 0.6}
B1_TAGS = (B1_DEFAULT,) + tuple(t for t, _, _ in B1_SETTINGS)
B1_SEEDS = tuple(range(1, 31))
B1_WALL = ("wall_s", "reaction_ms_p50", "reaction_ms_p90", "recovery_ms")      # + every t_*_ms timer
B1_METRICS = (("completion", "completion"), ("gate_open_share", "cons_rate"), ("lt_occupancy", "lt_occupancy"))
# The share of UAV-slots with burst on, per setting, is exact from the run row: the simulator charges delta_e_w * slot_s
# for every UAV-slot with B_j = 1 (tpa/env/simulator.py:486-487) and the run row sums it (e_burst_j,
# tpa/run/metrics.py:95, 148), so share = e_burst_j / (delta_e_w slot_s M slots).
B1_DERIVED = ("burst_on_share",)
B1_ALL = tuple(n for n, _ in B1_METRICS) + B1_DERIVED


def b1_burst_on_share(rr: dict, ov: dict) -> float:
    """Share of UAV-slots with burst on in one run row, with the row's own configuration (default.toml + overrides)."""
    from tpa.config import load_config
    cfg = load_config(overrides=ov)
    den = float(cfg.burst.delta_e_w) * float(cfg.scenario.slot_s) * int(cfg.scenario.n_uav) * float(rr.get("slots", math.nan))
    e = float(rr.get("e_burst_j", math.nan))
    return e / den if den > 0 and np.isfinite(e) else math.nan


def b1_parse(rid: str) -> tuple[int, str]:
    if not rid.startswith("b1-A0-s"):
        raise ValueError(rid)
    s, tag = rid[len("b1-A0-s"):].split("-", 1)
    if tag not in B1_TAGS:
        raise ValueError(rid)
    return int(s), tag


def b1_rows(a, main_rows: list[dict]) -> list[dict] | None:
    """The rows of the sensitivity queue. Guard: every sensitivity row is done or failed; a row that is still open is
    accepted only if its source main row failed or it was stopped deliberately (reported missing). Without the queue
    file the analysis refuses (the rows are registered), unless --no-b1 is given explicitly (then None: no section 7
    table)."""
    if not Path(a.b1_db).exists():
        if a.no_b1:
            return None
        sys.exit(f"no B1 queue file at {a.b1_db}: the B1 rows are registered (ANALYSIS_PLAN A16B1.4) and read with the main "
                 f"campaign; pass --b1-db (and --b1-root if the B1 run directories are elsewhere), or --no-b1 to analyse "
                 f"without them (recorded in the outputs)")
    rows = q(a.b1_db, "SELECT run_id, arm, seed, phase, state, overrides, code_hash, error FROM runs ORDER BY run_id")
    src_state = {r["run_id"]: r["state"] for r in main_rows}
    # the registered rows: 15 per main A0 row of the queue (seeds 1-30 in the campaign), no more, no fewer
    a0_seeds = sorted(int(r["seed"]) for r in main_rows if r["run_id"] == f"main-A0-s{int(r['seed'])}-sinusoidal")
    want = {f"b1-A0-s{s}-{t}" for s in a0_seeds for t in B1_TAGS}
    have = {r["run_id"] for r in rows}
    if want - have:
        sys.exit(f"the B1 queue lacks {len(want - have)} registered rows (e.g. {sorted(want - have)[:3]}); it holds {len(have)}: "
                 f"refused (ANALYSIS_PLAN A16B1.2: 15 rows per main A0 row)")
    if have - want:
        sys.exit(f"the B1 queue holds {len(have - want)} rows that are not registered (e.g. {sorted(have - want)[:3]}): refused")
    open_ = [r["run_id"] for r in rows if r["state"] not in ("done", "failed") and not stopped_by_ruling(r)
             and src_state.get(f"main-A0-s{int(r['seed'])}-sinusoidal") != "failed"]
    if open_:
        sys.exit(f"{len(open_)} B1 rows are not finished (e.g. {open_[:3]}): the B1 rows are read with the main campaign, "
                 f"once every one of them is done (ANALYSIS_PLAN A16B1.4)")
    return rows


def _run_row_of(d: Path, heavy: float) -> dict | None:
    """The run row (row_kind run, heavy load, test run 0) of one evaluation directory; None if absent."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted(d.rglob("*.parquet")) if d.is_dir() else []
    if not files:
        return None
    df = pa.concat_tables([pq.read_table(f) for f in files], promote_options="default").to_pandas()
    sel = df[(df["row_kind"] == "run") & np.isclose(df["load"].astype(float), heavy) & (df["test_run"].astype(int) == 0)]
    sel = sel.drop_duplicates(subset=[c for c in ("row_kind", "load", "test_run", "t_end") if c in sel.columns])
    if len(sel) != 1:
        return None
    return sel.iloc[0].to_dict()


def _same(u, v) -> bool:
    fu = isinstance(u, (float, np.floating)) and np.isnan(u)
    fv = isinstance(v, (float, np.floating)) and np.isnan(v)
    return (fu and fv) or (not fu and not fv and u == v)


def b1_analysis(a, rows: list[dict], heavy: float) -> dict:
    """Per setting the paired difference to the default row of the same seed (completion, gate-open share, end-of-run
    long-term occupancy, burst-on share), mean and 95 % t-interval; the levels beside them; the reproduction check of
    the default rows against the main test's first heavy run. Descriptive only: no test, no p-value, no Holm."""
    import pandas as pd
    recs, missing, errors, full = [], [], [], {}
    for r in rows:
        try:
            seed, tag = b1_parse(r["run_id"])
        except ValueError:
            errors.append(f"{r['run_id']}: not a B1 run id")
            continue
        key, val = next(((k, v) for t, k, v in B1_SETTINGS if t == tag), (None, None))
        ov = json.loads(r["overrides"] or "{}")
        if ov.get("run.test_runs") != 1 or (key is not None and ov.get(key) != val) or r["arm"] != "A0" or int(r["seed"]) != seed:
            errors.append(f"{r['run_id']}: queue row does not carry its setting (overrides {ov.get(key) if key else '-'})")
            continue
        if r["state"] != "done":
            missing.append(r["run_id"])
            continue
        rr = _run_row_of(Path(a.b1_root) / r["phase"] / r["run_id"] / "test", heavy)
        if rr is None:
            missing.append(r["run_id"])
            continue
        try:
            ck = json.loads((Path(a.b1_root) / r["phase"] / r["run_id"] / "RESULT.json").read_text())["b1"]["source_ckpt_sha256"]
        except (OSError, ValueError, KeyError, TypeError):
            ck = None
        full[(seed, tag)] = rr
        recs.append(dict(run_id=r["run_id"], seed=seed, tag=tag, key=key or "-", value=val, code_hash=r["code_hash"][:12],
                         source_ckpt_sha256=ck, **{name: float(rr.get(col, math.nan)) for name, col in B1_METRICS},
                         burst_on_share=b1_burst_on_share(rr, ov)))
    runs = pd.DataFrame(recs, columns=["run_id", "seed", "tag", "key", "value", "code_hash", "source_ckpt_sha256"]
                        + list(B1_ALL))
    # one checkpoint per seed, the main row's selected one: otherwise the seed's sensitivity rows are left out.
    # Built from the plain records, not the DataFrame: pandas turns a missing (None) hash into NaN.
    excluded, compared = {}, []
    for seed in sorted({rec["seed"] for rec in recs}):
        hs = [rec["source_ckpt_sha256"] for rec in recs if rec["seed"] == seed]
        if any(not isinstance(h, str) or len(h) != 64 for h in hs) or len(set(hs)) != 1:
            excluded[seed] = "its B1 rows carry different or missing checkpoint hashes"
            continue
        p = a.runs_root / "main" / f"main-A0-s{seed}-sinusoidal" / "ckpt_best.pt"
        if p.is_file():
            compared.append(seed)
            if _sha256(p) != hs[0]:
                excluded[seed] = "the checkpoint hash of its B1 rows differs from the main row's ckpt_best.pt"
    for s, why in sorted(excluded.items()):
        errors.append(f"seed {s}: {why}; its B1 rows are left out")
    runs = runs.assign(excluded=runs["seed"].isin(list(excluded)))
    all_runs, runs = runs, runs[~runs["excluded"]]
    dflt = runs[runs["tag"] == B1_DEFAULT].set_index("seed")
    table = []
    for tag, key, val in B1_SETTINGS:
        s = runs[runs["tag"] == tag].set_index("seed")
        seeds = sorted(set(s.index) & set(dflt.index))
        rec = dict(tag=tag, key=key, value=val, main_value=B1_MAIN_VALUES[key], n=len(seeds), seeds=seeds)
        for name in B1_ALL:
            x = s.loc[seeds, name].to_numpy(float) if seeds else np.zeros(0)
            y = dflt.loc[seeds, name].to_numpy(float) if seeds else np.zeros(0)
            d, mx, my = PA.mean_ci(x - y), PA.mean_ci(x), PA.mean_ci(y)
            rec.update({f"{name}_diff_mean": d["mean"], f"{name}_diff_ci95": d["ci95"],
                        f"{name}_diff_ci_lo": d["mean"] - d["ci95"], f"{name}_diff_ci_hi": d["mean"] + d["ci95"],
                        f"{name}_mean": mx["mean"], f"{name}_ci95": mx["ci95"], f"{name}_default_mean": my["mean"],
                        f"{name}_default_ci95": my["ci95"]})
        table.append(rec)
    repro = []
    for seed in sorted(dflt.index):
        b = full[(seed, B1_DEFAULT)]
        m = _run_row_of(a.runs_root / "main" / f"main-A0-s{seed}-sinusoidal" / "test", heavy)
        if m is None:
            repro.append(dict(seed=seed, main_found=False, identical=False, n_columns=0, differing=""))
            continue
        cols = sorted((set(b) | set(m)) - set(B1_WALL))
        cols = [c for c in cols if not (c.startswith("t_") and c.endswith("_ms"))]
        diff = [c for c in cols if not _same(b.get(c, math.nan), m.get(c, math.nan))]
        rec = dict(seed=seed, main_found=True, identical=not diff, n_columns=len(cols), differing=";".join(diff))
        for name, col in B1_METRICS:
            rec[f"{name}_b1"], rec[f"{name}_main"] = float(b.get(col, math.nan)), float(m.get(col, math.nan))
            rec[f"{name}_absdiff"] = abs(rec[f"{name}_b1"] - rec[f"{name}_main"])
        repro.append(rec)
    repro = pd.DataFrame(repro)
    n_found = int(repro["main_found"].sum()) if len(repro) else 0
    summ = dict(n_rows=len(rows), n_done=sum(r["state"] == "done" for r in rows), missing=missing, errors=errors,
                excluded_seeds=sorted(excluded), ckpt_compared_seeds=compared, n_default_seeds=int(len(dflt)), repro_seeds=n_found,
                repro_identical=int(repro["identical"].sum()) if len(repro) else 0,
                repro_max_absdiff={name: (float(repro.loc[repro["main_found"], f"{name}_absdiff"].max()) if n_found else math.nan)
                                   for name, _ in B1_METRICS},
                # default-row completion against the main test's first heavy run, bit for bit
                repro_completion_differs=[int(r["seed"]) for _, r in repro.iterrows()
                                          if r["main_found"] and not _same(r["completion_b1"], r["completion_main"])]
                                         if n_found else [])
    return dict(runs=all_runs, table=pd.DataFrame(table), repro=repro, summary=summ)


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def b1_tables(b: dict, heavy: float) -> list[str]:
    s = b["summary"]
    L = ["", f"## 7. Sensitivity of the frozen A0 policies (Amendment 16, Part B1; heavy load {heavy}, the first registered "
             f"test run of each seed; descriptive only, no test, no Holm)", "",
         f"B1 rows: {s['n_rows']}, done {s['n_done']}; missing (failed, never run, or no table): {s['missing'] or 'none'}; "
         f"inconsistent rows or seeds: {s['errors'] or 'none'}. Checkpoint of every seed's B1 rows compared with the main "
         f"row's ckpt_best.pt for {len(s['ckpt_compared_seeds'])} seeds. Each difference is setting - default on the same seed "
         f"(same checkpoint, same trace, same code path); mean over seeds with the 95 % t-interval. Gate-open share = the "
         f"run's share of gate-open slots (cons_rate); LT occupancy = long-term slots filled at the end of the run; "
         f"burst-on share = the share of UAV-slots with burst mode on (A16B1.7).", "",
         "| setting | key | value (main) | n | completion diff (pp) | 95 % CI (pp) | gate-open share: setting / default | "
         "diff (95 % CI) | LT occupancy: setting / default | diff (95 % CI) | burst-on share: setting / default | diff (95 % CI) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in b["table"].iterrows():
        L.append(f"| {r['tag']} | {r['key']} | {r['value']:g} ({r['main_value']:g}) | {r['n']} | "
                 f"{fmt(r['completion_diff_mean'])} | [{fmt(r['completion_diff_ci_lo'])}, {fmt(r['completion_diff_ci_hi'])}] | "
                 f"{r['gate_open_share_mean']:.4f} / {r['gate_open_share_default_mean']:.4f} | "
                 f"{r['gate_open_share_diff_mean']:.4f} (+- {r['gate_open_share_diff_ci95']:.2g}) | "
                 f"{r['lt_occupancy_mean']:.2f} / {r['lt_occupancy_default_mean']:.2f} | "
                 f"{r['lt_occupancy_diff_mean']:.2f} (+- {r['lt_occupancy_diff_ci95']:.2g}) | "
                 f"{r['burst_on_share_mean']:.4f} / {r['burst_on_share_default_mean']:.4f} | "
                 f"{r['burst_on_share_diff_mean']:.4f} (+- {r['burst_on_share_diff_ci95']:.2g}) |")
    mx = s["repro_max_absdiff"]
    dif = s["repro_completion_differs"]
    L += ["", f"Default B1 row vs the main test's first heavy run, completion bit for bit (A16B1.7): "
              f"{'identical for every one of the ' + str(s['repro_seeds']) + ' seeds' if not dif else 'differs for seeds ' + str(dif)}."]
    L += ["", f"Reproduction check (A16B1.3): of {s['repro_seeds']} seeds with a main test table, {s['repro_identical']} default "
              f"rows are identical to the main test's first heavy run in every column but the wall-clock ones; largest "
              f"absolute differences: completion {mx['completion']:.3g}, gate-open share {mx['gate_open_share']:.3g}, "
              f"LT occupancy {mx['lt_occupancy']:.3g} (`b1_reproduction.csv`). Tables: `b1_sensitivity.csv`, `b1_runs.csv`."]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("main", "robust", "tune"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--db", type=Path, default=ROOT / "runs" / "runs.db")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--b1-db", type=Path, default=None, help="Amendment 16 B1 queue (default <runs-root>/b1/runs_b1.db)")
    ap.add_argument("--b1-root", type=Path, default=None, help="B1 runs root (default <runs-root>/b1)")
    ap.add_argument("--no-b1", action="store_true", help="--phase main without the B1 rows (recorded in the outputs)")
    a = ap.parse_args()
    a.b1_root = a.b1_root or (a.b1_db.parent if a.b1_db else a.runs_root / "b1")   # queue file inside its runs root
    a.b1_db = a.b1_db or a.runs_root / "b1" / "runs_b1.db"
    if a.phase == "tune" and not a.rehearsal:
        sys.exit("tuning rows are read only with --rehearsal")
    if a.rehearsal and a.phase != "tune":
        sys.exit("--rehearsal is for --phase tune only (never main or robust)")
    a.block = "val_full" if a.phase == "tune" else "test"
    tag = "REHEARSAL on tuning data (validation block val_full, validation seeds; selected configuration per tuned " \
          "arm; NOT a result)" if a.rehearsal else f"phase {a.phase}, block {a.block}"
    a.out.mkdir(parents=True, exist_ok=True)
    rows, done = load_rows(a)
    b1r = b1_rows(a, rows) if a.phase == "main" and not a.rehearsal else None      # sensitivity guard, before any read
    trials, missing_tables = build_trials(a, done)
    loads = sorted({round(float(x), 6) for x in trials["load"]})
    if len(loads) != 3:
        sys.exit(f"expected 3 loads, found {loads}")
    light, medium, heavy = loads
    workloads = sorted(set(trials["workload"]))
    wl0 = "sinusoidal" if "sinusoidal" in workloads else workloads[0]
    failed = [r["run_id"] for r in rows if r["state"] == "failed"]
    stopped = [r["run_id"] for r in rows if stopped_by_ruling(r)]
    n_by_arm = trials[np.isclose(trials["load"].astype(float), heavy) & (trials["workload"] == wl0)] \
        .groupby("arm")["seed"].nunique().to_dict()

    contrasts = []
    if a.phase != "robust":
        contrasts.append(run_contrast(trials, "H1", "A0 - A3, heavy", "A0", "A3", wl0, heavy))
        for b in S1:
            contrasts.append(run_contrast(trials, "S1", f"A0 - {b}, heavy", "A0", b, wl0, heavy))
        for x, y in S2:
            contrasts.append(run_contrast(trials, "S2", f"{x} - {y}, heavy", x, y, wl0, heavy))
        for b in S3_B:
            for L, name in ((light, "light"), (medium, "medium"), (heavy, "heavy")):
                contrasts.append(run_contrast(trials, "S3", f"A0 - {b}, {name}", "A0", b, wl0, L))
        for L, name in ((light, "light"), (medium, "medium"), (heavy, "heavy")):
            contrasts.append(run_contrast(trials, "S4", f"A0 - A9, {name}", "A0", "A9", wl0, L))
    else:
        for wl in workloads:
            for b in ROBUST_ARMS:
                contrasts.append(run_contrast(trials, f"R:{wl}", f"A0 - {b}, heavy, {wl}", "A0", b, wl, heavy))
    # Holm within each secondary family (H1 is a single confirmatory test: no adjustment)
    for fam in sorted({c["family"] for c in contrasts} - {"H1"}):
        idx = [i for i, c in enumerate(contrasts) if c["family"] == fam]
        adj = A.holm([contrasts[i]["p"] for i in idx])
        for i, p in zip(idx, adj):
            contrasts[i]["p_holm"] = float(p) if np.isfinite(p) else math.nan
    for c in contrasts:
        c.setdefault("p_holm", math.nan)
    interaction = None
    if a.phase != "robust":
        s4 = {c["label"].split(", ")[1]: c for c in contrasts if c["family"] == "S4"}
        if s4["heavy"]["n"] >= 2 and s4["light"]["n"] >= 2 and s4["heavy"]["seeds"] == s4["light"]["seeds"]:
            dd = np.asarray(s4["heavy"]["diffs"]) - np.asarray(s4["light"]["diffs"])
            from scipy import stats as st
            t = st.ttest_1samp(dd, 0.0)
            interaction = dict(n=int(dd.size), mean_pp=100 * float(dd.mean()), t=float(t.statistic), p=float(t.pvalue))
    # tier C (effect size and CI; a p-value only for A0 - A0-tmem25 at heavy load) and the exploratory comparisons
    # (NOT registered): outside every Holm family
    tier_c, exploratory = [], []
    if a.phase != "robust":
        for x, y in TIER_C:
            for Lv, name in ((heavy, "heavy"), (medium, "medium"), (light, "light")):
                tier_c.append(descriptive_contrast(trials, "C", f"{x} - {y}, {name}", x, y, wl0, Lv,
                                                   with_p=(y == TMEM25 and Lv == heavy)))
        for x, b in EXPLORATORY:
            exploratory.append(descriptive_contrast(trials, "X", f"{x} - {b}, heavy", x, b, wl0, heavy))

    # descriptive table: mean and 95 % t-CI per (arm, workload, load) over seeds
    import pandas as pd
    from scipy import stats as st
    desc_rows = []
    metrics = [m for m in DESCRIPTIVE if m in trials.columns]
    for (arm, wl, L), g in trials.groupby(["arm", "workload", "load"]):
        rec = dict(arm=arm, workload=wl, load=float(L), n=int(g["seed"].nunique()))
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            rec[m] = float(v.mean()) if v.size else math.nan
            rec[m + "_ci95"] = float(st.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size)) if v.size >= 2 else math.nan
        desc_rows.append(rec)
    desc = pd.DataFrame(desc_rows).sort_values(["workload", "load", "arm"])
    # per-cycle statistics on the sinusoid (main / rehearsal); P10 on MMPP (robust)
    partA, cyc = None, None
    if a.phase != "robust":
        cyc = partA_cycles(a, [r for r in done if r["workload"] == wl0])
        partA = partA_summaries(cyc, trials, wl0, loads) if len(cyc) else None
    p10 = partA_p10(trials, loads) if a.phase == "robust" and "mmpp" in workloads else None
    b1 = b1_analysis(a, b1r, heavy) if b1r is not None else None                   # evaluation-only sensitivity rows

    # outputs (in a rehearsal every table carries the REHEARSAL tag in a column, the JSON files in a field)
    if a.rehearsal:
        trials = trials.assign(tag=tag)
        desc = desc.assign(tag=tag)
        if cyc is not None:
            cyc = cyc.assign(tag=tag)
        if partA is not None:
            partA["summary"] = partA["summary"].assign(tag=tag)
            partA["contrasts"] = partA["contrasts"].assign(tag=tag)
        for c in contrasts + tier_c + exploratory:
            c["tag"] = tag
    trials.to_csv(a.out / "trials.csv", index=False)
    desc.to_csv(a.out / "descriptive.csv", index=False)
    pd.DataFrame([{k: v for k, v in c.items() if k not in ("diffs", "seeds", "unpaired_seeds", "tost", "normality")}
                  | dict(ci_lo=c["ci"][0], ci_hi=c["ci"][1], p_tost=c["tost"]["p_tost"],
                         equivalent=c["tost"]["equivalent"], shapiro_p=c["normality"]["shapiro_p"],
                         wilcoxon_p=c["normality"]["wilcoxon_p"]) for c in contrasts]).to_csv(a.out / "contrasts.csv", index=False)
    pd.DataFrame([{k: v for k, v in c.items() if k not in ("seeds", "unpaired_seeds", "ci")} | dict(ci_lo=c["ci"][0], ci_hi=c["ci"][1])
                  for c in tier_c]).to_csv(a.out / "tierc.csv", index=False)
    pd.DataFrame([{k: v for k, v in c.items() if k not in ("seeds", "unpaired_seeds", "ci")} | dict(ci_lo=c["ci"][0], ci_hi=c["ci"][1])
                  for c in exploratory]).to_csv(a.out / "exploratory.csv", index=False)
    if partA is not None:
        cyc.to_csv(a.out / "per_cycle_runs.csv", index=False)
        partA["summary"].to_csv(a.out / "per_cycle_summary.csv", index=False)
        partA["contrasts"].to_csv(a.out / "per_cycle_contrasts.csv", index=False)
        (a.out / "p6_p9.json").write_text(json.dumps({k: partA[k] for k in ("p6", "p9", "load_dependence", "kappa_phi", "errors")} | dict(tag=tag),
                                                     indent=1, default=str))
    if p10 is not None:
        (a.out / "p10.json").write_text(json.dumps(p10, indent=1, default=str))
    if b1 is not None:
        b1["table"].assign(seeds=[" ".join(map(str, s)) for s in b1["table"]["seeds"]]).to_csv(a.out / "b1_sensitivity.csv", index=False)
        b1["runs"].to_csv(a.out / "b1_runs.csv", index=False)
        b1["repro"].to_csv(a.out / "b1_reproduction.csv", index=False)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = [f"# Campaign analysis: {tag}", "",
         f"Generated {now} by `scripts/analyze_campaign.py`. Rows of phase `{a.phase}` in the queue: {len(rows)}; done {len(done)}; "
         f"failed (reported missing): {failed or 'none'}; stopped by a ruling (reported missing): {stopped or 'none'}; run directories without a `{a.block}` table: {missing_tables or 'none'}. "
         f"Code hashes in the data: {sorted(set(trials['code_hash']))}. Workloads: {workloads}. Loads (light, medium, heavy): "
         f"{light}, {medium}, {heavy}. Seeds per arm at heavy load: {n_by_arm}. alpha = {ALPHA}; SESOI = {100 * SESOI:.1f} pp.",
         ""]
    if a.rehearsal:
        L += ["**REHEARSAL: every number here comes from validation seeds and the validation block of tuning runs; "
              "n is 1-2 seeds per arm; p-values and CIs are exercised, not interpreted. No main or robust row was read.**", ""]
    if a.phase != "robust":
        h = contrasts[0]
        L += ["## 1. Primary hypothesis H1: A0 vs A3, heavy-load completion (paired by seed)", "",
              f"n = {h['n']} pairs (seeds {h['seeds']}); mean A0 {fmt(h['mean_a'])} %, A3 {fmt(h['mean_b'])} %; difference "
              f"{fmt(h['mean_diff'])} pp, 95 % CI [{fmt(h['ci'][0])}, {fmt(h['ci'][1])}] pp; t = {fmt(h['t'], False, 4)}, "
              f"p = {fmt(h['p'], False, 3)}; Cohen's d (pooled) = {fmt(h['cohen_d_pooled'], False, 3)}, d_z = {fmt(h['d_z'], False, 3)}; "
              f"MDE at this SD_d = {fmt(h['mde_pp'] / 100 if np.isfinite(h['mde_pp']) else math.nan)} pp.",
              f"TOST equivalence (+-{100 * SESOI:.1f} pp): p = {fmt(h['tost']['p_tost'], False, 3)} -> "
              f"{'equivalent' if h['tost']['equivalent'] else 'not shown equivalent' if h['tost']['equivalent'] is not None else 'n/a'}. "
              f"Normality: Shapiro p = {fmt(h['normality']['shapiro_p'], False, 3)}; {h['normality']['note'] or 'no sensitivity test needed'}"
              f"{'' if not np.isfinite(h['normality']['wilcoxon_p']) else '; Wilcoxon p = ' + fmt(h['normality']['wilcoxon_p'], False, 3)}.", ""]
    L += ["## 2. Contrasts (Holm within each family; completion, pp)", "",
          "| family | contrast | n | mean diff | 95 % CI | d | d_z | p | Holm p | TOST p | MDE (pp) | Shapiro p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in contrasts:
        L.append(f"| {md(c['family'])} | {md(c['label'])} | {c['n']} | {fmt(c['mean_diff'])} | [{fmt(c['ci'][0])}, {fmt(c['ci'][1])}] | "
                 f"{fmt(c['cohen_d_pooled'], False, 3)} | {fmt(c['d_z'], False, 3)} | {fmt(c['p'], False, 3)} | "
                 f"{fmt(c['p_holm'], False, 3)} | {fmt(c['tost']['p_tost'], False, 3)} | "
                 f"{fmt(c['mde_pp'] / 100 if np.isfinite(c['mde_pp']) else math.nan)} | {fmt(c['normality']['shapiro_p'], False, 3)} |")
    if interaction:
        L += ["", f"S4 interaction (difference at heavy - difference at light, one-sample t): n = {interaction['n']}, mean "
              f"{interaction['mean_pp']:.2f} pp, t = {interaction['t']:.3g}, p = {interaction['p']:.3g}."]
    elif a.phase != "robust":
        L += ["", "S4 interaction: not computable (fewer than 2 complete pairs at both loads)."]
    for title, cs in (("## 2a. Tier C: effect size and 95 % CI (A7.1, A13.2, A13.3; not in any Holm family; the only p-value is the "
                       "A13.3 paired t of A0 - A0-tmem25 at heavy load)", tier_c),
                      ("## 2b. EXPLORATORY, NOT REGISTERED (Amendment 16 A3): A8 - B_x and A9 - B_x at heavy load, descriptive "
                       "(separates burst mode and visiting order from learning; no p-values, no Holm)", exploratory)):
        if cs:
            L += ["", title, "", "| contrast | n | mean diff (pp) | 95 % CI (pp) | d | d_z | paired t p (A13.3 only) |",
                  "|---|---|---|---|---|---|---|"]
            for c in cs:
                L.append(f"| {md(c['label'])} | {c['n']} | {fmt(c['mean_diff'])} | [{fmt(c['ci'][0])}, {fmt(c['ci'][1])}] | "
                         f"{fmt(c['cohen_d_pooled'], False, 3)} | {fmt(c['d_z'], False, 3)} | "
                         f"{fmt(c['p_paired_t'], False, 3) if 'p_paired_t' in c else '-'} |")
    L += ["", "## 3. Descriptive outcomes (mean over seeds, +- 95 % t-CI half-width; wall-clock columns excluded)", ""]
    cols = metrics[:8]
    L += ["| arm | workload | load | n | " + " | ".join(cols) + " |", "|---|---|---|---|" + "---|" * len(cols)]
    for _, r in desc.iterrows():
        L.append(f"| {md(r['arm'])} | {md(r['workload'])} | {r['load']:.3f} | {int(r['n'])} | " +
                 " | ".join(f"{r[m]:.4g} +- {r[m + '_ci95']:.2g}" if np.isfinite(r[m + '_ci95']) else f"{r[m]:.4g}" for m in cols) + " |")
    L += ["", f"Full descriptive table (all {len(metrics)} metrics): `descriptive.csv`; trial table: `trials.csv`; contrasts: `contrasts.csv`.", ""]

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = []
    for fam in sorted({c["family"] for c in contrasts}):
        p = a.out / f"fig_forest_{fam.replace(':', '_').replace('|', '_').replace('=', '-')}.png"
        forest([c for c in contrasts if c["family"] == fam], f"{fam} ({tag[:40]})", p)
        figs.append(p.name)
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, g in trials[trials["workload"] == wl0].groupby("arm"):
        m = g.groupby("load")["completion"].mean()
        ax.plot(m.index.astype(float), 100 * m.values, marker="o", ms=3, lw=1, label=arm)
    ax.set_xlabel("offered load"); ax.set_ylabel("completion (%)"); ax.set_title(f"completion vs load, {wl0} ({tag[:30]})", fontsize=9)
    ax.legend(fontsize=6, ncol=3); fig.tight_layout(); fig.savefig(a.out / "fig_completion_vs_load.png", dpi=130); plt.close(fig)
    figs.append("fig_completion_vs_load.png")
    ms = minute_series(a, [r for r in done if r["workload"] == wl0], ("A0", "A3", "A9"), heavy)
    if ms:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for arm, s in ms.items():
            ax.plot(s.index.astype(float), 100 * s.values, lw=1, label=arm)
        ax.set_xlabel("minutes since the start of the evaluation run (mean over seeds and runs)"); ax.set_ylabel("completion (%)")
        ax.set_title(f"per-minute completion, heavy load ({tag[:30]})", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.out / "fig_minute_series_heavy.png", dpi=130); plt.close(fig)
        figs.append("fig_minute_series_heavy.png")
    L += ["## 4. Figures", ""] + [f"- `{f}`" for f in figs]
    if partA is not None:
        L += partA_tables(partA)
    if p10 is not None:
        L += p10_tables(p10)
    if b1 is not None:
        L += b1_tables(b1, heavy)
    elif a.phase == "main" and not a.rehearsal:
        L += ["", f"## 7. Amendment 16, Part B1: NOT READ (--no-b1 given; no B1 queue file at `{a.b1_db}`); no sensitivity table."]
    (a.out / "TABLES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    summary = dict(utc=now, tag=tag, phase=a.phase, block=a.block, rows=len(rows), done=len(done), failed=failed, stopped=stopped,
                   missing_tables=missing_tables, loads=loads, workloads=workloads, n_by_arm=n_by_arm,
                   contrasts=[{k: v for k, v in c.items() if k != "diffs"} for c in contrasts],
                   interaction=interaction, figures=figs, code_hashes=sorted(set(trials["code_hash"])),
                   tier_c=tier_c, exploratory=exploratory, p10=p10,
                   b1=None if b1 is None else dict(summary=b1["summary"], table=b1["table"].to_dict("records")),
                   b1_not_read=bool(a.phase == "main" and not a.rehearsal and b1 is None),
                   partA=None if partA is None else dict(contrasts=partA["contrasts"].to_dict("records"), p6=partA["p6"],
                                                         p9=partA["p9"], load_dependence=partA["load_dependence"],
                                                         errors=partA["errors"]))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"{tag}: {len(done)} runs, {len(trials)} trials, {len(contrasts)} contrasts, {len(figs)} figures -> {a.out}")


if __name__ == "__main__":
    main()
