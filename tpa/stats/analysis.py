"""Paired comparisons, effect sizes and multiplicity control.

- Primary: A0 vs A3, heavy-load completion, paired t over seeds, alpha = 0.05.
- Secondary contrasts: Holm step-down adjusted p-values.
- Cohen's d on the pooled SD, with d_z (mean diff / SD of diffs) also reported.
- The trial unit is the mean of the test runs of one (arm, seed, workload, load).

All tests are two-sided.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import stats as _st

TRIAL_KEYS = ("arm", "seed", "workload", "load")


def _as1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a


def cohen_d_pooled(a, b) -> float:
    """Cohen's d = (mean a - mean b) / s_p, s_p = sqrt(((n_a-1)s_a^2 + (n_b-1)s_b^2)/(n_a+n_b-2)).

    For paired data (n_a = n_b) s_p = sqrt((s_a^2 + s_b^2)/2).
    """
    a, b = _as1d(a), _as1d(b)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    diff = a.mean() - b.mean()
    if sp2 == 0:
        return 0.0 if diff == 0 else math.copysign(math.inf, diff)
    return float(diff / math.sqrt(sp2))


def _t_from(mean: float, se: float, df: float, alpha: float):
    if se == 0:
        t = 0.0 if mean == 0 else math.copysign(math.inf, mean)
        p = 1.0 if mean == 0 else 0.0
        return t, p, (mean, mean)
    t = mean / se
    p = float(2.0 * _st.t.sf(abs(t), df))
    q = float(_st.t.ppf(1.0 - alpha / 2.0, df))
    return float(t), p, (float(mean - q * se), float(mean + q * se))


def paired(a, b, alpha: float = 0.05) -> dict:
    """Paired t-test of a vs b (difference a - b), two-sided.

    Returns {n, df, mean_a, mean_b, mean_diff, sd_diff, se, ci (1-alpha), t, p,
    cohen_d_pooled, d_z}. Pairs with a NaN on either side are dropped (counted
    in ``n_dropped``).
    """
    a, b = _as1d(a), _as1d(b)
    if a.shape != b.shape:
        raise ValueError(f"paired(): length mismatch {a.size} vs {b.size}")
    keep = ~(np.isnan(a) | np.isnan(b))
    n_drop = int((~keep).sum())
    a, b = a[keep], b[keep]
    n = a.size
    out = {"n": n, "n_dropped": n_drop, "df": n - 1, "alpha": alpha}
    if n < 2:
        out.update(mean_a=float(a.mean()) if n else math.nan, mean_b=float(b.mean()) if n else math.nan,
                   mean_diff=math.nan, sd_diff=math.nan, se=math.nan, ci=(math.nan, math.nan),
                   t=math.nan, p=math.nan, cohen_d_pooled=math.nan, d_z=math.nan)
        return out
    d = a - b
    md = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t, p, ci = _t_from(md, se, n - 1, alpha)
    if sd == 0:
        dz = 0.0 if md == 0 else math.copysign(math.inf, md)
    else:
        dz = md / sd
    out.update(mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=md, sd_diff=sd,
               se=se, ci=ci, t=t, p=p, cohen_d_pooled=cohen_d_pooled(a, b), d_z=float(dz))
    return out


def welch(a, b, alpha: float = 0.05) -> dict:
    """Welch's unequal-variance t-test of a vs b (difference a - b), two-sided.

    df = (s_a^2/n_a + s_b^2/n_b)^2 / ((s_a^2/n_a)^2/(n_a-1) + (s_b^2/n_b)^2/(n_b-1)).
    NaNs are dropped per sample.
    """
    a, b = _as1d(a), _as1d(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = a.size, b.size
    out = {"n_a": na, "n_b": nb, "alpha": alpha}
    if na < 2 or nb < 2:
        out.update(mean_diff=math.nan, se=math.nan, df=math.nan, ci=(math.nan, math.nan),
                   t=math.nan, p=math.nan, cohen_d_pooled=math.nan)
        return out
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = math.sqrt(va + vb)
    md = float(a.mean() - b.mean())
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)) if se > 0 else float(na + nb - 2)
    t, p, ci = _t_from(md, se, df, alpha)
    out.update(mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=md, se=se,
               df=float(df), ci=ci, t=t, p=p, cohen_d_pooled=cohen_d_pooled(a, b))
    return out


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values, monotone, in the input order.

    Sorted ascending p_(1) <= ... <= p_(m):  p~_(i) = max_{j<=i} min(1, (m-j+1) p_(j)).
    NaN entries stay NaN and do not count towards m.
    """
    p = _as1d(p)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    if np.any((pv < 0) | (pv > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(pv, kind="mergesort")          # stable: ties keep input order
    adj_sorted = np.minimum(1.0, (m - np.arange(m)) * pv[order])
    adj_sorted = np.maximum.accumulate(adj_sorted)    # enforce monotonicity
    adj = np.empty(m)
    adj[order] = adj_sorted
    out[ok] = adj
    return out


def holm_reject(p: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    adj = holm(p)
    return np.where(np.isnan(adj), False, adj <= alpha)


# ============================================================== trial table
def trial_table(path: str | Path, metrics: Iterable[str] | None = None,
                run_col: str = "test_run", row_kind_col: str = "row_kind",
                run_kind: str = "run"):
    """Per-(arm, seed, workload, load) trial means from the test Parquet files.

    Reads every ``*.parquet`` under ``path`` recursively. Expected columns: the
    keys ``arm, seed, workload, load``, the test-run index ``test_run`` and
    numeric metric columns. If a ``row_kind`` column exists, only rows with
    ``row_kind == "run"`` (full-run summaries) are used, so per-minute series in
    the same files do not enter the trial means. Each test run is first reduced
    to its mean (a no-op for one summary row per run), then the trial value is
    the mean over test runs (the trial unit is the mean of the test runs, 3 by
    default). Returns a DataFrame with the keys, ``n_runs`` and one column per metric.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    import pyarrow as pa

    files = sorted(Path(path).rglob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=list(TRIAL_KEYS) + ["n_runs"])
    df = pa.concat_tables([pq.read_table(f) for f in files],
                          promote_options="default").to_pandas()
    missing = [k for k in TRIAL_KEYS + (run_col,) if k not in df.columns]
    if missing:
        raise KeyError(f"trial_table: test logs lack columns {missing}")
    if row_kind_col in df.columns:
        df = df[df[row_kind_col] == run_kind]
    keys = list(TRIAL_KEYS)
    if metrics is None:
        metrics = [c for c in df.columns
                   if c not in keys + [run_col, row_kind_col] and pd.api.types.is_numeric_dtype(df[c])
                   and not pd.api.types.is_bool_dtype(df[c])]
    metrics = list(metrics)
    per_run = df.groupby(keys + [run_col], dropna=False)[metrics].mean().reset_index()
    g = per_run.groupby(keys, dropna=False)
    trials = g[metrics].mean()
    trials.insert(0, "n_runs", g.size())
    return trials.reset_index().sort_values(keys).reset_index(drop=True)


def contrast(trials, metric: str, arm_a: str, arm_b: str, workload: str, load: float,
             alpha: float = 0.05) -> dict:
    """Paired comparison of two arms on one (workload, load), paired by seed."""
    sub = trials[(trials["workload"] == workload) & np.isclose(trials["load"].astype(float), load)]
    A = sub[sub["arm"] == arm_a].set_index("seed")[metric]
    B = sub[sub["arm"] == arm_b].set_index("seed")[metric]
    seeds = sorted(set(A.index) & set(B.index))
    res = paired(A.loc[seeds].to_numpy(), B.loc[seeds].to_numpy(), alpha=alpha)
    res.update(metric=metric, arm_a=arm_a, arm_b=arm_b, workload=workload, load=load,
               seeds=seeds, unpaired_seeds=sorted(set(A.index) ^ set(B.index)))
    return res
