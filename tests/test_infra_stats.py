import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpa.stats.analysis import (cohen_d_pooled, contrast, holm, holm_reject, paired,  # noqa: E402
                                trial_table, welch)


def test_holm_hand_example():
    # sorted: .005 .01 .03 .04 -> x4 x3 x2 x1 = .02 .03 .06 .04 -> monotone .02 .03 .06 .06
    p = [0.01, 0.04, 0.03, 0.005]
    adj = holm(p)
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert list(holm_reject(p, 0.05)) == [True, False, False, True]


def test_holm_monotone_and_capped():
    p = np.array([0.04, 0.001, 0.0101, 0.9, 0.012])
    adj = holm(p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= 0)          # monotone in the p order
    assert np.all(adj >= p) and np.all(adj <= 1)
    # hand: sorted .001 .0101 .012 .04 .9 -> .005 .0404 .036 .08 .9 -> .005 .0404 .0404 .08 .9
    np.testing.assert_allclose(adj, [0.08, 0.005, 0.0404, 0.9, 0.0404])
    assert holm([0.6, 0.7])[1] == 1.0 and holm([0.6, 0.7])[0] == 1.0
    nan = holm([0.01, np.nan, 0.02])
    assert math.isnan(nan[1]) and np.allclose(nan[[0, 2]], [0.02, 0.02])


def test_paired_hand_values():
    a = [1, 2, 3, 4, 5]
    b = [0, 1, 1, 3, 2]
    r = paired(a, b)
    # d = [1,1,2,1,3], mean 1.6, sd sqrt(0.8), se 0.4, t = 4
    assert r["n"] == 5 and r["df"] == 4
    assert r["mean_diff"] == pytest.approx(1.6)
    assert r["t"] == pytest.approx(4.0)
    assert r["d_z"] == pytest.approx(1.6 / math.sqrt(0.8))
    # pooled SD: var_a = 2.5, var_b = 1.3 -> sqrt(1.9)
    assert r["cohen_d_pooled"] == pytest.approx(1.6 / math.sqrt(1.9))
    q = stats.t.ppf(0.975, 4)
    assert r["ci"][0] == pytest.approx(1.6 - q * 0.4) and r["ci"][1] == pytest.approx(1.6 + q * 0.4)
    ref = stats.ttest_rel(a, b)
    assert r["t"] == pytest.approx(ref.statistic) and r["p"] == pytest.approx(ref.pvalue)


def test_paired_vs_scipy_random():
    g = np.random.Generator(np.random.PCG64(42))
    a = g.normal(0.8, 0.1, 30)
    b = a - g.normal(0.02, 0.03, 30)
    r = paired(a, b)
    ref = stats.ttest_rel(a, b)
    assert r["t"] == pytest.approx(ref.statistic, rel=1e-12)
    assert r["p"] == pytest.approx(ref.pvalue, rel=1e-10)
    ci = ref.confidence_interval(0.95)
    assert r["ci"][0] == pytest.approx(ci.low) and r["ci"][1] == pytest.approx(ci.high)
    assert cohen_d_pooled(a, b) == pytest.approx(
        (a.mean() - b.mean()) / math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))


def test_paired_nan_and_degenerate():
    r = paired([1, 2, np.nan, 4], [0, 1, 1, np.nan])
    assert r["n"] == 2 and r["n_dropped"] == 2
    z = paired([1, 2, 3], [1, 2, 3])
    assert z["t"] == 0 and z["p"] == 1.0
    with pytest.raises(ValueError):
        paired([1, 2], [1, 2, 3])


def test_welch_vs_scipy():
    g = np.random.Generator(np.random.PCG64(7))
    a, b = g.normal(1.0, 1.0, 12), g.normal(0.3, 2.5, 20)
    r = welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert r["t"] == pytest.approx(ref.statistic) and r["p"] == pytest.approx(ref.pvalue)
    assert r["df"] == pytest.approx(ref.df)
    ci = ref.confidence_interval(0.95)
    assert r["ci"][0] == pytest.approx(ci.low) and r["ci"][1] == pytest.approx(ci.high)


def test_trial_table_and_contrast(tmp_path):
    rows = []
    for arm, off in (("A0", 0.1), ("A3", 0.0)):
        for seed in (1, 2, 3):
            for run in range(3):
                rows.append(dict(arm=arm, seed=seed, workload="sinusoidal", load=0.85, test_run=run,
                                 row_kind="run", completion=0.5 + off + 0.01 * seed + 0.001 * run))
                rows.append(dict(arm=arm, seed=seed, workload="sinusoidal", load=0.85, test_run=run,
                                 row_kind="minute", completion=-99.0))
    d = tmp_path / "A" / "test"
    d.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df[df.arm == "A0"].to_parquet(d / "part-00001.parquet")
    df[df.arm == "A3"].to_parquet(tmp_path / "A" / "part-00001.parquet")
    t = trial_table(tmp_path)
    assert len(t) == 6 and set(t["n_runs"]) == {3}
    v = t[(t.arm == "A0") & (t.seed == 2)]["completion"].item()
    assert v == pytest.approx(0.5 + 0.1 + 0.02 + 0.001)
    c = contrast(t, "completion", "A0", "A3", "sinusoidal", 0.85)
    assert c["n"] == 3 and c["mean_diff"] == pytest.approx(0.1)
