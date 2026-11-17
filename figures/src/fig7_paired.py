"""The primary contrast seed by seed (one column).

(a) heavy-load completion of A0 against A3, one point per paired test seed, with the identity line
and the registered +-1 pp equivalence band. (b) the paired differences A0 - A3 in ascending order,
their mean and 95 % t interval, and the same band. Reads trials.csv of the registered analysis.
Usage: python3 fig7_paired.py ANALYSIS_DIR OUT.pdf
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S  # noqa: E402

d, out = Path(sys.argv[1]), Path(sys.argv[2])
t = pd.read_csv(d / "trials.csv")
h = t[(t["load"] == t["load"].max()) & (t["workload"] == "sinusoidal")]
a0 = h[h["arm"] == "A0"].set_index("seed")["completion"] * 100
a3 = h[h["arm"] == "A3"].set_index("seed")["completion"] * 100
s = a0.index.intersection(a3.index)
a0, a3 = a0[s].to_numpy(), a3[s].to_numpy()
diff = a0 - a3
n = len(diff)
m = diff.mean()
ci = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)

S.apply()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.COL_W_IN, 1.80), gridspec_kw={"width_ratios": [1, 1.12]})
lo, hi = np.floor(min(a0.min(), a3.min()) - 0.6), np.ceil(max(a0.max(), a3.max()) + 0.6)
xx = np.array([lo, hi])
ax.fill_between(xx, xx - 1, xx + 1, color=S.BLUE_LIGHT, alpha=0.55, lw=0, zorder=0)
ax.plot(xx, xx, color=S.GRAY_MID, lw=0.6, zorder=1)
ax.scatter(a3, a0, s=11, facecolor=S.BLUE, edgecolor="white", lw=0.4, zorder=3)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("single store A3 (%)"); ax.set_ylabel("TPA, A0 (%)")
ax.text(0.04, 0.96, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

o = np.sort(diff)
x = np.arange(1, n + 1)
bx.axhspan(-1, 1, color=S.BLUE_LIGHT, alpha=0.55, lw=0, zorder=0)
bx.axhline(0, color=S.GRAY_MID, lw=0.6, zorder=1)
bx.axhspan(m - ci, m + ci, color=S.ORANGE, alpha=0.28, lw=0, zorder=1.5)
bx.axhline(m, color=S.ORANGE, lw=1.0, zorder=2)
bx.vlines(x, 0, o, color=S.BLUE, lw=0.6, alpha=0.55, zorder=2.5)
bx.scatter(x, o, s=11, facecolor=S.BLUE, edgecolor="white", lw=0.4, zorder=3)
lim = max(2.6, np.ceil(np.abs(o).max() * 10) / 10 + 0.2)
bx.set_ylim(-lim, lim); bx.set_xlim(0, n + 1)
bx.set_xticks([1, 10, 20, 30])
bx.set_xlabel("seeds, ordered"); bx.set_ylabel("A0 " + "−" + " A3 (pp)", labelpad=1)
bx.text(0.04, 0.96, "(b)", transform=bx.transAxes, va="top", fontweight="bold")
bx.text(n + 0.6, -lim + 0.12, "mean and 95% interval", color=S.ORANGE, ha="right", va="bottom")
bx.text(0.6, 1.0, "±1 pp", color=S.BLUE, ha="left", va="bottom")
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
fig.tight_layout(pad=0.25, w_pad=0.9)
fig.savefig(out, metadata={"Title": "Fig. 7 paired seeds", "Author": None, "Creator": None, "Producer": None,
                           "CreationDate": None})
print(f"wrote {out}  n={n} mean={m:.2f} ci=[{m-ci:.2f},{m+ci:.2f}] within1={(np.abs(diff)<1).sum()}")
