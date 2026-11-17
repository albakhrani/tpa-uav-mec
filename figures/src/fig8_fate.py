"""What happens to the offered tasks at heavy load (one column).

(a) Shares of all arrivals that completed, were dropped from the full queue, or were not admitted
before their deadline (admitted tasks that missed are below 0.01 % and are not drawn).
(b) Tasks admitted per slot, mean and 95 % t interval over seeds. Reads trials.csv.
Usage: python3 fig8_fate.py ANALYSIS_DIR OUT.pdf
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

ARMS = ["A0", "A3", "A9", "A10", "A12", "B_HUANG", "B_SONG", "B_QIN", "B_ZHOU"]
NAME = {"B_ZHOU": r"B$_{\mathrm{Zhou}}$", "B_QIN": r"B$_{\mathrm{Qin}}$", "B_HUANG": r"B$_{\mathrm{Huang}}$",
        "B_SONG": r"B$_{\mathrm{Song}}$", "A0": "TPA (A0)"}
KIND = {"A0": "learned, memory", "A3": "learned, memory", "A9": "learned", "A10": "learned", "A12": "rule",
        "B_HUANG": "learned", "B_SONG": "learned", "B_QIN": "rule", "B_ZHOU": "rule"}
GREEN, AMBER, GRAY = "#14936a", "#e2a100", "#b9b9b9"

d, out = Path(sys.argv[1]), Path(sys.argv[2])
t = pd.read_csv(d / "trials.csv")
h = t[(t["load"] == t["load"].max()) & (t["workload"] == "sinusoidal")]
S.apply()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.COL_W_IN, 2.05), gridspec_kw={"width_ratios": [1.55, 1]}, sharey=True)
ys = np.arange(len(ARMS))[::-1]
print("arm n completed dropped notadm admissions/slot")
for y, arm in zip(ys, ARMS):
    g = h[h["arm"] == arm]
    n = len(g)
    c = (g["completed"] / g["arrivals"]).mean() * 100
    dr = (g["n_overflow"] / g["arrivals"]).mean() * 100
    ex = (g["n_expired"] / g["arrivals"]).mean() * 100
    ax.barh(y, c, color=GREEN, height=0.68, lw=0)
    ax.barh(y, dr, left=c, color=GRAY, height=0.68, lw=0)
    ax.barh(y, ex, left=c + dr, color=AMBER, height=0.68, lw=0)
    ax.text(c - 1.0, y, f"{c:.1f}", ha="right", va="center", color="white", fontsize=7.5)
    adm = g["assigned"] / g["slots"]
    m = adm.mean(); ci = stats.t.ppf(0.975, n - 1) * adm.std(ddof=1) / np.sqrt(n)
    rule = KIND[arm] == "rule"
    bx.errorbar(m, y, xerr=ci, fmt="D" if rule else "o", ms=3.6, color=S.INK if rule else S.BLUE,
                mfc="white" if rule else S.BLUE, mew=0.8, elinewidth=0.8, capsize=0)
    print(arm, n, round(c, 2), round(dr, 2), round(ex, 2), round(m, 2), round(ci, 2))
ax.set_yticks(ys); ax.set_yticklabels([NAME.get(a, a) for a in ARMS])
ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("share of offered tasks (%)")
ax.tick_params(axis="y", length=0)
ax.axhline(3.5, color=S.GRAY_MID, lw=0.5, ls=(0, (2, 2)))
bx.axhline(3.5, color=S.GRAY_MID, lw=0.5, ls=(0, (2, 2)))
bx.axvline(32, color=S.GRAY_MID, lw=0.6, ls=(0, (1, 1.5)))
bx.text(31.3, ys[0] + 0.05, "visit cap", rotation=90, ha="right", va="top", color=S.GRAY_DARK)
bx.set_xlim(8, 34); bx.set_xticks([10, 20, 30])
bx.set_xlabel("admitted per slot")
bx.tick_params(axis="y", length=0)
bx.grid(axis="x", color=S.GRAY_LIGHT, lw=0.5, zorder=0); bx.set_axisbelow(True)
for a in (ax, bx):
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
hd = [Patch(color=GREEN, label="completed"), Patch(color=GRAY, label="dropped from queue"),
      Patch(color=AMBER, label="not admitted")]
fig.legend(handles=hd, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.005), handlelength=0.9,
           columnspacing=0.8, handletextpad=0.35)
ax.set_title("(a)", loc="left", fontweight="bold", pad=2)
bx.set_title("(b)", loc="left", fontweight="bold", pad=2)
fig.tight_layout(pad=0.25, w_pad=0.6, rect=(0, 0, 1, 0.93))
fig.savefig(out, metadata={"Title": "Fig. 8 task fate", "Author": None, "Creator": None, "Producer": None,
                           "CreationDate": None})
print("wrote", out)
