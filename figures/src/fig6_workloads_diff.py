"""Paired differences in heavy-load completion on the four workloads (one column).

For each workload: TPA (A0) minus the single store A3, the memoryless A9 and the recurrent A10, mean and
95 % interval over paired seeds, with the +-1 pp equivalence band of the primary test. The label under
each workload name is its predictive gap G_w(100) from Table III (a derived number, not a result).
Reads contrasts.csv of the main analysis and of the analysis of the other workloads; the A10 row of the
sinusoid is registered as A10 - A0 and is shown with its sign reversed.
Usage: python3 fig6_workloads_diff.py MAIN_DIR ROBUST_DIR OUT.pdf
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S  # noqa: E402

WL = [("sinusoidal", "Sinusoid", "0.0015"), ("mmpp", "MMPP", "lt"),
      ("sinusoidal|mobility=gauss_markov", "Mobility", "n/a"), ("pareto", "On-off", "0.044")]
CMP = [("A3", "#d4561f", "s", "single store A3"), ("A9", "#14936a", "v", "memoryless A9"),
       ("A10", "#c28a00", "X", "recurrent A10")]
MINUS = "−"
dm, dr, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
c = pd.concat([pd.read_csv(dm / "contrasts.csv"), pd.read_csv(dr / "contrasts.csv")])
c = c[(c["metric"] == "completion") & (c["load"] == c["load"].max())]


def row(wl, b):
    r = c[(c["workload"] == wl) & (c["arm_a"] == "A0") & (c["arm_b"] == b)]
    if len(r):
        r = r.iloc[0]
        return r["mean_diff"] * 100, r["ci_lo"] * 100, r["ci_hi"] * 100, int(r["n"])
    r = c[(c["workload"] == wl) & (c["arm_a"] == b) & (c["arm_b"] == "A0")].iloc[0]
    return -r["mean_diff"] * 100, -r["ci_hi"] * 100, -r["ci_lo"] * 100, int(r["n"])


S.apply()
fig, ax = plt.subplots(figsize=(S.COL_W_IN, 2.25))
ax.axvspan(-1, 1, color=S.BLUE_LIGHT, alpha=0.5, lw=0, zorder=0)
ax.axvline(0, color=S.GRAY_MID, lw=0.6, zorder=1)
y = 0.0
yt, yl = [], []
for wi, (wl, name, gap) in enumerate(WL):
    if wi:
        ax.axhline(y - 0.2, color=S.GRAY_LIGHT, lw=0.6, zorder=0.5)
    ys = []
    for b, col, mk, _ in CMP:
        y -= 1
        m, lo, hi, n = row(wl, b)
        ax.plot([lo, hi], [y, y], color=col, lw=1.1, zorder=3, solid_capstyle="butt")
        ax.plot([m], [y], mk, color=col, ms=4.6 if mk != "X" else 5.2, mew=0, zorder=4)
        ys.append(y)
        print(name, b, f"{m:.2f} [{lo:.2f}, {hi:.2f}] n={n}")
    ax.text(-5.95, ys[1] + 0.48, name, ha="left", va="center", fontweight="bold")
    ax.text(-5.95, ys[1] - 0.52, {"n/a": "$G_w$ not defined", "lt": r"$G_w < 10^{-11}$"}.get(gap, f"$G_w$ = {gap}"), ha="left", va="center",
            color=S.GRAY_DARK)
    ax.text(3.05, ys[1], f"$n$ = {n}", ha="right", va="center", color=S.GRAY_DARK)
    y -= 0.6
ax.set_ylim(y - 0.1, -0.3)
ax.set_xlim(-6.1, 3.1)
ax.set_xticks([-4, -3, -2, -1, 0, 1, 2])
ax.set_yticks([])
ax.set_xlabel(f"TPA {MINUS} comparator, heavy-load completion (pp)")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
hd = [Line2D([], [], color=col, marker=mk, lw=1.1, ms=4.6, mew=0, label=lab) for _, col, mk, lab in CMP]
fig.legend(handles=hd, loc="upper center", ncol=3, columnspacing=1.0, handlelength=1.4, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.25, rect=(0, 0, 1, 0.93))
fig.savefig(out, metadata={"Title": "Fig. 6 workloads", "Author": None, "Creator": None, "Producer": None,
                           "CreationDate": None})
print("wrote", out)
