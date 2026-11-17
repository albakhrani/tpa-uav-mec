"""Memory mechanism on the test seeds (one column).

(a) Long-term occupancy of TPA at the end of each test run, by load (one point per run, 90 per load),
against the occupancy bound of Proposition 2 at rho_merge = 0.95. (b) Share of slots in cycles 2-3 on
which the gate opened, per seed, against the phase imbalance kappa_phi of the seed, with least-squares
lines. Reads per_cycle_runs.csv and p6_p9.json of the registered analysis.
Usage: python3 fig9_mechanism_on_test_seeds.py ANALYSIS_DIR OUT.pdf
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S  # noqa: E402

COL = ["#1f63b5", "#d4561f", "#14936a"]
MK = ["o", "s", "^"]
NAMES = ["light", "medium", "heavy"]
d, out = Path(sys.argv[1]), Path(sys.argv[2])
pc = pd.read_csv(d / "per_cycle_runs.csv")
pc = pc[pc["arm"] == "A0"]
j = json.loads((d / "p6_p9.json").read_text())
kap = {int(k): v for k, v in j["kappa_phi"].items()}
loads = sorted(pc["load"].unique())
last = pc["cycle"].max()
c23 = sorted(pc["cycle"].unique())[1:]
rng = np.random.default_rng(0)
S.apply()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.COL_W_IN, 1.85), gridspec_kw={"width_ratios": [1, 1.25]})
for i, L in enumerate(loads):
    occ = pc[(pc["load"] == L) & (pc["cycle"] == last)]["lt_end_hi"].to_numpy()
    xj = i + rng.uniform(-0.27, 0.27, size=len(occ))
    ax.scatter(xj, occ, s=6, color=COL[i], alpha=0.6, lw=0, zorder=3)
    ax.plot([i - 0.33, i + 0.33], [np.median(occ)] * 2, color=S.INK, lw=1.2, zorder=4)
    print(NAMES[i], "occ n", len(occ), "max", occ.max(), "median", np.median(occ))
ax.axhline(19, color=S.INK, lw=0.8, ls=(0, (4, 2)))
ax.text(2.45, 19.4, "bound, 19", ha="right", va="bottom")
ax.set_xticks(range(3)); ax.set_xticklabels(NAMES)
ax.set_ylim(-0.8, 23.5); ax.set_yticks([0, 5, 10, 15, 20]); ax.set_xlim(-0.55, 2.55)
ax.set_ylabel(r"occupied $\mathbf{M}_L$ entries", labelpad=2)
ax.text(0.04, 0.985, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
for i, L in enumerate(loads):
    g = pc[(pc["load"] == L) & (pc["cycle"].isin(c23))].groupby("seed")["open_share"].mean() * 100
    k = np.array([kap[s] for s in g.index])
    r = np.corrcoef(k, g.to_numpy())[0, 1]
    b, a = np.polyfit(k, g.to_numpy(), 1)
    kk = np.array([k.min(), k.max()])
    bx.plot(kk, a + b * kk, color=COL[i], lw=0.8, zorder=2)
    bx.scatter(k, g, s=9, marker=MK[i], facecolor=COL[i], edgecolor="white", lw=0.3, zorder=3,
               label=f"{NAMES[i]}, $r$ = {r:.2f}")
    print(NAMES[i], "open share mean", round(g.mean(), 2), "r", round(r, 3))
bx.set_xlabel(r"phase imbalance $\kappa_\varphi$"); bx.set_ylabel("gate open (% of slots)", labelpad=2)
bx.set_ylim(bottom=-0.8)
bx.legend(loc="upper left", bbox_to_anchor=(0.08, 1.04), handletextpad=0.1, borderaxespad=0.1, labelspacing=0.15)
bx.text(0.015, 0.985, "(b)", transform=bx.transAxes, va="top", fontweight="bold")
for a_ in (ax, bx):
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
fig.tight_layout(pad=0.25, w_pad=0.8)
fig.savefig(out, metadata={"Title": "Fig. 9 mechanism on test seeds", "Author": None, "Creator": None,
                           "Producer": None, "CreationDate": None})
print("wrote", out)
