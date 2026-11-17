"""Supplement figure (figS_seed.pdf): per-seed paired differences in heavy-load completion for every heavy-load
contrast of the supplement table of all registered contrasts.

Usage: python3 figS_seed.py <analysis_dir> <out.pdf>
One dot per seed (paired difference, first-named minus second-named controller, pp), the mean as a vertical tick,
and the +-1 pp equivalence band. Sign convention as in the forest figure: A0 minus the comparator unless both are named.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROWS = [("A0", "A3"), None,
        ("A0", "A1"), ("A0", "A2"), ("A0", "A4"), ("A0", "A5"), ("A0", "A6"), ("A0", "A7"), ("A0", "A8"),
        ("A0", "A9"), ("A0", "A14"), ("A2", "A1"), None,
        ("A0", "A10"), ("A10", "A6"), ("A11", "A9"), ("A0", "A12"), None,
        ("A0", "B_ZHOU"), ("A0", "B_QIN"), ("A0", "B_HUANG"), ("A0", "B_SONG"), None,
        ("A0", "A0-tmem25"), ("A0", "A13"), ("A0", "A15")]
MINUS = "−"
NAMES = {"B_ZHOU": r"B$_{\mathrm{Zhou}}$", "B_QIN": r"B$_{\mathrm{Qin}}$", "B_HUANG": r"B$_{\mathrm{Huang}}$",
         "B_SONG": r"B$_{\mathrm{Song}}$", "A0-tmem25": "(A0, 25 ms)"}


def main():
    d = Path(sys.argv[1]); out = Path(sys.argv[2])
    t = pd.read_csv(d / "trials.csv")
    heavy = t["load"].max()
    h = t[(t["load"] == heavy) & (t["workload"] == "sinusoidal")]
    fs.apply()
    n_slots = len(ROWS)
    fig, ax = plt.subplots(figsize=(fs.COL_W_IN, 0.155 * n_slots + 0.55))
    ax.axvspan(-1, 1, color=fs.BLUE_LIGHT, alpha=0.45, lw=0, zorder=0)
    ax.axvline(0, color=fs.GRAY_MID, lw=0.5, zorder=1)
    labels, ys = [], []
    for i, row in enumerate(ROWS):
        y = n_slots - i
        if row is None:
            continue
        a, b = row
        x = h[h["arm"] == a].set_index("seed")["completion"]
        z = h[h["arm"] == b].set_index("seed")["completion"]
        s = x.index.intersection(z.index)
        v = (x[s] - z[s]).to_numpy() * 100.0
        jitter = (np.arange(len(v)) % 5 - 2) * 0.045
        ax.scatter(v, y + jitter, s=5, color=fs.BLUE, alpha=0.75, lw=0, zorder=3)
        ax.plot([v.mean()] * 2, [y - 0.32, y + 0.32], color=fs.INK, lw=1.2, zorder=4)
        labels.append(f"{NAMES.get(a, a)} {MINUS} {NAMES.get(b, b)}  (n = {len(v)})")
        ys.append(y)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(0.3, n_slots + 0.7)
    ax.set_xlabel("paired difference in heavy-load completion (pp)")
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=fs.GRAY_LIGHT, lw=0.5, zorder=0)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, metadata={"Title": "Per-seed paired differences", "Creator": None, "Producer": None,
                               "CreationDate": None})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
