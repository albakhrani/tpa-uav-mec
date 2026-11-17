#!/usr/bin/env python3
"""Supplement Fig. S1: heavy-load completion (%) of A0, A3, A9 and A10 across workloads, one IEEEtran column.

Workloads in the order of the paper's figure: sinusoidal, MMPP, Gauss-Markov
user mobility, heavy-tailed on-off; mean and 95 % t interval over seeds. Under each workload: its
predictive gap G_w(100) from the paper's predictive-gap table (a declared number, not a result) and the
number of seeds read from the files.

Reads descriptive.csv (arm, workload, load, n, completion, completion_ci95 as fractions) and
summary.json (loads, phase/block/tag) from the analysis directory given as first argument, and
from every --robust directory if the robustness family was analysed separately (it ran from its
own queue and is analysed on its own). Rows are never pooled: an (arm, workload) found in two
directories stops the script, and so do directories of different kinds (rehearsal, synthetic,
final). Workload names are matched as the trainer writes them ('mmpp', 'pareto',
'sinusoidal|mobility=gauss_markov'); an unknown name is listed and skipped, as is an absent one.

Usage:
  python3 fig6_workloads.py ANALYSIS_DIR [OUT.pdf] [--robust DIR ...] [--final]
Without --final the figure is watermarked and must go under figures/rehearsal/.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import figstyle as S

# ---------------------------------------------------------------- shared block (same in fig4/5/6)
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent
REH_DIR = FIG_DIR / "rehearsal"
MAIN_WORKLOAD = "sinusoidal"
SYNTH_MARKER = "SYNTHETIC_DATA_NOT_RESULTS.txt"
WM_TEXT = {"REHEARSAL": "REHEARSAL DATA, NOT RESULTS", "SYNTHETIC": "SYNTHETIC",
           "FINAL": "PREVIEW, NOT FOR SUBMISSION"}
TMEM_ALIASES = {"A0-tmem25", "A0_tmem25", "A0-tmem25ms", "A0_tmem25ms", "A0-25ms", "A0_25ms"}
DISPLAY = {"B_ZHOU": r"B$_{\mathrm{Zhou}}$", "B_QIN": r"B$_{\mathrm{Qin}}$",
           "B_HUANG": r"B$_{\mathrm{Huang}}$", "B_SONG": r"B$_{\mathrm{Song}}$",
           "A0-tmem25": "(A0, 25 ms)"}
MINUS = "−"


def say(msg):
    print("  " + msg)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def to_int(x):
    v = num(x)
    return int(v) if math.isfinite(v) else 0


def read_csv(path):
    if not path.exists():
        sys.exit(f"missing {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_summary(d):
    p = d / "summary.json"
    if not p.exists():
        sys.exit(f"missing {p} (needed to tell rehearsal from final data)")
    return json.loads(p.read_text())


def data_kind(d, summary):
    """FINAL, REHEARSAL or SYNTHETIC, from the marker file and summary.json phase/block/tag."""
    phase = str(summary.get("phase", "")).lower()
    block = str(summary.get("block", "")).lower()
    tag = str(summary.get("tag", "")).upper()
    if (d / SYNTH_MARKER).exists() or "synthetic" in phase or "SYNTHETIC" in tag:
        return "SYNTHETIC"
    if phase == "tune" or block.startswith("val") or "REHEARSAL" in tag:
        return "REHEARSAL"
    return "FINAL"


def canon(arm):
    arm = str(arm).strip()
    return "A0-tmem25" if arm in TMEM_ALIASES else arm


def disp(arm):
    return DISPLAY.get(arm, arm)


def resolve_out(arg_out, final, stem, kind):
    if arg_out:
        out = Path(arg_out).resolve()
    else:
        out = FIG_DIR / f"{stem}.pdf" if final else REH_DIR / f"{stem}_{kind if kind != 'FINAL' else 'PREVIEW'}.pdf"
    in_reh = REH_DIR.resolve() in out.resolve().parents
    if final and in_reh:
        sys.exit("--final writes the submission figure; it must not go under rehearsal/")
    if not final and not in_reh:
        sys.exit(f"without --final the output must go under {REH_DIR}")
    return out


def watermark(fig, text):
    """Light diagonal watermark across the whole figure."""
    w, h = fig.get_size_inches()
    probe = fig.text(0, 0, text, fontsize=10)
    width_pt = probe.get_window_extent(fig.canvas.get_renderer()).width * 72.0 / fig.dpi
    probe.remove()
    size = 10.0 * 0.80 * math.hypot(w, h) * 72.0 / width_pt
    fig.text(0.5, 0.5, text, fontsize=size, color="#8c8c8c", alpha=0.30, ha="center", va="center",
             rotation=math.degrees(math.atan2(h, w)), rotation_mode="anchor", zorder=100)


def save(fig, out, title):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Title": title, "Author": None, "Creator": None, "Producer": None,
                               "CreationDate": None})
    print("wrote", out)
# ---------------------------------------------------------------- end of shared block


# ---------------------------------------------------------------- controller styles (same in fig5/6)
# Fixed per controller in fig5_load.py and fig6_workloads.py: (color, marker, filled).
# Colors: one per controller, darkened for print as in figstyle.py. A7 and A9 can look alike under
# color-vision deficiency, so their markers differ in shape.
STYLE = {
    "A0": ("#1f63b5", "o", True),
    "A3": ("#d4561f", "s", True),
    "A5": ("#6250d6", "D", True),
    "A7": ("#c93b3a", "^", True),
    "A9": ("#14936a", "v", True),
    "A10": ("#c28a00", "X", True),
    "A12": ("#d35d8e", "P", True),
    "B_ZHOU": ("#555555", "s", False),
    "B_QIN": ("#555555", "D", False),
    "B_HUANG": ("#555555", "^", False),
    "B_SONG": ("#555555", "v", False),
}
MS = {"o": 3.9, "s": 3.4, "D": 3.2, "^": 4.1, "v": 4.1, "X": 4.4, "P": 4.4}
LS = {"B_ZHOU": (0, (3.0, 1.5)), "B_QIN": (0, (3.0, 1.5)), "B_HUANG": (0, (1.0, 1.2)),
      "B_SONG": (0, (1.0, 1.2))}                      # rule baselines dashed, learned ones dotted


def marker_kw(a):
    """Filled markers get a thin white ring (separates neighbours); X and P are too thin for one.
    Baselines are hollow."""
    col, mk, filled = STYLE[a]
    if not filled:
        return dict(marker=mk, ms=MS[mk] * 0.92, mfc="white", mec=col, mew=0.8)
    ring = mk not in ("X", "P")
    return dict(marker=mk, ms=MS[mk], mfc=col, mec="white" if ring else col, mew=0.4 if ring else 0.0)
# ---------------------------------------------------------------- end of controller styles

ARMS = ["A0", "A3", "A9", "A10"]                     # the arms of the paper's figure
# (key, label, G_w(100) as printed): the paper's predictive-gap table, column G_w(Delta = 100).
# Sinusoid: heavy-load row. On-off: "as run" row (each run starts a sojourn at slot 0; the
# stationary value is 0.057). Gauss-Markov mobility has no row in the table.
WORKLOADS = [
    ("sinusoidal", "Sinusoid", "0.0015"),
    ("mmpp", "MMPP", r"$<10^{-11}$"),
    ("mobility", "Mobility", "–"),
    ("onoff", "On-off", "0.044"),
]
MOBILITY_NAMES = {"mobility", "gauss_markov", "gm", "gaussmarkov", "gauss-markov"}
ONOFF_NAMES = {"pareto", "onoff", "on-off", "on_off", "pareto_onoff"}


def classify(w):
    parts = [p.strip().lower() for p in str(w).split("|")]
    base, opts = parts[0], parts[1:]
    mob = [o for o in opts if o.startswith("mobility=")]
    if base == "sinusoidal" and not opts:
        return "sinusoidal"
    if base in MOBILITY_NAMES and not opts:
        return "mobility"
    if base == "sinusoidal" and len(opts) == 1 and mob and mob[0].split("=", 1)[1] in MOBILITY_NAMES:
        return "mobility"
    if base == "mmpp" and not opts:
        return "mmpp"
    if base in ONOFF_NAMES and not opts:
        return "onoff"
    return None


def collect(args):
    dirs = [Path(args.analysis_dir).resolve()] + [Path(r).resolve() for r in (args.robust or [])]
    kinds, heavy = set(), None
    val, src, unknown = {}, {}, set()
    for i, d in enumerate(dirs):
        summary = read_summary(d)
        kind = data_kind(d, summary)
        kinds.add(kind)
        say(f"data: {d}  kind={kind}  phase={summary.get('phase')!r}  block={summary.get('block')!r}")
        desc = read_csv(d / "descriptive.csv")
        loads = sorted(num(x) for x in summary.get("loads", [])) or \
            sorted({num(r["load"]) for r in desc})
        if i == 0:
            heavy = loads[-1]
            say(f"heavy load = {heavy:g}")
        elif all(abs(l - heavy) > 1e-6 for l in loads):
            say(f"WARNING {d} has no load {heavy:g} (its loads: {loads}); nothing taken from it")
        for r in desc:
            if abs(num(r["load"]) - heavy) > 1e-6:
                continue
            a, wk = canon(r["arm"]), classify(r["workload"])
            if wk is None:
                unknown.add(r["workload"])
                continue
            n, m, h = to_int(r["n"]), num(r["completion"]), num(r["completion_ci95"])
            if a not in ARMS or n <= 0 or not math.isfinite(m):
                continue
            if (a, wk) in val:
                sys.exit(f"{a} on {r['workload']!r} at load {heavy:g} is in both {src[(a, wk)]} and {d}; "
                         "the families are not pooled, so pass each workload from one directory only")
            val[(a, wk)] = (100.0 * m, 100.0 * h, n)
            src[(a, wk)] = d
    if len(kinds) > 1:
        sys.exit(f"refusing to combine directories of different kinds {sorted(kinds)}")
    kind = kinds.pop()
    if args.final and kind != "FINAL":
        sys.exit(f"refusing --final: the directories hold {kind} data")
    if unknown:
        say(f"note: workloads not in Fig. 6, skipped: {sorted(unknown)}")
    present = [w for w in WORKLOADS if any((a, w[0]) in val for a in ARMS)]
    absent = [w[1] for w in WORKLOADS if w not in present]
    if absent:
        say(f"note: workloads absent from the files and skipped: {absent}")
    return present, val, kind


def draw(present, val, kind, args, out):
    S.apply()
    W, H = S.COL_W_IN, 2.40
    fig = plt.figure(figsize=(W, H))
    L, R, B, T = 0.62, 0.04, 0.50, 0.24
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    arms = [a for a in ARMS if any((a, w[0]) in val for w in present)]
    k = len(arms)
    gw = 0.60
    offs = np.linspace(-gw / 2, gw / 2, k) if k > 1 else np.zeros(1)
    lo_all, hi_all = [], []
    for j, a in enumerate(arms):
        col = STYLE[a][0]
        for g, w in enumerate(present):
            if (a, w[0]) not in val:
                continue
            m, h, n = val[(a, w[0])]
            x = g + offs[j]
            if math.isfinite(h):
                ax.plot([x, x], [m - h, m + h], color=col, lw=0.8, solid_capstyle="butt", zorder=3)
                lo_all.append(m - h)
                hi_all.append(m + h)
            else:
                lo_all.append(m)
                hi_all.append(m)
            ax.plot([x], [m], ls="none", zorder=4, **marker_kw(a))

    lo = max(0.0, 5 * math.floor(min(lo_all) / 5))
    hi = min(100.0, 5 * math.ceil(max(hi_all) / 5))
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.set_ylabel("heavy-load completion (%)", labelpad=3)
    ax.set_xlim(-0.5, len(present) - 0.5)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([w[1] for w in present])
    ax.tick_params(axis="x", length=0, pad=3)
    for g in range(1, len(present)):
        ax.axvline(g - 0.5, color="#d9d9d9", lw=0.4, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#e4e4e4", lw=0.4)
    ax.set_axisbelow(True)

    # two table rows under the workload names: G_w(100) and n
    def ns(w):
        v = sorted({val[(a, w)][2] for a in ARMS if (a, w) in val})
        return f"{v[0]}" if len(v) == 1 else f"{v[0]}–{v[-1]}"

    for row, (head, cells) in enumerate([(r"$G_w(100)$", [w[2] for w in present]),
                                         (r"$n$", [ns(w[0]) for w in present])]):
        dy = -(16.5 + 10.5 * row)
        for g, c in enumerate(cells):
            ax.annotate(c, (g, 0), xycoords=("data", "axes fraction"), xytext=(0, dy),
                        textcoords="offset points", ha="center", va="top", fontsize=8)
        ax.annotate(head, (0, 0), xycoords="axes fraction", xytext=(-4, dy),
                    textcoords="offset points", ha="right", va="top", fontsize=8)

    handles = [Line2D([], [], ls="none", **marker_kw(a)) for a in arms]
    fig.legend(handles, [disp(a) for a in arms], loc="center", ncol=len(arms),
               bbox_to_anchor=((L + (W - R - L) / 2) / W, (H - 0.11) / H), handlelength=1.0,
               handletextpad=0.3, columnspacing=1.2, borderaxespad=0.0, borderpad=0.0)

    if not args.final:
        watermark(fig, WM_TEXT[kind])
    title = "Fig. 6 workloads" if args.final else f"Fig. 6 workloads ({WM_TEXT[kind]})"
    save(fig, out, title)
    say(f"size {W:.3f} x {H:.2f} in; workloads {[w[1] for w in present]}; arms {arms}; "
        f"axis {lo:g}-{hi:g} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis_dir")
    ap.add_argument("out", nargs="?", help="output PDF (default ../fig6_workloads.pdf with --final, "
                                           "else ../rehearsal/fig6_workloads_<KIND>.pdf)")
    ap.add_argument("--robust", action="append", metavar="DIR",
                    help="analysis directory of the robustness family, if separate (repeatable)")
    ap.add_argument("--final", action="store_true", help="real analysis: no watermark, submission file")
    args = ap.parse_args()
    print("Fig. 6 (workloads)")
    present, val, kind = collect(args)
    if not present:
        sys.exit("no workload of Fig. 6 found in the files")
    out = resolve_out(args.out, args.final, "fig6_workloads", kind)
    draw(present, val, kind, args, out)


if __name__ == "__main__":
    main()
