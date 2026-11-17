#!/usr/bin/env python3
"""Task completion (%) against load, one IEEEtran column, panels (a) and (b) side by side.

(a) the controllers with 30 paired seeds: A0, A3, A5, A7, A9, A10, A12;
(b) A0 and the four baselines B_Zhou, B_Qin, B_Huang, B_Song on their paired seeds.
Mean and 95 % t interval over seeds; the two panels share the completion axis.

Reads the analysis directory given as first argument:
  descriptive.csv  arm, workload, load, n, completion, completion_ci95 (half-width; fractions)
  trials.csv       per-seed completion; used only for A0 in panel (b), which is averaged over the
                   seeds it shares with the baselines (the check that trials.csv reproduces
                   descriptive.csv for A0 on all its seeds runs first)
  summary.json     loads, phase/block/tag (rehearsal or final)
The controller lists and their order follow the paper's load figure; a controller
that is absent from the files is skipped. Colors and markers are fixed per controller and are the
same in fig6_workloads.py (supplement Fig. S1).

Usage:
  python3 fig5_load.py ANALYSIS_DIR [OUT.pdf] [--final]
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

PANEL_A = ["A0", "A3", "A5", "A7", "A9", "A10", "A12"]      # the paper's load figure, panel (a)
PANEL_B = ["A0", "B_ZHOU", "B_QIN", "B_HUANG", "B_SONG"]    # (b)
LOAD_NAMES = ["light", "medium", "heavy"]


def t_mean_ci(v):
    from scipy import stats
    v = np.asarray(v, float)
    n = len(v)
    half = float(stats.t.ppf(0.975, n - 1)) * float(v.std(ddof=1)) / math.sqrt(n) if n > 1 else math.nan
    return float(v.mean()), half, n


def collect(args):
    d = Path(args.analysis_dir).resolve()
    summary = read_summary(d)
    kind = data_kind(d, summary)
    say(f"data: {d}  kind={kind}  phase={summary.get('phase')!r}  block={summary.get('block')!r}")
    if args.final and kind != "FINAL":
        sys.exit(f"refusing --final: {d} holds {kind} data")
    desc = read_csv(d / "descriptive.csv")
    trials = read_csv(d / "trials.csv")
    loads = sorted(num(x) for x in summary.get("loads", []))
    if not loads:
        loads = sorted({num(r["load"]) for r in desc if r["workload"] == MAIN_WORKLOAD})
    say("loads: " + ", ".join(f"{x:g}" for x in loads))

    def li(x):
        for i, l in enumerate(loads):
            if abs(num(x) - l) < 1e-6:
                return i
        return None

    # descriptive values: (arm, load index) -> (mean %, half-width %, n)
    val = {}
    for r in desc:
        if r["workload"] != MAIN_WORKLOAD or li(r["load"]) is None:
            continue
        n, m, h = to_int(r["n"]), num(r["completion"]), num(r["completion_ci95"])
        if n > 0 and math.isfinite(m):
            val[(canon(r["arm"]), li(r["load"]))] = (100.0 * m, 100.0 * h, n)

    # per-seed completion from trials.csv: (arm, load index) -> {seed: value}
    seeds = {}
    for r in trials:
        if r["workload"] != MAIN_WORKLOAD or li(r["load"]) is None:
            continue
        seeds.setdefault((canon(r["arm"]), li(r["load"])), {})[to_int(r["seed"])] = num(r["completion"])

    # check: trials.csv reproduces descriptive.csv for A0 on all its seeds
    for i in range(len(loads)):
        if ("A0", i) in val and ("A0", i) in seeds:
            m, h, n = t_mean_ci(list(seeds[("A0", i)].values()))
            dm, dh, dn = val[("A0", i)]
            if n != dn or abs(100 * m - dm) > 1e-6 or (n > 1 and abs(100 * h - dh) > 1e-6):
                sys.exit(f"trials.csv does not reproduce descriptive.csv for A0 at load {loads[i]:g}: "
                         f"n {n} vs {dn}, mean {100 * m:.4f} vs {dm:.4f} %, half-width {100 * h:.4f} vs "
                         f"{dh:.4f}; panel (b) recomputes A0 from trials.csv, so check the files first")

    panel_a = {a: {i: val[(a, i)] for i in range(len(loads)) if (a, i) in val} for a in PANEL_A}
    panel_a = {a: v for a, v in panel_a.items() if v}

    panel_b = {}
    for a in PANEL_B[1:]:
        v = {i: val[(a, i)] for i in range(len(loads)) if (a, i) in val}
        if v:
            panel_b[a] = v
    if panel_b:
        a0 = {}
        for i in range(len(loads)):
            base_seeds = set()
            for b in panel_b:
                base_seeds |= set(seeds.get((b, i), {}))
            a0_seeds = seeds.get(("A0", i), {})
            common = sorted(base_seeds & set(a0_seeds))
            for b in panel_b:
                if set(seeds.get((b, i), {})) != base_seeds:
                    say(f"WARNING baseline {b} at load {loads[i]:g} lacks seeds of the others; A0 in (b) "
                        "uses the union of the baselines' seeds")
            if len(common) >= 2:
                m, h, n = t_mean_ci([a0_seeds[s] for s in common])
                a0[i] = (100.0 * m, 100.0 * h, n)
                say(f"(b) A0 at load {loads[i]:g}: {n} seeds shared with the baselines "
                    f"({common[0]}..{common[-1]}), {100 * m:.2f} % +- {100 * h:.2f}")
        panel_b = dict({"A0": a0}, **panel_b) if a0 else panel_b
    missing = [a for a in PANEL_A if a not in panel_a] + [a for a in PANEL_B[1:] if a not in panel_b]
    if missing:
        say(f"note: absent from the files and skipped: {missing}")
    return loads, panel_a, panel_b, kind


def n_text(panel):
    ns = sorted({v[2] for series in panel.values() for v in series.values()})
    if not ns:
        return ""
    return rf"$n$ = {ns[0]}" if len(ns) == 1 else rf"$n$ = {ns[0]}{chr(8211)}{ns[-1]}"


def draw_panel(ax, panel, order, loads, gw):
    arms = [a for a in order if a in panel]
    k = len(arms)
    offs = np.linspace(-gw / 2, gw / 2, k) if k > 1 else np.zeros(1)
    for j, a in enumerate(arms):
        col = STYLE[a][0]
        idx = sorted(panel[a])
        xs = np.array(idx, float) + offs[j]
        ms = np.array([panel[a][i][0] for i in idx])
        hs = np.array([panel[a][i][1] for i in idx])
        ax.plot(xs, ms, color=col, lw=0.65, ls=LS.get(a, "-"), alpha=0.9, zorder=2)
        for x, m, h in zip(xs, ms, hs):
            if math.isfinite(h):
                ax.plot([x, x], [m - h, m + h], color=col, lw=0.8, solid_capstyle="butt", zorder=3)
        ax.plot(xs, ms, ls="none", zorder=4, **marker_kw(a))
    ax.set_xlim(-0.5, len(loads) - 0.5)
    ax.set_xticks(range(len(loads)))
    names = LOAD_NAMES if len(loads) == 3 else [""] * len(loads)
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", length=0, pad=2.5)
    for i, l in enumerate(loads):                      # second row: the load value
        ax.annotate(f"{l:g}", (i, 0), xycoords=("data", "axes fraction"), xytext=(0, -13.5),
                    textcoords="offset points", ha="center", va="top", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#e4e4e4", lw=0.4)
    ax.set_axisbelow(True)


def legend_handle(a):
    return Line2D([], [], color=STYLE[a][0], lw=0.65, ls=LS.get(a, "-"), **marker_kw(a))


def draw(loads, pa, pb, kind, args, out):
    S.apply()
    W, H = S.COL_W_IN, 2.60
    fig = plt.figure(figsize=(W, H))
    L, R, GAP = 0.44, 0.04, 0.10
    B, T_LEG, T_TITLE = 0.33, 0.34, 0.17
    plot_h = H - B - T_LEG - T_TITLE
    avail = W - L - R - (GAP if pb else 0.0)
    wa = avail * 0.56 if pb else avail                        # (a) holds 7 controllers, (b) 5
    ax_a = fig.add_axes([L / W, B / H, wa / W, plot_h / H])
    axes = [ax_a]
    draw_panel(ax_a, pa, PANEL_A, loads, gw=0.66)
    if pb:
        ax_b = fig.add_axes([(L + wa + GAP) / W, B / H, (avail - wa) / W, plot_h / H], sharey=ax_a)
        draw_panel(ax_b, pb, PANEL_B, loads, gw=0.60)
        ax_b.tick_params(axis="y", labelleft=False)
        axes.append(ax_b)

    # shared completion axis: every interval, clamped to [0, 100], on 5 % steps
    vals = [(m - h, m + h) if math.isfinite(h) else (m, m)
            for p in (pa, pb) for s in p.values() for (m, h, n) in s.values()]
    lo = max(0.0, 5 * math.floor(min(v[0] for v in vals) / 5))
    hi = min(100.0, 5 * math.ceil(max(v[1] for v in vals) / 5))
    ax_a.set_ylim(lo, hi)
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax_a.set_ylabel("task completion (%)", labelpad=3)
    x_mid = (L + (W - R)) / 2 / W
    ax_a.annotate(r"load $\ell$", (0, 0), xycoords="axes fraction", xytext=(-4, -13.5),
                  textcoords="offset points", ha="right", va="top", fontsize=8)

    for ax, tag, p in zip(axes, ("(a)", "(b)"), (pa, pb)):
        bb = ax.get_position()
        fig.text(bb.x0 - (0.40 / W if tag == "(a)" else 0.0), bb.y1 + 0.05 / H, f"{tag}  {n_text(p)}",
                 ha="left", va="bottom", fontsize=8)

    row1 = [a for a in PANEL_A if a in pa]
    row2 = [a for a in PANEL_B[1:] if a in pb]
    y1 = (H - 0.10) / H
    kw = dict(handlelength=1.5, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0,
              borderpad=0.0, frameon=False)
    fig.legend([legend_handle(a) for a in row1], [disp(a) for a in row1], loc="center",
               bbox_to_anchor=(x_mid, y1), ncol=len(row1), **kw)
    if row2:
        fig.legend([legend_handle(a) for a in row2], [disp(a) for a in row2], loc="center",
                   bbox_to_anchor=(x_mid, (H - 0.25) / H), ncol=len(row2), **kw)

    if not args.final:
        watermark(fig, WM_TEXT[kind])
    title = "Fig. 5 load" if args.final else f"Fig. 5 load ({WM_TEXT[kind]})"
    save(fig, out, title)
    say(f"size {W:.3f} x {H:.2f} in; (a) {row1}; (b) {['A0'] if 'A0' in pb else []}{row2}; "
        f"axis {lo:g}-{hi:g} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis_dir")
    ap.add_argument("out", nargs="?", help="output PDF (default ../fig5_load.pdf with --final, "
                                           "else ../rehearsal/fig5_load_<KIND>.pdf)")
    ap.add_argument("--final", action="store_true", help="real analysis: no watermark, submission file")
    args = ap.parse_args()
    print("Fig. 5 (completion against load)")
    loads, pa, pb, kind = collect(args)
    if not pa and not pb:
        sys.exit("no controller of Fig. 5 found in the files")
    out = resolve_out(args.out, args.final, "fig5_load", kind)
    draw(loads, pa, pb, kind, args, out)


if __name__ == "__main__":
    main()
