#!/usr/bin/env python3
"""Forest plot of paired differences in heavy-load completion (pp), one IEEEtran column.

Reads the analysis directory given as first argument:
  contrasts.csv  paired contrasts: n, mean_diff, ci_lo, ci_hi (fractions), p, p_holm, family, arms, load
  summary.json   loads, phase/block/tag (rehearsal or final), and the S4 interaction (n, mean_pp, t, p)
  trials.csv     per-seed completion; used only for the 95 % interval of the S4 interaction, which
                 summary.json does not hold (its n, mean and t are checked against summary.json)

Rows, their order and their grouping follow the paper's forest figure (fixed lists below); a row
that is absent from the files is skipped. n, p values, families and arm names come
from the files. A contrast stored in the other direction (A10 - A0) is drawn as A0 - A10: mean and
interval change sign, the two-sided p does not. Marker shape gives n, hollow markers are the
untested effect sizes, and p < 0.05 is printed in bold. The p column is Holm-adjusted within each
family; the load interaction and the 25 ms run are single tests and print their unadjusted p.

Usage:
  python3 fig4_forest.py ANALYSIS_DIR [OUT.pdf] [--final] [--with-baselines] [--xlim LO HI]
Without --final the figure is watermarked and must go under figures/rehearsal/.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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


# Paper order (forest figure; Section VI "Evaluation and Statistics").
# (header, kind, rows); a row is (arm_a, arm_b) in the direction the paper writes it, or
# (arm_a, arm_b, row_kind) in a "mixed" group, where each row carries its own kind.
GROUPS = [
    ("Primary contrast (H1)", "tested", [("A0", "A3")]),
    ("Ablation family", "tested",
     [("A0", a) for a in ("A1", "A2", "A4", "A5", "A6", "A7", "A8", "A9", "A14")]),
    ("Memory family", "tested", [("A0", "A10"), ("A10", "A6"), ("A11", "A9"), ("A0", "A12")]),
    ("Load interaction: A0 " + MINUS + " A9", "interaction", [("A0", "A9")]),
    # one header for the rows outside the Holm families: the registered single paired t-test of the
    # 25 ms run (filled, unadjusted p) and the two untested effect sizes (hollow, no p)
    ("Single test and effect sizes", "mixed",
     [("A0", "A0-tmem25", "single"), ("A0", "A13", "untested"), ("A0", "A15", "untested")]),
]
BASELINE_GROUP = ("Baseline family", "tested",
                  [("A0", b) for b in ("B_ZHOU", "B_QIN", "B_HUANG", "B_SONG")])

SHAPES = ["o", "s", "D", "^", "v", "p"]                       # by n, largest n first
MS = {"o": 3.9, "s": 3.5, "D": 3.3, "^": 4.2, "v": 4.2, "p": 4.1}
BAND = "#dfe8f4"                                              # +-1 pp equivalence margin
GRID = "#e4e4e4"
SESOI_PP = 1.0


def fmt_p(p):
    if p is None or not math.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    if p < 0.1:
        s = f"{p:.3f}"
        return f"{p:.4f}" if s == "0.050" else s
    return f"{p:.2f}"


def interaction_row(summary, trials, heavy, light):
    """S4 interaction (A0 - A9 at heavy minus at light), in pp, with its 95 % t interval."""
    from scipy import stats

    def seed_map(arm, load):
        return {int(num(r["seed"])): num(r["completion"]) for r in trials
                if canon(r["arm"]) == arm and r["workload"] == MAIN_WORKLOAD
                and abs(num(r["load"]) - load) < 1e-6}

    m = [seed_map("A0", heavy), seed_map("A9", heavy), seed_map("A0", light), seed_map("A9", light)]
    seeds = sorted(set(m[0]) & set(m[1]) & set(m[2]) & set(m[3]))
    if len(seeds) < 2:
        return None
    d = np.array([(m[0][s] - m[1][s]) - (m[2][s] - m[3][s]) for s in seeds]) * 100.0
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    t = mean / se if se > 0 else math.nan
    p = float(2 * stats.t.sf(abs(t), n - 1)) if se > 0 else math.nan
    half = float(stats.t.ppf(0.975, n - 1)) * se
    ref = summary.get("interaction")
    if ref:
        bad = (int(ref.get("n", -1)) != n or abs(num(ref.get("mean_pp")) - mean) > 1e-6
               or (math.isfinite(t) and abs(num(ref.get("t")) - t) > 1e-6))
        if bad:
            sys.exit(f"S4 interaction from trials.csv (n={n}, mean={mean:.6f} pp, t={t:.6f}) does not "
                     f"match summary.json {ref}; not plotting")
        # the interaction is judged by its interval (P4), outside the Holm family of the three loads:
        # print its unadjusted p, which is below 0.05 exactly when the 95 % interval excludes zero
        p_show = num(ref.get("p")) if "p" in ref else p
        if math.isfinite(p) and math.isfinite(p_show) and abs(p_show - p) > 1e-6:
            say(f"WARNING S4 interaction: p in summary.json ({p_show:.6g}) differs from the paired t-test "
                f"on trials.csv ({p:.6g}); printing the summary.json value")
        if "p_holm" in ref:
            say(f"note: summary.json also gives p_holm={num(ref['p_holm']):.4g} for the interaction; "
                "Fig. 4 prints the unadjusted p, as its caption says")
        say(f"S4 interaction: n={n}, mean {mean:+.3f} pp, 95% CI [{mean - half:+.3f}, {mean + half:+.3f}] "
            f"(CI from trials.csv; n, mean, t match summary.json), unadjusted p={p_show:.4g}")
    else:
        p_show = p
        say("summary.json has no 'interaction'; computed from trials.csv alone")
    return dict(mean=mean, lo=mean - half, hi=mean + half, n=n, p=p_show)


def collect(args):
    d = Path(args.analysis_dir).resolve()
    summary = read_summary(d)
    kind = data_kind(d, summary)
    say(f"data: {d}  kind={kind}  phase={summary.get('phase')!r}  block={summary.get('block')!r}")
    if args.final and kind != "FINAL":
        sys.exit(f"refusing --final: {d} holds {kind} data")
    contrasts = read_csv(d / "contrasts.csv")
    # the final analysis writes tier C (A13, A15, the 25 ms run) to tierc.csv, outside contrasts.csv;
    # its only p is the paired t of A0 - A0-tmem25 at heavy load (column p_paired_t)
    tierc_path = d / "tierc.csv"
    if tierc_path.exists():
        have = {(r["arm_a"], r["arm_b"], round(num(r["load"]), 6)) for r in contrasts}
        added = 0
        for r in read_csv(tierc_path):
            key = (r["arm_a"], r["arm_b"], round(num(r["load"]), 6))
            if key in have:
                continue
            contrasts.append(dict(r, metric="completion", p=r.get("p_paired_t", ""), p_holm="",
                                  family=r.get("family") or "C"))
            added += 1
        say(f"tier C: {added} rows read from tierc.csv")
    trials = read_csv(d / "trials.csv")
    loads = sorted(num(x) for x in summary.get("loads", []))
    if not loads:
        loads = sorted({num(r["load"]) for r in contrasts if r["workload"] == MAIN_WORKLOAD})
    light, heavy = loads[0], loads[-1]
    say(f"heavy load = {heavy:g}, light load = {light:g}")

    cands = defaultdict(list)                  # (arm_a, arm_b) -> rows at heavy load, main workload
    for r in contrasts:
        if r.get("metric", "completion") != "completion" or r.get("workload") != MAIN_WORKLOAD:
            continue
        if abs(num(r["load"]) - heavy) > 1e-6:
            continue
        if to_int(r.get("n")) <= 0 or not math.isfinite(num(r.get("mean_diff"))):
            continue                           # arm not run (n = 0)
        cands[(canon(r["arm_a"]), canon(r["arm_b"]))].append(r)

    def lookup(a, b):
        if (a, b) in cands:
            return cands[(a, b)], 1.0
        if (b, a) in cands:
            return cands[(b, a)], -1.0
        return [], 0.0

    groups = list(GROUPS)
    if args.with_baselines:
        groups.insert(3, BASELINE_GROUP)

    tested_fams = {}
    for header, gkind, rows in groups:
        if gkind == "tested":
            c = Counter(r["family"] for a, b, *_ in rows for r in lookup(a, b)[0])
            if c:
                tested_fams[header] = c.most_common(1)[0][0]

    out_groups, absent = [], []
    for header, gkind, rows in groups:
        items = []
        if gkind == "interaction":
            it = interaction_row(summary, trials, heavy, light)
            if it:
                out_groups.append((header, [dict(it, label="heavy " + MINUS + " light", tested=True)]))
            else:
                absent.append("A0 - A9 interaction")
            continue
        for row in rows:
            a, b = row[0], row[1]
            rk = row[2] if len(row) > 2 else gkind
            rs, sign = lookup(a, b)
            if not rs:
                absent.append(f"{a} - {b}")
                continue
            if rk == "tested":
                fam = tested_fams[header]
                pick = [r for r in rs if r["family"] == fam] or rs
            elif rk == "single":
                pick = rs
            else:
                pick = [r for r in rs if r["family"] not in tested_fams.values()] or rs
            r = pick[0]
            lo, hi = sorted((sign * num(r["ci_lo"]) * 100.0, sign * num(r["ci_hi"]) * 100.0))
            p_holm, p_raw = num(r.get("p_holm")), num(r.get("p"))
            label = f"{disp(a)} {MINUS} {disp(b)}"
            if sign < 0:
                say(f"{label}: file stores {r['label']!r}; sign flipped")
            if rk == "tested":
                p_show = p_holm if math.isfinite(p_holm) else p_raw
                fam_size = sum(1 for rr in contrasts if rr["family"] == r["family"]
                               and to_int(rr.get("n")) > 0)
                if not math.isfinite(p_holm) and fam_size > 1:
                    say(f"WARNING {label}: family {r['family']} has {fam_size} contrasts but no Holm p; "
                        "printing the unadjusted p")
            elif rk == "single":
                p_show = p_raw          # registered single paired t-test outside every Holm family
            else:
                p_show = None
                if math.isfinite(p_holm) or r["family"] in tested_fams.values():
                    say(f"WARNING {label}: the paper lists it as an untested effect size, but the file "
                        f"puts it in family {r['family']} (Holm p {p_holm:.3g}); drawn hollow, no p")
            items.append(dict(label=label, mean=sign * num(r["mean_diff"]) * 100.0, lo=lo, hi=hi,
                              n=to_int(r["n"]), p=p_show, tested=rk in ("tested", "single")))
        if items:
            out_groups.append((header, items))

    # membership check: each tested family in the file against the paper's list for its group
    for header, gkind, rows in groups:
        fam = tested_fams.get(header)
        if not fam:
            continue
        paper = {frozenset(ab[:2]) for ab in rows}
        infile = {frozenset((canon(r["arm_a"]), canon(r["arm_b"]))) for r in contrasts
                  if r["family"] == fam and r.get("workload") == MAIN_WORKLOAD
                  and abs(num(r["load"]) - heavy) < 1e-6 and to_int(r.get("n")) > 0}
        extra = sorted(" - ".join(sorted(x)) for x in infile - paper)
        if extra:
            say(f"WARNING family {fam} ({header}) in the file also holds {extra} at heavy load; its Holm "
                f"p values are adjusted over {len(infile)} contrasts, the paper's list has {len(paper)}")
    if absent:
        say(f"note: absent from the files (n = 0 or no row), skipped: {absent}")
    return out_groups, kind


def text_width_in(fig, s, **kw):
    t = fig.text(0, 0, s, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def draw(groups, kind, args, out):
    S.apply()
    lines = []
    for header, items in groups:
        lines.append(("header", header))
        lines += [("row", it) for it in items]
    rows = [p for t, p in lines if t == "row"]
    if not rows:
        sys.exit("no contrast of Fig. 4 found in the files")

    W = S.COL_W_IN
    PITCH = 0.12                       # one line = 8.6 pt
    HEAD_GAP = 0.40                    # extra space above each group header but the first, in lines
    TOP, BOT = 0.27, 0.37
    ys, y = [], 0.0
    for i, (t, _) in enumerate(lines):
        y += HEAD_GAP if (t == "header" and i > 0) else 0.0
        ys.append(y)
        y += 1.0
    span = y
    H = TOP + PITCH * span + BOT
    fig = plt.figure(figsize=(W, H))

    X_HEAD, INDENT = 0.02, 0.09
    lab_w = max(text_width_in(fig, it["label"], fontsize=8) for it in rows)
    p_w = max([text_width_in(fig, "<0.001", fontsize=8, fontweight="bold"),
               text_width_in(fig, r"$p$", fontsize=8)])
    ax_l = X_HEAD + INDENT + lab_w + 0.10
    ax_r = W - 0.03 - p_w - 0.10
    ax = fig.add_axes([ax_l / W, BOT / H, (ax_r - ax_l) / W, PITCH * span / H])
    ax.set_ylim(span - 0.5, -0.5)

    # x range: every interval, 0 and the margin, unless --xlim is given
    if args.xlim:
        x0, x1 = args.xlim
    else:
        x0 = min([it["lo"] for it in rows] + [-SESOI_PP])
        x1 = max([it["hi"] for it in rows] + [SESOI_PP])
        pad = 0.05 * (x1 - x0)
        x0, x1 = x0 - pad, x1 + pad
    ax.set_xlim(x0, x1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))

    ax.axvspan(-SESOI_PP, SESOI_PP, color=BAND, lw=0, zorder=0)
    ax.xaxis.grid(True, color=GRID, lw=0.4, zorder=0.5)
    ax.set_axisbelow(True)
    ax.axvline(0, color=S.GRAY_MID, lw=0.6, zorder=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel("paired difference in heavy-load completion (pp)", labelpad=2)

    ns = sorted({it["n"] for it in rows}, reverse=True)
    if len(ns) > len(SHAPES):
        sys.exit(f"more distinct n than marker shapes: {ns}")
    shape = {n: SHAPES[i] for i, n in enumerate(ns)}

    lab_tf = transforms.blended_transform_factory(fig.transFigure, ax.transData)
    clipped = 0
    for y, (t, p) in zip(ys, lines):
        if t == "header":
            fig.text(X_HEAD / W, y, p, transform=lab_tf, ha="left", va="center_baseline",
                     fontsize=8, style="italic", color=S.INK)
            continue
        it = p
        fig.text((X_HEAD + INDENT) / W, y, it["label"], transform=lab_tf, ha="left",
                 va="center_baseline", fontsize=8, color=S.INK)
        lo, hi = max(it["lo"], x0), min(it["hi"], x1)
        ax.plot([lo, hi], [y, y], color=S.INK, lw=0.9, solid_capstyle="butt", zorder=3)
        for edge, beyond, mk in ((x0, it["lo"] < x0, "<"), (x1, it["hi"] > x1, ">")):
            if beyond:
                clipped += 1
                ax.plot([edge], [y], marker=mk, ms=3.2, color=S.INK, mew=0, zorder=3, clip_on=False)
        m = shape[it["n"]]
        mx = min(max(it["mean"], x0), x1)
        ax.plot([mx], [y], ls="none", marker=m, ms=MS[m], mfc=S.INK if it["tested"] else "white",
                mec=S.INK, mew=0.8, zorder=4, clip_on=False)
        if it["p"] is not None:
            fig.text((W - 0.03) / W, y, fmt_p(it["p"]), transform=lab_tf, ha="right",
                     va="center_baseline", fontsize=8, color=S.INK,
                     fontweight="bold" if it["p"] < 0.05 else "normal")
    if clipped:
        say(f"{clipped} interval end(s) beyond --xlim, drawn as arrowheads at the axis edge")

    # top band: marker key (n), the margin, and the p column header
    y_key = (H - TOP / 2 - 0.01) / H
    handles = [Line2D([], [], ls="none", marker=shape[n], ms=MS[shape[n]], mfc=S.INK, mec=S.INK,
                      mew=0.8) for n in ns]
    labels = [rf"$n$ = {n}" for n in ns]
    handles.append(Patch(facecolor=BAND, edgecolor="none"))
    labels.append(r"$\pm$1 pp")
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(X_HEAD / W - 0.005, y_key),
               ncol=len(handles), handlelength=1.0, handletextpad=0.35, columnspacing=0.9,
               borderaxespad=0.0, borderpad=0.0)
    fig.text((W - 0.03) / W, y_key, r"$p$", ha="right", va="center", fontsize=8, color=S.INK)

    if not args.final:
        watermark(fig, WM_TEXT[kind])
    title = "Fig. 4 forest" if args.final else f"Fig. 4 forest ({WM_TEXT[kind]})"
    save(fig, out, title)
    say(f"size {W:.3f} x {H:.2f} in, {len(rows)} rows in {len(groups)} groups, n values {ns}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis_dir")
    ap.add_argument("out", nargs="?", help="output PDF (default ../fig4_forest.pdf with --final, "
                                           "else ../rehearsal/fig4_forest_<KIND>.pdf)")
    ap.add_argument("--final", action="store_true", help="real analysis: no watermark, submission file")
    ap.add_argument("--with-baselines", action="store_true",
                    help="add the baseline family at heavy load (not in the main.tex placeholder)")
    ap.add_argument("--xlim", nargs=2, type=float, metavar=("LO", "HI"),
                    help="fixed axis range in pp; interval ends beyond it get arrowheads")
    args = ap.parse_args()
    print("Fig. 4 (forest)")
    groups, kind = collect(args)
    out = resolve_out(args.out, args.final, "fig4_forest", kind)
    draw(groups, kind, args, out)


if __name__ == "__main__":
    main()
