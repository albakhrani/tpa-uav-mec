"""Shared matplotlib style for the IEEEtran figures (Times-compatible STIX, 8 pt, TrueType embedding)."""
import matplotlib as mpl

COL_W_IN = 252 / 72.27     # IEEEtran \columnwidth = 252 pt = 3.487 in (include at width=\columnwidth: scale 1.000)
TEXT_W_IN = 516 / 72.27    # IEEEtran \textwidth = 516 pt = 7.140 in

# Colors: blue = derived (formula), orange = measured, grays = structure.
# Blue and orange are also the colors of A0 and A3 in the result figures (STYLE in fig5_load.py).
BLUE = "#1f63b5"
BLUE_LIGHT = "#b4cdee"
ORANGE = "#d4561f"
GRAY_DARK = "#333333"
GRAY_MID = "#7a7a7a"
GRAY_LIGHT = "#ececec"
INK = "#111111"


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.edgecolor": GRAY_DARK,
        "xtick.color": GRAY_DARK,
        "ytick.color": GRAY_DARK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.25,
        "legend.columnspacing": 0.9,
        "pdf.fonttype": 42,          # embed TrueType, no Type 3 fonts
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "axes.unicode_minus": True,
        "text.usetex": False,
    })
