"""Per-cycle statistics and P10 on MMPP: analysis-only statistics used by
scripts/analyze_campaign.py. Pure functions on evaluation tables, outside tpa/ and config/. They read nothing by
themselves: analyze_campaign.py decides when evaluation tables are read (its guards keep it outcome-blind until the
registered analysis runs).

What the records allow (tpa/run/evaluator.py, tpa/run/metrics.py): one run row per (load, test run) and 180 minute
rows of 60 slots each; per minute the share of gate-open slots (cons_rate), arrivals, completed and mem_util; per run
the long-term occupancy after the last slot (lt_occupancy). Nothing per slot, and no workload (MMPP) state.

P10, gate on MMPP. (a) open fraction per trial by load for each memory variant: exact, from the run rows'
    cons_rate (no opening is possible in the first w = 100 slots, so the post-warm-up share is cons_rate * N / (N - w)).
    (b) the clustering ratio is NOT computable from the records (no per-slot gate decision, no MMPP state), so P10
    is reported as not computable and (a) is shown only as a descriptive table.
    p10_clustering_ratio() implements the frozen definition for synthetic or future per-slot data, and
    mmpp_base_share() gives the stationary share of slots within w slots after a level change of any zone; for the
    configured MMPP (p_stay 0.98 / 0.95, Z = 6, w = 100) it is 1 - 8e-7, so the frozen ratio is ~1 whatever the gate
    does: the statistic could not discriminate even with per-slot records. On a finite run it is even biased above 1
    for ANY gate (about N / (N - c1 - 1) ~ 1.0007: the slots up to the first level change are 'far' and lie in the
    100-slot warm-up, where the gate cannot open), so with a t-interval over seeds it would have confirmed P10 for
    uniform openings too.
Per-cycle reporting on the sinusoid, from the minute rows. Cycle c of a 3P test run = minutes 60(c-1) .. 60c - 1
    (run-start origin; P = 3600 slots = 60 minute rows). Per cycle: completion = sum completed / sum arrivals; open slots
    = sum 60 * cons_rate; open share of the cycle's slots (and of its post-warm-up slots in cycle 1); open phase bins:
    8 bins of P/8 = 450 slots from the run start ("origin t0"; the bins are NOT phase-aligned
    across seeds, whose zone phases differ, so the per-run counts carry the structure); a minute whose 60 slots
    straddle a bin edge (minutes 7, 22, 37, 52 of each cycle, 30 | 30) is split in proportion to its slots (point
    value), with sharp bounds from the half capacities (each half holds at most 30 openings); number of open intervals
    per cycle = circular runs of open bins, (i) bins whose lower bound is > 0 (certainly open), (ii) bins holding >= 5 %
    of the cycle's openings; cycles with no opening and cycles with all 8 bins open are flagged separately (not
    resolvable at P/8);
    long-term occupancy at the end of each cycle: exact for cycle 3 (the run row's lt_occupancy), bounds for cycles 1-2
    from the minute means L = (K_S + K_L) * mem_util - S, S = minute mean of min(i + 1, K_S), because L never decreases
    within a run (tpa/memory/stores.py: L_filled is only set, and reset only at a run start).
Exploratory comparisons, not registered: A8 - B_x and A9 - B_x at heavy load (analyze_campaign.py; descriptive only).
"""
from __future__ import annotations

import math

import numpy as np

P_SLOTS = 3600
MINUTE_SLOTS = 60
MIN_PER_CYCLE = P_SLOTS // MINUTE_SLOTS          # 60
N_CYCLES = 3
PHASE_BINS = 8
BIN_SLOTS = P_SLOTS // PHASE_BINS                # 450
P10_W = 100
WARMUP_SLOTS = 100                               # memory.window_w: sigma_p = 0 and no consolidation in the first w slots
K_S, K_L = 100, 500                              # memory.k_short, memory.k_long (config/default.toml, never tuned)
OPEN_BIN_SHARE = 0.05                            # interval rule (ii): a bin is open if it holds >= 5 % of the openings
P6_BOUND = 19                                    # P6: LT occupancy <= 19 (rho_merge = 0.95)


# ---------------------------------------------------------------- P10 on MMPP
def level_change_mask(state) -> np.ndarray:
    """(n, Z) per-zone state -> (n,) bool: True at slot t iff some zone's state at t differs from slot t - 1
    (slot 0: False; simultaneous changes of several zones count once)."""
    s = np.asarray(state)
    if s.ndim == 1:
        s = s[:, None]
    m = np.zeros(s.shape[0], bool)
    if s.shape[0] > 1:
        m[1:] = np.any(s[1:] != s[:-1], axis=1)
    return m


def within_after(change_mask, w: int = P10_W) -> np.ndarray:
    """(n,) bool: slot t lies within w slots after a level change, i.e. there is a change at c with 0 < t - c <= w."""
    ch = np.asarray(change_mask, bool)
    n = ch.size
    out = np.zeros(n, bool)
    for c in np.flatnonzero(ch):
        out[c + 1:min(n, c + w + 1)] = True
    return out


def p10_clustering_ratio(open_mask, change_mask, w: int = P10_W) -> float:
    """Frozen P10 (b): share of openings (slots with the gate open) within w slots after a level change of any zone,
    divided by the share of all slots within w slots after a level change (1 = uniform openings; P10 predicts > 1).
    NaN if there is no opening or no slot within w after a change."""
    o = np.asarray(open_mask, bool)
    near = within_after(change_mask, w)
    if o.size != near.size:
        raise ValueError("open_mask and change_mask differ in length")
    if not o.any() or not near.any():
        return math.nan
    return float(near[o].mean() / near.mean())


def mmpp_base_share(p_stay, w: int = P10_W, n_zones: int = 6) -> float:
    """Stationary probability that a slot lies within w slots after a level change of any of n_zones independent
    two-state chains with stay probabilities p_stay = (p_low, p_high): 1 - (pi_low p_low^w + pi_high p_high^w)^Z."""
    p0, p1 = float(p_stay[0]), float(p_stay[1])
    pi0 = (1.0 - p1) / ((1.0 - p0) + (1.0 - p1))
    none = pi0 * p0 ** w + (1.0 - pi0) * p1 ** w
    return float(1.0 - none ** n_zones)


def post_warmup_share(cons_rate: float, n_slots: int, warmup: int = WARMUP_SLOTS) -> float:
    """Open share of the post-warm-up slots, given the share over all n_slots (no opening is possible in the warm-up)."""
    return float(cons_rate) * n_slots / (n_slots - warmup) if np.isfinite(cons_rate) else math.nan


# ---------------------------------------------------------------- per cycle on the sinusoid
def minute_bins(j_in_cycle: int) -> tuple[int, int, float]:
    """Phase bin(s) of minute j of a cycle (run-start origin): (bin_a, bin_b, share_a) with share_a = the share of the
    minute's 60 slots that lie in bin_a (1.0 and bin_b = bin_a when the minute lies in one bin)."""
    u0 = j_in_cycle * MINUTE_SLOTS
    b0, b1 = u0 // BIN_SLOTS, (u0 + MINUTE_SLOTS - 1) // BIN_SLOTS
    if b0 == b1:
        return int(b0), int(b0), 1.0
    return int(b0), int(b1), (b1 * BIN_SLOTS - u0) / MINUTE_SLOTS


def short_term_fill_mean(j: int, k_s: int = K_S) -> float:
    """Minute mean of S_fill(i) = min(i + 1, K_S) over the slots i = 60 j .. 60 j + 59 of a run."""
    i = np.arange(j * MINUTE_SLOTS, (j + 1) * MINUTE_SLOTS)
    return float(np.minimum(i + 1, k_s).mean())


def lt_minute_mean(mem_util: float, j: int, flags) -> float:
    """Minute-mean long-term occupancy from mem_util (= minute mean of (S_fill + L) / (K_S + K_L), tpa/control.py
    mem_fill). flags: tpa.arms.ArmFlags of the arm. NaN for memoryless arms and for A3 (flat store, no M_L)."""
    if flags is None or not flags.memory or flags.flat:
        return math.nan
    if not flags.use_ml:
        return 0.0
    s = short_term_fill_mean(j) if flags.use_ms else 0.0
    v = (K_S + K_L) * mem_util - s
    return float(round(60.0 * v) / 60.0) if np.isfinite(v) else math.nan   # the exact minute mean lies on the 1/60 grid


def count_circular_intervals(open_bins) -> int:
    """Number of maximal circular runs of open bins (all open: 1; none: 0)."""
    b = np.asarray(open_bins, bool)
    if not b.any():
        return 0
    if b.all():
        return 1
    return int(np.sum(b & ~np.roll(b, 1)))


def cycle_rows(minutes, run_row, flags) -> list[dict]:
    """Per-cycle values of ONE test run. minutes: DataFrame of its 180 minute rows (columns t_end, arrivals,
    completed, cons_rate, mem_util); run_row: its run row (dict-like with lt_occupancy, may be None); flags: ArmFlags.
    Returns 3 dicts (cycle 1..3) and, for P6, the rows opened in cycles 2-3."""
    m = minutes.sort_values("t_end")
    j = np.rint((m["t_end"].to_numpy(dtype=float) - float(m["t_end"].min())) / MINUTE_SLOTS).astype(int)
    if len(j) != N_CYCLES * MIN_PER_CYCLE or not np.array_equal(j, np.arange(N_CYCLES * MIN_PER_CYCLE)):
        raise ValueError(f"expected {N_CYCLES * MIN_PER_CYCLE} consecutive minute rows, got {len(j)}")
    arr = m["arrivals"].to_numpy(dtype=float)
    done = m["completed"].to_numpy(dtype=float)
    opens = np.rint(MINUTE_SLOTS * m["cons_rate"].to_numpy(dtype=float)).astype(int)
    mu = m["mem_util"].to_numpy(dtype=float) if "mem_util" in m.columns else np.full(len(j), math.nan)
    L = np.array([lt_minute_mean(u, int(jj), flags) for u, jj in zip(mu, j)])
    has_lt = flags is not None and flags.memory and flags.use_ml and not flags.flat     # a long-term store M_L exists
    lt_run = float(run_row["lt_occupancy"]) if has_lt and run_row is not None and "lt_occupancy" in run_row \
        and run_row["lt_occupancy"] is not None and np.isfinite(float(run_row["lt_occupancy"])) else math.nan
    out = []
    for c in range(1, N_CYCLES + 1):
        sl = slice((c - 1) * MIN_PER_CYCLE, c * MIN_PER_CYCLE)
        a_, d_, o_ = arr[sl], done[sl], opens[sl]
        n_open = int(o_.sum())
        pt, lo, hi = np.zeros(PHASE_BINS), np.zeros(PHASE_BINS), np.zeros(PHASE_BINS)
        for jc in range(MIN_PER_CYCLE):
            ba, bb, sa = minute_bins(jc)
            v = float(o_[jc])
            if ba == bb:
                pt[ba] += v; lo[ba] += v; hi[ba] += v
            else:                                   # straddler: each half holds at most its own slots (sharp bounds)
                na = int(round(sa * MINUTE_SLOTS)); nb = MINUTE_SLOTS - na
                pt[ba] += sa * v; pt[bb] += (1.0 - sa) * v
                lo[ba] += max(0.0, v - nb); lo[bb] += max(0.0, v - na)
                hi[ba] += min(v, na); hi[bb] += min(v, nb)
        slots = P_SLOTS - (WARMUP_SLOTS if c == 1 else 0)
        open_i, open_ii = lo > 0, (pt >= OPEN_BIN_SHARE * n_open) if n_open > 0 else np.zeros(PHASE_BINS, bool)
        rec = dict(cycle=c, arrivals=float(a_.sum()), completed=float(d_.sum()),
                   completion=float(d_.sum() / a_.sum()) if a_.sum() > 0 else math.nan,
                   open_slots=n_open, open_share=n_open / P_SLOTS, open_share_postwarmup=n_open / slots,
                   n_intervals_certain=count_circular_intervals(open_i), n_intervals_5pct=count_circular_intervals(open_ii),
                   no_opening=bool(n_open == 0), all_bins_open_certain=bool(open_i.all()), all_bins_open_5pct=bool(open_ii.all()))
        for b in range(PHASE_BINS):
            rec[f"bin{b}_open"] = float(pt[b]); rec[f"bin{b}_lo"] = float(lo[b]); rec[f"bin{b}_hi"] = float(hi[b])
        if c < N_CYCLES:
            rec["lt_end_lo"], rec["lt_end_hi"] = float(L[c * MIN_PER_CYCLE - 1]), float(L[c * MIN_PER_CYCLE])
        elif np.isfinite(lt_run):                   # exact: the run row's occupancy after the last slot
            rec["lt_end_lo"] = rec["lt_end_hi"] = lt_run
            rec["lt_check_ok"] = bool(not np.isfinite(L[-1]) or L[-1] <= lt_run + 1e-9)
        elif has_lt:                                # run row missing: the last minute mean is only a lower bound
            rec["lt_end_lo"], rec["lt_end_hi"], rec["lt_check_ok"] = float(L[-1]), math.nan, None
        else:                                       # no M_L: 0 for A1 (lt_minute_mean), NaN for A3 and memoryless arms
            rec["lt_end_lo"] = rec["lt_end_hi"] = float(L[-1])
            rec["lt_check_ok"] = None
        out.append(rec)
    out[2]["lt_opened_c23_lo"] = out[2]["lt_end_lo"] - out[0]["lt_end_hi"]
    out[2]["lt_opened_c23_hi"] = out[2]["lt_end_hi"] - out[0]["lt_end_lo"]
    return out


def interval_category(n_intervals: int, no_opening: bool, all_open: bool) -> str:
    """'none' (no opening), 'all8' (all 8 bins open: not resolvable at P/8), '1', '2' or '>=3' (the P9 falsifier side)."""
    if no_opening:
        return "none"
    if all_open:
        return "all8"
    return str(n_intervals) if n_intervals <= 2 else ">=3"


def pooled_completion(rows: list[dict], cycles=(2, 3)) -> float:
    a = sum(r["arrivals"] for r in rows if r["cycle"] in cycles)
    d = sum(r["completed"] for r in rows if r["cycle"] in cycles)
    return d / a if a > 0 else math.nan


def pooled_open_share(rows: list[dict], cycles=(2, 3)) -> float:
    n = sum(r["open_slots"] for r in rows if r["cycle"] in cycles)
    return n / (P_SLOTS * len(cycles))


# ---------------------------------------------------------------- P9: phase imbalance
def kappa_phi(zone_phase) -> float:
    """Registered per-seed phase imbalance kappa_phi = |sum_z exp(2 i phi_z)| / Z (unweighted; used for P9)."""
    ph = np.asarray(zone_phase, float)
    return float(abs(np.exp(2j * ph).sum()) / ph.size)


def kappa_phi_of_seed(seed: int, cfg) -> float:
    """kappa_phi of a test seed's world (the zone phases are the world stream's draw, tpa/env/world.py)."""
    from tpa.env.world import build_world
    from tpa.rng import Streams
    return kappa_phi(build_world(cfg, Streams(int(seed), "train").world).zone_phase)


def correlation(x, y) -> dict:
    """Pearson r with a Fisher-z 95 % CI and Spearman rho over paired finite values."""
    from scipy import stats as st
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    out = dict(n=n, pearson_r=math.nan, ci95=(math.nan, math.nan), spearman_rho=math.nan)
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return out
    r = float(np.corrcoef(x, y)[0, 1])
    z, se = math.atanh(max(min(r, 0.999999), -0.999999)), 1.0 / math.sqrt(n - 3)
    out.update(pearson_r=r, ci95=(math.tanh(z - 1.959964 * se), math.tanh(z + 1.959964 * se)),
               spearman_rho=float(st.spearmanr(x, y).statistic))
    return out


def mean_ci(v) -> dict:
    """Mean and 95 % t half-width over finite values."""
    from scipy import stats as st
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return dict(n=0, mean=math.nan, ci95=math.nan)
    hw = float(st.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)) if n >= 2 else math.nan
    return dict(n=n, mean=float(v.mean()), ci95=hw)
