"""Diagnostic helper for the long-term store (no new behaviour), and the phase
consistency criterion of retrieval (a mechanism check).

    out = run_diagnostic(mem, mepr, params, inputs, store=None)

`inputs` is an iterable of (ctx, zone_arr, zone_mean, truth_bin[, uav_occ]) per slot,
where ctx is the query input [S_sys; tau_bar] (q_dim), truth_bin is the harness label
written with the entry and uav_occ (M,) is the per-UAV occupancy o_t (input of the
zone-resolved candidate; zeros if absent). The helper runs exactly the rollout slot
pipeline of tpa/control.py (encode -> write_short -> MEPR -> candidate z_t / key ->
consolidate -> observe_candidate -> retrieve) on the given objects and returns per-slot
arrays:

    sigma_p, consolidated, lt_occupancy (filled LT slots), n_distinct (filled slots whose
    content has cos < distinct_cos (0.99) with every other filled slot), maxcos (max
    cosine of the candidate to the existing keys / contents at the write, nan without a
    write), top1_age,
    top1_level ('S' / 'L' / 'F' / '' ), event_kind ('' / new / merge / replace / soft /
    fifo), plus mode (lt_addressing), n_events and the LT content/key cosine matrices
    at the end (for inspection).

It mutates `mem` and `mepr` (call reset() first for a fresh segment). It never reads a
test seed or a workload generator; the caller supplies the inputs.

    pc = phase_consistency(driver, period, w)

Phase consistency: hits / slots over the evaluation window [w + P, w + 3P) of a synthetic
periodic driver, pure numpy over driver.info (see the function).

Mechanism checks under arrival keys (key-group count of the long-term keys, age-matched
same-phase share):

    n_groups = greedy_key_groups(K, cos=0.95)      # leader count of the filled L_key rows
    n_iso    = isolated_key_count(K, cos=0.95)     # rows with max cosine to any other < 0.95
    share    = age_matched_phase_share(info, period, w, T)   # from retrieve(min_age=P/2)

Phase coverage of the long-term store (mechanism check):

    pc = phase_coverage(L_key, L_time, L_filled, period, t0)
"""
from __future__ import annotations

import numpy as np
import torch

DISTINCT_COS = 0.99


def _cos_matrix(rows: np.ndarray) -> np.ndarray:
    if rows.shape[0] == 0:
        return np.zeros((0, 0))
    n = np.linalg.norm(rows, axis=1, keepdims=True)
    u = rows / np.maximum(n, 1e-12)
    return u @ u.T


def n_distinct(rows: np.ndarray, thr: float = DISTINCT_COS) -> int:
    """Rows whose cosine with every other row is < thr."""
    m = rows.shape[0]
    if m == 0:
        return 0
    C = _cos_matrix(rows)
    np.fill_diagonal(C, -np.inf)
    return int((C.max(axis=1) < thr).sum()) if m > 1 else 1


def run_diagnostic(mem, mepr, params, inputs, store=None, distinct_cos: float = DISTINCT_COS,
                   t0: int = 0) -> dict:
    sig, cons, occ, ndist, age, lvl, kind, mcos = [], [], [], [], [], [], [], []
    t = int(t0)
    for item in inputs:
        ctx, zone_arr, zone_mean, truth_bin = item[:4]
        uav_occ = item[4] if len(item) > 4 else None
        ctx = np.asarray(ctx, np.float64)
        x = np.concatenate([ctx, mem.mbar_short()])
        if store is not None:
            store.record(t, x, ctx, zone_arr, zone_mean, uav_occ=uav_occ)
        with torch.no_grad():
            e = params.encode(x)
        mem.write_short(e, int(truth_bin), t)
        P, s_p, c = mepr.step_buffered(e, zone_arr, zone_mean, uav_occ=uav_occ)
        # Slot ordering: z_t -> key (inside consolidate, from mu_{t-1}) -> write -> mu update
        z = mem.candidate_for(mepr)
        ev = mem.consolidate(z, int(truth_bin), t, p_ref=mepr.P_ref_used,
                             v_ref=mepr.v_ref_used) if c else None
        if z is not None:
            mem.observe_candidate(z)
        _, info = mem.retrieve(ctx, t)
        if mem.flags.flat:
            rows = mem.F_raw[: mem.F_fill].cpu().numpy()
            occ.append(int(mem.F_fill))
        else:
            rows = mem.L_raw[torch.as_tensor(mem.L_filled)].cpu().numpy() if mem.flags.use_ml \
                else np.zeros((0, mem.d))
            occ.append(int(mem.L_filled.sum()) if mem.flags.use_ml else 0)
        sig.append(float(s_p))
        cons.append(bool(c))
        ndist.append(n_distinct(rows, distinct_cos))
        age.append(int(info["top_age"]))
        lvl.append(info["top_level"] or "")
        kind.append(ev["mode"] if ev is not None else "")
        mcos.append(float(ev["maxcos"]) if ev is not None else np.nan)
        t += 1
    out = dict(sigma_p=np.array(sig), consolidated=np.array(cons), lt_occupancy=np.array(occ),
               n_distinct=np.array(ndist), top1_age=np.array(age), top1_level=np.array(lvl),
               event_kind=np.array(kind), maxcos=np.array(mcos), mode=mem.addressing,
               candidate_repr=params.repr, n_events=int(mem.n_events))
    if mem.flags.use_ml and not mem.flags.flat:
        f = torch.as_tensor(mem.L_filled)
        out["content_cos"] = _cos_matrix(mem.L_raw[f].cpu().numpy())
        out["key_cos"] = _cos_matrix(mem.L_key[f].cpu().numpy())
    return out


def greedy_key_groups(K, cos: float = 0.95) -> int:
    """Key-group count of the long-term keys: greedy leader count of the rows of K (n, Z)
    taken in slot order (the caller passes the filled L_key rows in the order it wants;
    the pre-registered check uses slot order). The first row starts the first group; a
    later row starts a new group iff its cosine to EVERY earlier group leader is < cos.
    Returns the number of groups (0 for no rows). Cosine on float64 with a 1e-12 norm
    floor."""
    K = np.asarray(K, np.float64)
    if K.ndim != 2 or K.shape[0] == 0:
        return 0
    U = K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-12)
    leaders = []
    for u in U:
        if not leaders or (np.asarray(leaders) @ u < cos).all():
            leaders.append(u)
    return len(leaders)


def isolated_key_count(K, cos: float = 0.95) -> int:
    """Isolated-key count (logged alongside the key-group count): number of rows of K
    (n, Z) whose maximum cosine to any OTHER row is < cos. A single row counts as
    isolated; no rows -> 0."""
    K = np.asarray(K, np.float64)
    if K.ndim != 2 or K.shape[0] == 0:
        return 0
    if K.shape[0] == 1:
        return 1
    C = _cos_matrix(K)
    np.fill_diagonal(C, -np.inf)
    return int((C.max(axis=1) < cos).sum())


def age_matched_phase_share(info, period: int, w: int, T: int | None = None) -> float:
    """Age-matched same-phase share: over the evaluation slots t in [w + P, w + 3P), the share
    whose retrieval info (from retrieve(..., min_age=P/2); a dict per slot, or a driver
    with .info) has top_old_age >= 0 and a folded phase offset min(age mod P, P - age
    mod P) <= P / 8. A missing slot, None info, or top_old_age < 0 (no entry older than
    min_age, no displaced memory under A14, no M_L) is a non-hit. T (optional) is the
    run length and must cover the window. Chance 0.25."""
    info = info if isinstance(info, dict) else info.info
    P, w = int(period), int(w)
    lo, hi = w + P, w + 3 * P
    if T is not None and int(T) < hi:
        raise ValueError(f"age_matched_phase_share: run length {T} < w + 3P = {hi}")
    hits = []
    for t in range(lo, hi):
        i = info.get(t)
        age = -1 if not i else int(i.get("top_old_age", -1))
        if age < 0:
            hits.append(False)
            continue
        m = age % P
        hits.append(min(m, P - m) <= P / 8)
    return float(np.mean(hits)) if hits else 0.0


def phase_coverage(L_key, L_time, L_filled, period: int, t0: int) -> dict:
    """Phase coverage and key structure of the long-term store at block end.

    For every filled slot j: bin b_j = floor(((L_time[j] - t0) mod P) * 8 / P) in {0..7}
    (integer arithmetic: ((L_time - t0) mod P) * 8 // P, exact for integer slots).
    Returns dict(bins = the 8 occupancy counts (list of int), covered = every bin holds
    at least one filled slot, n_within / within = number and mean cosine of the key pairs
    of filled slots that share a bin (nan if none), n_between / between = the same over
    pairs in different bins (nan if none), key_pass = n_within > 0 and n_between > 0 and
    within > between). Cosine in float64 with a 1e-12 norm floor; unfilled slots are
    ignored; a store with fewer than two filled slots has no pairs."""
    P = int(period)
    if P < 1:
        raise ValueError("phase_coverage: period must be >= 1")
    f = np.asarray(L_filled, bool)
    idx = np.flatnonzero(f)
    tm = np.asarray(L_time, np.int64)[idx]
    b = ((tm - int(t0)) % P) * 8 // P
    bins = np.bincount(b, minlength=8)[:8]
    n = idx.size
    within = between = float("nan")
    n_within = n_between = 0
    if n >= 2:
        C = _cos_matrix(np.asarray(L_key, np.float64)[idx])
        iu, ju = np.triu_indices(n, 1)
        same = b[iu] == b[ju]
        c = C[iu, ju]
        n_within, n_between = int(same.sum()), int((~same).sum())
        if n_within:
            within = float(c[same].mean())
        if n_between:
            between = float(c[~same].mean())
    key_pass = bool(n_within > 0 and n_between > 0 and within > between)
    return dict(bins=[int(x) for x in bins], covered=bool((bins > 0).all()),
                n_within=n_within, within=within, n_between=n_between, between=between,
                key_pass=key_pass)


def phase_hit(info: dict | None, period: int) -> bool:
    """Phase-consistency hit rule for one slot's retrieval info dict: the top-1 entry is a
    long-term entry (top_level == 'L'), written in an earlier cycle (age >= P/2) and at a
    phase within +-P/8 of the current one: min(age mod P, P - age mod P) <= P/8. An M_S
    top-1, an empty retrieval, no info (A14 "no displaced memory") or an A4 read (no
    top-1) is a non-hit."""
    if not info or info.get("empty", True) or info.get("top_level") != "L":
        return False
    age = int(info.get("top_age", -1))
    if age < 0 or age < period / 2:
        return False
    m = age % period
    return min(m, period - m) <= period / 8


def phase_consistency(driver, period: int, w: int, T: int | None = None) -> float:
    """Phase consistency of retrieval on the synthetic periodic driver
    (tests/test_memory_driver.py Driver, or any object with an `info` dict of per-slot
    retrieval info dicts, or that dict itself).

    Evaluation window: every slot t in [w + P, w + 3P) (P = period, w = MEPR window);
    hit per `phase_hit`; returns hits / number of slots in the window. Chance level for an
    earlier-cycle write at a uniformly random phase is 0.25. T (optional) is the run
    length and must cover the window (T >= w + 3P). Raises if a slot of the window is
    missing from the info dict."""
    info = driver if isinstance(driver, dict) else driver.info
    P, w = int(period), int(w)
    lo, hi = w + P, w + 3 * P
    if T is not None and int(T) < hi:
        raise ValueError(f"phase_consistency: run length {T} < w + 3P = {hi}")
    missing = [t for t in range(lo, hi) if t not in info]
    if missing:
        raise ValueError(f"phase_consistency: {len(missing)} slots of [{lo}, {hi}) missing "
                         f"from driver.info (first {missing[0]})")
    hits = np.array([phase_hit(info[t], P) for t in range(lo, hi)], dtype=bool)
    return float(hits.mean()) if hits.size else 0.0
