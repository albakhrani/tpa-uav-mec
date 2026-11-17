"""Pattern-aware DRM, Algorithm 3; Eqs. (2), (36)-(41), (52)-(56).

Action convention (networks.py): a (q_max, 2M) in [-1, 1]; a[k, :M] are the
preference scores a_ij of window slot k, a[k, M:] map to f~_ij = (a + 1)/2.
Row k of `a` belongs to queue index win[k].
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ..env.queueing import (comp_energy, f_required, mm1_wait, predict_total_time,
                            t_mem_seconds, t_trans)
from ..types import Assignment, Decision
from .burst import Burst, _flag


def _rows_norm(P) -> np.ndarray:
    """Vector 2-norm of each zone row of P (Z, d)."""
    return np.linalg.norm(np.asarray(P, dtype=np.float64), axis=1)


class Admission(NamedTuple):
    """Result of `slot_admissible`.
    ok      : candidate admissible (itself satisfiable AND every earlier admitted task
              still meets its deadline, after `raises` if allow_raise)
    f_req   : candidate's minimal f at its rate with itself + earlier pairs interfering
    rates   : (len(queue), M) predicted rates with interferers = earlier pairs + candidate
    raises  : {position in `assigned`: new f} needed to keep earlier tasks satisfiable
              (empty unless allow_raise=True)
    delay   : candidate's device-busy wait (s) before its upload can start"""
    ok: bool
    f_req: float
    rates: np.ndarray
    raises: dict
    delay: float = 0.0


class RowRates:
    """Predicted rates for a subset of queue indices (SlotObs.rate_rows_fn).
    rates[i, j] and rates.row(i) work for the indices it holds; any other index raises,
    so a hidden dependency on an uncomputed row cannot pass silently."""
    __slots__ = ("rows", "data")

    def __init__(self, rows: dict, data: np.ndarray):
        self.rows, self.data = rows, data

    def __getitem__(self, key):
        i, j = key
        return self.data[self.rows[int(i)], j]

    def row(self, i: int) -> np.ndarray:
        return self.data[self.rows[int(i)]]


def _rates_row(rates, i: int) -> np.ndarray:
    return rates.row(i) if isinstance(rates, RowRates) else rates[int(i)]


def device_busy(obs, device: int) -> float:
    """Seconds of upload already committed on `device`'s single radio at slot start
    (SlotObs.device_busy_s; None / missing = idle)."""
    b = getattr(obs, "device_busy_s", None) or {}
    return float(b.get(int(device), 0.0))


def _t_other(task, R, lam, mu, eta_j, t_mem_s, cfg, delay: float = 0.0) -> float:
    """Eq. 13 without T_comp; `delay` = device-busy wait added to T_trans."""
    return delay + t_trans(task.size_bits, task.complexity, R, cfg) + mm1_wait(lam, mu, eta_j) + t_mem_s


def slot_admissible(obs, assigned, cand, cfg, t_mem_s: float = 0.0, F_res=None, E_res=None,
                    eta=None, h_min=None, f_cand=None, allow_raise: bool = False,
                    rtol: float = 1e-12, rows_fn=None) -> Admission:
    """Residual-rate-budget admission check for interference_set = "slot_assigned".

    assigned : list of (task_index, uav, f) already admitted in this slot, in order.
    cand     : (task_index, uav) candidate pair.
    With interferers = assigned pairs + cand, rates are computed ONCE via obs.rate_fn
    (the rate the simulator fixes for every slot-t upload if no further pair is added).
      (a) the candidate is satisfiable at its rate: h >= h_min (Eq. 2), R >= R_min
          (Eq. 40), f_req = c phi / (rem - T_trans - T_queue - T_mem) finite (Eq. 39) and,
          if given, f_req <= F_res[j] (Eq. 37) and E_res[j] - E_comp(f_req) >= E_min (Eq. 41);
          if f_cand is given it must be >= f_req;
      (b) EVERY earlier admitted task still meets its remaining deadline with its
          allocated f at its new rate (Eq. 13 with T_trans at the new rate) and keeps
          R >= R_min.
    allow_raise: an earlier task failing (b) only on time may instead have its f raised to
    its new f_req, if F_res / E_res (residuals after all `assigned`) cover all raises plus
    the candidate's f_req.
    Serial device radio: each task's T_trans is preceded by its device's busy
    time = obs.device_busy_s[dev] (uploads in flight at slot start) + the T_trans, at the
    NEW rates, of the same device's uploads admitted earlier in this slot (in order).
    Pure: no RNG, no state.
    rows_fn: obs.rate_rows_fn; when given, only the rows this check reads (the
    assigned tasks and the candidate) are computed, with the same values (RowRates).
    """
    i, j = int(cand[0]), int(cand[1])
    eta = np.asarray(obs.eta if eta is None else eta, dtype=np.float64)
    if h_min is None:
        h_min = obs.h_min if getattr(obs, "h_min", None) is not None else cfg.channel.h_min
    h_thr = float(h_min)
    r_min = float(cfg.channel.r_min_mbps) * 1e6
    dt = float(cfg.scenario.slot_s)
    pairs = [(int(a[0]), int(a[1])) for a in assigned] + [(i, j)]
    if rows_fn is not None:
        rates = rows_fn(pairs, [int(a[0]) for a in assigned] + [i])
    else:
        rates = np.asarray(obs.rate_fn(pairs), dtype=np.float64)
    fail = Admission(False, math.inf, rates, {})

    def rem(task):
        return task.deadline_s - (obs.t - task.t_arr) * dt

    # (b) earlier admitted tasks at their new rates, with the device chain
    raises: dict = {}
    need_f = np.zeros(len(eta))
    need_e = np.zeros(len(eta))
    acc: dict = {}                                          # device -> busy seconds so far
    for pos, (ai, aj, af) in enumerate(assigned):
        ai, aj, af = int(ai), int(aj), float(af)
        tk = obs.queue[ai]
        Ra = float(rates[ai, aj])
        if Ra < r_min:
            return fail
        d = acc.get(tk.device, device_busy(obs, tk.device))
        acc[tk.device] = d + t_trans(tk.size_bits, tk.complexity, Ra, cfg)
        budget = rem(tk) - _t_other(tk, Ra, float(obs.lam_eff[aj]), float(obs.mu[aj]),
                                    float(eta[aj]), t_mem_s, cfg, d)
        fa = f_required(tk.cycles, tk.complexity, budget)
        if fa <= af * (1.0 + rtol):
            continue
        if not allow_raise or not math.isfinite(fa):
            return fail
        raises[pos] = fa
        need_f[aj] += fa - af
        need_e[aj] += (comp_energy(tk.cycles, fa, tk.complexity, cfg)
                       - comp_energy(tk.cycles, af, tk.complexity, cfg))
    # (a) candidate (behind every earlier upload of its device)
    task = obs.queue[i]
    R = float(rates[i, j])
    if obs.h_now[i, j] < h_thr or R < r_min:
        return fail
    delay = acc.get(task.device, device_busy(obs, task.device))
    f_req = f_required(task.cycles, task.complexity,
                       rem(task) - _t_other(task, R, float(obs.lam_eff[j]), float(obs.mu[j]),
                                            float(eta[j]), t_mem_s, cfg, delay))
    if not math.isfinite(f_req):
        return fail
    if f_cand is not None and f_cand < f_req * (1.0 - rtol):
        return fail
    need_f[j] += f_req
    need_e[j] += comp_energy(task.cycles, f_req, task.complexity, cfg)
    if F_res is not None and np.any(need_f > np.asarray(F_res, np.float64) * (1.0 + rtol)):
        return fail
    if E_res is not None and np.any(np.asarray(E_res, np.float64) - need_e
                                    < np.asarray(obs.e_min, np.float64)):
        return fail
    return Admission(True, f_req, rates, raises, delay)


def slot_consts(cfg, obs):
    """Per-slot constants for `slot_admissible_fast` (read once per slot instead of
    once per call): (tx_overhead, kappa_c, r_min bit/s, slot_s, e_min (M,))."""
    return (float(cfg.task.tx_overhead), float(cfg.uav.kappa_c), float(cfg.channel.r_min_mbps) * 1e6,
            float(cfg.scenario.slot_s), np.asarray(obs.e_min, dtype=np.float64))


def slot_admissible_fast(obs, assigned, cand, consts, mm1, t_mem_s, F_res, E_res, eta, h_min,
                         allow_raise, rows_fn, rtol: float = 1e-12) -> Admission:
    """Fast variant of `slot_admissible` with (i) the predicted rates restricted to the rows it reads
    (rows_fn = obs.rate_rows_fn), (ii) the per-UAV Eq. 16 wait `mm1` precomputed for the slot
    and (iii) the config constants hoisted (`consts` = slot_consts). Every arithmetic
    expression is the one of `slot_admissible` / `_t_other` / `t_trans` / `f_required` /
    `comp_energy`, in the same association order, on the same float64 inputs, so the result
    is bit-identical (tests/test_d8_equivalence.py). F_res, E_res and eta are required."""
    ovh, kappa_c, r_min, dt, e_min = consts
    i, j = int(cand[0]), int(cand[1])
    eta = np.asarray(eta, dtype=np.float64)
    h_thr = float(h_min)
    queue, t = obs.queue, obs.t
    pairs = [(int(a[0]), int(a[1])) for a in assigned] + [(i, j)]
    rates = rows_fn(pairs, [int(a[0]) for a in assigned] + [i])
    fail = Admission(False, math.inf, rates, {})
    busy = getattr(obs, "device_busy_s", None) or {}
    raises: dict = {}
    need_f = np.zeros(len(eta))
    need_e = np.zeros(len(eta))
    acc: dict = {}                                          # device -> busy seconds so far
    for pos, (ai, aj, af) in enumerate(assigned):
        ai, aj, af = int(ai), int(aj), float(af)
        tk = queue[ai]
        Ra = float(rates[ai, aj])
        if Ra < r_min:
            return fail
        dev = tk.device
        d = acc.get(dev)
        if d is None:
            d = float(busy.get(int(dev), 0.0))
        tt = math.inf if Ra <= 0.0 else tk.size_bits * (1.0 + ovh * tk.complexity) / Ra   # t_trans
        acc[dev] = d + tt
        rem = tk.deadline_s - (t - tk.t_arr) * dt
        budget = rem - (((d + tt) + mm1[aj]) + t_mem_s)                                  # _t_other
        fa = math.inf if budget <= 0.0 else tk.cycles * tk.complexity / budget            # f_required
        if fa <= af * (1.0 + rtol):
            continue
        if not allow_raise or not math.isfinite(fa):
            return fail
        raises[pos] = fa
        need_f[aj] += fa - af
        need_e[aj] += (kappa_c * tk.cycles * fa * fa * tk.complexity
                       - kappa_c * tk.cycles * af * af * tk.complexity)                   # comp_energy
    task = queue[i]
    R = float(rates[i, j])
    if obs.h_now[i, j] < h_thr or R < r_min:
        return fail
    dev = task.device
    delay = acc.get(dev)
    if delay is None:
        delay = float(busy.get(int(dev), 0.0))
    tt = math.inf if R <= 0.0 else task.size_bits * (1.0 + ovh * task.complexity) / R
    rem = task.deadline_s - (t - task.t_arr) * dt
    budget = rem - (((delay + tt) + mm1[j]) + t_mem_s)
    f_req = math.inf if budget <= 0.0 else task.cycles * task.complexity / budget
    if not math.isfinite(f_req):
        return fail
    need_f[j] += f_req
    need_e[j] += kappa_c * task.cycles * f_req * f_req * task.complexity
    if np.any(need_f > np.asarray(F_res, np.float64) * (1.0 + rtol)):
        return fail
    if np.any(np.asarray(E_res, np.float64) - need_e < e_min):
        return fail
    return Admission(True, f_req, rates, raises, delay)


class EntryRates:
    """The rate entries one admission check read, keyed (task_index, uav)."""
    __slots__ = ("d",)

    def __init__(self, d: dict):
        self.d = d

    def __getitem__(self, key):
        return self.d[(int(key[0]), int(key[1]))]


def slot_admissible_scan(obs, assigned, cand, consts, mm1, t_mem_s, F_res, E_res, eta, h_min,
                         allow_raise, scan, rtol: float = 1e-12) -> Admission:
    """Variant of `slot_admissible_fast` with the rates taken from channel.FdmaScan.entries
    (one dgemm per check, elementwise rate expressions at the entries read). Same
    arithmetic and branch order as `slot_admissible`."""
    ovh, kappa_c, r_min, dt, e_min = consts
    i, j = int(cand[0]), int(cand[1])
    eta = np.asarray(eta, dtype=np.float64)
    h_thr = float(h_min)
    queue, t = obs.queue, obs.t
    task = queue[i]
    devs = [queue[int(a[0])].device for a in assigned] + [task.device]
    uavs = [int(a[1]) for a in assigned] + [j]
    re = scan.entries(task.device, j, devs, uavs)
    keyed = {(int(a[0]), int(a[1])): float(re[k]) for k, a in enumerate(assigned)}
    keyed[(i, j)] = float(re[-1])
    rates = EntryRates(keyed)
    fail = Admission(False, math.inf, rates, {})
    busy = getattr(obs, "device_busy_s", None) or {}
    raises: dict = {}
    need_f = np.zeros(len(eta))
    need_e = np.zeros(len(eta))
    acc: dict = {}
    for pos, (ai, aj, af) in enumerate(assigned):
        ai, aj, af = int(ai), int(aj), float(af)
        tk = queue[ai]
        Ra = float(re[pos])
        if Ra < r_min:
            return fail
        dev = tk.device
        d = acc.get(dev)
        if d is None:
            d = float(busy.get(int(dev), 0.0))
        tt = math.inf if Ra <= 0.0 else tk.size_bits * (1.0 + ovh * tk.complexity) / Ra   # t_trans
        acc[dev] = d + tt
        rem = tk.deadline_s - (t - tk.t_arr) * dt
        budget = rem - (((d + tt) + mm1[aj]) + t_mem_s)                                  # _t_other
        fa = math.inf if budget <= 0.0 else tk.cycles * tk.complexity / budget            # f_required
        if fa <= af * (1.0 + rtol):
            continue
        if not allow_raise or not math.isfinite(fa):
            return fail
        raises[pos] = fa
        need_f[aj] += fa - af
        need_e[aj] += (kappa_c * tk.cycles * fa * fa * tk.complexity
                       - kappa_c * tk.cycles * af * af * tk.complexity)                   # comp_energy
    R = float(re[-1])
    if obs.h_now[i, j] < h_thr or R < r_min:
        return fail
    dev = task.device
    delay = acc.get(dev)
    if delay is None:
        delay = float(busy.get(int(dev), 0.0))
    tt = math.inf if R <= 0.0 else task.size_bits * (1.0 + ovh * task.complexity) / R
    rem = task.deadline_s - (t - task.t_arr) * dt
    budget = rem - (((delay + tt) + mm1[j]) + t_mem_s)
    f_req = math.inf if budget <= 0.0 else task.cycles * task.complexity / budget
    if not math.isfinite(f_req):
        return fail
    need_f[j] += f_req
    need_e[j] += kappa_c * task.cycles * f_req * f_req * task.complexity
    if np.any(need_f > np.asarray(F_res, np.float64) * (1.0 + rtol)):
        return fail
    if np.any(np.asarray(E_res, np.float64) - need_e < e_min):
        return fail
    return Admission(True, f_req, rates, raises, delay)


def finalize_predictions(obs, decision, cfg, t_mem_s: float = 0.0, eta=None) -> list:
    """Recompute every assignment's t_total_pred ONCE at the end of the slot
    (Eq. 13), so J1 sees the latency the slot's decisions actually imply:
      * rate  = FINAL rate, obs.rate_fn(all slot pairs) when rate_fn exists (the rate the
                simulator fixes for slot-t uploads), else obs.rates_pred (unchanged);
      * f     = each task's final allocation (after any slot_raise_f raise);
      * wait  = final serial-radio wait: obs.device_busy_s[dev] + T_trans (final rates) of
                the same device's uploads that precede it in `decision.assignments`
                (the order in which the simulator queues them on the radio).
    Mutates decision.assignments[k].t_total_pred, stores the previous (admission-time)
    values in decision.util_terms["t_total_admit"] and returns them. Pure otherwise."""
    asg = decision.assignments
    admit = [float(a.t_total_pred) for a in asg]
    if decision.util_terms is None:
        decision.util_terms = {}
    decision.util_terms["t_total_admit"] = admit
    if not asg:
        return admit
    eta = np.asarray(obs.eta if eta is None else eta, dtype=np.float64)
    rate_fn = getattr(obs, "rate_fn", None)
    pairs = [(int(a.task_index), int(a.uav)) for a in asg]
    rates = (np.asarray(rate_fn(pairs), dtype=np.float64) if rate_fn is not None
             else np.asarray(obs.rates_pred, dtype=np.float64))
    acc: dict = {}
    for a in asg:
        tk = obs.queue[int(a.task_index)]
        j = int(a.uav)
        R = float(rates[int(a.task_index), j])
        d = acc.get(tk.device, device_busy(obs, tk.device))
        acc[tk.device] = d + t_trans(tk.size_bits, tk.complexity, R, cfg)
        a.t_total_pred = d + predict_total_time(tk.size_bits, tk.cycles, tk.complexity, R,
                                                float(a.f), float(obs.lam_eff[j]),
                                                float(obs.mu[j]), float(eta[j]), t_mem_s, cfg)
    return admit


class DRM:
    def __init__(self, cfg, flags=None, arm_key: str = "A0", f_max=None, eta=None):
        D = cfg.drm
        self.cfg = cfg
        self.arm_key = arm_key
        self.pattern = bool(_flag(flags, "pattern_terms", True))
        self.burst_on = bool(_flag(flags, "burst_on", True))
        self.beta_c = float(_flag(flags, "beta_c", D.beta_c))
        self.a_T, self.a_M, self.a_D = float(D.alpha_t), float(D.alpha_m), float(D.alpha_d)
        self.b_E, self.b_R, self.b_L, self.b_A = (float(D.beta_e), float(D.beta_r),
                                                  float(D.beta_l), float(D.beta_a))
        # A6 / memoryless: chi = 0, gamma_p = 0, omega_L = 1 (beta_b = 0 in Burst)
        self.gamma_p = float(D.gamma_p) if self.pattern else 0.0
        self.omega_l = float(D.omega_l) if self.pattern else 1.0
        self.eta_min = float(D.eta_min)
        self.q_max = int(cfg.task.q_max)
        self.M = int(cfg.scenario.n_uav)
        self.Z = int(cfg.scenario.n_zones)
        self.h_min = float(cfg.channel.h_min)
        self.r_min = float(cfg.channel.r_min_mbps) * 1e6
        self.slot_s = float(cfg.scenario.slot_s)
        self.t_mem = t_mem_seconds(cfg, arm_key)
        self.f_max = None if f_max is None else np.asarray(f_max, dtype=np.float64)
        self.eta = None if eta is None else np.asarray(eta, dtype=np.float64)
        # P_j: assignment-weighted zone mix per UAV, EWM with time constant w slots
        self.mix_decay = 1.0 - 1.0 / float(cfg.memory.window_w)
        self.zone_mix = np.zeros((self.M, self.Z))
        self.p_max = 0.0                      # running max ||P||_2 (Eq. 55, log only)
        self.burst = Burst(cfg, flags)
        self.last_pi = np.zeros(0)
        self.n_raises = 0                     # diagnostic: earlier same-slot allocations raised (test coverage)
        # slot_assigned mode (obs.rate_fn set): residual-rate-budget admission.
        # slot_raise_f (ONE rule for DRM and the baseline executor): a candidate
        # is admitted if every earlier same-slot task stays satisfiable, raising earlier
        # tasks' f within residual capacity / energy where needed. Read from
        # cfg.drm.slot_raise_f (default True); the arm flag may override for diagnostics.
        # slot_admission=False keeps only the prefix check of `_pair`, without the slot
        # admission check (diagnostic only).
        self.slot_admission = bool(_flag(flags, "slot_admission", True))
        self.raise_earlier = bool(_flag(flags, "slot_raise_f", cfg.drm.get("slot_raise_f", True)))
        # cfg.drm.impl = "fast" runs Algorithm 3 through _run_fast (vectorised
        # feasibility, utility-ordered admission, row-restricted rate prediction); "reference"
        # runs the plain per-pair loop (_run_reference). Decisions are bit-identical
        # (tests/test_d8_equivalence.py); only the diagnostic count n_admission_checks differs.
        impl = str(cfg.drm.get("impl", "reference"))
        if impl not in ("fast", "reference"):
            raise ValueError(f"drm.impl must be 'fast' or 'reference', got {impl!r}")
        self.fast = impl == "fast"

    def reset(self) -> None:
        self.zone_mix[:] = 0.0
        self.p_max = 0.0
        self.burst.reset()

    # ------------------------------------------------------------ helpers
    def _world(self, obs):
        f_max = getattr(obs, "f_max", None)
        eta = getattr(obs, "eta", None)
        f_max = self.f_max if f_max is None else np.asarray(f_max, np.float64)
        eta = self.eta if eta is None else np.asarray(eta, np.float64)
        if f_max is None or eta is None:
            raise ValueError("DRM needs F_max and eta (pass to DRM(...) or via SlotObs)")
        return f_max, eta

    def _pattern_on(self, P) -> bool:
        return self.pattern and P is not None

    def pattern_strength(self, obs, P) -> np.ndarray:
        """||P_i||_2 / max_z ||P_z||_2 per queued task (P_i = zone row of the task)."""
        Q = len(obs.queue)
        if not self._pattern_on(P) or Q == 0:
            return np.zeros(Q)
        rn = _rows_norm(P)
        mx = rn.max()
        if mx <= 0.0:
            return np.zeros(Q)
        z = np.array([t.zone for t in obs.queue], dtype=np.int64)
        return rn[z] / mx

    # ------------------------------------------------------------ Eqs. 52-53
    def pattern_priority(self, obs, P, sigma) -> np.ndarray:
        """Eq. (53): chi_i = ||P_i||_2 / max_j ||P_j||_2 * sigma_p. Zero for A6/memoryless."""
        return self.pattern_strength(obs, P) * float(sigma)

    def priority(self, obs, P, sigma) -> np.ndarray:
        """Eq. (52): pi_i = a_T rho_i + a_M chi_i + a_D exp(-(T_max - (t - t_arr))/T_max)."""
        q = obs.queue
        if len(q) == 0:
            return np.zeros(0)
        rho = np.array([t.priority for t in q], dtype=np.float64)
        tmax = np.array([t.deadline_s for t in q], dtype=np.float64)
        age = (obs.t - np.array([t.t_arr for t in q], dtype=np.float64)) * self.slot_s
        chi = self.pattern_priority(obs, P, sigma)
        return self.a_T * rho + self.a_M * chi + self.a_D * np.exp(-(tmax - age) / tmax)

    def select_window(self, pi, obs=None) -> np.ndarray:
        """Indices of the q_max highest-pi queued tasks, descending pi,
        ties broken by ascending tid (by queue index if obs is not given)."""
        pi = np.asarray(pi, dtype=np.float64)
        if pi.size == 0:
            return np.zeros(0, dtype=np.int64)
        tie = (np.array([t.tid for t in obs.queue], dtype=np.int64) if obs is not None
               else np.arange(pi.size))
        order = np.lexsort((tie, -pi))
        return order[: self.q_max].astype(np.int64)

    # ------------------------------------------------------------ Eq. 56
    def uav_pattern_share(self, P) -> np.ndarray:
        """Second term of Eq. (56): ||P_j||_2 / ||P||_F, with P_j the assignment-
        weighted mix of zone rows served by UAV j (DRM-internal EWM over ~w slots)."""
        if not self._pattern_on(P):
            return np.zeros(self.M)
        P = np.asarray(P, dtype=np.float64)
        nf = np.linalg.norm(P)
        if nf <= 0.0:
            return np.zeros(self.M)
        tot = self.zone_mix.sum(axis=1, keepdims=True)
        w = np.divide(self.zone_mix, tot, out=np.zeros_like(self.zone_mix), where=tot > 0)
        Pj = w @ P                                                   # (M, d)
        return np.linalg.norm(Pj, axis=1) / nf

    def load_index(self, cap_bound, residual, share) -> np.ndarray:
        """Eq. (56): L_j = w_L (used capacity)/cap_bound_j + (1 - w_L) ||P_j||/||P||_F.
        Used capacity = cap_bound - residual (all reservations on j, incl. this slot's);
        normalised by the CURRENT capacity bound incl. burst, so L_j in [0, 1]."""
        cap = np.asarray(cap_bound, np.float64)
        used = np.divide(cap - residual, cap, out=np.zeros_like(cap), where=cap > 0)
        return self.omega_l * used + (1.0 - self.omega_l) * share

    # ------------------------------------------------------------ feasibility
    def _h_min(self, obs) -> float:
        """Eq. 2 threshold in force: obs.h_min when the simulator supplies it
        (channel.h_min_rule), else cfg.channel.h_min."""
        h = getattr(obs, "h_min", None)
        return self.h_min if h is None else float(h)

    def _pair(self, obs, i, j, F_res, E_res, eta, rates=None, h_min=None, delay=0.0):
        """Satisfiability of Eqs. (2), (37)-(41) for (i, j) under residual budgets.
        `rates` (len(queue), M): predicted rates in force (default obs.rates_pred).
        Returns (feasible, f_req, f_hi) with f_hi the largest f meeting Eqs. 37 and 41."""
        task = obs.queue[i]
        if obs.h_now[i, j] < (self._h_min(obs) if h_min is None else h_min):  # Eq. 2
            return False, math.inf, 0.0
        R = float((obs.rates_pred if rates is None else rates)[i, j])
        if R < self.r_min:                                                 # Eq. 40
            return False, math.inf, 0.0
        rem = task.deadline_s - (obs.t - task.t_arr) * self.slot_s        # Eq. 39 (from arrival)
        other = _t_other(task, R, float(obs.lam_eff[j]), float(obs.mu[j]), float(eta[j]),
                         self.t_mem, self.cfg, delay)      # delay: device-busy wait (serial radio)
        f_req = f_required(task.cycles, task.complexity, rem - other)
        if not math.isfinite(f_req) or f_req > F_res[j]:                  # Eq. 37 (residual)
            return False, f_req, 0.0
        room = E_res[j] - float(obs.e_min[j])                              # Eq. 41
        if room - comp_energy(task.cycles, f_req, task.complexity, self.cfg) < 0.0:
            return False, f_req, 0.0
        k = self.cfg.uav.kappa_c * task.cycles * task.complexity
        f_e = math.sqrt(room / k) if k > 0 else math.inf
        return True, f_req, min(float(F_res[j]), f_e)

    def _pair_vec(self, obs, i, F_res, E_res, eta, rates, h_min, delay=0.0, mm1=None):
        """Vectorised `_pair` for task i over ALL UAVs at once: (ok (M,) bool, f_req (M,), f_hi (M,)).
        Every scalar expression of `_pair` / `_t_other` / `t_trans` / `f_required` /
        `comp_energy` is evaluated elementwise in the same association order, so each entry
        equals the scalar result bit for bit (float64 IEEE ops; no fused operations).
        `mm1` = the per-UAV Eq. 16 wait (mm1_wait, task-independent), precomputed per slot.
        Entries with ok False carry f_hi = 0.0 and an f_req that may be inf, as `_pair`."""
        task = obs.queue[i]
        cfg = self.cfg
        R = np.asarray(_rates_row(rates, i), dtype=np.float64)
        hv = np.asarray(obs.h_now[i], dtype=np.float64)
        F = np.asarray(F_res, dtype=np.float64)
        E = np.asarray(E_res, dtype=np.float64)
        e_min = np.asarray(obs.e_min, dtype=np.float64)
        if mm1 is None:
            mm1 = np.array([mm1_wait(float(obs.lam_eff[j]), float(obs.mu[j]), float(eta[j]))
                            for j in range(self.M)], dtype=np.float64)
        ok = ~(hv < h_min) & ~(R < self.r_min)                            # Eqs. 2, 40
        rem = task.deadline_s - (obs.t - task.t_arr) * self.slot_s        # Eq. 39 (from arrival)
        num = task.size_bits * (1.0 + cfg.task.tx_overhead * task.complexity)   # t_trans numerator
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            tt = np.where(R <= 0.0, np.inf, num / R)                      # Eq. 14
            other = ((delay + tt) + mm1) + self.t_mem                     # _t_other, same order
            budget = rem - other
            cp = task.cycles * task.complexity
            f_req = np.where(budget <= 0.0, np.inf, cp / budget)          # f_required
            ok &= np.isfinite(f_req) & ~(f_req > F)                       # Eq. 37 (residual)
            room = E - e_min                                              # Eq. 41
            ce = ((cfg.uav.kappa_c * task.cycles) * f_req) * f_req * task.complexity   # comp_energy
            ok &= ~((room - ce) < 0.0)
            k = cfg.uav.kappa_c * task.cycles * task.complexity
            f_e = np.sqrt(room / k) if k > 0 else np.full(self.M, math.inf)
            f_hi = np.where(ok, np.minimum(F, f_e), 0.0)
        return ok, f_req, f_hi

    def _pair_batch(self, obs, tasks, F_res, E_res, eta, rates, h_min, delays, mm1):
        """Batched `_pair_vec` for several tasks at once: (ok (T,M), f_req (T,M), f_hi (T,M)).
        `rates` is (T,M) (rows of the tasks, in order), `delays` (T,). Same expressions and
        association order as `_pair` / `_pair_vec`, broadcast over the task axis."""
        cfg = self.cfg
        T = len(tasks)
        if T == 0:
            z = np.zeros((0, self.M))
            return z.astype(bool), z, z
        q = obs.queue
        dl = np.array([q[i].deadline_s for i in tasks], dtype=np.float64)
        ta = np.array([q[i].t_arr for i in tasks], dtype=np.float64)
        sb = np.array([q[i].size_bits for i in tasks], dtype=np.float64)
        ph = np.array([q[i].complexity for i in tasks], dtype=np.float64)
        cy = np.array([q[i].cycles for i in tasks], dtype=np.float64)
        R = np.asarray(rates, dtype=np.float64)
        hv = np.asarray(obs.h_now[tasks], dtype=np.float64)
        F = np.asarray(F_res, dtype=np.float64)[None, :]
        E = np.asarray(E_res, dtype=np.float64)
        e_min = np.asarray(obs.e_min, dtype=np.float64)
        delays = np.asarray(delays, dtype=np.float64)[:, None]
        ok = ~(hv < h_min) & ~(R < self.r_min)                            # Eqs. 2, 40
        rem = (dl - (obs.t - ta) * self.slot_s)[:, None]                 # Eq. 39
        num = (sb * (1.0 + cfg.task.tx_overhead * ph))[:, None]          # t_trans numerator
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            tt = np.where(R <= 0.0, np.inf, num / R)                      # Eq. 14
            other = ((delays + tt) + mm1[None, :]) + self.t_mem           # _t_other
            budget = rem - other
            cp = (cy * ph)[:, None]
            f_req = np.where(budget <= 0.0, np.inf, cp / budget)          # f_required
            ok &= np.isfinite(f_req) & ~(f_req > F)                       # Eq. 37
            room = (E - e_min)[None, :]                                   # Eq. 41
            kc = (cfg.uav.kappa_c * cy)[:, None]
            ce = (kc * f_req) * f_req * ph[:, None]                       # comp_energy
            ok &= ~((room - ce) < 0.0)
            k = (cfg.uav.kappa_c * cy * ph)[:, None]
            f_e = np.where(k > 0, np.sqrt(room / np.where(k > 0, k, 1.0)), np.inf)
            f_hi = np.where(ok, np.minimum(F, f_e), 0.0)
        return ok, f_req, f_hi

    # ------------------------------------------------------------ Eq. 54
    def allocation(self, f_tilde: float, cap_bound_j: float, p_strength_i: float) -> float:
        """Eq. (54): f* = f~ * cap_bound_j * (1 + gamma_p ||P_i||/max||P||) * delta,
        delta = 1; cap_bound already includes eta (not multiplied again)."""
        return float(f_tilde) * float(cap_bound_j) * (1.0 + self.gamma_p * float(p_strength_i))

    # ------------------------------------------------------------ Alg. 3
    def run(self, obs, win, a, P, sigma, P_prev=None, P_ref=None) -> Decision:
        if self.fast:
            return self._run_fast(obs, win, a, P, sigma, P_prev, P_ref)
        return self._run_reference(obs, win, a, P, sigma, P_prev, P_ref)

    def _run_fast(self, obs, win, a, P, sigma, P_prev=None, P_ref=None) -> Decision:
        """Algorithm 3 with (i) `_pair_batch` (batched `_pair_vec`) instead of the per-UAV `_pair` loop,
        (ii) when the utility does not depend on the admission result (beta_c = 0: every
        arm but A13), candidates visited in descending utility (ties: ascending j) and the
        first admitted one taken -- the same argmax as the reference loop with far fewer
        admission checks -- and (iii) rate prediction restricted to the rows that are read
        (obs.rate_rows_fn). For beta_c != 0 every candidate is checked as in the reference.
        Every value that reaches a decision is computed by the same expression as in
        `_run_reference`; only util_terms["n_admission_checks"] (a diagnostic count) differs."""
        M = self.M
        f_max, eta = self._world(obs)
        win = np.asarray(win, dtype=np.int64)
        a = np.asarray(a, dtype=np.float64)
        pi = self.priority(obs, P, sigma)                                  # l.2
        self.last_pi = pi
        chi = self.pattern_priority(obs, P, sigma)
        pstr = self.pattern_strength(obs, P)
        cap = np.asarray(obs.cap_bound, dtype=np.float64)
        F_res = np.asarray(obs.residual_f, dtype=np.float64).copy()
        E_res = np.asarray(obs.energy, dtype=np.float64).copy()
        share = self.uav_pattern_share(P)
        L = self.load_index(cap, F_res, share)
        h_min = self._h_min(obs)
        rate_fn = getattr(obs, "rate_fn", None)
        rows_fn = getattr(obs, "rate_rows_fn", None)
        scan_fn = getattr(obs, "fdma_scan_fn", None)
        rates = np.asarray(obs.rates_pred, dtype=np.float64)
        pairs: list[tuple[int, int]] = []
        admit = rate_fn is not None and self.slot_admission
        admit_rate: list[float] = []
        admit_delay: list[float] = []
        dev_acc: dict = {}
        n_adm_calls = 0
        mm1 = np.array([mm1_wait(float(obs.lam_eff[j]), float(obs.mu[j]), float(eta[j]))
                        for j in range(M)], dtype=np.float64)
        consts = slot_consts(self.cfg, obs)
        scan = scan_fn(()) if (admit and scan_fn is not None) else None

        def admission(i, j):
            n = [(x.task_index, x.uav, x.f) for x in assignments]
            if scan is not None:
                return slot_admissible_scan(obs, n, (i, j), consts, mm1, self.t_mem, F_res, E_res,
                                            eta, h_min, self.raise_earlier, scan)
            if rows_fn is not None:
                return slot_admissible_fast(obs, n, (i, j), consts, mm1, self.t_mem, F_res, E_res,
                                            eta, h_min, self.raise_earlier, rows_fn)
            return slot_admissible(obs, n, (i, j), self.cfg, self.t_mem, F_res=F_res, E_res=E_res,
                                   eta=eta, h_min=h_min, allow_raise=self.raise_earlier)
        if win.size:
            tids = np.array([obs.queue[i].tid for i in win], dtype=np.int64)
            order = np.lexsort((tids, -pi[win]))
        else:
            order = np.zeros(0, dtype=np.int64)
        wtasks = [int(win[k]) for k in order]                              # tasks in visiting order
        nW = len(wtasks)
        # r_max pre-pass: slot-start residuals, slot-start rates, no device delay (as the reference)
        R0 = rates[wtasks] if nW else np.zeros((0, M))
        ok0, _, _ = self._pair_batch(obs, wtasks, F_res, E_res, eta, R0, h_min, np.zeros(nW), mm1)
        n_feas = int(ok0.sum())
        r_max = float(R0[ok0].max()) if n_feas else 0.0
        use_c = self.beta_c != 0.0 and r_max > 0
        cpu_frac = np.zeros(M)
        rate_frac = np.zeros(M)
        assignments: list[Assignment] = []
        # per-prefix batch of the prefix check for the tasks still to visit; redone after each
        # accepted assignment (the only event that changes residuals, device delays or rates)
        Rw = R0                                                            # (nW, M) rows of wtasks
        batch_from = 0
        bok = bfr = bfh = None
        for pos in range(nW):                                              # l.3
            i = wtasks[pos]
            k = int(order[pos])
            if bok is None:
                rest = wtasks[pos:]
                dls = [dev_acc.get(obs.queue[q].device, device_busy(obs, obs.queue[q].device)) for q in rest]
                bok, bfr, bfh = self._pair_batch(obs, rest, F_res, E_res, eta, Rw[pos:], h_min, dls, mm1)
                batch_from = pos
            b = pos - batch_from
            ok, f_req_v, f_hi_v = bok[b], bfr[b], bfh[b]                   # l.5
            pref = a[k, :M]
            f_tilde = np.clip((a[k, M:2 * M] + 1.0) / 2.0, 0.0, 1.0)
            dev = obs.queue[i].device
            dly0 = dev_acc.get(dev, device_busy(obs, dev))
            cand = np.flatnonzero(ok)
            if cand.size == 0:
                continue                                                   # l.14: re-queue
            R_pref = Rw[pos]
            base_u = self.b_E * eta + self.b_R * chi[i] + self.b_L * (1.0 - L) + self.b_A * pref   # l.6
            best_j, best_u, best = -1, -math.inf, None
            if not admit:
                u = base_u + (self.beta_c * R_pref / r_max if use_c else 0.0)
                jc = int(cand[int(np.argmax(u[cand]))])                   # first max = lowest j
                best_j, best_u = jc, float(u[jc])
                best = (float(f_req_v[jc]), float(f_hi_v[jc]), float(R_pref[jc]), {}, dly0)
            elif not use_c:
                # utility independent of the admission result: first admitted in utility order
                u = base_u + 0.0
                for jc in cand[np.lexsort((cand, -u[cand]))]:
                    j = int(jc)
                    n_adm_calls += 1
                    adm = admission(i, j)
                    if not adm.ok:
                        continue
                    f_req, raises = adm.f_req, adm.raises
                    dF = sum(fn - assignments[p].f for p, fn in raises.items()
                             if assignments[p].uav == j)
                    f_hi = max(min(float(f_hi_v[j]), float(F_res[j]) - dF), f_req)
                    best_j, best_u = j, float(u[j])
                    best = (f_req, f_hi, float(adm.rates[i, j]), raises, adm.delay)
                    break
            else:
                for jc in cand:                                            # A13: as the reference
                    j = int(jc)
                    n_adm_calls += 1
                    adm = admission(i, j)
                    if not adm.ok:
                        continue
                    f_req, raises = adm.f_req, adm.raises
                    dF = sum(fn - assignments[p].f for p, fn in raises.items()
                             if assignments[p].uav == j)
                    f_hi = max(min(float(f_hi_v[j]), float(F_res[j]) - dF), f_req)
                    R = float(adm.rates[i, j])
                    u = (self.b_E * eta[j] + self.b_R * chi[i] + self.b_L * (1.0 - L[j])
                         + self.b_A * pref[j]
                         + (self.beta_c * R / r_max if (self.beta_c != 0.0 and r_max > 0) else 0.0))
                    if u > best_u:
                        best_j, best_u, best = j, u, (f_req, f_hi, R, raises, adm.delay)
            if best_j < 0:
                continue                                                   # l.14: re-queue
            j = best_j
            f_req, f_hi, R, raises, dly = best
            self.n_raises += len(raises)
            for p, fn in raises.items():
                x = assignments[p]
                tk = obs.queue[x.task_index]
                F_res[x.uav] -= fn - x.f
                E_res[x.uav] -= (comp_energy(tk.cycles, fn, tk.complexity, self.cfg)
                                 - comp_energy(tk.cycles, x.f, tk.complexity, self.cfg))
                cpu_frac[x.uav] += (fn - x.f) / f_max[x.uav]
                x.f = fn
            f_star = self.allocation(f_tilde[j], cap[j], pstr[i])
            f = float(np.clip(max(f_star, f_req), 0.0, f_hi))             # l.11
            task = obs.queue[i]
            F_res[j] -= f                                                  # l.12
            E_res[j] -= comp_energy(task.cycles, f, task.complexity, self.cfg)
            L = self.load_index(cap, F_res, share)
            t_pred = predict_total_time(task.size_bits, task.cycles, task.complexity, R, f,
                                        float(obs.lam_eff[j]), float(obs.mu[j]), float(eta[j]),
                                        self.t_mem, self.cfg) + dly
            dev_acc[task.device] = dly + t_trans(task.size_bits, task.complexity, R, self.cfg)
            assignments.append(Assignment(task_index=i, uav=j, f=f, t_total_pred=t_pred))
            admit_rate.append(R)
            admit_delay.append(dly)
            cpu_frac[j] += f / f_max[j]
            if r_max > 0:
                rate_frac[j] += R / r_max
            if rate_fn is not None:                                       # slot_assigned: re-predict rates
                pairs.append((i, j))
                if scan is not None:
                    scan = scan_fn(list(pairs))
                    Rw = np.full((nW, M), np.nan)
                    if pos + 1 < nW:
                        rest = wtasks[pos + 1:]
                        Rw[pos + 1:] = scan.rows([obs.queue[q].device for q in rest])
                elif rows_fn is not None:
                    rr = rows_fn(list(pairs), win)
                    Rw = np.array([rr.row(q) for q in wtasks]) if nW else Rw
                else:
                    rates = np.asarray(rate_fn(list(pairs)), dtype=np.float64)
                    Rw = rates[wtasks]
            bok = None                                                     # residuals changed: re-batch
        dec = Decision(assignments=assignments, load_index=L)
        B_next, budget = self.burst.step(obs, dec, P if self._pattern_on(P) else None,
                                         P_prev, P_ref, pi=pi, f_max=f_max)
        dec.burst_next = B_next
        dec.burst_budget = budget
        self.zone_mix *= self.mix_decay
        for asg in assignments:
            self.zone_mix[asg.uav, obs.queue[asg.task_index].zone] += 1.0
        eta_target = float("nan")
        if self._pattern_on(P):
            pn = float(np.linalg.norm(np.asarray(P, np.float64), 2))
            self.p_max = max(self.p_max, pn)
            if self.p_max > 0:
                eta_target = self.eta_min + (1.0 - self.eta_min) * pn / self.p_max
        dec.util_terms = {"admit_rate": admit_rate, "admit_delay": admit_delay, "n_admission_checks": n_adm_calls,
                          "cpu_frac": cpu_frac, "rate_frac": rate_frac, "r_max": r_max,
                          "n_feasible_pairs": n_feas, "burst_budget_frac": budget,
                          "burst_phi": self.burst.last_phi.copy(),
                          "eta_res_target": eta_target,
                          "eta_res": float(np.sum(cap - F_res) / np.sum(cap)) if np.sum(cap) > 0 else 0.0}
        finalize_predictions(obs, dec, self.cfg, self.t_mem, eta)          # final latency predictions (J1)
        return dec

    def _run_reference(self, obs, win, a, P, sigma, P_prev=None, P_ref=None) -> Decision:
        """The plain Algorithm 3 loop (cfg.drm.impl = "reference"), the reference for the
        equivalence test (tests/test_d8_equivalence.py)."""
        M = self.M
        f_max, eta = self._world(obs)
        win = np.asarray(win, dtype=np.int64)
        a = np.asarray(a, dtype=np.float64)
        pi = self.priority(obs, P, sigma)                                  # l.2
        self.last_pi = pi
        chi = self.pattern_priority(obs, P, sigma)
        pstr = self.pattern_strength(obs, P)
        cap = np.asarray(obs.cap_bound, dtype=np.float64)
        F_res = np.asarray(obs.residual_f, dtype=np.float64).copy()
        E_res = np.asarray(obs.energy, dtype=np.float64).copy()
        share = self.uav_pattern_share(P)
        L = self.load_index(cap, F_res, share)
        h_min = self._h_min(obs)
        # interference_set = "slot_assigned": rates are re-predicted after every
        # assignment with the pairs already assigned in this slot as extra interferers.
        rate_fn = getattr(obs, "rate_fn", None)
        rates = np.asarray(obs.rates_pred, dtype=np.float64)
        pairs: list[tuple[int, int]] = []
        admit = rate_fn is not None and self.slot_admission
        admit_rate: list[float] = []
        admit_delay: list[float] = []
        # serial device radio: device -> busy seconds (in flight at slot start + this slot's earlier
        # uploads of that device at their admission rates)
        dev_acc: dict = {}
        n_adm_calls = 0
        # rows of `a` follow win; visit in descending pi (ties by tid)
        if win.size:
            tids = np.array([obs.queue[i].tid for i in win], dtype=np.int64)
            order = np.lexsort((tids, -pi[win]))
        else:
            order = np.zeros(0, dtype=np.int64)
        # R_max: max predicted rate over pairs feasible against the slot-start residuals
        # and slot-start rates. Kept fixed within the slot (also under rate_fn): residuals
        # only shrink and added interferers only lower rates, so every later feasible
        # pair has R <= R_max and R/R_max stays in [0, 1].
        r_max, n_feas = 0.0, 0
        for k in order:
            i = int(win[k])
            for j in range(M):
                ok, _, _ = self._pair(obs, i, j, F_res, E_res, eta, rates, h_min)
                if ok:
                    n_feas += 1
                    r_max = max(r_max, float(rates[i, j]))
        cpu_frac = np.zeros(M)
        rate_frac = np.zeros(M)
        assignments: list[Assignment] = []
        for k in order:                                                    # l.3
            i = int(win[k])
            pref = a[k, :M]
            f_tilde = np.clip((a[k, M:2 * M] + 1.0) / 2.0, 0.0, 1.0)
            best_j, best_u, best = -1, -math.inf, None
            for j in range(M):                                             # l.4
                dly = dev_acc.get(obs.queue[i].device, device_busy(obs, obs.queue[i].device))
                ok, f_req, f_hi = self._pair(obs, i, j, F_res, E_res, eta, rates, h_min, dly)  # l.5
                if not ok:
                    continue
                raises: dict = {}
                if admit:
                    # The prefix check above is a necessary condition (an added
                    # interferer only lowers rates); the full check adds the candidate itself
                    # as an interferer and re-tests every earlier task of this slot
                    n_adm_calls += 1
                    adm = slot_admissible(obs, [(x.task_index, x.uav, x.f) for x in assignments],
                                          (i, j), self.cfg, self.t_mem, F_res=F_res, E_res=E_res,
                                          eta=eta, h_min=h_min, allow_raise=self.raise_earlier)
                    if not adm.ok:
                        continue
                    f_req, raises = adm.f_req, adm.raises
                    dF = sum(fn - assignments[p].f for p, fn in raises.items()
                             if assignments[p].uav == j)
                    f_hi = max(min(f_hi, float(F_res[j]) - dF), f_req)
                    R = float(adm.rates[i, j])
                    dly = adm.delay
                else:
                    R = float(rates[i, j])
                u = (self.b_E * eta[j] + self.b_R * chi[i] + self.b_L * (1.0 - L[j])
                     + self.b_A * pref[j]
                     + (self.beta_c * R / r_max if (self.beta_c != 0.0 and r_max > 0) else 0.0))  # l.6
                if u > best_u:
                    best_j, best_u, best = j, u, (f_req, f_hi, R, raises, dly)
            if best_j < 0:
                continue                                                   # l.14: re-queue
            j = best_j
            f_req, f_hi, R, raises, dly = best
            self.n_raises += len(raises)
            for p, fn in raises.items():            # slot_raise_f: raised f -> Assignment.f, F_res,
                                                    # E_res, cpu_frac; finalize_predictions re-derives
                                                    # t_total_pred from the raised f at the final rates
                x = assignments[p]
                tk = obs.queue[x.task_index]
                F_res[x.uav] -= fn - x.f
                E_res[x.uav] -= (comp_energy(tk.cycles, fn, tk.complexity, self.cfg)
                                 - comp_energy(tk.cycles, x.f, tk.complexity, self.cfg))
                cpu_frac[x.uav] += (fn - x.f) / f_max[x.uav]
                x.f = fn                        # t_total_pred kept at its admission-time value
            f_star = self.allocation(f_tilde[j], cap[j], pstr[i])
            f = float(np.clip(max(f_star, f_req), 0.0, f_hi))             # l.11
            task = obs.queue[i]
            F_res[j] -= f                                                  # l.12
            E_res[j] -= comp_energy(task.cycles, f, task.complexity, self.cfg)
            L = self.load_index(cap, F_res, share)
            t_pred = predict_total_time(task.size_bits, task.cycles, task.complexity, R, f,
                                        float(obs.lam_eff[j]), float(obs.mu[j]), float(eta[j]),
                                        self.t_mem, self.cfg) + dly
            dev_acc[task.device] = dly + t_trans(task.size_bits, task.complexity, R, self.cfg)
            assignments.append(Assignment(task_index=i, uav=j, f=f, t_total_pred=t_pred))
            admit_rate.append(R)
            admit_delay.append(dly)
            cpu_frac[j] += f / f_max[j]
            if r_max > 0:
                rate_frac[j] += R / r_max
            if rate_fn is not None:                                       # slot_assigned: re-predict rates
                pairs.append((i, j))
                rates = np.asarray(rate_fn(list(pairs)), dtype=np.float64)
        dec = Decision(assignments=assignments, load_index=L)
        # l.15: burst for slot t+1
        B_next, budget = self.burst.step(obs, dec, P if self._pattern_on(P) else None,
                                         P_prev, P_ref, pi=pi, f_max=f_max)
        dec.burst_next = B_next
        dec.burst_budget = budget
        # P_j zone-mix EWM (after the slot's assignments)
        self.zone_mix *= self.mix_decay
        for asg in assignments:
            self.zone_mix[asg.uav, obs.queue[asg.task_index].zone] += 1.0
        # Eq. 55 design target (log only)
        eta_target = float("nan")
        if self._pattern_on(P):
            pn = float(np.linalg.norm(np.asarray(P, np.float64), 2))
            self.p_max = max(self.p_max, pn)
            if self.p_max > 0:
                eta_target = self.eta_min + (1.0 - self.eta_min) * pn / self.p_max
        dec.util_terms = {"admit_rate": admit_rate, "admit_delay": admit_delay, "n_admission_checks": n_adm_calls,
                          "cpu_frac": cpu_frac, "rate_frac": rate_frac, "r_max": r_max,
                          "n_feasible_pairs": n_feas, "burst_budget_frac": budget,
                          "burst_phi": self.burst.last_phi.copy(),
                          "eta_res_target": eta_target,
                          "eta_res": float(np.sum(cap - F_res) / np.sum(cap)) if np.sum(cap) > 0 else 0.0}
        finalize_predictions(obs, dec, self.cfg, self.t_mem, eta)          # final latency predictions (J1)
        return dec

    # ------------------------------------------------------------ checkpoint
    def get_state(self) -> dict:
        return {"zone_mix": self.zone_mix.copy(), "p_max": self.p_max,
                "burst": self.burst.get_state()}

    def set_state(self, st: dict) -> None:
        self.zone_mix = np.asarray(st["zone_mix"], dtype=np.float64).copy()
        self.p_max = float(st["p_max"])
        self.burst.set_state(st["burst"])
