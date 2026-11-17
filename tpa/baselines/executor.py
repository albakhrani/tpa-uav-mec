"""Shared pattern-free executor for the published-method baselines
(B_ZHOU/B_QIN/B_HUANG/B_SONG) and the common scaffolding of the baseline and control-arm controllers.

The executor keeps everything that is *not* the method identical to TPA:
  * feasibility = satisfiability via DRM._pair: Eq. 2 (h >= h_min),
    Eq. 40 (R >= R_min), Eqs. 37/39 (some f <= F_res meets the remaining deadline;
    f_req = minimal such f, from queueing.f_required) and Eq. 41 (energy);
  * the same delay predictor (Eq. 13, queueing.predict_total_time) for t_total_pred;
  * the same queue and physics (the simulator).
What the method decides:
  * the 32-task window is FIFO (top-32 by arrival time, ties by tid) -- no TPA priority;
  * the assignment rule `choose(i, k, cands)` and the allocation rule `alloc(j, items, F0_j)`.
The allocation actually reserved is f = clip(max(f_method, f_req), 0, F_res),
applied sequentially with room kept for the f_req of the tasks still to be served on
the same UAV, so a small (even all-zero) f_method never makes an assigned task infeasible
and the residual capacity is never exceeded (Eq. 37).
No burst (B_j = 0, budget 0) and no pattern terms (chi = 0, gamma_p = 0, omega_L = 1).

Two passes per slot:
  pass 1 (assignment): tasks in FIFO order; each is assigned to choose(i, feasible
          pairs) where feasibility is checked against residuals reduced by the f_req
          (and E_comp(f_req)) of the tasks already assigned this slot;
  pass 2 (allocation): f_method from the method's rule over the tasks assigned to each
          UAV (e.g. equal split for Zhou), then clipped as above.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..drm.drm import (DRM, _rates_row, finalize_predictions, slot_admissible, slot_admissible_fast,
                       slot_admissible_scan, slot_consts)
from ..env.queueing import comp_energy, mm1_wait, predict_total_time
from ..learner.objectives import AdaptiveWeights, Normalizer, reward, slot_objectives
from ..run.timing import ModuleTimer
from ..types import Assignment, Decision


def opt(cfg, dotted: str, default):
    """Optional config value `section.key`.
    Falls back to `default` when the key is absent from config/default.toml."""
    node = cfg
    for part in dotted.split("."):
        try:
            if part not in node:
                return default
            node = node[part]
        except TypeError:
            return default
    return node


def fifo_window(obs, q_max: int) -> np.ndarray:
    """Baseline window: the q_max earliest-arrived queued tasks (ties by tid), in that order."""
    if not obs.queue:
        return np.zeros(0, dtype=np.int64)
    t_arr = np.array([k.t_arr for k in obs.queue], dtype=np.int64)
    tid = np.array([k.tid for k in obs.queue], dtype=np.int64)
    return np.lexsort((tid, t_arr))[:q_max].astype(np.int64)


class Executor:
    """Pattern-free executor (see module docstring)."""

    def __init__(self, cfg, arm_key: str, h_min: float):
        self.cfg = cfg
        self.M = int(cfg.scenario.n_uav)
        self.q_max = int(cfg.task.q_max)
        # DRM is used only for its satisfiability helper (_pair) and constants;
        # pattern terms and burst are off.
        self.drm = DRM(cfg, {"pattern_terms": False, "burst_on": False, "beta_c": 0.0}, arm_key)
        self.drm.h_min = float(h_min)
        self.t_mem = self.drm.t_mem
        # diagnostics only: slot_admission False = prefix check only (no slot admission check);
        # allow_raise False = refuse a candidate instead of raising earlier tasks' f
        self.slot_admission = True
        self.allow_raise = True

    def window(self, obs) -> np.ndarray:
        return fifo_window(obs, self.q_max)

    def admit(self, obs, admitted, cand, F_t, E_t, eta, rows_fn=None, consts=None, mm1=None, scan=None):
        """Slot admission used by pass 1: tpa.drm.slot_admissible -- the SAME rule as DRM
        (residual rate budgets; serial device radio: slot-start busy from
        obs.device_busy_s plus the earlier same-slot uploads of the device at the new
        rates) -- with allow_raise = self.allow_raise (True by default).
        Returns None (refused) or {f_req, R, raises: {pos: f_new}, wait = Admission.delay}."""
        if scan is not None and consts is not None:               # fast path (FdmaScan), same values
            a = slot_admissible_scan(obs, [(x[0], x[1], x[2]) for x in admitted], cand, consts, mm1,
                                     self.t_mem, F_t, E_t, eta, self.drm.h_min, self.allow_raise, scan)
        elif rows_fn is not None and consts is not None:          # fast path, same values
            a = slot_admissible_fast(obs, [(x[0], x[1], x[2]) for x in admitted], cand, consts, mm1,
                                     self.t_mem, F_t, E_t, eta, self.drm.h_min, self.allow_raise, rows_fn)
        else:
            a = slot_admissible(obs, [(x[0], x[1], x[2]) for x in admitted], cand, self.cfg,
                                t_mem_s=self.t_mem, F_res=F_t, E_res=E_t, eta=eta,
                                h_min=self.drm.h_min, allow_raise=self.allow_raise)
        if not a.ok:
            return None
        i, j = int(cand[0]), int(cand[1])
        return dict(f_req=float(a.f_req), R=float(a.rates[i, j]), raises=dict(a.raises),
                    wait=float(a.delay))

    def run(self, obs, win, choose, alloc) -> Decision:
        """choose(i, k, cands) -> j or -1; cands = list of (j, f_req, f_hi, R) feasible pairs,
        k the window row of queue index i.
        alloc(j, items, F0_j) -> list of f_method (one per item); items = list of
        (i, k, f_req, R) in service order; F0_j = slot-start residual of UAV j.
        cfg.drm.impl = "fast" -> _run_fast (vectorised feasibility, row-restricted rate
        prediction, and -- for a `choose` carrying `prefix_order` (Huang, Song), whose choice
        does not depend on the admission values -- first-admitted-in-preference-order);
        "reference" -> _run_reference, the plain two-pass loop."""
        if self.drm.fast:
            return self._run_fast(obs, win, choose, alloc)
        return self._run_reference(obs, win, choose, alloc)

    def _run_fast(self, obs, win, choose, alloc) -> Decision:
        cfg, M, drm = self.cfg, self.M, self.drm
        f_max, eta = drm._world(obs)
        h_min_obs = getattr(obs, "h_min", None)
        if h_min_obs is not None:                              # Eq. 2 threshold in force
            drm.h_min = float(h_min_obs)
        rate_fn = getattr(obs, "rate_fn", None)
        rows_fn = getattr(obs, "rate_rows_fn", None)
        scan_fn = getattr(obs, "fdma_scan_fn", None)
        win = np.asarray(win, dtype=np.int64)
        wl = [int(i) for i in win]
        nW = len(wl)
        F0 = np.asarray(obs.residual_f, dtype=np.float64).copy()
        E0 = np.asarray(obs.energy, dtype=np.float64).copy()
        h_min = drm._h_min(obs)
        mm1 = np.array([mm1_wait(float(obs.lam_eff[j]), float(obs.mu[j]), float(eta[j]))
                        for j in range(M)], dtype=np.float64)
        consts = slot_consts(cfg, obs)
        rates0 = np.asarray(obs.rates_pred, dtype=np.float64)
        Rw = rates0[wl] if nW else np.zeros((0, M))
        ok0, _, _ = drm._pair_batch(obs, wl, F0, E0, eta, Rw, h_min, np.zeros(nW), mm1)
        n_feas = int(ok0.sum())
        r_max = float(Rw[ok0].max()) if n_feas else 0.0
        if rate_fn is None:
            obs_cur = obs
        else:
            obs_cur = replace(obs, rates_pred=rates0.copy(), rate_fn=None)
        scan = scan_fn(()) if (rate_fn is not None and self.slot_admission and scan_fn is not None) else None
        prefix_order = getattr(choose, "prefix_order", None)
        pairs: list = []
        admitted: list = []                       # [task_index, uav, f_floor, k, R, wait_s]
        F_t, E_t = F0.copy(), E0.copy()
        per_uav: list[list] = [[] for _ in range(M)]
        n_adm_calls = 0
        bok = bfr = bfh = None
        batch_from = 0
        for k, i in enumerate(wl):
            if bok is None:                                    # prefix check for the tasks still to visit
                bok, bfr, bfh = drm._pair_batch(obs_cur, wl[k:], F_t, E_t, eta, Rw[k:], h_min,
                                                np.zeros(nW - k), mm1)
                batch_from = k
            b = k - batch_from
            ok, f_req_v, f_hi_v = bok[b], bfr[b], bfh[b]
            cand_js = [int(j) for j in np.flatnonzero(ok)]
            if not cand_js:
                continue                                       # re-queued (Alg. 3 l.14)
            R_pref = Rw[k]
            cands = []
            adm_of = {}
            if rate_fn is not None and self.slot_admission:
                js = prefix_order(i, k, cand_js) if prefix_order is not None else cand_js
                for j in js:
                    n_adm_calls += 1
                    adm = self.admit(obs, admitted, (i, j), F_t, E_t, eta, rows_fn, consts, mm1, scan)
                    if adm is None:
                        continue
                    cands.append((j, adm["f_req"], max(float(f_hi_v[j]), adm["f_req"]), adm["R"]))
                    adm_of[j] = adm
                    if prefix_order is not None:               # the rule takes the first admitted
                        break
            else:
                cands = [(j, float(f_req_v[j]), float(f_hi_v[j]), float(R_pref[j])) for j in cand_js]
            if not cands:
                continue
            j = int(choose(i, k, cands))
            if j < 0:
                continue
            f_req, R = next((c[1], c[3]) for c in cands if c[0] == j)
            task = obs.queue[i]
            if rate_fn is not None:
                wait = 0.0
                if j in adm_of:                                # apply raises of earlier tasks
                    drm.n_raises += len(adm_of[j]["raises"])
                    for pos, fn in adm_of[j]["raises"].items():
                        a = admitted[pos]
                        tk = obs.queue[a[0]]
                        F_t[a[1]] -= fn - a[2]
                        E_t[a[1]] -= (comp_energy(tk.cycles, fn, tk.complexity, cfg)
                                      - comp_energy(tk.cycles, a[2], tk.complexity, cfg))
                        a[2] = fn
                    wait = adm_of[j]["wait"]
                F_t[j] -= f_req
                E_t[j] -= comp_energy(task.cycles, f_req, task.complexity, cfg)
                admitted.append([i, j, f_req, k, R, wait])
                pairs.append((i, j))
                if scan is not None:
                    scan = scan_fn(list(pairs))
                    Rw = np.full((nW, M), np.nan)
                    if k + 1 < nW:
                        Rw[k + 1:] = scan.rows([obs.queue[q].device for q in wl[k + 1:]])
                elif rows_fn is not None:
                    rr = rows_fn(list(pairs), win)
                    Rw = np.array([rr.row(q) for q in wl])
                else:
                    obs_cur.rates_pred = np.asarray(rate_fn(list(pairs)), np.float64)
                    Rw = obs_cur.rates_pred[wl]
            else:
                F_t[j] -= f_req
                E_t[j] -= comp_energy(task.cycles, f_req, task.complexity, cfg)
                per_uav[j].append((i, k, f_req, R))
            bok = None                                         # residuals changed: re-batch
        wait_of = {}
        if rate_fn is not None:                                # floors after all raises
            for i, j, f_fl, k, R, wait in admitted:
                per_uav[j].append((i, k, f_fl, R))
                wait_of[i] = wait
        return self._pass2(obs, win, alloc, F0, E0, eta, f_max, per_uav, wait_of, r_max, n_feas, rate_fn,
                           n_adm_calls)

    def _pass2(self, obs, win, alloc, F0, E0, eta, f_max, per_uav, wait_of, r_max, n_feas, rate_fn,
               n_adm_calls=None) -> Decision:
        """Pass 2 of the fast path (allocation); the same statements as in _run_reference."""
        cfg, M, drm = self.cfg, self.M, self.drm
        F_res, E_res = F0.copy(), E0.copy()
        cpu_frac, rate_frac = np.zeros(M), np.zeros(M)
        assignments: list[Assignment] = []
        for j in range(M):
            items = per_uav[j]
            if not items:
                continue
            f_m = np.asarray(alloc(j, items, float(F0[j])), dtype=np.float64)
            reqs = np.array([it[2] for it in items])
            for n, (i, k, f_req1, R) in enumerate(items):
                if rate_fn is None:
                    ok, f_req, f_hi = drm._pair(obs, i, j, F_res, E_res, eta)
                    if not ok:
                        continue
                else:
                    f_req = f_req1
                    tk = obs.queue[i]
                    e_res = sum(comp_energy(obs.queue[it[0]].cycles, it[2], obs.queue[it[0]].complexity, cfg)
                                for it in items[n + 1:])
                    room = float(E_res[j] - obs.e_min[j]) - e_res
                    kk = cfg.uav.kappa_c * tk.cycles * tk.complexity
                    f_e = math.sqrt(max(room, 0.0) / kk) if kk > 0 else math.inf
                    f_hi = max(min(float(F_res[j]), f_e), f_req)
                reserve = float(reqs[n + 1:].sum())
                upper = min(f_hi, F_res[j] - reserve)
                upper = max(upper, f_req) if f_req <= f_hi else f_hi
                fm = float(f_m[n]) if math.isfinite(f_m[n]) else 0.0
                f = float(np.clip(max(fm, f_req), 0.0, upper))           # feasibility clip (see module docstring)
                if not f > 0.0:
                    continue
                task = obs.queue[i]
                F_res[j] -= f
                E_res[j] -= comp_energy(task.cycles, f, task.complexity, cfg)
                t_pred = predict_total_time(task.size_bits, task.cycles, task.complexity, R, f,
                                            float(obs.lam_eff[j]), float(obs.mu[j]), float(eta[j]),
                                            self.t_mem, cfg) + wait_of.get(i, 0.0)
                assignments.append(Assignment(task_index=i, uav=j, f=f, t_total_pred=t_pred))
                cpu_frac[j] += f / f_max[j]
                if r_max > 0:
                    rate_frac[j] += R / r_max
        assignments.sort(key=lambda a: int(np.where(win == a.task_index)[0][0]))   # FIFO order
        cap = np.asarray(obs.cap_bound, dtype=np.float64)
        used = np.divide(cap - F_res, cap, out=np.zeros(M), where=cap > 0)
        dec = Decision(assignments=assignments, burst_next=np.zeros(M),
                       load_index=used, burst_budget=np.zeros(M))
        dec.util_terms = {"cpu_frac": cpu_frac, "rate_frac": rate_frac, "r_max": r_max,
                          "n_feasible_pairs": n_feas, "burst_budget_frac": np.zeros(M),
                          "eta_res": float(np.sum(cap - F_res) / np.sum(cap)) if np.sum(cap) > 0 else 0.0}
        if n_adm_calls is not None:
            dec.util_terms["n_admission_checks"] = n_adm_calls      # diagnostic
        finalize_predictions(obs, dec, cfg, self.t_mem, eta)   # final latency predictions, as in DRM
        return dec

    def _run_reference(self, obs, win, choose, alloc) -> Decision:
        """The plain two-pass loop (cfg.drm.impl = "reference"; reference for the equivalence test)."""
        cfg, M, drm = self.cfg, self.M, self.drm
        f_max, eta = drm._world(obs)
        h_min_obs = getattr(obs, "h_min", None)
        if h_min_obs is not None:                              # Eq. 2 threshold in force
            drm.h_min = float(h_min_obs)
        rate_fn = getattr(obs, "rate_fn", None)
        win = np.asarray(win, dtype=np.int64)
        F0 = np.asarray(obs.residual_f, dtype=np.float64).copy()
        E0 = np.asarray(obs.energy, dtype=np.float64).copy()
        # R_max over pairs feasible against slot-start residuals and slot-start rates (J2)
        r_max, n_feas = 0.0, 0
        for i in win:
            for j in range(M):
                ok, _, _ = drm._pair(obs, int(i), j, F0, E0, eta)
                if ok:
                    n_feas += 1
                    r_max = max(r_max, float(obs.rates_pred[i, j]))
        # ---- pass 1: assignment against residuals net of committed f_req.
        # interference_set = "slot_assigned": when obs.rate_fn is given, the predicted
        # rates of the remaining tasks are recomputed after every assignment with the pairs
        # assigned so far in this slot as interferers; feasibility, the method's choice
        # (Zhou's rate, Qin's finish time) and the task's t_total_pred all use the rates
        # in force when the task is considered. rate_fn None: slot-start rates throughout.
        if rate_fn is None:
            obs_cur = obs
        else:                                                  # prefix rates (copy of obs)
            obs_cur = replace(obs, rates_pred=np.asarray(obs.rates_pred, np.float64).copy(),
                              rate_fn=None)
        # Slot admission: under slot_assigned the simulator fixes every slot-t
        # upload's rate with ALL slot-t pairs interfering and sends a device's uploads one
        # after another on its single radio. A pair is a candidate only if it passes the
        # prefix check (_pair at the prefix rates; necessary condition) AND the admission
        # check `admit` (= drm.slot_admissible with allow_raise=True, incl. the
        # serial-radio wait): with interferers = earlier pairs + candidate, the candidate
        # is satisfiable and every earlier task of this slot still meets its deadline,
        # if necessary after RAISING its admitted f to its new f_req within the residual
        # capacity / energy (Admission.raises). Raising is the executor's analogue of DRM
        # holding f >= f_req with slack: without it, any second different device lowers the
        # first task's rate and is refused. Each task's admitted f (raised as
        # later candidates require) is its floor; pass 2 only RAISES f above that floor,
        # which only shortens T_comp, so every admission stays valid. The candidate's f_req
        # and R (used by the method's rule and t_total_pred) are those with itself included;
        # t_total_pred also includes its serial-radio wait at admission.
        pairs: list = []
        admitted: list = []                       # [task_index, uav, f_floor, k, R, wait_s]
        F_t, E_t = F0.copy(), E0.copy()
        per_uav: list[list] = [[] for _ in range(M)]
        for k, i in enumerate(win):
            i = int(i)
            cands = []
            adm_of = {}
            for j in range(M):
                ok, f_req, f_hi = drm._pair(obs_cur, i, j, F_t, E_t, eta)
                if not ok:
                    continue
                R = float(obs_cur.rates_pred[i, j])
                if rate_fn is not None and self.slot_admission:
                    adm = self.admit(obs, admitted, (i, j), F_t, E_t, eta)
                    if adm is None:
                        continue
                    f_req, R = adm["f_req"], adm["R"]
                    f_hi = max(f_hi, f_req)
                    adm_of[j] = adm
                cands.append((j, f_req, f_hi, R))
            if not cands:
                continue                                       # re-queued (Alg. 3 l.14)
            j = int(choose(i, k, cands))
            if j < 0:
                continue
            f_req, R = next((c[1], c[3]) for c in cands if c[0] == j)
            task = obs.queue[i]
            if rate_fn is not None:
                wait = 0.0
                if j in adm_of:                                # apply raises of earlier tasks
                    drm.n_raises += len(adm_of[j]["raises"])
                    for pos, fn in adm_of[j]["raises"].items():
                        a = admitted[pos]
                        tk = obs.queue[a[0]]
                        F_t[a[1]] -= fn - a[2]
                        E_t[a[1]] -= (comp_energy(tk.cycles, fn, tk.complexity, cfg)
                                      - comp_energy(tk.cycles, a[2], tk.complexity, cfg))
                        a[2] = fn
                    wait = adm_of[j]["wait"]
                F_t[j] -= f_req
                E_t[j] -= comp_energy(task.cycles, f_req, task.complexity, cfg)
                admitted.append([i, j, f_req, k, R, wait])
                pairs.append((i, j))
                obs_cur.rates_pred = np.asarray(rate_fn(list(pairs)), np.float64)
            else:
                F_t[j] -= f_req
                E_t[j] -= comp_energy(task.cycles, f_req, task.complexity, cfg)
                per_uav[j].append((i, k, f_req, R))
        wait_of = {}
        if rate_fn is not None:                                # floors after all raises
            for i, j, f_fl, k, R, wait in admitted:
                per_uav[j].append((i, k, f_fl, R))
                wait_of[i] = wait
        # ---- pass 2: allocation, sequential with room kept for later f_req
        F_res, E_res = F0.copy(), E0.copy()
        cpu_frac, rate_frac = np.zeros(M), np.zeros(M)
        assignments: list[Assignment] = []
        for j in range(M):
            items = per_uav[j]
            if not items:
                continue
            f_m = np.asarray(alloc(j, items, float(F0[j])), dtype=np.float64)
            reqs = np.array([it[2] for it in items])
            for n, (i, k, f_req1, R) in enumerate(items):
                if rate_fn is None:
                    ok, f_req, f_hi = drm._pair(obs, i, j, F_res, E_res, eta)
                    if not ok:
                        continue
                else:
                    # admitted in pass 1 at f_req1; only raise (bounded by capacity left
                    # after the later tasks' f_req and by the energy left after theirs)
                    f_req = f_req1
                    tk = obs.queue[i]
                    e_res = sum(comp_energy(obs.queue[it[0]].cycles, it[2], obs.queue[it[0]].complexity, cfg)
                                for it in items[n + 1:])
                    room = float(E_res[j] - obs.e_min[j]) - e_res
                    kk = cfg.uav.kappa_c * tk.cycles * tk.complexity
                    f_e = math.sqrt(max(room, 0.0) / kk) if kk > 0 else math.inf
                    f_hi = max(min(float(F_res[j]), f_e), f_req)
                reserve = float(reqs[n + 1:].sum())
                upper = min(f_hi, F_res[j] - reserve)
                upper = max(upper, f_req) if f_req <= f_hi else f_hi
                fm = float(f_m[n]) if math.isfinite(f_m[n]) else 0.0
                f = float(np.clip(max(fm, f_req), 0.0, upper))           # feasibility clip (see module docstring)
                if not f > 0.0:
                    continue
                task = obs.queue[i]
                F_res[j] -= f
                E_res[j] -= comp_energy(task.cycles, f, task.complexity, cfg)
                t_pred = predict_total_time(task.size_bits, task.cycles, task.complexity, R, f,
                                            float(obs.lam_eff[j]), float(obs.mu[j]), float(eta[j]),
                                            self.t_mem, cfg) + wait_of.get(i, 0.0)
                assignments.append(Assignment(task_index=i, uav=j, f=f, t_total_pred=t_pred))
                cpu_frac[j] += f / f_max[j]
                if r_max > 0:
                    rate_frac[j] += R / r_max
        assignments.sort(key=lambda a: int(np.where(win == a.task_index)[0][0]))   # FIFO order
        cap = np.asarray(obs.cap_bound, dtype=np.float64)
        used = np.divide(cap - F_res, cap, out=np.zeros(M), where=cap > 0)
        dec = Decision(assignments=assignments, burst_next=np.zeros(M),
                       load_index=used, burst_budget=np.zeros(M))
        dec.util_terms = {"cpu_frac": cpu_frac, "rate_frac": rate_frac, "r_max": r_max,
                          "n_feasible_pairs": n_feas, "burst_budget_frac": np.zeros(M),
                          "eta_res": float(np.sum(cap - F_res) / np.sum(cap)) if np.sum(cap) > 0 else 0.0}
        finalize_predictions(obs, dec, cfg, self.t_mem, eta)   # final latency predictions, as in DRM
        return dec


# ---------------------------------------------------------------- shared rules
def argmax_pref_choose(a: np.ndarray, M: int):
    """Learned baselines (Huang, Song): j = argmax over feasible pairs of a_ij (ties: lowest j).
    `prefix_order` (used by the fast path): the same preference as a total order over candidate UAVs; it depends
    on a_ij only, so the first admitted candidate in this order is the argmax over the
    admitted ones."""
    def choose(i, k, cands):
        return max(cands, key=lambda c: (a[k, c[0]], -c[0]))[0]

    def prefix_order(i, k, js):
        return sorted(js, key=lambda j: (-a[k, j], j))
    choose.prefix_order = prefix_order
    return choose


def ftilde_alloc(a: np.ndarray, M: int, cap_bound: np.ndarray):
    """Learned baselines: f_method = f~_ij * cap_bound_j, f~ = (a[k, M+j] + 1)/2 (action map as in DRM)."""
    def alloc(j, items, F0):
        return [float(np.clip((a[k, M + j] + 1.0) / 2.0, 0.0, 1.0)) * float(cap_bound[j])
                for (_, k, _, _) in items]
    return alloc


# ---------------------------------------------------------------- controller scaffolding
class BaseController:
    """Common parts of the baseline and control-arm controllers: config, timer, objectives -> reward
    (same slot_objectives / Normalizer / Eq. 59 weights as TPAController).
    Memory-free: J4 = 0, sigma_p = 0 unless a subclass supplies one (A12)."""

    fixed_weights: bool | None = None          # None: follow arm flag adaptive_w
    # get_state() keys that eval_clone takes from the trained controller; the best
    # and final checkpoints hold exactly these (tpa.run.trainer._net_state). Rule controllers
    # learn nothing: only the frozen normaliser. Learning subclasses add their learned state.
    LEARNED_STATE_KEYS: tuple[str, ...] = ("norm",)

    def __init__(self, cfg, arm_key: str, streams, r_ref: float, h_min: float,
                 device: str = "cpu", timer: ModuleTimer | None = None):
        from ..arms import arm
        self.cfg, self.arm_key, self.streams = cfg, arm_key, streams
        self.flags = arm(arm_key)
        self.r_ref, self.h_min, self.device = float(r_ref), float(h_min), device
        self.timer = timer or ModuleTimer()
        self.norm = Normalizer(cfg)
        fixed = (not self.flags.adaptive_w) if self.fixed_weights is None else self.fixed_weights
        self.weights = AdaptiveWeights(cfg, fixed=fixed)
        self.seg_id = -1
        self.slots_seen = 0
        self.last: dict = {}
        self._sigma = 0.0

    def _reward(self, dec, outcome, train: bool):
        J = slot_objectives(dec, outcome, None, self.cfg)
        if train:
            self.norm.observe(J)
        Jn = self.norm.normalize(J)
        w = self.weights.step(Jn, self._sigma)
        return J, reward(Jn, w)

    def learned_state_keys(self) -> tuple[str, ...]:
        return tuple(self.LEARNED_STATE_KEYS)

    def _base_state(self) -> dict:
        return dict(norm=self.norm.get_state(), weights=self.weights.get_state(),
                    seg_id=self.seg_id, slots_seen=self.slots_seen, sigma=self._sigma)

    def _set_base_state(self, st: dict) -> None:
        self.norm.set_state(st["norm"])
        self.weights.set_state(st["weights"])
        self.seg_id = int(st["seg_id"])
        self.slots_seen = int(st["slots_seen"])
        self._sigma = float(st["sigma"])
