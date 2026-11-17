"""Metric accumulation. One accumulator per episode (training) or per test run
(evaluation), with optional per-minute rows.

Definitions (as implemented):
  completion       completed-by-deadline / all arrivals (all-arrivals denominator)
  cpu_util         time-mean reserved capacity / nominal F_max*eta  (burst share separate)
  cpu_eff          completed effective cycles / provisioned nominal cycles   ("by completed tasks")
  bw_eff           mean device-uplink busy fraction x (completed bits / uploaded-task bits)
  mem_util         filled memory slots / (K_S + K_L)  (0 for memory-free arms)
  resource_eff     0.5 cpu_eff + 0.3 bw_eff + 0.2 mem_util     (declared composite)
  resource_eff_cb  0.6 cpu_eff + 0.4 bw_eff                    (CPU+BW only)
  hit_rate         top-1 entry older than P/2 and |label - truth| <= tol bins (circular);
                   empty store = miss; undefined (NaN) for memory-free arms and A4
  energy_eff       completed Gcycles per kJ of (comp + memory) energy
  proc_time_s      mean realized on-board computation time of completed tasks
  wait_s/trans_s/comp_s  realized latency components of finished (completed + late) tasks:
                   controller-queue wait (t_assign - t_arr), upload (t_upload_done - t_assign),
                   computation (t_done - t_upload_done), in seconds
  pred_bias        median of realized (t_done - t_assign) / predicted T_total (Eq. 13) over
                   finished tasks (predictor-bias log); pred_bias_n = sample count
"""
from __future__ import annotations

import math

import numpy as np


class SlotAccumulator:
    def __init__(self, cfg, has_mem: bool, hit_defined: bool, minute_slots: int = 60):
        self.cfg = cfg
        self.has_mem, self.hit_defined = has_mem, hit_defined
        self.P = int(cfg.workload.period_slots)
        self.bins = int(cfg.memory.hit_bins)
        self.tol = int(cfg.memory.hit_tol_bins)
        self.minute = int(minute_slots)
        self.cap_mem = cfg.memory.k_short + cfg.memory.k_long
        self.rows_min: list[dict] = []
        self.pred: dict[int, float] = {}            # tid -> predicted T_total at assignment
        self.bias_tot: list[float] = []
        self._reset_tot()
        self._reset_min()

    def _zero(self):
        return dict(slots=0, arrivals=0, completed=0, failed=0, late=0, assigned=0,
                    cpu_util=0.0, burst_share=0.0, bw_util=0.0, cyc_done=0.0, cyc_prov=0.0,
                    bits_done=0.0, bits_assigned=0.0, e_comp=0.0, e_mem=0.0, e_burst=0.0,
                    e_up=0.0, e_down=0.0, e_hover=0.0, hit=0, hit_n=0, proc_sum=0.0,
                    proc_n=0, reward=0.0, J=np.zeros(4), sigma=0.0, cons=0, mem_fill=0.0,
                    n_expired=0, n_overflow=0, n_aborted=0, wait=0.0, trans=0.0, comp=0.0,
                    fin_n=0, bias=[])

    def _reset_tot(self):
        self.tot = self._zero()

    def _reset_min(self):
        self.cur = self._zero()

    def add(self, t, obs, dec, out, ctrl_last: dict, truth_bin: int, info: dict,
            mem_fill: float, cap_nominal: np.ndarray):
        dt = float(self.cfg.scenario.slot_s)
        for a in dec.assignments:
            self.pred[obs.queue[a.task_index].tid] = float(a.t_total_pred)
        for tk in out.failed:
            self.pred.pop(tk.tid, None)
        fin = list(out.completed) + list(out.late)
        comps = []
        for tk in fin:
            if tk.t_assign < 0 or tk.t_done < 0:
                continue
            w_ = (tk.t_assign - tk.t_arr) * dt
            tr = (tk.t_upload_done - tk.t_assign) * dt
            cp = (tk.t_done - tk.t_upload_done) * dt
            p = self.pred.pop(tk.tid, None)
            b = ((tk.t_done - tk.t_assign) * dt / p) if p and math.isfinite(p) and p > 0 else None
            comps.append((w_, tr, cp, b))
        for acc in (self.tot, self.cur):
            acc["slots"] += 1
            acc["arrivals"] += out.arrivals
            acc["completed"] += len(out.completed)
            acc["failed"] += len(out.failed)
            acc["late"] += len(out.late)
            acc["assigned"] += len(dec.assignments)
            cu = np.asarray(out.cpu_util, dtype=float)
            acc["cpu_util"] += float(np.minimum(cu, 1.0).mean())
            acc["burst_share"] += float(np.maximum(cu - 1.0, 0.0).mean())
            acc["bw_util"] += float(out.bw_util)
            acc["cyc_done"] += float(sum(tk.eff_cycles for tk in out.completed))
            acc["cyc_prov"] += float(cap_nominal.sum()) * float(self.cfg.scenario.slot_s)
            acc["bits_done"] += float(sum(tk.size_bits for tk in out.completed))
            acc["bits_assigned"] += float(sum(obs.queue[a.task_index].size_bits
                                              for a in dec.assignments))
            acc["e_comp"] += float(np.sum(out.comp_energy_j))
            acc["e_mem"] += float(out.mem_energy_j) * len(cap_nominal)
            acc["e_burst"] += float(np.sum(out.burst_energy_j))
            acc["e_up"] += float(out.uplink_energy_j)
            acc["e_down"] += float(out.downlink_energy_j)
            acc["e_hover"] += float(out.hover_energy_j)
            pt = (out.info or {}).get("proc_times_s", [])
            acc["proc_sum"] += float(np.sum(pt)) if len(pt) else 0.0
            acc["proc_n"] += len(pt)
            for k in ("n_expired", "n_overflow", "n_aborted"):
                acc[k] += int((out.info or {}).get(k, 0))
            for w_, tr, cp, b in comps:
                acc["wait"] += w_
                acc["trans"] += tr
                acc["comp"] += cp
                acc["fin_n"] += 1
                if b is not None:
                    acc["bias"].append(b)
            acc["reward"] += float(info.get("reward", 0.0))
            acc["J"] = acc["J"] + np.asarray(info.get("J", np.zeros(4)))
            acc["sigma"] += float(ctrl_last.get("sigma", 0.0))
            acc["cons"] += int(bool(ctrl_last.get("cons", False)))
            acc["mem_fill"] += float(mem_fill)
            if self.hit_defined:
                ri = ctrl_last.get("rinfo") or {}
                acc["hit_n"] += 1
                if not ri.get("empty", True) and ri.get("top_label", -1) >= 0 and truth_bin >= 0:
                    d = abs(int(ri["top_label"]) - int(truth_bin)) % self.bins
                    d = min(d, self.bins - d)
                    if ri.get("top_age", -1) > self.P // 2 and d <= self.tol:
                        acc["hit"] += 1
        if self.cur["slots"] >= self.minute:
            self.rows_min.append(self.summary(self.cur, kind="minute", t_end=t))
            self._reset_min()

    def summary(self, acc=None, kind="run", t_end=None) -> dict:
        a = self.tot if acc is None else acc
        n = max(a["slots"], 1)
        cpu_eff = a["cyc_done"] / a["cyc_prov"] if a["cyc_prov"] > 0 else 0.0
        bw_eff = (a["bw_util"] / n) * (a["bits_done"] / a["bits_assigned"]
                                       if a["bits_assigned"] > 0 else 0.0)
        mem_util = (a["mem_fill"] / n) / self.cap_mem if self.has_mem else 0.0
        e_kj = (a["e_comp"] + a["e_mem"]) / 1e3
        J = a["J"] / n
        return dict(
            row_kind=kind, t_end=-1 if t_end is None else int(t_end), slots=a["slots"],
            arrivals=a["arrivals"], completed=a["completed"], failed=a["failed"], late=a["late"],
            assigned=a["assigned"],
            completion=a["completed"] / a["arrivals"] if a["arrivals"] else math.nan,
            cpu_util=a["cpu_util"] / n, burst_share=a["burst_share"] / n,
            cpu_eff=cpu_eff, bw_eff=bw_eff, mem_util=mem_util,
            resource_eff=0.5 * cpu_eff + 0.3 * bw_eff + 0.2 * mem_util,
            resource_eff_cb=0.6 * cpu_eff + 0.4 * bw_eff,
            hit_rate=(a["hit"] / a["hit_n"]) if (self.hit_defined and a["hit_n"]) else math.nan,
            energy_eff=(a["cyc_done"] / 1e9) / e_kj if e_kj > 0 else math.nan,
            e_comp_j=a["e_comp"], e_mem_j=a["e_mem"], e_burst_j=a["e_burst"],
            e_uplink_j=a["e_up"], e_downlink_j=a["e_down"], e_hover_j=a["e_hover"],
            proc_time_s=a["proc_sum"] / a["proc_n"] if a["proc_n"] else math.nan,
            reward_mean=a["reward"] / n, J1=J[0], J2=J[1], J3=J[2], J4=J[3],
            sigma_mean=a["sigma"] / n, cons_rate=a["cons"] / n,
            wait_s=a["wait"] / a["fin_n"] if a["fin_n"] else math.nan,
            trans_s=a["trans"] / a["fin_n"] if a["fin_n"] else math.nan,
            comp_s=a["comp"] / a["fin_n"] if a["fin_n"] else math.nan,
            pred_bias=float(np.median(a["bias"])) if a["bias"] else math.nan,
            pred_bias_n=len(a["bias"]),
            n_expired=a["n_expired"], n_overflow=a["n_overflow"], n_aborted=a["n_aborted"])
