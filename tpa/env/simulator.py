"""Slot simulator: realized physics for every arm.

Time: slot t covers [t, t+1) in slot units (seconds = slot units * delta_t).
Tasks that arrive in slot t are released at time t. All finish times are
continuous within the slot (float slot units in Task.t_upload_done / t_done).

Realized task life cycle
------------------------
queue --(assignment at slot t, f reserved on UAV j from t until compute done, E_comp
charged, Eq. 17)--> upload --(bits integrated slot by slot at the realized rate, see
"Rates" below; packet loss via task.loss_u)--> compute at f --> done at t_done.
completed iff (t_done - t_arr) * delta_t <= T_max, otherwise late.

Modelling details:
* A device has one radio: its assigned uploads are sent one after another in
  assignment order (the "head" upload is the one in progress).
* Rates depend on channel.interference_set. "in_progress": piecewise constant per slot, computed
  from the uploads in progress after this slot's assignments. "slot_assigned" (the default): fixed
  at a task's assignment, with the uploads assigned in that slot as interferers, for its whole upload.
* T_mem (Eq. 13) delays the start of uploads assigned in slot t to t + T_mem.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..types import SlotObs, SlotOutcome
from . import channel as ch
from .energy import (comp_energy, downlink_energy, downlink_rate, hover_power,
                     propulsion_power, step_energy)
from .mobility import GaussMarkov, Patrol
from .queueing import t_mem_seconds
from .workload import Workload, mean_offered_cycles
from .world import zone_of

S_SYS_DIM_PER_UAV = ("energy_frac", "eta", "reserved_frac", "backlog_s", "burst", "burst_budget_frac")
GM_NAMES = ("gauss_markov", "gm", "gaussmarkov")
_ENV_STREAMS = ("world", "arrivals", "channel", "mobility", "loss")


def s_sys_layout(M: int, Z: int) -> list:
    """Names of the 49 S_sys entries, in order."""
    names = [f"uav{j}.{k}" for j in range(M) for k in S_SYS_DIM_PER_UAV]
    names += [f"zone{z}.{k}" for z in range(Z) for k in ("arr_ratio", "queued_frac")]
    names += ["queue_frac", "mean_slack"]
    names += [f"uav{j}.mean_pred_rate_ratio" for j in range(M)]
    return names


class Simulator:
    def __init__(self, cfg, world, streams, arm_key: str = "A0", workload_kind: str = "sinusoidal",
                 load: float = 0.55, mobility: str = "static", patrol: bool = False,
                 packet_loss: float | None = None, t0: int | None = None, horizon: int | None = None):
        self.cfg, self.world, self.streams, self.arm_key = cfg, world, streams, arm_key
        self.M, self.N, self.Z = world.n_uav, world.n_devices, world.n_zones
        self.dt = float(cfg.scenario.slot_s)
        self.q_max = int(cfg.task.q_max)
        self.queue_cap = int(cfg.task.queue_cap_factor) * self.q_max
        self.max_tx = int(cfg.task.max_tx_attempts)
        self.t_mem_s = float(t_mem_seconds(cfg, arm_key))
        self.p_loss = float(cfg.workload.packet_loss if packet_loss is None else packet_loss)
        self.workload = Workload(cfg, world, streams.arrivals, workload_kind, load, t0=t0, horizon=horizon)
        if mobility in GM_NAMES:
            self.gm = GaussMarkov(cfg, streams.mobility, n=self.N)
        elif mobility == "static":
            self.gm = None
        else:
            raise ValueError(f"unknown mobility {mobility!r}; use 'static' or 'gauss_markov'")
        self.patrol = Patrol(cfg, world) if patrol else None
        self.hover_w = hover_power(cfg)
        self.fly_w = propulsion_power(self.patrol.speed(), cfg) if self.patrol else self.hover_w
        self.r_ref = ch.r_ref(world, cfg)
        self.e_min = float(cfg.uav.e_min_frac) * world.battery_j
        self.rho_b = float(cfg.burst.rho_b)
        self.ewm_beta = float(cfg.drm.rate_ewm)
        self.zone_beta = float(cfg.workload.zone_ewm)
        self.cycles_prior = mean_offered_cycles(cfg)
        self.h_min = ch.h_min_value(world, cfg)               # Eq. 2 threshold (channel.h_min_rule)
        self.interference_set = str(cfg.channel.interference_set)
        if self.interference_set not in ("in_progress", "slot_assigned"):
            raise ValueError(f"channel.interference_set must be 'in_progress' or 'slot_assigned', "
                             f"got {self.interference_set!r}")
        self.abort_at_deadline = bool(cfg.uav.abort_at_deadline)
        self.access = str(cfg.channel.access)
        if self.access not in ("shared", "fdma_per_uav"):
            raise ValueError(f"channel.access must be 'shared' or 'fdma_per_uav', got {self.access!r}")

        # ---- runtime state (all of it goes through get_state/set_state)
        self.device_pos = world.device_pos.copy()
        self.uav_pos = world.uav_pos.copy()
        self.energy = world.battery_j.copy()
        self.burst = np.zeros(self.M, dtype=bool)
        self.burst_budget = np.zeros(self.M)
        self.queue: list = []          # unassigned tasks, FIFO by arrival
        self.inflight: list = []       # assigned tasks, in assignment order
        self.retry_slot: dict = {}     # tid -> first slot an upload may restart after a loss
        self.fixed_rate: dict = {}     # tid -> rate fixed at assignment ("slot_assigned" mode)
        self.lam_eff = np.zeros(self.M)
        self.ewm_cycles = np.full(self.M, self.cycles_prior)
        self.zone_m = np.zeros(self.Z)
        self.zone_n = 0
        self.t = None                  # slot in progress (between begin_slot and end_slot)
        self.next_t = None
        self.pending_failed: list = []
        self.obs_queue: list = []
        self.h_now = None
        self.xi = None
        self.last_arrivals: list = []
        self.last_zone_arrivals = np.zeros(self.Z)
        self.last_info: dict = {}

    # ------------------------------------------------------------------ helpers
    def cap_bound(self) -> np.ndarray:
        """Capacity bound F_max eta (1 + rho_b B_j) (Eqs. 37-38 with burst)."""
        w = self.world
        return w.f_max * w.eta * (1.0 + self.rho_b * self.burst.astype(float))

    def reserved(self) -> np.ndarray:
        r = np.zeros(self.M)
        for task in self.inflight:
            r[task.uav] += task.f_alloc
        return r

    def residual(self) -> np.ndarray:
        return np.maximum(self.cap_bound() - self.reserved(), 0.0)

    def mu(self) -> np.ndarray:
        """Service rate mu_j (tasks/s) for Eq. 16: F_max / EWM of assigned c*phi.
        Cold start: the EWM starts at E[c phi] of the task distribution (cfg ranges),
        so mu_j > 0 from slot 0. eta is NOT folded in here: Eq. 16 applies it as mu_j eta_j."""
        return self.world.f_max / self.ewm_cycles

    def zone_mean(self) -> np.ndarray:
        """Online long-run mean of observed per-zone arrivals (no generator leakage).
        zone_ewm >= 1: cumulative mean since this simulator started (one memory segment or
        one test run), because an EWM with a time constant < P tracks the cycle and
        halves the recurrent pattern. zone_ewm < 1: bias-corrected EWM (legacy)."""
        if self.zone_n == 0:
            return np.zeros(self.Z)
        if self.zone_beta >= 1.0:
            return self.zone_m / self.zone_n
        return self.zone_m / (1.0 - self.zone_beta ** self.zone_n)

    def _active(self) -> np.ndarray:
        """x(t): (N,M) bool, the upload in progress (head) of each device."""
        act = np.zeros((self.N, self.M), dtype=bool)
        seen = set()
        for task in self.inflight:
            if task.bits_left > 0.0 and task.device not in seen:
                seen.add(task.device)
                act[task.device, task.uav] = True
        return act

    def _rates(self, h: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Rates for every pair given the interferer set x: Eq. (7) under access="shared",
        the per-UAV FDMA model (channel.rate_fdma, a model change) under "fdma_per_uav"."""
        return ch.rate_access(h, self.world.device_ptx, active, self.world.eta, self.cfg)

    def _device_busy_s(self, R_now: np.ndarray) -> dict:
        """SlotObs.device_busy_s: device -> seconds of upload already committed
        on its single radio at slot start = sum over its pending uploads (in order) of
        remaining bits / rate, with rate = the upload's fixed rate ("slot_assigned") or the
        current predicted rate R_now[dev, uav] ("in_progress"). Idle devices are omitted;
        a zero rate gives inf. Pure: no RNG, no state change."""
        busy: dict = {}
        sa = self.interference_set == "slot_assigned"
        for task in self.inflight:
            if task.bits_left > 0.0:
                r = self.fixed_rate[task.tid] if sa else float(R_now[task.device, task.uav])
                sec = task.bits_left / r if r > 0.0 else math.inf
                busy[task.device] = busy.get(task.device, 0.0) + sec
        return busy

    def _make_rate_fn(self):
        """SlotObs.rate_fn for this slot: rate_fn(pairs) -> (len(queue), M) predicted bit/s
        for the queued tasks, given [(task_index, uav), ...] already assigned earlier in
        THIS slot. Interferers: those pairs ("slot_assigned"), or the uploads in progress
        at the start of the slot plus those pairs ("in_progress"). Uses this slot's
        fading. Pure: a snapshot is captured; no RNG, no state change."""
        h = self.h_now.copy()
        base = self._prediction_set().copy()
        qdev = np.array([k.device for k in self.obs_queue], dtype=int)
        M, rates = self.M, self._rates

        def rate_fn(pairs=()):
            act = base.copy()
            for idx, j in pairs:
                act[qdev[int(idx)], int(j)] = True
            if len(qdev) == 0:
                return np.zeros((0, M))
            return rates(h, act)[qdev]
        return rate_fn

    def _make_rate_rows_fn(self):
        """SlotObs.rate_rows_fn(pairs, qidx) -> RowRates holding, for the queue indices
        `qidx` only, the same float64 values as rate_fn(pairs)[qidx] (channel.rate_fdma_rows
        under "fdma_per_uav"; the full computation under "shared"). Pure, like rate_fn."""
        from ..drm.drm import RowRates
        h = self.h_now.copy()
        base = self._prediction_set().copy()
        qdev = np.array([k.device for k in self.obs_queue], dtype=int)
        M, rates, cfg, w = self.M, self._rates, self.cfg, self.world
        fdma = self.access == "fdma_per_uav"
        p_kj = w.device_ptx[:, None] * h if fdma else None    # same expression as rate_fdma; h fixed in the slot

        def rate_rows_fn(pairs, qidx):
            act = base.copy()
            for idx, j in pairs:
                act[qdev[int(idx)], int(j)] = True
            qidx = [int(q) for q in qidx]
            if not qidx:
                return RowRates({}, np.zeros((0, M)))
            devs = qdev[qidx]
            if fdma:
                R = ch.rate_fdma_rows(h, w.device_ptx, act, w.eta, cfg, devs, p_kj)
            else:
                R = rates(h, act)[devs]
            return RowRates({q: r for r, q in enumerate(qidx)}, R)
        return rate_rows_fn

    def _make_fdma_scan_fn(self):
        """SlotObs.fdma_scan_fn(pairs) -> channel.FdmaScan for the prefix base | pairs
        (None under access = "shared": the fast path then falls back to rate_rows_fn)."""
        if self.access != "fdma_per_uav":
            return None
        h = self.h_now.copy()
        base = self._prediction_set().copy()
        qdev = np.array([k.device for k in self.obs_queue], dtype=int)
        w, cfg = self.world, self.cfg
        p_kj = w.device_ptx[:, None] * h

        def fdma_scan_fn(pairs):
            act = base.copy()
            for idx, j in pairs:
                act[qdev[int(idx)], int(j)] = True
            return ch.FdmaScan(h, w.device_ptx, act, w.eta, cfg, p_kj)
        return fdma_scan_fn

    def predict_rates(self, active_set: np.ndarray) -> np.ndarray:
        """Predicted rates (len(obs.queue), M) of the currently queued tasks (Eq. 7 or rate_fdma,
        per channel.access) for a given interferer set x (N,M) bool, with this slot's channel h(t).
        For 'slot_assigned' pass the set already assigned earlier in this slot; for
        'in_progress' the simulator's own prediction uses `_active()`."""
        R = self._rates(self.h_now, np.asarray(active_set, dtype=bool))
        qdev = np.array([k.device for k in self.obs_queue], dtype=int)
        return R[qdev] if len(qdev) else np.zeros((0, self.M))

    def active_from_assignments(self, pairs) -> np.ndarray:
        """(N,M) bool interferer set from (task_index into obs.queue, uav) pairs."""
        act = np.zeros((self.N, self.M), dtype=bool)
        for idx, j in pairs:
            act[self.obs_queue[int(idx)].device, int(j)] = True
        return act

    def _prediction_set(self) -> np.ndarray:
        if self.interference_set == "slot_assigned":
            return np.zeros((self.N, self.M), dtype=bool)    # nothing assigned yet in slot t
        return self._active()

    def uav_state(self) -> np.ndarray:
        """Eq. (3): s_j = [p_j (x, y, H), E_j, C_j (current capacity bound), eta_j]  (M,6)."""
        H = np.full(self.M, self.cfg.scenario.hover_altitude_m)
        return np.column_stack([self.uav_pos, H, self.energy, self.cap_bound(), self.world.eta])

    def truth_phase_bin(self, t: int) -> int:
        """Harness only: generator phase bin of slot t (-1 for aperiodic kinds)."""
        return self.workload.truth_bin(t)

    # ------------------------------------------------------------------ slot API
    def begin_slot(self, t: int) -> SlotObs:
        if self.t is not None:
            raise RuntimeError(f"begin_slot({t}) called before end_slot({self.t})")
        if self.next_t is not None and t != self.next_t:
            raise RuntimeError(f"slots must be consecutive: expected {self.next_t}, got {t}")
        cfg, w = self.cfg, self.world
        self.t = int(t)
        # 1. mobility
        if self.gm is not None:
            self.device_pos = self.gm.step(self.device_pos)
        if self.patrol is not None:
            self.uav_pos = self.patrol.uav_pos(t)
        # 2. fading, full N x M every slot (CRN)
        self.xi = ch.draw_fading(self.streams.channel, w.kappa)
        self.h_now = ch.channel_gain(self.device_pos, self.uav_pos, w, self.xi, cfg)
        # 3. arrivals
        new = self.workload.step(t, self.device_pos)
        self.last_arrivals = new
        self.queue.extend(new)
        za = np.bincount(np.array([k.zone for k in new], dtype=int), minlength=self.Z).astype(float)
        self.last_zone_arrivals = za
        if self.zone_beta >= 1.0:
            self.zone_m = self.zone_m + za
        else:
            self.zone_m = self.zone_beta * self.zone_m + (1.0 - self.zone_beta) * za
        self.zone_n += 1
        # 4. expiry (Alg. 3 line 14): t - t_arr > T_max
        keep, expired = [], []
        for task in self.queue:
            (expired if (t - task.t_arr) * self.dt > task.deadline_s else keep).append(task)
        # 5. overflow: drop the oldest beyond queue_cap
        overflow = []
        if len(keep) > self.queue_cap:
            keep.sort(key=lambda k: (k.t_arr, k.tid))
            overflow, keep = keep[:len(keep) - self.queue_cap], keep[len(keep) - self.queue_cap:]
        self.queue = keep
        # 5b. abort in-flight tasks past their deadline (cfg.uav.abort_at_deadline): failed, f freed now
        aborted = []
        if self.abort_at_deadline:
            alive = []
            for task in self.inflight:
                (aborted if (t - task.t_arr) * self.dt > task.deadline_s else alive).append(task)
            if aborted:
                self.inflight = alive
                for task in aborted:
                    self.retry_slot.pop(task.tid, None)
                    self.fixed_rate.pop(task.tid, None)
        self.pending_failed = expired + overflow + aborted
        self._n_expired, self._n_overflow, self._n_aborted = len(expired), len(overflow), len(aborted)
        # 5c. per-UAV in-flight occupancy at slot start (after 5b abort, before this
        #     slot's assignments) = tasks currently assigned to UAV j / q_max
        occ = np.bincount(np.array([task.uav for task in self.inflight], dtype=int),
                          minlength=self.M).astype(np.float64) / self.q_max
        # 6. predicted rates: interferers = uploads in progress (carried over from t-1), or
        #    none in 'slot_assigned' mode (nothing assigned yet in slot t)
        R_all = self._rates(self.h_now, self._prediction_set())
        qdev = np.array([k.device for k in self.queue], dtype=int)
        rates_pred = R_all[qdev] if len(qdev) else np.zeros((0, self.M))
        h_q = self.h_now[qdev] if len(qdev) else np.zeros((0, self.M))
        # 7. S_sys
        S = self._build_s_sys(t, rates_pred, za)
        self.obs_queue = list(self.queue)
        return SlotObs(t=self.t, S_sys=S, queue=self.obs_queue, rates_pred=rates_pred, h_now=h_q,
                       zone_arrivals=za.copy(), zone_mean=self.zone_mean(), residual_f=self.residual(),
                       cap_bound=self.cap_bound(), energy=self.energy.copy(), e_min=self.e_min.copy(),
                       burst=self.burst.copy(), lam_eff=self.lam_eff.copy(), mu=self.mu(),
                       f_max=w.f_max.copy(), eta=w.eta.copy(), h_min=float(self.h_min),
                       rate_fn=self._make_rate_fn(), device_busy_s=self._device_busy_s(R_all),
                       uav_occupancy=occ, rate_rows_fn=self._make_rate_rows_fn(),
                       fdma_scan_fn=self._make_fdma_scan_fn())

    def _build_s_sys(self, t: int, rates_pred: np.ndarray, za: np.ndarray) -> np.ndarray:
        """S_sys(t): exactly 6M + 2Z + 2 + M = 49 dims (no time/phase features)."""
        w, M, Z = self.world, self.M, self.Z
        cap = self.cap_bound()
        backlog = np.zeros(M)
        for task in self.inflight:
            backlog[task.uav] += task.cycles_left
        per_uav = np.column_stack([self.energy / w.battery_j, w.eta, self.reserved() / cap,
                                   backlog / w.f_max, self.burst.astype(float), self.burst_budget])
        zm = self.zone_mean()
        arr_ratio = np.divide(za, zm, out=np.zeros(Z), where=zm > 0)
        qz = np.bincount(np.array([k.zone for k in self.queue], dtype=int), minlength=Z) / self.q_max
        per_zone = np.column_stack([arr_ratio, qz])
        if self.queue:
            slack = np.mean([(k.deadline_s - (t - k.t_arr) * self.dt) / k.deadline_s for k in self.queue])
            link = rates_pred.mean(axis=0) / self.r_ref
        else:
            slack, link = 0.0, np.zeros(M)
        glob = np.array([len(self.queue) / self.q_max, slack])
        return np.concatenate([per_uav.ravel(), per_zone.ravel(), glob, link]).astype(np.float64)

    def end_slot(self, decision, mem_energy_j: float = 0.0) -> SlotOutcome:
        if self.t is None:
            raise RuntimeError("end_slot called without begin_slot")
        cfg, w, t, dt = self.cfg, self.world, self.t, self.dt
        t_end = t + 1.0
        obsq = self.obs_queue
        # ---------------- apply assignments
        res = self.residual()
        cap = self.cap_bound()
        comp = np.zeros(self.M)
        n_assign = np.zeros(self.M)
        cyc_assign = np.zeros(self.M)
        taken = set()
        for a in decision.assignments:
            idx, j, f = int(a.task_index), int(a.uav), float(a.f)
            assert 0 <= idx < len(obsq), f"task_index {idx} out of range"
            assert idx not in taken, f"task_index {idx} assigned twice"
            assert 0 <= j < self.M, f"bad uav {j}"
            assert f > 0.0 and math.isfinite(f), f"allocation must be positive and finite, got {f}"
            assert f <= res[j] * (1.0 + 1e-12) + 1e-9 * cap[j], \
                f"f={f:.6g} exceeds residual {res[j]:.6g} on UAV {j} (Eq. 37)"
            taken.add(idx)
            res[j] = max(res[j] - f, 0.0)
            task = obsq[idx]
            task.uav, task.f_alloc, task.t_assign, task.attempts = j, f, t, 0
            task.bits_left = task.size_bits * (1.0 + cfg.task.tx_overhead * task.complexity)  # Eq. 14
            task.cycles_left = task.eff_cycles                                              # c phi, Eq. 15
            comp[j] += comp_energy(task.cycles, f, task.complexity, cfg)                    # Eq. 18 at assignment
            n_assign[j] += 1
            cyc_assign[j] += task.eff_cycles
            self.inflight.append(task)
        if taken:
            ids = {obsq[i].tid for i in taken}
            self.queue = [k for k in self.queue if k.tid not in ids]

        # ---------------- uploads: slot-wise integration
        slot_assigned = self.interference_set == "slot_assigned"
        if slot_assigned:
            # literal x_ij(t): interferers = uploads assigned in slot t; each new task's rate is
            # fixed at the value after all slot-t assignments and used for its whole T_trans
            pairs = [(a.task_index, a.uav) for a in decision.assignments]
            R_new = self._rates(self.h_now, self.active_from_assignments(pairs))
            for idx in taken:
                k = obsq[idx]
                self.fixed_rate[k.tid] = float(R_new[k.device, k.uav])
            R = None
        else:
            R = self._rates(self.h_now, self._active())      # interferers: uploads in progress
        t_mem_slots = self.t_mem_s / dt
        tx_time = np.zeros(self.N)
        upload_failed = []
        by_dev: dict = {}
        for task in self.inflight:
            if task.bits_left > 0.0:
                by_dev.setdefault(task.device, []).append(task)
        for dev, lst in by_dev.items():
            tau = float(t)
            for task in lst:
                if self.retry_slot.get(task.tid, t) > t:
                    break
                st = max(tau, t + t_mem_slots) if task.t_assign == t else tau
                if st >= t_end:
                    break
                r = self.fixed_rate[task.tid] if slot_assigned else R[dev, task.uav]
                if not r > 0.0:
                    break
                need = task.bits_left / r / dt
                if st + need <= t_end:
                    fin = st + need
                    tx_time[dev] += need * dt
                    u = task.loss_u[task.attempts]
                    task.attempts += 1
                    if self.p_loss > 0.0 and u < self.p_loss:          # attempt lost
                        task.bits_left = task.size_bits * (1.0 + cfg.task.tx_overhead * task.complexity)
                        if task.attempts >= self.max_tx:
                            upload_failed.append(task)
                        else:
                            self.retry_slot[task.tid] = t + 1
                        break
                    task.bits_left = 0.0
                    task.t_upload_done = fin
                    self.retry_slot.pop(task.tid, None)
                    self.fixed_rate.pop(task.tid, None)
                    tau = fin
                else:
                    tx_time[dev] += (t_end - st) * dt
                    task.bits_left -= r * (t_end - st) * dt
                    break
        if upload_failed:
            ids = {k.tid for k in upload_failed}
            self.inflight = [k for k in self.inflight if k.tid not in ids]
            for k in upload_failed:
                self.retry_slot.pop(k.tid, None)
                self.fixed_rate.pop(k.tid, None)

        # ---------------- computation at f; reservation released at completion
        completed, late, done_ids = [], [], set()
        held = np.zeros(self.M)            # integral of reserved f over the slot (cycles)
        dl_energy, proc_times = 0.0, []
        for task in self.inflight:
            end_hold = t_end
            if task.bits_left <= 0.0 and task.cycles_left > 0.0:
                st = max(float(t), task.t_upload_done)
                need = task.cycles_left / task.f_alloc / dt
                if st + need <= t_end:
                    task.t_done = st + need
                    task.cycles_left = 0.0
                    end_hold = task.t_done
                    done_ids.add(task.tid)
                    proc_times.append(task.eff_cycles / task.f_alloc)
                    rdl = downlink_rate(self.h_now[task.device, task.uav], w.eta[task.uav], cfg)
                    dl_energy += float(downlink_energy(task.size_bits, rdl, cfg))
                    ok = (task.t_done - task.t_arr) * dt <= task.deadline_s     # Eq. 39 realized
                    (completed if ok else late).append(task)
                else:
                    task.cycles_left -= task.f_alloc * (t_end - st) * dt
            held[task.uav] += task.f_alloc * (end_hold - t)
        if done_ids:
            self.inflight = [k for k in self.inflight if k.tid not in done_ids]
        for k in upload_failed:            # held from assignment/slot start to slot end
            held[k.uav] += k.f_alloc

        # ---------------- energy, Eq. 17 (burst draw while B_j = 1 in this slot)
        burst_e = cfg.burst.delta_e_w * dt * self.burst.astype(float)
        self.energy = step_energy(self.energy, comp, mem_energy_j, self.burst, cfg)

        nominal = w.f_max * w.eta
        cpu_util = held * dt / (nominal * dt)
        out = SlotOutcome(t=t, completed=completed, failed=self.pending_failed + upload_failed, late=late,
                          comp_energy_j=comp, mem_energy_j=float(mem_energy_j), burst_energy_j=burst_e,
                          uplink_energy_j=float(np.sum(w.device_ptx * tx_time)),
                          downlink_energy_j=dl_energy,
                          hover_energy_j=float(self.M * self.fly_w * dt),
                          cpu_util=cpu_util, bw_util=float(tx_time.sum() / (self.N * dt)),
                          arrivals=len(self.last_arrivals))
        self.last_info = {"n_expired": self._n_expired, "n_overflow": self._n_overflow,
                          "n_aborted": self._n_aborted,
                          "n_upload_failed": len(upload_failed), "proc_times_s": proc_times,
                          "burst_share": np.maximum(cpu_util - 1.0, 0.0),
                          "reserved_end": self.reserved()}
        out.info = self.last_info

        # ---------------- burst for t+1 (Alg. 3 line 15) and its budget (S_sys)
        bn = decision.burst_next
        self.burst = np.zeros(self.M, dtype=bool) if bn is None else np.asarray(bn, dtype=bool).copy()
        bb = decision.burst_budget
        self.burst_budget = np.zeros(self.M) if bb is None else np.asarray(bb, dtype=float).copy()

        # ---------------- EWMs for Eq. 16 (lambda_eff, mu)
        b = self.ewm_beta
        self.lam_eff = b * self.lam_eff + (1.0 - b) * n_assign / dt
        has = n_assign > 0
        self.ewm_cycles = np.where(has, b * self.ewm_cycles + (1.0 - b) * cyc_assign / np.maximum(n_assign, 1),
                                   self.ewm_cycles)

        self.pending_failed = []
        self.next_t = t + 1
        self.t = None
        return out

    # ------------------------------------------------------------------ episode control
    def flush(self) -> dict:
        """Episode boundary: clear queue and in-flight tasks. They are 'flushed',
        neither completed nor failed. Burst state and budget are reset; the
        lambda_eff / mu and zone EWMs are kept."""
        out = {"flushed_inflight": list(self.inflight), "flushed_queue": list(self.queue)}
        self.inflight, self.queue, self.obs_queue = [], [], []
        self.retry_slot, self.fixed_rate = {}, {}
        self.burst = np.zeros(self.M, dtype=bool)
        self.burst_budget = np.zeros(self.M)
        return out

    def reset_battery(self) -> None:
        self.energy = self.world.battery_j.copy()

    # ------------------------------------------------------------------ checkpoint
    _STATE_KEYS = ("device_pos", "uav_pos", "energy", "burst", "burst_budget", "queue", "inflight",
                   "retry_slot", "fixed_rate", "lam_eff", "ewm_cycles", "zone_m", "zone_n", "t", "next_t",
                   "pending_failed", "obs_queue", "h_now", "xi", "last_arrivals", "last_zone_arrivals")

    def get_state(self) -> dict:
        s = {k: getattr(self, k) for k in self._STATE_KEYS}
        s["_n_expired"] = getattr(self, "_n_expired", 0)
        s["_n_overflow"] = getattr(self, "_n_overflow", 0)
        s["_n_aborted"] = getattr(self, "_n_aborted", 0)
        s = copy.deepcopy(s)        # one call: shared Task references stay shared
        s["workload"] = copy.deepcopy(self.workload.get_state())
        s["gm"] = self.gm.get_state() if self.gm is not None else None
        s["streams"] = {n: copy.deepcopy(getattr(self.streams, n).bit_generator.state) for n in _ENV_STREAMS}
        return s

    def set_state(self, state: dict) -> None:
        s = copy.deepcopy(state)
        for k in self._STATE_KEYS:
            setattr(self, k, s[k])
        self._n_expired, self._n_overflow, self._n_aborted = s["_n_expired"], s["_n_overflow"], s["_n_aborted"]
        self.workload.set_state(s["workload"])
        if self.gm is not None:
            self.gm.set_state(s["gm"])
        for n, st in s["streams"].items():
            getattr(self.streams, n).bit_generator.state = st
