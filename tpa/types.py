"""Data types exchanged between modules. Units are SI
unless the name says otherwise: bits, cycles, seconds, cycles/s, J, W, Wh."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Task:
    tid: int                 # unique within a run
    device: int
    zone: int                # zone of the device at arrival (Eq. 11)
    t_arr: int               # arrival slot
    size_bits: float         # s_i
    cycles: float            # c_i
    deadline_s: float        # T_i^max, measured from arrival
    priority: float          # rho_i
    complexity: float        # phi_i
    loss_u: np.ndarray       # pre-drawn U(0,1) per uplink attempt (CRN; packet loss)
    truth_bin: int           # generator phase bin at arrival (harness only — never shown to a controller)
    # runtime fields (set by the simulator)
    attempts: int = 0
    uav: int = -1            # assigned UAV, -1 while queued
    f_alloc: float = 0.0     # reserved cycles/s
    t_assign: int = -1
    bits_left: float = 0.0
    cycles_left: float = 0.0
    t_upload_done: float = -1.0   # continuous slot-time
    t_done: float = -1.0

    @property
    def eff_cycles(self) -> float:
        return self.cycles * self.complexity


@dataclass
class SlotObs:
    """What the simulator exposes to a controller at the start of slot t
    (after arrivals, mobility and fading). No phase, clock or generator truth."""
    t: int
    S_sys: np.ndarray        # (49,) system state, 6M + 2Z + 2 + M entries
    queue: list              # list[Task], all queued (unassigned) tasks
    rates_pred: np.ndarray   # (len(queue), M) predicted rate bit/s (prev-slot interferers; none if slot_assigned)
    h_now: np.ndarray        # (len(queue), M) channel gain this slot (for Eq. 2 viability)
    zone_arrivals: np.ndarray  # (Z,) arrivals this slot
    zone_mean: np.ndarray    # (Z,) online long-run mean arrivals per zone (Simulator.zone_mean)
    residual_f: np.ndarray   # (M,) residual capacity bound now (cycles/s, incl. burst)
    cap_bound: np.ndarray    # (M,) current capacity bound F_max*eta*(1+rho_b*B_j)
    energy: np.ndarray       # (M,) E_j (J)
    e_min: np.ndarray        # (M,) E_min (J)
    burst: np.ndarray        # (M,) B_j currently active
    lam_eff: np.ndarray      # (M,) EWM effective arrival rate (tasks/s) per UAV
    mu: np.ndarray           # (M,) EWM service rate (tasks/s) per UAV
    f_max: np.ndarray | None = None   # (M,) nominal F_max (cycles/s); simulator always fills it
    eta: np.ndarray | None = None     # (M,) operational efficiency (time-invariant)
    h_min: float | None = None        # Eq. 2 threshold in force (channel.h_min_rule)
    device_busy_s: dict | None = None # device -> seconds of upload already committed on its single
                                      # radio at slot start (serial radio per device)
    rate_fn: object | None = None     # slot_assigned mode: rate_fn(pairs) -> (len(queue), M)
                                      # predicted bit/s given [(task_index, uav), ...] already
                                      # assigned in THIS slot (incremental interference)
    uav_occupancy: np.ndarray | None = None  # (M,) tasks currently assigned (in flight) to UAV j,
                                             # divided by q_max
    rate_rows_fn: object | None = None  # rate_rows_fn(pairs, qidx) -> RowRates with the rows of
                                        # queue indices `qidx` only (same values as rate_fn(pairs)[qidx])
    fdma_scan_fn: object | None = None  # fdma_scan_fn(pairs) -> channel.FdmaScan of that prefix
                                        # (None unless channel.access = "fdma_per_uav")


@dataclass
class Assignment:
    task_index: int          # index into SlotObs.queue
    uav: int
    f: float                 # cycles/s reserved
    t_total_pred: float      # predicted T_total (Eq. 13) used for J1


@dataclass
class Decision:
    assignments: list = field(default_factory=list)   # list[Assignment]
    burst_next: np.ndarray | None = None              # (M,) B_j for slot t+1 (Alg. 3 line 15)
    load_index: np.ndarray | None = None              # (M,) L_j after assignment (Eq. 56)
    burst_budget: np.ndarray | None = None            # (M,) remaining burst budget T_b/T_max (S_sys)
    util_terms: dict = field(default_factory=dict)    # per-slot quantities for J2


@dataclass
class SlotOutcome:
    t: int
    completed: list          # tasks finished by deadline this slot
    failed: list             # tasks expired/dropped this slot (deadline, overflow)
    late: list               # tasks finished after deadline (counted failed for completion)
    comp_energy_j: np.ndarray    # (M,) Eq. 18 energy charged this slot
    mem_energy_j: float          # kappa_m ||r||_1 charged per UAV this slot
    burst_energy_j: np.ndarray   # (M,) Delta_E_b * slot
    uplink_energy_j: float       # device side, sum P_i * time transmitting
    downlink_energy_j: float     # UAV side result transmission (log only)
    hover_energy_j: float        # log only
    cpu_util: np.ndarray         # (M,) reserved / nominal capacity
    bw_util: float               # fraction of devices' uplink time in use (log)
    arrivals: int
    info: dict = field(default_factory=dict)   # simulator diagnostics (sim.last_info)
