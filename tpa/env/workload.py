"""Workload model, Eqs. (10)-(12), plus the evaluation trace families.

Eq. (11): Lambda~_i(t) = Lambda_i(t) psi_{z(i)}(t) omega_{z(i)}, with
psi_z(t) = 1 + A sin(2 pi t / P + varphi_z) -- ONE phase per zone, and
z(i) the Voronoi zone of the device's current position.

CRN / stream use:
* Poisson counts are drawn for ALL N devices every slot from `rng_arrivals`.
* Task attributes are drawn for exactly the drawn counts (a (K,5) uniform
  block, then a (K, max_tx_attempts) block of loss uniforms). Counts never
  depend on the policy, so the stream position never does either.
* The modulating processes (burst onsets, MMPP states, Pareto on/off) use a
  private child generator seeded once at construction from `rng_arrivals`
  (one integer, for every kind), so they never interleave with the Poisson
  draws.

Base-rate normalisation (cfg.workload.load_basis):
* "compute": lambda = load * lambda_cpu,
  lambda_cpu = sum_j F_max^j eta_j / E[c phi]  (tasks/s), E[c phi] = E[c] E[phi].
* "binding": lambda = load * min(lambda_cpu, lambda_radio),
  lambda_radio = sum_j eta_j B log2(1 + SNRbar_j) / E[s (1 + varrho phi)],
  SNRbar_j = median over devices whose best UAV is j of P_i h_ij / (N0 B), with
  h from Eq. 4 at xi = 1 (path loss incl. kappa and Psi; no fading, no
  interference), from the world's initial geometry, once per world. A UAV
  that is no device's best UAV contributes 0.
The mean zone weight over the devices' initial zones is divided out.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from ..types import Task
from .world import zone_of
from . import channel as _ch

MB_TO_BITS = 8.0e6     # 1 MB = 1e6 bytes (unit conversion)
MEGA = 1.0e6           # Mcycles -> cycles
KINDS = ("sinusoidal", "shifted", "burst", "mmpp", "pareto", "trace")
PERIODIC = ("sinusoidal", "shifted", "burst")
MILAN_GRID = 100       # Milan dataset is a 100 x 100 cell grid (dataset property)


def mean_offered_bits(cfg) -> float:
    """E[s (1 + varrho phi)] per task (bits), Eq. 14 numerator, over the task ranges of the configuration."""
    tk = cfg.task
    return float(np.mean(tk.size_mb)) * MB_TO_BITS * (1.0 + tk.tx_overhead * float(np.mean(tk.complexity)))


def lambda_cpu(world, cfg) -> float:
    """Compute-side task rate at utilisation 1: sum_j F_max eta / E[c phi]  (tasks/s)."""
    return float(np.sum(world.f_max * world.eta)) / mean_offered_cycles(cfg)


def lambda_radio(world, cfg) -> float:
    """Radio-side task rate at utilisation 1:
    sum_j eta_j B log2(1 + SNRbar_j) / E[s(1+varrho phi)]  (tasks/s)."""
    c = cfg.channel
    n, m = world.kappa.shape
    h = _ch.channel_gain(world.device_pos, world.uav_pos, world, np.ones((n, m)), cfg)
    best = np.argmax(h, axis=1)
    snr = world.device_ptx[:, None] * h / (c.noise_psd_w_hz * c.bandwidth_hz)
    cap = 0.0
    for j in range(m):
        sel = best == j
        if sel.any():
            cap += world.eta[j] * c.bandwidth_hz * math.log2(1.0 + float(np.median(snr[sel, j])))
    return cap / mean_offered_bits(cfg)


def mean_offered_cycles(cfg) -> float:
    """E[c phi] per task (cycles), from the task ranges of the configuration."""
    tk = cfg.task
    return float(np.mean(tk.mcycles)) * MEGA * float(np.mean(tk.complexity))


class Workload:
    def __init__(self, cfg, world, rng_arrivals: np.random.Generator, kind: str = "sinusoidal",
                 load: float = 0.55, t0: int | None = None, horizon: int | None = None):
        if kind not in KINDS:
            raise ValueError(f"unknown workload kind {kind!r}; expected one of {KINDS}")
        self.cfg, self.world, self.rng, self.kind, self.load = cfg, world, rng_arrivals, kind, float(load)
        wl = cfg.workload
        self.A = float(wl.amplitude)
        self.P = int(wl.period_slots)
        self.dt = float(cfg.scenario.slot_s)
        self.n = world.n_devices
        self.Z = world.n_zones
        self.t0 = t0                      # first slot of the run (set on first step if None)
        self.horizon = int(horizon if horizon is not None else cfg.run.test_slots)
        self.max_tx = int(cfg.task.max_tx_attempts)
        self.next_tid = 0
        # private generator for the modulating processes (one draw, every kind)
        self.aux = np.random.Generator(np.random.PCG64(int(rng_arrivals.integers(0, 2**63 - 1))))

        self.load_basis = str(wl.load_basis)
        self.lambda_cpu = lambda_cpu(world, cfg)
        self.lambda_radio = lambda_radio(world, cfg)
        self.binding = "cpu" if self.lambda_cpu <= self.lambda_radio else "radio"
        if self.load_basis == "compute":
            task_rate = self.load * self.lambda_cpu                       # tasks/s
        elif self.load_basis == "binding":
            task_rate = self.load * min(self.lambda_cpu, self.lambda_radio)
        else:
            raise ValueError(f"workload.load_basis must be 'compute' or 'binding', got {self.load_basis!r}")
        self.task_rate = task_rate
        self.base0 = task_rate * self.dt / self.n                         # tasks/slot/device
        z0 = zone_of(world.device_pos, world)
        self.mean_omega = float(np.mean(world.zone_weights[z0]))
        self.base = self.base0 / self.mean_omega                          # Lambda_i, Eq. 11

        # burst family
        self.burst_len = int(round(float(wl.burst_frac) * self.P))
        self._burst_cycle = None
        self._burst_onset = 0
        # MMPP (2-state per zone)
        p_stay = np.asarray(wl.mmpp_p_stay, dtype=float)
        self.mmpp_mult = np.asarray(wl.mmpp_mult, dtype=float)
        self.mmpp_p_stay = p_stay
        leave = 1.0 - p_stay
        self.mmpp_pi = np.array([leave[1], leave[0]]) / leave.sum()       # stationary dist.
        self.mmpp_norm = float(self.mmpp_pi @ self.mmpp_mult)
        self.mmpp_state = (self.aux.random(self.Z) < self.mmpp_pi[1]).astype(int)
        # Pareto on/off (Type-I Pareto, mean pareto_mean_s => x_m = mean (a-1)/a)
        a = float(wl.pareto_shape)
        self.par_a = a
        self.par_xm = float(wl.pareto_mean_s) * (a - 1.0) / a
        self.par_on_mult = float(wl.pareto_on_mult)
        self.par_norm = self.par_on_mult * 0.5    # equal ON/OFF means -> duty 1/2
        self.par_on = self.aux.random(self.Z) < 0.5
        self.par_left = self._pareto(self.Z)
        self._advanced_t = None
        # trace hook
        self.trace_rates = None
        if kind == "trace":
            self.trace_rates = load_trace_zones(cfg)                      # (T, Z), mean-1

    # ------------------------------------------------------------ helpers
    def _pareto(self, k: int) -> np.ndarray:
        return self.par_xm * (1.0 + self.aux.pareto(self.par_a, size=k))   # seconds

    def phase_of(self, t: int) -> float:
        """Cycle phase 2 pi (t mod P)/P of the sinusoid (harness truth)."""
        return 2.0 * math.pi * ((t % self.P) / self.P)

    def truth_bin(self, t: int) -> int:
        """floor(phase / (2 pi / hit_bins)) for periodic kinds, -1 otherwise."""
        if self.kind not in PERIODIC:
            return -1
        bins = int(self.cfg.memory.hit_bins)
        return min(int((t % self.P) * bins // self.P), bins - 1)

    def zone_psi(self, t: int) -> np.ndarray:
        """psi_z(t) = 1 + A sin(2 pi t/P + varphi_z) (+ shift on the 2nd half for 'shifted')."""
        ph = self.world.zone_phase
        if self.kind == "shifted" and self.t0 is not None and (t - self.t0) >= self.horizon // 2:
            ph = ph + float(self.cfg.workload.shift_rad)
        return 1.0 + self.A * np.sin(2.0 * math.pi * t / self.P + ph)

    def _advance(self, t: int) -> None:
        """Advance the modulating processes exactly once per slot (policy-independent)."""
        if self._advanced_t == t:
            return
        if self.t0 is None:
            self.t0 = int(t)
        self._advanced_t = t
        if self.kind == "burst":
            cyc = (t - self.t0) // self.P
            if cyc != self._burst_cycle:
                self._burst_cycle = cyc
                self._burst_onset = int(self.aux.integers(0, self.P - self.burst_len + 1))
        elif self.kind == "mmpp":
            u = self.aux.random(self.Z)
            stay = self.mmpp_p_stay[self.mmpp_state]
            self.mmpp_state = np.where(u < stay, self.mmpp_state, 1 - self.mmpp_state)
        elif self.kind == "pareto":
            self.par_left = self.par_left - self.dt
            while np.any(self.par_left <= 0.0):
                idx = np.nonzero(self.par_left <= 0.0)[0]
                self.par_on[idx] = ~self.par_on[idx]
                self.par_left[idx] = self.par_left[idx] + self._pareto(len(idx))

    def in_burst(self, t: int) -> bool:
        if self.kind != "burst" or self._burst_cycle is None:
            return False
        k = (t - self.t0) % self.P
        return self._burst_onset <= k < self._burst_onset + self.burst_len

    # ------------------------------------------------------------ Eq. 10-11
    def rates(self, t: int, device_pos: np.ndarray) -> np.ndarray:
        """Eq. (11): slot-integrated intensity Lambda~_i(t) (expected tasks this slot), (N,).
        Zone from the current position (Voronoi); one phase per zone."""
        if self.t0 is None:
            self.t0 = int(t)
        z = zone_of(device_pos, self.world)
        omega = self.world.zone_weights[z]
        k = self.kind
        if k in PERIODIC:
            lam = self.base * self.zone_psi(t)[z] * omega
            if self.in_burst(t):
                lam = lam * float(self.cfg.workload.burst_mult)
            return lam
        if k == "mmpp":
            return self.base * self.mmpp_mult[self.mmpp_state][z] * omega / self.mmpp_norm
        if k == "pareto":
            mult = np.where(self.par_on, self.par_on_mult, 0.0)
            return self.base * mult[z] * omega / self.par_norm
        # trace: zone demand supplies shape and spatial weights (not omega_z on top)
        tr = self.trace_rates
        return self.base0 * tr[t % tr.shape[0]][z]

    def step(self, t: int, device_pos: np.ndarray) -> list:
        """Eq. (10): Poisson counts for ALL devices, then Eq. (12) task profiles."""
        if self.t0 is None:
            self.t0 = int(t)
        self._advance(t)
        lam = np.maximum(self.rates(t, device_pos), 0.0)
        counts = self.rng.poisson(lam)                       # N draws every slot
        K = int(counts.sum())
        if K == 0:
            return []
        tk = self.cfg.task
        u = self.rng.random((K, 5))
        loss = self.rng.random((K, self.max_tx))
        dev = np.repeat(np.arange(self.n), counts)
        zones = zone_of(device_pos[dev], self.world)
        lo = np.array([tk.size_mb[0], tk.mcycles[0], tk.deadline_s[0], tk.priority[0], tk.complexity[0]])
        hi = np.array([tk.size_mb[1], tk.mcycles[1], tk.deadline_s[1], tk.priority[1], tk.complexity[1]])
        v = lo + (hi - lo) * u
        tb = self.truth_bin(t)
        out = []
        for k in range(K):
            out.append(Task(tid=self.next_tid, device=int(dev[k]), zone=int(zones[k]), t_arr=int(t),
                            size_bits=float(v[k, 0] * MB_TO_BITS), cycles=float(v[k, 1] * MEGA),
                            deadline_s=float(v[k, 2]), priority=float(v[k, 3]),
                            complexity=float(v[k, 4]), loss_u=loss[k].copy(), truth_bin=tb))
            self.next_tid += 1
        return out

    # ------------------------------------------------------------ measurement helper
    def measured_zone_amplitude(self, T: int | None = None, device_pos: np.ndarray | None = None,
                                t_start: int | None = None) -> np.ndarray:
        """Relative amplitude of the realized per-zone arrival counts at period P:
        2 |sum_t (c_z(t) - mean) e^{-i 2 pi t/P}| / (T mean_z), per zone (Z,).
        Runs on a deep copy, so this workload's streams are NOT consumed."""
        w = copy.deepcopy(self)
        T = int(T if T is not None else 3 * self.P)
        pos = self.world.device_pos if device_pos is None else device_pos
        t0 = int(t_start if t_start is not None else (self.t0 if self.t0 is not None else 0))
        c = np.zeros((T, self.Z))
        for k in range(T):
            for task in w.step(t0 + k, pos):
                c[k, task.zone] += 1.0
        ts = np.arange(t0, t0 + T)
        mean = c.mean(axis=0)
        e = np.exp(-2j * math.pi * ts / self.P)
        amp = 2.0 * np.abs(((c - mean) * e[:, None]).sum(axis=0)) / (T * np.maximum(mean, 1e-300))
        return np.where(mean > 0, amp, 0.0)

    # ------------------------------------------------------------ checkpoint
    def get_state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "aux": self.aux.bit_generator.state,
                "t0": self.t0, "next_tid": self.next_tid, "burst_cycle": self._burst_cycle,
                "burst_onset": self._burst_onset, "mmpp_state": self.mmpp_state.copy(),
                "par_on": self.par_on.copy(), "par_left": self.par_left.copy(),
                "advanced_t": self._advanced_t}

    def set_state(self, s: dict) -> None:
        self.rng.bit_generator.state = s["rng"]
        self.aux.bit_generator.state = s["aux"]
        self.t0, self.next_tid = s["t0"], s["next_tid"]
        self._burst_cycle, self._burst_onset = s["burst_cycle"], s["burst_onset"]
        self.mmpp_state = s["mmpp_state"].copy()
        self.par_on, self.par_left = s["par_on"].copy(), s["par_left"].copy()
        self._advanced_t = s["advanced_t"]


# ---------------------------------------------------------------- trace hook
def load_trace_zones(cfg) -> np.ndarray:
    """Public-trace hook. `cfg.workload.trace_path` may point to
    * a .npy / .csv matrix (T_slots, Z) of per-zone demand, or
    * the Milan Telecom grid CSV (columns cell,time,internet).
    Returned series are mean-1 per zone. No data is shipped."""
    path = str(cfg.workload.trace_path)
    if not path:
        raise ValueError("workload kind 'trace' needs cfg.workload.trace_path "
                         "(a (T,Z) .npy/.csv matrix or the Milan cell,time,internet CSV); it is empty")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trace file not found: {p}")
    Z = int(cfg.scenario.n_zones)
    if p.suffix == ".npy":
        arr = np.load(p).astype(float)
    else:
        import pandas as pd
        df = pd.read_csv(p)
        if {"cell", "time", "internet"} <= set(df.columns):
            arr = _milan_counts(df, Z, float(cfg.scenario.slot_s))
        else:
            arr = df.to_numpy(dtype=float)
    if arr.ndim != 2 or arr.shape[1] != Z:
        raise ValueError(f"trace matrix must be (T, {Z}); got {arr.shape}")
    m = arr.mean(axis=0, keepdims=True)
    return arr / np.where(m > 0, m, 1.0)


def _milan_counts(df, Z: int, slot_s: float) -> np.ndarray:
    import pandas as pd
    t = df["time"]
    if np.issubdtype(t.dtype, np.number):
        ts = t.to_numpy(dtype=float)
        ts = ts / 1e3 if ts.mean() > 1e12 else ts            # ms vs s
    else:
        ts = pd.to_datetime(t).astype("int64").to_numpy() / 1e9
    bin_s = int(np.median(np.diff(np.unique(ts))))
    tb = ((ts - ts.min()) / bin_s).astype(int)
    cell = df["cell"].astype(int).to_numpy() - 1
    row, col = cell // MILAN_GRID, cell % MILAN_GRID
    side = int(math.ceil(math.sqrt(Z)))
    zr = np.minimum(row * side // MILAN_GRID, side - 1)
    zc = np.minimum(col * side // MILAN_GRID, side - 1)
    zone = np.minimum(zr * side + zc, Z - 1)
    counts = np.zeros((tb.max() + 1, Z))
    np.add.at(counts, (tb, zone), df["internet"].to_numpy(float))
    return np.repeat(counts, max(int(bin_s / slot_s), 1), axis=0)
