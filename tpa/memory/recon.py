"""Replay reconstruction of memory and retrieval.

SegmentStore keeps, per memory segment, the raw encoder inputs x, the query inputs,
zone arrivals / means, the per-UAV occupancy o_t (zone-resolved candidate), the rollout
encodings e (used for A3's older entries), the M_L event log (contents, slot keys, the
candidate's key k_t, P_ref / v_ref used) and the memory time read by retrieval at each
slot (A14 schedule).

reconstruct(params, cfg, flags, refs, grad) rebuilds, for every (store, t):
  M_S(t)  : the last K_S rows of x up to t re-encoded with the current W_S, Eq. 42.
  M_L(t)  : latest detached row_after per slot from the event log; the most recent
            event <= t is recomputed with the current W_z (candidate_source "pattern":
            P(t_e) recomputed from x with the current W_S; c_tau detached) or W_v
            ("zone": v(t_e) recomputed from the logged zone arrivals / zone mean window
            and o_{t_e}), and W_g -- one-step truncated gradient. In "soft" mode that
            event touches every filled slot: the log keeps per-slot before rows, keys
            and w, and the soft-addressed gate is recomputed (w from the keys and the
            current z; under key_centering from the keys and the LOGGED k_t, so w is
            then detached). Eq. 42 re-applied.
            Offset-free keys: with key_centering the attention keys of M_L are the slot
            keys read from the event log (latest event per slot, `flat_key`); mu is never
            recomputed and the keys carry no gradient.
  A3      : the flat FIFO of the last K_S + K_L entries; e entries of the last K_S
            slots re-encoded, older e entries and older z entries detached (rollout
            values), the most recent z recomputed (ungated FIFO write); Eq. 42. With
            key_centering the keys are e (logged rollout e) for e rows and the logged
            k_t for z rows.
  A14     : memory rebuilt at the stored snapshot slot s (mem_time[t]); query at t;
            r = 0 where no displaced memory existed.
  r(t)    : retrieval.attend with the current W_Q, W_K, W_V, W_omega.
  Arrival keys (memory.key_source = "arrivals"): the slot keys k_t = a_t - mu_{t-1} (R^Z) are
            recomputed from the logged zone arrivals / zone means by the same sequential
            code as the rollout (SegmentStore.arrival_keys_upto, a cache derived from
            _za / _zm, so rollout and replay see one value); the keys of the stored
            entries come from the event log (M_L, A3 z rows) or from that sequence at
            the entry's write slot (M_S, A3 e rows); the query at t is k_t of the LIVE t
            (also under A14); r = 0 where k_t is undefined (t - t0 < w - 1). No gradient
            flows through keys; W_Q / W_K receive none on this path.
  A15 (key-shuffled control, flags.key_shuffled): the store logs pi(t) per slot
            (log_pi / pi, -1 = none); replay takes the M_S keys of slot i and the A3
            e-row keys from write_keys_upto (= k_{pi(i)} from the log, never re-drawn),
            the M_L keys, the A3 z-row keys and the soft address from the event log
            (logged after the substitution at rollout), and the query from the TRUE k_t
            of the live slot.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch

from . import EPS, MemoryFlags, q_dim, x_dim
from .mepr import compute_P_batch, compute_v_batch
from .retrieval import attend
from .stores import (DELAY_SLOTS_DEFAULT, KEY_CENTERING_DEFAULT, KEY_MEAN_HORIZON_DEFAULT, KEY_SOURCE_DEFAULT,
                     KEY_TEMPERATURE_DEFAULT, SOFT_TAU_DEFAULT, key_dim, project_norm, soft_update)

_CK = 64      # event-table checkpoint interval (events)


class _Grow:
    """Growable numpy array along axis 0."""

    def __init__(self, shape_tail, dtype, fill=0, cap=1024):
        self.tail, self.dtype, self.fill = tuple(shape_tail), dtype, fill
        self.a = np.full((cap,) + self.tail, fill, dtype)

    def ensure(self, n):
        if n > self.a.shape[0]:
            new = np.full((max(n, 2 * self.a.shape[0]),) + self.tail, self.fill, self.dtype)
            new[: self.a.shape[0]] = self.a
            self.a = new


class SegmentStore:
    """Per-segment replay store. Slot rows are indexed by t - t0.

    The rollout loop (tpa/control.py) calls record(t, x, qin, zone_arr, zone_mean, uav_occ)
    every slot; an attached TwoLevelMemory logs e_t, events and mem_time itself."""

    def __init__(self, cfg, flags: MemoryFlags, capacity: int | None = None):
        mc = cfg.memory
        self.flags = flags
        self.d, self.KL = int(mc.d), int(mc.k_long)
        Z = int(cfg.scenario.n_zones)
        self.Z = Z
        self.M = int(cfg.scenario.n_uav)
        cap = int(capacity or max(int(cfg.run.segment_slots), int(cfg.run.test_slots)))
        # Stored key dimension (Z under key_source = "arrivals", d otherwise) and the
        # arrival-key sequence cache (derived from _za / _zm; see arrival_keys_upto)
        self.key_source = str(mc.get("key_source", KEY_SOURCE_DEFAULT))
        self.arrivals = self.key_source == "arrivals"
        self.shuffled = bool(getattr(flags, "key_shuffled", False))     # A15 (key-shuffled control)
        if self.shuffled and not self.arrivals:
            raise ValueError("A15 (key_shuffled) is defined only under memory.key_source = 'arrivals'")
        self.kd = key_dim(cfg)
        self.w = int(mc.window_w)
        if self.arrivals and self.w < 1:
            raise ValueError("key_source='arrivals' needs memory.window_w >= 1")
        self.alpha = 1.0 / int(mc.get("key_mean_horizon_slots", KEY_MEAN_HORIZON_DEFAULT))
        self._ak = _Grow((Z,), np.float32, 0, cap)
        self._ak_reset()
        self._pi = _Grow((), np.int64, -1, cap)       # A15: logged shuffle slot pi(t) (-1 = none)
        self.t0 = None
        self.n = 0
        self._x = _Grow((x_dim(cfg),), np.float32, 0, cap)
        self._q = _Grow((q_dim(cfg),), np.float32, 0, cap)
        self._za = _Grow((Z,), np.float64, 0, cap)
        self._zm = _Grow((Z,), np.float64, 0, cap)
        self._e = _Grow((self.d,), np.float32, 0, cap)
        self._uo = _Grow((self.M,), np.float32, 0, cap)  # uav_occupancy o_t (zone-resolved candidate)
        self._mt = _Grow((), np.int64, -2, cap)          # -2 = not logged (non-A14: t itself)
        self._dl = _Grow((), np.int64, -1, cap)
        ecap = 256
        self.n_ev = 0
        self._ev_t = _Grow((), np.int64, -1, ecap)
        self._ev_k = _Grow((), np.int64, -1, ecap)
        self._ev_mode = []
        self._ev_b = _Grow((self.d,), np.float32, 0, ecap)
        self._ev_a = _Grow((self.d,), np.float32, 0, ecap)
        self._ev_key = _Grow((self.kd,), np.float32, 0, ecap)  # slot key after the event
        # per touched slot (ragged; one row per event in the hard modes, every filled slot
        # in a "soft" event): slot index, content before / after, key, soft weight w
        self._ev_off = _Grow((), np.int64, 0, ecap + 1)   # flat offsets, n_ev + 1 entries
        self._ev_off.a[0] = 0
        self._fk = _Grow((), np.int64, -1, ecap)
        self._fb = _Grow((self.d,), np.float32, 0, ecap)
        self._fa = _Grow((self.d,), np.float32, 0, ecap)
        self._fkey = _Grow((self.kd,), np.float32, 0, ecap)
        self._fw = _Grow((), np.float32, 0, ecap)
        self.n_flat = 0
        self._ev_pref = _Grow((Z, self.d), np.float32, 0, ecap)   # P_ref used for z ("novelty")
        self._ev_vref = _Grow((Z + self.M,), np.float32, 0, ecap)  # v_ref used for z (zone-resolved candidate)
        self._ev_ckey = _Grow((self.kd,), np.float32, 0, ecap)    # the candidate's key k_t
        self._ev_maxcos = _Grow((), np.float32, -1, ecap)
        self._ev_src = _Grow((), np.int64, -1, ecap)              # A16: delay_src (slot of the key), -1 = immediate
        self.delayed = bool(getattr(flags, "delayed", False))      # A16 (delayed consolidation)
        self.delay = int(mc.get("delay_slots", DELAY_SLOTS_DEFAULT)) if self.delayed else 0
        self._table = np.full(self.KL, -1, np.int64)      # slot -> latest FLAT row index
        self._ck = [self._table.copy()]                   # table after _CK * i events

    # ------------------------------------------------- arrival keys (derived)
    def _ak_reset(self) -> None:
        self._ak_n = 0                        # rows [0, _ak_n) of _ak are computed
        self._ak_S = np.zeros(self.Z)         # mu numerator S (float64); mu = S / c, normalised EMA
        self._ak_c = 0.0                      # mu normaliser c (c_0 = 0)
        self._ak_nd = 0                       # defined a_t folded into mu so far

    def arrival_defined(self, n: int) -> np.ndarray:
        """(n,) bool: row i (slot t0 + i) has a defined arrival key iff i >= w - 1."""
        return np.arange(int(n)) >= self.w - 1

    def arrival_keys_upto(self, n: int) -> np.ndarray:
        """The arrival keys k_i (float32 (n, Z)) of rows [0, n) (n <= self.n), rows
        with undefined keys (i < w - 1) left at zero. Row i: a_i = mean over rows
        i-w+1..i of _za, elementwise / max(_zm[i], EPS) (float64); k_i = a_i - mu_{i-1}
        (k_i = a_i for the first defined row), cast to float32; then mu is advanced with
        a_i: S <- (1-alpha) S + alpha a_i, c <- (1-alpha) c + alpha, mu = S / c, alpha =
        1 / key_mean_horizon_slots. Sequential and cached: the rollout (key_for_slot)
        and reconstruction read the same rows, so they agree exactly."""
        n = int(n)
        if n > self.n:
            raise ValueError(f"arrival_keys_upto({n}): only {self.n} slots recorded")
        if n > self._ak_n:
            self._ak.ensure(n)
            za, zm, w, al = self._za.a, self._zm.a, self.w, self.alpha
            for i in range(self._ak_n, n):
                if i < w - 1:
                    continue
                a = za[i - w + 1:i + 1].mean(axis=0) / np.maximum(zm[i], EPS)
                k = a if self._ak_nd == 0 else a - self._ak_S / self._ak_c
                self._ak.a[i] = k.astype(np.float32)
                self._ak_S = (1.0 - al) * self._ak_S + al * a
                self._ak_c = (1.0 - al) * self._ak_c + al
                self._ak_nd += 1
            self._ak_n = n
        return self._ak.a[:n]

    def arrival_key(self, t: int):
        """k_t (float32 (Z,) copy) of a recorded slot t, or None while undefined."""
        if self.t0 is None:
            raise ValueError("arrival_key: nothing recorded")
        i = int(t) - self.t0
        if i < 0 or i >= self.n:
            raise ValueError(f"arrival_key: slot {t} not recorded")
        if i < self.w - 1:
            return None
        return self.arrival_keys_upto(i + 1)[i].copy()

    # -------------------------------------------------------- A15 shuffle log
    def pi_of(self, t: int) -> int:
        """Logged shuffle slot pi(t) of a recorded slot t (absolute slot), -1 if none."""
        if self.t0 is None:
            raise ValueError("pi_of: nothing recorded")
        i = int(t) - self.t0
        if i < 0 or i >= self.n:
            raise ValueError(f"pi_of: slot {t} not recorded")
        return int(self._pi.a[i])

    def log_pi(self, t: int, pi: int) -> None:
        """Log pi(t) (the memory draws it once per slot, write_key_for_slot). pi must be a
        recorded slot with a defined key, <= t."""
        i = self._row(t)
        j = int(pi) - self.t0
        if j < self.w - 1 or j > i:
            raise ValueError(f"log_pi: pi({t}) = {pi} is not an earlier defined slot")
        if self._pi.a[i] >= 0 and self._pi.a[i] != int(pi):
            raise ValueError(f"log_pi: pi({t}) already logged as {self._pi.a[i]}")
        self._pi.a[i] = int(pi)

    @property
    def pi(self) -> np.ndarray:
        """(n,) int64 logged pi(t) per slot (row i = slot t0 + i); -1 where none (undefined
        key, or not A15)."""
        return self._pi.a[: self.n]

    def write_keys_upto(self, n: int) -> np.ndarray:
        """The WRITE keys (float32 (n, Z)) of rows [0, n): the arrival keys themselves
        (arrival_keys_upto) unless key_shuffled, in which case row i carries k_{pi(i)} read
        from the log (rows whose pi is unlogged, e.g. undefined keys, are zero; a row with
        a defined key but no logged pi raises: replay never re-draws)."""
        seq = self.arrival_keys_upto(n)
        if not self.shuffled:
            return seq
        n = int(n)
        pi = self._pi.a[:n]
        defined = self.arrival_defined(n)
        if (defined & (pi < 0)).any():
            i = int(np.flatnonzero(defined & (pi < 0))[0])
            raise ValueError(f"write_keys_upto: slot {self.t0 + i} has a defined key but no logged pi")
        out = np.zeros_like(seq)
        out[defined] = seq[pi[defined] - self.t0]
        return out

    # ---------------------------------------------------------------- writing
    def _row(self, t: int) -> int:
        if self.t0 is None:
            self.t0 = int(t)
        i = int(t) - self.t0
        if i < 0:
            raise ValueError("slot before segment start")
        for g in (self._x, self._q, self._za, self._zm, self._e, self._uo, self._mt, self._dl,
                  self._pi):
            g.ensure(i + 1)
        self.n = max(self.n, i + 1)
        return i

    def record(self, t: int, x, qin, zone_arr, zone_mean, uav_occ=None) -> None:
        i = self._row(t)
        if i < self._ak_n:                    # a re-recorded row invalidates the key cache
            self._ak_reset()
        self._x.a[i] = np.asarray(x, np.float32)
        self._q.a[i] = np.asarray(qin, np.float32)
        self._za.a[i] = zone_arr
        self._zm.a[i] = zone_mean
        self._uo.a[i] = 0.0 if uav_occ is None else np.asarray(uav_occ, np.float32).reshape(self.M)

    def log_e(self, t: int, e) -> None:
        i = self._row(t)
        self._e.a[i] = e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e

    def log_mem_time(self, t: int, mem_time: int, delta: int) -> None:
        i = self._row(t)
        self._mt.a[i], self._dl.a[i] = mem_time, delta

    def add_event(self, ev: dict) -> None:
        if self.n_ev and ev["t"] < self._ev_t.a[self.n_ev - 1]:
            raise ValueError("events must be chronological")
        i = self.n_ev
        for g in (self._ev_t, self._ev_k, self._ev_b, self._ev_a, self._ev_key, self._ev_pref,
                  self._ev_vref, self._ev_ckey, self._ev_maxcos, self._ev_src):
            g.ensure(i + 1)
        self._ev_src.a[i] = int(ev.get("delay_src", -1))                 # A16
        self._ev_key.a[i] = ev.get("key", np.zeros(self.kd, np.float32))
        pr = ev.get("p_ref")
        self._ev_pref.a[i] = 0.0 if pr is None else pr
        vr = ev.get("v_ref")
        self._ev_vref.a[i] = 0.0 if vr is None else vr
        ck = ev.get("cand_key")
        self._ev_ckey.a[i] = self._ev_key.a[i] if ck is None else ck
        self._ev_maxcos.a[i] = ev.get("maxcos", -1.0)
        self._ev_t.a[i], self._ev_k.a[i] = ev["t"], ev["k"]
        self._ev_b.a[i], self._ev_a.a[i] = ev["row_before"], ev["row_after"]
        self._ev_mode.append(ev["mode"])
        ks = np.asarray(ev.get("ks", [ev["k"]]), np.int64)
        rb = np.asarray(ev.get("rows_before", ev["row_before"][None]), np.float32)
        ra = np.asarray(ev.get("rows_after", ev["row_after"][None]), np.float32)
        kk = np.asarray(ev.get("keys", ev.get("key", np.zeros(self.kd, np.float32))[None]), np.float32)
        w = np.asarray(ev.get("w", np.ones(1, np.float32)), np.float32)
        f0, f1 = self.n_flat, self.n_flat + ks.size
        for g in (self._fk, self._fb, self._fa, self._fkey, self._fw):
            g.ensure(f1)
        self._ev_off.ensure(i + 2)
        self._fk.a[f0:f1], self._fb.a[f0:f1], self._fa.a[f0:f1] = ks, rb, ra
        self._fkey.a[f0:f1], self._fw.a[f0:f1] = kk, w
        self._ev_off.a[i + 1] = f1
        self.n_flat = f1
        self.n_ev += 1
        if ev["mode"] != "fifo":
            self._table[ks] = np.arange(f0, f1)
        if self.n_ev % _CK == 0:
            self._ck.append(self._table.copy())

    # ---------------------------------------------------------------- reading
    @property
    def x(self):
        return self._x.a[: self.n]

    @property
    def qin(self):
        return self._q.a[: self.n]

    @property
    def zone_arr(self):
        return self._za.a[: self.n]

    @property
    def zone_mean(self):
        return self._zm.a[: self.n]

    @property
    def e(self):
        return self._e.a[: self.n]

    @property
    def uav_occ(self):
        return self._uo.a[: self.n]

    @property
    def mem_time(self):
        return self._mt.a[: self.n]

    @property
    def delta(self):
        return self._dl.a[: self.n]

    @property
    def events(self) -> dict:
        n = self.n_ev
        return dict(t=self._ev_t.a[:n], k=self._ev_k.a[:n], mode=list(self._ev_mode),
                    row_before=self._ev_b.a[:n], row_after=self._ev_a.a[:n], key=self._ev_key.a[:n],
                    maxcos=self._ev_maxcos.a[:n], p_ref=self._ev_pref.a[:n],
                    v_ref=self._ev_vref.a[:n], cand_key=self._ev_ckey.a[:n],
                    delay_src=self._ev_src.a[:n])

    def n_events_upto(self, t: int) -> int:
        return int(np.searchsorted(self._ev_t.a[: self.n_ev], t, side="right"))

    def table_at(self, n: int) -> np.ndarray:
        """Latest flat row index per M_L slot after the first n events (-1 = empty).
        Index into `flat_after` for the slot's content."""
        c = n // _CK
        tab = self._ck[c].copy()
        lo = c * _CK
        if n > lo:
            f0, f1 = int(self._ev_off.a[lo]), int(self._ev_off.a[n])
            np.maximum.at(tab, self._fk.a[f0:f1], np.arange(f0, f1))
        return tab

    @property
    def flat_after(self):
        return self._fa.a[: self.n_flat]

    @property
    def flat_key(self):
        """Slot key after each touched-slot row (index with table_at, like flat_after)."""
        return self._fkey.a[: self.n_flat]

    def event_rows(self, i: int):
        """(ks, rows_before, keys, w) of event i over its touched slots."""
        f0, f1 = int(self._ev_off.a[i]), int(self._ev_off.a[i + 1])
        return self._fk.a[f0:f1], self._fb.a[f0:f1], self._fkey.a[f0:f1], self._fw.a[f0:f1]

    def memory_time(self, t: int) -> int:
        """Slot whose memory retrieval read at slot t (A14); t itself otherwise."""
        m = int(self._mt.a[t - self.t0])
        return t if m == -2 else m

    # ------------------------------------------------------------- checkpoint
    def get_state(self) -> dict:
        n, ne = self.n, self.n_ev
        return dict(t0=self.t0, n=n, x=self.x.copy(), qin=self.qin.copy(),
                    zone_arr=self.zone_arr.copy(), zone_mean=self.zone_mean.copy(),
                    e=self.e.copy(), uav_occ=self.uav_occ.copy(),
                    mem_time=self.mem_time.copy(), delta=self.delta.copy(),
                    pi=self.pi.copy(),                                        # A15
                    n_ev=ne, ev_t=self._ev_t.a[:ne].copy(), ev_k=self._ev_k.a[:ne].copy(),
                    ev_mode=list(self._ev_mode), ev_b=self._ev_b.a[:ne].copy(),
                    ev_a=self._ev_a.a[:ne].copy(), ev_key=self._ev_key.a[:ne].copy(),
                    ev_pref=self._ev_pref.a[:ne].copy(), ev_maxcos=self._ev_maxcos.a[:ne].copy(),
                    ev_vref=self._ev_vref.a[:ne].copy(), ev_ckey=self._ev_ckey.a[:ne].copy(),
                    **({"ev_src": self._ev_src.a[:ne].copy()} if self.delayed else {}),   # A16 only
                    ev_off=self._ev_off.a[:ne + 1].copy(), fk=self._fk.a[:self.n_flat].copy(),
                    fb=self._fb.a[:self.n_flat].copy(), fa=self._fa.a[:self.n_flat].copy(),
                    fkey=self._fkey.a[:self.n_flat].copy(), fw=self._fw.a[:self.n_flat].copy())

    def set_state(self, s: dict) -> None:
        self.t0, self.n = s["t0"], int(s["n"])
        self._ak_reset()                      # derived cache: rebuilt from _za / _zm on demand
        for g, key in ((self._x, "x"), (self._q, "qin"), (self._za, "zone_arr"),
                       (self._zm, "zone_mean"), (self._e, "e"), (self._uo, "uav_occ"),
                       (self._mt, "mem_time"), (self._dl, "delta"), (self._pi, "pi")):
            g.ensure(self.n)
            g.a[:] = g.fill
            if key in s:
                g.a[: self.n] = s[key]
        self.n_ev, self.n_flat = 0, 0
        self._ev_mode = []
        self._ev_off.a[0] = 0
        self._table = np.full(self.KL, -1, np.int64)
        self._ck = [self._table.copy()]
        for i in range(int(s["n_ev"])):
            f0, f1 = int(s["ev_off"][i]), int(s["ev_off"][i + 1])
            self.add_event(dict(t=int(s["ev_t"][i]), k=int(s["ev_k"][i]), mode=s["ev_mode"][i],
                                row_before=s["ev_b"][i], row_after=s["ev_a"][i],
                                key=s["ev_key"][i], ks=s["fk"][f0:f1], rows_before=s["fb"][f0:f1],
                                rows_after=s["fa"][f0:f1], keys=s["fkey"][f0:f1], w=s["fw"][f0:f1],
                                p_ref=s["ev_pref"][i], maxcos=float(s["ev_maxcos"][i]),
                                v_ref=None if "ev_vref" not in s else s["ev_vref"][i],
                                cand_key=None if "ev_ckey" not in s else s["ev_ckey"][i],
                                delay_src=-1 if "ev_src" not in s else int(s["ev_src"][i])))


# ====================================================================== reconstruct
def _window_rows(store: SegmentStore, te: int, wl: int):
    """Indices (relative to t0) of the MEPR window ending at slot te (oldest first)."""
    lo = max(store.t0, te - wl + 1)
    return np.arange(lo - store.t0, te - store.t0 + 1)


def _gather(params, cfg, flags: MemoryFlags, refs):
    """Numpy-side gathering for a batch of refs. Returns a dict of arrays."""
    mc = cfg.memory
    d, KS, KL = int(mc.d), int(mc.k_short), int(mc.k_long)
    KF = KS + KL
    wl = int(mc.window_w) + 1 if flags.use_ms else 1
    B = len(refs)
    xd, qd, Z = x_dim(cfg), q_dim(cfg), int(cfg.scenario.n_zones)
    M_uav = int(cfg.scenario.n_uav)
    kc = bool(mc.get("key_centering", KEY_CENTERING_DEFAULT))
    arr = str(mc.get("key_source", KEY_SOURCE_DEFAULT)) == "arrivals"     # arrival keys
    kd = key_dim(cfg)
    w = int(mc.window_w)
    G = dict(B=B, valid=np.ones(B, bool), qin=np.zeros((B, qd), np.float32), key_centering=kc,
             arrivals=arr, key_tau=float(mc.get("key_temperature", KEY_TEMPERATURE_DEFAULT)))
    if arr:
        G["qkey"] = np.zeros((B, Z), np.float32)                  # q_t = k_t of the live slot
    # --- memory time per ref
    tms = np.zeros(B, np.int64)
    for b, (st, t) in enumerate(refs):
        tm = st.memory_time(t) if flags.displaced else int(t)
        tms[b] = tm
        G["qin"][b] = st.qin[t - st.t0]
        if tm < 0:
            G["valid"][b] = False
        if arr:
            if t - st.t0 < w - 1:
                G["valid"][b] = False                             # key undefined: r = 0
            else:
                G["qkey"][b] = st.arrival_keys_upto(t - st.t0 + 1)[t - st.t0]
    # --- last event per ref (needed for M_L and A3) and its P window
    need_ev = flags.use_ml or flags.flat
    if need_ev:
        G["has_ev"] = np.zeros(B, bool)
        G["X_pw"] = np.zeros((B, wl, xd), np.float32)
        G["A_pw"] = np.zeros((B, wl, Z), np.float32)
        G["zm"] = np.ones((B, Z), np.float32)
        G["m_pw"] = np.zeros((B, wl), bool)
        G["n_ev"] = np.zeros(B, np.int64)
        G["p_ref"] = np.zeros((B, Z, d), np.float32)
        G["v_ref"] = np.zeros((B, Z + M_uav), np.float32)        # zone-resolved candidate
        G["uo"] = np.zeros((B, M_uav), np.float32)               # o_{t_e}
        if getattr(flags, "delayed", False):                      # A16: next-Delta content window of the last event
            dw = max(wl, int(mc.get("delay_slots", DELAY_SLOTS_DEFAULT)))
            G["A_dw"] = np.zeros((B, dw, Z), np.float32)
            G["m_dw"] = np.zeros((B, dw), bool)
        G["ckey"] = np.zeros((B, kd), np.float32)                # k_{t_e} (logged key, soft address)
    if flags.flat:
        G["F_base"] = np.zeros((B, KF + KS + 1, d), np.float32)   # + unique dummy rows
        G["F_key"] = np.zeros((B, KF + KS + 1, kd), np.float32)   # slot keys (same layout)
        G["F_mask"] = np.zeros((B, KF), bool)
        G["pos_rec"] = np.tile(np.arange(KF, KF + KS), (B, 1))     # dummy by default
        G["pos_z"] = np.full(B, KF + KS, np.int64)
        G["X_st"] = np.zeros((B, KS, xd), np.float32)
        G["m_st"] = np.zeros((B, KS), bool)
    else:
        if flags.use_ms:
            G["X_st"] = np.zeros((B, KS, xd), np.float32)
            G["m_st"] = np.zeros((B, KS), bool)
            if arr:
                G["keys_S"] = np.zeros((B, KS, Z), np.float32)   # M_S slot keys (arrival keys)
                G["m_st_key"] = np.zeros((B, KS), bool)          # keyed rows (attended)
        if flags.use_ml:
            G["rows_L"] = np.zeros((B, KL, d), np.float32)
            G["keys_L"] = np.zeros((B, KL, kd), np.float32)       # slot keys (detached)
            G["m_L"] = np.zeros((B, KL), bool)
            # touched slots of the last event, padded to the batch max (1 in the hard modes)
            nmax = 1
            for b in range(B):
                st, tm = refs[b][0], int(tms[b])
                if tm >= 0:
                    n = st.n_events_upto(tm)
                    if n > 0:
                        nmax = max(nmax, int(st._ev_off.a[n] - st._ev_off.a[n - 1]))
            G["ks"] = np.zeros((B, nmax), np.int64)
            G["m_ks"] = np.zeros((B, nmax), bool)
            G["before"] = np.zeros((B, nmax, d), np.float32)
            G["keys"] = np.zeros((B, nmax, kd), np.float32)
    for b, (st, t) in enumerate(refs):
        tm = int(tms[b])
        if tm < 0:
            continue
        t0 = st.t0
        if "X_st" in G:
            lo = max(t0, tm - KS + 1)
            rows = np.arange(lo - t0, tm - t0 + 1)
            G["X_st"][b, : rows.size] = st.x[rows]
            G["m_st"][b, : rows.size] = True
            if arr and not flags.flat:
                seq = st.write_keys_upto(tm - t0 + 1)          # k_{pi(i)} under A15
                G["keys_S"][b, : rows.size] = seq[rows]
                G["m_st_key"][b, : rows.size] = rows >= w - 1
        if need_ev:
            n = st.n_events_upto(tm)
            G["n_ev"][b] = n
            if n > 0:
                e_last = n - 1
                te = int(st._ev_t.a[e_last])
                G["has_ev"][b] = True
                G["p_ref"][b] = st._ev_pref.a[e_last]
                G["v_ref"][b] = st._ev_vref.a[e_last]
                G["ckey"][b] = st._ev_ckey.a[e_last]
                G["uo"][b] = st.uav_occ[te - t0]
                wr = _window_rows(st, te, wl)
                G["X_pw"][b, wl - wr.size:] = st.x[wr]
                G["A_pw"][b, wl - wr.size:] = st.zone_arr[wr]
                G["m_pw"][b, wl - wr.size:] = True
                G["zm"][b] = st.zone_mean[te - t0]
                if "A_dw" in G:                                    # A16
                    src = int(st._ev_src.a[e_last])
                    if src >= 0:                                   # delayed write: rows src+1 .. te = src + Delta
                        dr = np.arange(src - t0 + 1, te - t0 + 1)
                        G["A_dw"][b, : dr.size] = st.zone_arr[dr]
                        G["m_dw"][b, : dr.size] = True
                        G["uo"][b] = st.uav_occ[dr].mean(axis=0)
                    else:                                          # immediate write (Delta = 0): the A0 window
                        G["A_dw"][b, : wr.size] = st.zone_arr[wr]
                        G["m_dw"][b, : wr.size] = True
        if flags.use_ml and not flags.flat:
            n = int(G["n_ev"][b])
            if n > 0:
                tab = st.table_at(n)
                filled = tab >= 0
                G["rows_L"][b, filled] = st.flat_after[tab[filled]]
                G["keys_L"][b, filled] = st.flat_key[tab[filled]]
                G["m_L"][b] = filled
                ks, rb, kk, _ = st.event_rows(n - 1)
                G["ks"][b, : ks.size], G["m_ks"][b, : ks.size] = ks, True
                G["before"][b, : ks.size], G["keys"][b, : ks.size] = rb, kk
        if flags.flat:
            # entries of the last KF slots in write order: per slot e first, then z (if any)
            s_lo = max(t0, tm - KF + 1)
            slots = np.arange(s_lo, tm + 1)
            n = int(G["n_ev"][b])
            ev_lo = int(np.searchsorted(st._ev_t.a[:n], s_lo, side="left"))
            ev_idx = np.arange(ev_lo, n)
            key = np.concatenate([slots * 2, st._ev_t.a[ev_idx] * 2 + 1])
            kind = np.concatenate([np.zeros(slots.size, np.int64), np.ones(ev_idx.size, np.int64)])
            ref = np.concatenate([slots, ev_idx])
            order = np.argsort(key, kind="stable")[-KF:]
            kind, ref = kind[order], ref[order]
            m = order.size
            G["F_mask"][b, :m] = True
            is_e = kind == 0
            G["F_base"][b, :m][is_e] = st.e[ref[is_e] - t0]
            G["F_base"][b, :m][~is_e] = st._ev_a.a[ref[~is_e]]
            if arr:                                                  # "arrivals": slot keys
                seq = st.write_keys_upto(tm - t0 + 1)                # k_{pi} under A15
                G["F_key"][b, :m][is_e] = seq[ref[is_e] - t0]        # e rows: key of the write slot
                G["F_mask"][b, :m][is_e] = (ref[is_e] - t0) >= w - 1  # unkeyed e rows not attended
            else:
                G["F_key"][b, :m][is_e] = st.e[ref[is_e] - t0]       # e rows: key = e_t
            G["F_key"][b, :m][~is_e] = st._ev_key.a[ref[~is_e]]      # z rows: key = k_t
            # the most recent K_S e entries are re-encoded
            pos = np.arange(m)
            rec = is_e & (ref > tm - KS)
            rpos, rslot = pos[rec], ref[rec]
            G["pos_rec"][b, : rpos.size] = rpos
            # X_st rows were filled oldest-first over [max(t0, tm-KS+1), tm]; align to rslot
            lo = max(t0, tm - KS + 1)
            assert np.array_equal(rslot, np.arange(lo, tm + 1))
            if n > 0 and (~is_e).any() and ref[~is_e][-1] == n - 1:
                G["pos_z"][b] = pos[~is_e][-1]
            else:
                G["has_ev"][b] = False
    return G


def reconstruct(params, cfg, flags: MemoryFlags, refs, grad: bool = True) -> torch.Tensor:
    """Replayed retrieval r(t) (B, d) for refs = [(SegmentStore, t), ...] with the current params.

    grad=True keeps the graph (one-step truncated through the last M_L write);
    grad=False runs under no_grad (targets r(t+1), r(t-1))."""
    G = _gather(params, cfg, flags, refs)
    mc = cfg.memory
    d, KS, KL = int(mc.d), int(mc.k_short), int(mc.k_long)
    KF = KS + KL
    lag = int(mc.corr_lag)
    ms_max, ml_max = float(mc.m_max_s), float(mc.m_max_l)
    mf_max = float(mc.get("m_max_flat", math.sqrt(ms_max ** 2 + ml_max ** 2)))
    tau = float(mc.get("soft_tau", SOFT_TAU_DEFAULT))
    dev = params.device
    T = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    B = G["B"]
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx:
        qin = T(G["qin"])
        need_ev = flags.use_ml or flags.flat
        # one encoder pass over all rows that need re-encoding
        chunks = []
        if "X_st" in G:
            chunks.append(T(G["X_st"]))
        if need_ev:
            chunks.append(T(G["X_pw"]))
        E = params.encode(torch.cat(chunks, dim=1)) if chunks else None
        off = 0
        E_st = E_pw = None
        if "X_st" in G:
            E_st = E[:, off:off + KS]
            off += KS
        if need_ev:
            E_pw = E[:, off:]
        z_new = None
        arr = bool(G["arrivals"])
        kc = bool(G["key_centering"]) or arr        # explicit keys on every keyed level
        qkey = T(G["qkey"]) if arr else None        # "arrivals": query = the live slot's key
        if need_ev:
            P = compute_P_batch(E_pw, T(G["A_pw"]), T(G["zm"]), T(G["m_pw"]), lag)
            if "A_dw" in G:                                                              # A16 delayed content
                v = compute_v_batch(T(G["A_dw"]), T(G["zm"]), T(G["m_dw"]), T(G["uo"]))
            else:
                v = compute_v_batch(T(G["A_pw"]), T(G["zm"]), T(G["m_pw"]), T(G["uo"]))   # zone-resolved v_t
            z = params.candidate_from(P, v, T(G["p_ref"]), T(G["v_ref"]))    # (B, d)
            z_new = z
        if flags.flat:
            base = T(G["F_base"])
            m_st = T(G["m_st"]).to(torch.float32)[..., None]
            src = E_st * m_st
            M = base.scatter(1, T(G["pos_rec"])[..., None].expand(B, KS, d), src)
            hz = T(G["has_ev"])
            pz = torch.where(hz, T(G["pos_z"]), torch.full_like(T(G["pos_z"]), KF + KS))
            M = M.scatter(1, pz[:, None, None].expand(B, 1, d), z_new[:, None, :])
            M = project_norm(M[:, :KF], mf_max)
            Kf = T(G["F_key"])[:, :KF] if kc else None                     # slot keys (detached)
            levels = [(M, T(G["F_mask"]), Kf)]
        else:
            levels = []
            if flags.use_ms:
                m_st = T(G["m_st"])
                MS = project_norm(E_st * m_st.to(torch.float32)[..., None], ms_max)
                if arr:
                    levels.append((MS, m_st & T(G["m_st_key"]), T(G["keys_S"])))
                else:
                    levels.append((MS, m_st))
            else:
                levels.append(None)
            if flags.use_ml:
                # Eq. (45) on the touched slots of the most recent event (soft_update with a
                # single valid slot is exactly the hard-mode gated write)
                m_ks = T(G["m_ks"]) & T(G["has_ev"])[:, None]
                zq = T(G["ckey"]) if kc else None       # soft address from the logged k_t
                new, _ = soft_update(T(G["before"]), T(G["keys"]), z_new, m_ks, tau, params, zq=zq)
                ML_raw = T(G["rows_L"])                                     # detached rows
                bidx = torch.arange(B, device=dev)[:, None].expand_as(m_ks)
                ML_raw[bidx[m_ks], T(G["ks"])[m_ks]] = new[m_ks]
                KL_keys = T(G["keys_L"]) if kc else None                    # slot keys (detached)
                levels.append((project_norm(ML_raw, ml_max), T(G["m_L"]), KL_keys))
            else:
                levels.append(None)
        r, _, _ = attend(params, qin, levels, flags.mean_pool, qkey=qkey, key_tau=G["key_tau"])
        r = r * T(G["valid"]).to(r.dtype)[:, None]
    return r


def reconstruct_P(params, cfg, flags: MemoryFlags, refs, grad: bool = False) -> torch.Tensor:
    """P(t) (B, Z, d) recomputed from x with the current W_S (Eq. 44)."""
    mc = cfg.memory
    wl = int(mc.window_w) + 1 if flags.use_ms else 1
    xd, Z = x_dim(cfg), int(cfg.scenario.n_zones)
    B = len(refs)
    X = np.zeros((B, wl, xd), np.float32)
    A = np.zeros((B, wl, Z), np.float32)
    zm = np.ones((B, Z), np.float32)
    m = np.zeros((B, wl), bool)
    for b, (st, t) in enumerate(refs):
        wr = _window_rows(st, int(t), wl)
        X[b, wl - wr.size:] = st.x[wr]
        A[b, wl - wr.size:] = st.zone_arr[wr]
        m[b, wl - wr.size:] = True
        zm[b] = st.zone_mean[t - st.t0]
    dev = params.device
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx:
        E = params.encode(torch.as_tensor(X, device=dev))
        return compute_P_batch(E, torch.as_tensor(A, device=dev), torch.as_tensor(zm, device=dev),
                               torch.as_tensor(m, device=dev), int(mc.corr_lag))
