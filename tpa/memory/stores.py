"""Trainable memory parameters and the rollout-time two-level memory.

Eq. (19) M = {M_S, M_L}; Eqs. (21)-(22) short-term write with encoder
e_t = tanh(W_S x_t); Eqs. (23)-(24) = Eq. (45) gated long-term write with
merge / LRU addressing; Eq. (42) Frobenius projection; Eqs. (25)-(26), (46) retrieval
(retrieval.attend). Arms: A1 (no M_L), A2 (no M_S), A3 (flat FIFO), A4 (mean
pooling), A14 (time-displaced retrieval).

Offset-free keys and the zone-resolved candidate:
- Offset-free keys (memory.key_centering): the long-term candidate z_t is computed
  every slot in which it is defined; the memory keeps a bias-corrected exponential moving
  mean mu of the candidates (alpha = 1 / memory.key_mean_horizon_slots) and the KEY of a
  write is k_t = z_t - mu_{t-1} (k_t = z_t at the first candidate of a segment). The merge
  test is max_k cos(k_t, L_key[k]) >= rho_merge, a new / replaced slot stores L_key = k_t,
  and retrieval uses L_key (unprojected, detached) as the attention keys of M_L (values =
  the projected contents). Contents keep the full z_t and the gate is unchanged. A3: the
  z-kind rows of the flat store get key k_t, e-kind rows key e_t. Order within a slot:
  z_t -> k_t = key_of(z_t) -> [consolidate(z_t, key=k_t) if the gate is open] ->
  update_mu(z_t) (observe_candidate). key_centering = false: keys = raw z, retrieval keys =
  contents.
- Zone-resolved candidate (memory.candidate_source = "zone"): z_t = repr(W_v v_t) with
  v_t = [mean over the MEPR window of A[tau, z] / zone_mean_t[z] ; uav_occupancy_t]
  (MEPR.v); "pattern" keeps z_t = repr(W_z vec(P)). Under "novelty", v_t - v_ref with v_ref
  kept like P_ref (MEPR.v_ref_used).

Arrival keys (memory.key_source = "arrivals"; "candidate" = the candidate-derived keys above,
the code fallback):
- The key of slot t is the ARRIVAL key k_t = a_t - mu_{t-1} in R^Z: a_t = mean over the last
  w = memory.window_w recorded slots of the per-zone arrivals za, divided elementwise by the
  long-run zone mean m_t recorded with slot t (both from the attached SegmentStore; see
  SegmentStore.arrival_keys_upto for the exact arithmetic), mu the normalised EMA
  (S/c, c_0 = 0, alpha = 1/key_mean_horizon_slots) over every slot with a defined a_t, the
  first defined a_t left uncentred. Defined once w slots are recorded (t - t0 >= w - 1);
  before that there is no long-term write and retrieval returns r = 0 with an empty info.
  No learned parameters on the key side. The key is computed ONCE per slot
  (key_for_slot(t), cached in the store) and the same value serves the merge test
  (cos(k_t, L_key[j]) >= rho_merge), every write of the slot and the retrieval query.
  mu_{t-1} is used for the key of slot t and mu is advanced with a_t right after, inside
  the sequential cache, so the ordering key -> writes -> mu update holds by construction.
- Every stored entry carries the key of its write slot: M_L (L_key), M_S (S_key) and
  the A3 flat store (F_key); rows written before the key was defined are unkeyed
  (S_keyed / F_keyed False) and are not attended. Retrieval logits are cos(k_t, K_j) /
  key_temperature per level (retrieval.attend, one head on the key side). The candidate
  mean mu of the offset-free keys is not advanced under "arrivals" (observe_candidate is a
  no-op).
- retrieve(..., min_age): info gains top_old_slot / top_old_age, the long-term entry with
  the highest cos(k_t, L_key) among entries of age >= min_age on the memory actually read
  (the A14 snapshot when displaced); diagnostic only, never touches r(t) or the LRU clock.

Key-shuffled control A15 (flags.key_shuffled; "arrivals" only):
- The WRITE key of slot t is k_{pi(t)}, the arrival key of a uniformly random earlier
  defined slot pi(t) ~ U{t' : t0 + w - 1 <= t' < t} (pi(t) = t at the first defined slot,
  where no earlier key exists). pi(t) is drawn ONCE per slot with a defined key, in slot
  order, from rng_a15 (streams.a15 / the driver's default_rng(seed + 3000)), independent
  of the gate: the draw is made by write_key_for_slot(t) on its first call in slot t --
  write_short, which every caller (control.py, the driver, diag.run_diagnostic) runs
  before consolidate and retrieve -- and logged in the SegmentStore (log_pi; -1 = none);
  later calls in the same slot (consolidate) read the log, and reconstruction reads the
  log too (never re-draws). k_{pi(t)} serves the long-term merge test, the L_key of a
  new / replaced slot, the S_key of the M_S row and the F_key of the A3 rows written at
  t. The retrieval query at t stays the TRUE k_t (key_for_slot). Contents untouched.
  Raises under key_source = "candidate" and without rng_a15.

Delayed consolidation, A16, the forecasting write rule (flags.delayed; "arrivals" keys, use_ml,
candidate_source "zone"):
- At slot s with the gate open, consolidate takes the key k_s = write_key_for_slot(s) (the true
  arrival key) and PUSHES (s, k_s, label_s) onto a pending list; nothing is written at s. At
  slot t = s + Delta (memory.delay_slots, override-only key) the entry is written by
  _flush_due(t) -- called at the end of write_short(t) every slot and again after the push
  inside consolidate(t) -- with content y_s = delayed_content(s) = repr(W_v [ mean over slots
  s+1..s+Delta of za / zone_mean of slot s+Delta ; mean over the same slots of uav_occ ])
  (compute_v over the NEXT-Delta rows of the attached SegmentStore through the same W_v and
  repr as A0's candidate), key k_s, the label of s, and the event logged with t = the write
  slot and delay_src = s. Delta = 0 writes the slot's own candidate z_s at s itself (the A0
  path, bit for bit). Pending entries whose write slot lies beyond the segment end are
  dropped at reset. Everything else (M_S, gate, retrieval query k_t, SAC, DRM) is A0. Arms
  without the flag do not use this path.

Eq. (42) projection convention: the projection M_l <- M_l * min(1, M_max / ||M_l||_F)
is applied to the matrix that every reader sees (retrieval, mbar_S, J4 norms) after
every write. The pre-projection ("raw") rows are kept internally, so repeated
projections do not compound; this makes the projected matrix a function of the
current raw contents only, which replay reconstruction (recon.py) reproduces exactly.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import EPS, MemoryFlags, q_dim, x_dim
from .retrieval import attend, cos_rows, safe_norm   # noqa: F401  (cos_rows / safe_norm re-exported)

SOFT_TAU_DEFAULT = 0.1   # memory.soft_tau fallback (default.toml sets the key)
ADDRESSING_MODES = ("fixed_key", "content", "soft")
CANDIDATE_REPRS = ("tanh", "layernorm", "novelty")
CANDIDATE_REPR_DEFAULT = "tanh"   # memory.candidate_repr fallback if the key is absent (default.toml sets it)
CANDIDATE_SOURCES = ("pattern", "zone")
CANDIDATE_SOURCE_DEFAULT = "pattern"   # memory.candidate_source fallback (candidate from the pattern P)
KEY_CENTERING_DEFAULT = False          # memory.key_centering fallback (uncentred keys)
KEY_MEAN_HORIZON_DEFAULT = 7200        # memory.key_mean_horizon_slots fallback (2P)
KEY_SOURCES = ("candidate", "arrivals")
KEY_SOURCE_DEFAULT = "candidate"       # memory.key_source fallback (keys from the candidate z_t)
KEY_TEMPERATURE_DEFAULT = 0.1          # memory.key_temperature fallback (tau_key, declared a priori)
DELAY_SLOTS_DEFAULT = 300              # memory.delay_slots (A16): override-only key, stage-1 tunable


def key_dim(cfg) -> int:
    """Dimension of the stored slot keys: Z under key_source = "arrivals", d otherwise."""
    mc = cfg.memory
    src = str(mc.get("key_source", KEY_SOURCE_DEFAULT))
    if src not in KEY_SOURCES:
        raise ValueError(f"unknown memory.key_source {src!r}")
    return int(cfg.scenario.n_zones) if src == "arrivals" else int(mc.d)


def soft_update(before: torch.Tensor, keys: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                tau: float, params, zq: torch.Tensor | None = None):
    """Soft-addressed Eq. (45) over the touched slots (the "soft" addressing mode).

    before, keys: (..., n, d); z: (..., d); mask: (..., n) bool (valid slots).
    w = softmax_k(cos(zq, K_k) / tau) over the valid slots, zq = the candidate's addressing
    vector (the centred key k_t under key centering; default zq = z, uncentred);
    s_k = sigmoid(W_g [before_k; z]); g_k = 1 - w_k (1 - s_k);
    after_k = before_k * g_k + z * (1 - g_k) (the content written is always the full z).
    With a single valid slot w = 1 and this reduces to the hard-mode gated write, so the
    same function serves "content" / "fixed_key" events (n = 1) in reconstruction.
    Returns (after, w)."""
    logits = cos_rows(keys, z if zq is None else zq) / tau
    logits = logits.masked_fill(~mask, float("-inf"))
    anyv = mask.any(dim=-1, keepdim=True)
    # a row with no valid slot (ref without any event) would give softmax(-inf...) = NaN,
    # which poisons the backward pass even though the row is never used: zero it instead
    logits = torch.where(anyv, logits, torch.zeros_like(logits))
    w = torch.softmax(logits, dim=-1) * mask.to(logits.dtype)
    zz = z[..., None, :].expand_as(before)
    s_ = params.gate(before, zz)
    g = 1.0 - w[..., None] * (1.0 - s_)
    after = before * g + zz * (1.0 - g)
    return after, w


def project_norm(M: torch.Tensor, m_max: float) -> torch.Tensor:
    """Eq. (42): ||M_l||_F <= M_max^l, by uniform scaling. M is (..., K, d)."""
    M64 = M.to(torch.float64)
    nrm = torch.sqrt(torch.clamp((M64 * M64).sum(dim=(-2, -1), keepdim=True), min=EPS * EPS))
    scale = torch.clamp(m_max / nrm, max=1.0)
    return M * scale.to(M.dtype)


class MemoryParams(nn.Module):
    """Trainable memory parameters: W_S, W_z, W_g, W_Q, W_K, W_V, W_omega, and
    W_v (the zone-resolved candidate map R^{Z+M} -> R^d).

    Initialisation uses torch's default Linear init inside a forked, seeded RNG
    context (no global RNG consumption); pass `seed` from Streams.torch_seed().
    W_v is created LAST inside that context, so the other layers get the same seeded
    initial values as without W_v (tests/test_memory_d4.py checks this).
    """

    def __init__(self, cfg, seed: int = 0, device: str | torch.device = "cpu"):
        super().__init__()
        mc = cfg.memory
        self.d = int(mc.d)
        self.n_heads = int(mc.n_heads)
        assert self.d % self.n_heads == 0
        self.d_k = self.d // self.n_heads
        self.Z = int(cfg.scenario.n_zones)
        self.M = int(cfg.scenario.n_uav)
        self.x_dim, self.q_dim = x_dim(cfg), q_dim(cfg)
        self.eps = float(mc.get("eps", EPS))
        self.repr = str(mc.get("candidate_repr", CANDIDATE_REPR_DEFAULT))
        if self.repr not in CANDIDATE_REPRS:
            raise ValueError(f"unknown memory.candidate_repr {self.repr!r}")
        self.source = str(mc.get("candidate_source", CANDIDATE_SOURCE_DEFAULT))   # "pattern" | "zone"
        if self.source not in CANDIDATE_SOURCES:
            raise ValueError(f"unknown memory.candidate_source {self.source!r}")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.W_S = nn.Linear(self.x_dim, self.d)                 # Eq. (22)
            self.W_z = nn.Linear(self.Z * self.d, self.d)            # Eq. (45) candidate z(t)
            self.W_g = nn.Linear(2 * self.d, self.d)                 # Eq. (45) gate g(t)
            self.W_Q = nn.Linear(self.q_dim, self.d)                 # Eq. (46) query
            self.W_K = nn.Linear(self.d, self.d, bias=False)         # Eq. (46) keys
            self.W_V = nn.Linear(self.d, self.d, bias=False)         # Eq. (46) values
            self.W_omega = nn.Linear(self.q_dim, 2)                  # Eq. (26) level weights
            # W_v: created after the layers above so that their seeded init does not depend on it
            self.W_v = nn.Linear(self.Z + self.M, self.d)            # candidate from v_t
        self.to(device)

    @property
    def device(self) -> torch.device:
        return self.W_S.weight.device

    def _t(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

    def encode(self, x) -> torch.Tensor:
        """Eq. (22): e_t = tanh(W_S x_t), x_t = [S_sys; tau_bar(mean,max); mbar_S]."""
        return torch.tanh(self.W_S(self._t(x)))

    def candidate(self, P, P_ref=None) -> torch.Tensor:
        """Eq. (45): the candidate z(t) from P (..., Z, d). memory.candidate_repr:
        - "tanh":      z = tanh(W_z vec(P))
        - "layernorm": z = LayerNorm(W_z vec(P)), no affine parameters, eps = memory.eps
        - "novelty":   z = LayerNorm(W_z vec(P - P_ref)), P_ref (detached) = the MEPR
                       reference the slot's sigma_p was measured against."""
        P = self._t(P)
        if self.repr == "novelty":
            if P_ref is None:
                raise ValueError("candidate_repr='novelty' needs P_ref (mepr.P_ref_used)")
            P = P - self._t(P_ref).detach()
        h = self.W_z(P.reshape(*P.shape[:-2], self.Z * self.d))
        return self._repr(h)

    def _repr(self, h: torch.Tensor) -> torch.Tensor:
        if self.repr == "tanh":
            return torch.tanh(h)
        return F.layer_norm(h, (self.d,), eps=self.eps)

    def candidate_zone(self, v, v_ref=None) -> torch.Tensor:
        """The candidate z(t) from the zone-resolved vector v (..., Z + M),
        v = [mean_window A[tau, z] / zone_mean[z] ; uav_occupancy]: z = repr(W_v v), with
        repr as in `candidate`; "novelty": repr(W_v (v - v_ref)), v_ref (detached) the MEPR
        reference (mepr.v_ref_used), required."""
        v = self._t(v)
        if self.repr == "novelty":
            if v_ref is None:
                raise ValueError("candidate_repr='novelty' needs v_ref (mepr.v_ref_used)")
            v = v - self._t(v_ref).detach()
        return self._repr(self.W_v(v))

    def candidate_from(self, P, v, p_ref=None, v_ref=None) -> torch.Tensor:
        """Dispatcher on memory.candidate_source: "pattern" -> candidate(P, p_ref);
        "zone" -> candidate_zone(v, v_ref)."""
        if self.source == "zone":
            return self.candidate_zone(v, v_ref)
        return self.candidate(P, p_ref)

    def gate(self, row, z) -> torch.Tensor:
        """Eqs. (24)/(45): g = sigmoid(W_g [M_L[k]; z]) (Phi = P enters through z)."""
        return torch.sigmoid(self.W_g(torch.cat([self._t(row), self._t(z)], dim=-1)))

    def query(self, qin) -> torch.Tensor:
        """Eq. (46): q(t) = W_Q [S_sys; tau_bar]."""
        return self.W_Q(self._t(qin))

    def level_weights(self, qin) -> torch.Tensor:
        """Eq. (26): omega_l = softmax(W_omega q) over (S, L)."""
        return torch.softmax(self.W_omega(self._t(qin)), dim=-1)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().copy()


class TwoLevelMemory:
    """Rollout-time memory state (detached float32 torch tensors on the params device).

    Per slot (slot timeline and ordering):
        store.record(t, x, qin, za, zone_mean, uav_occ)   (the caller; FIRST in the slot)
        e = params.encode(x)           (no_grad)
        mem.write_short(e, label, t)   (no-op in A2)
        ... MEPR (P, v, sigma_p, cons) ...
        z = mem.candidate_for(mepr)    (None while undefined / A1)   -> k_t = mem.key_of(z)
        if cons: mem.consolidate(z, label, t, p_ref=mepr.P_ref_used, v_ref=mepr.v_ref_used)
        if z is not None: mem.observe_candidate(z)        (mu_t <- mu_{t-1}, z_t)
        r, info = mem.retrieve(qin, t)

    The key of a write uses mu_{t-1} (consolidate computes key_of(z) before the caller
    updates mu with z_t); observe_candidate must be called AFTER consolidate.
    Under key_source = "arrivals": the same call sequence; the key of slot t is
    key_for_slot(t) (from the store's record of the slot, computed once and cached), used
    by write_short, consolidate and retrieve alike; observe_candidate is a no-op.
    A15 (flags.key_shuffled): the WRITE key of slot t (write_short, consolidate) is
    write_key_for_slot(t) = k_{pi(t)}; the retrieve query stays key_for_slot(t).

    If a SegmentStore is attached (attach_store), write_short logs e_t, consolidate
    logs the event and retrieve logs the memory time used (A14 schedule); the caller
    must NOT log these again (it logs x, qin, zone arrays, uav_occ via store.record).
    """

    def __init__(self, cfg, flags: MemoryFlags, params: MemoryParams, rng_a14=None,
                 rng_a15=None):
        mc = cfg.memory
        self.cfg, self.flags, self.params = cfg, flags, params
        self.d, self.KS, self.KL = int(mc.d), int(mc.k_short), int(mc.k_long)
        self.KF = self.KS + self.KL                       # A3 flat store size (600)
        self.Z = int(cfg.scenario.n_zones)
        self.w = int(mc.window_w)
        self.key_source = str(mc.get("key_source", KEY_SOURCE_DEFAULT))       # "candidate" | "arrivals"
        if self.key_source not in KEY_SOURCES:
            raise ValueError(f"unknown memory.key_source {self.key_source!r}")
        self.arrivals = self.key_source == "arrivals"
        self.shuffled = bool(getattr(flags, "key_shuffled", False))           # A15 (key-shuffled control)
        if self.shuffled and not self.arrivals:
            raise ValueError("A15 (key_shuffled) is defined only under memory.key_source = 'arrivals'")
        self.rng_a15 = rng_a15
        if self.shuffled and rng_a15 is None:
            raise ValueError("A15 needs rng_a15 = streams.a15")
        self.delayed = bool(getattr(flags, "delayed", False))                 # A16 (delayed consolidation)
        self.delay = int(mc.get("delay_slots", DELAY_SLOTS_DEFAULT)) if self.delayed else 0
        if self.delayed:
            if not self.arrivals:
                raise ValueError("A16 (delayed) is defined only under memory.key_source = 'arrivals'")
            if not flags.use_ml or flags.flat:
                raise ValueError("A16 (delayed) needs the long-term store (use_ml, not flat)")
            if self.delay < 0:
                raise ValueError("memory.delay_slots must be >= 0")
            if params.source != "zone" or params.repr == "novelty":
                raise ValueError("A16 (delayed) needs memory.candidate_source = 'zone' and a non-novelty repr")
        # A16': the content map W_v of the long-term content is FROZEN at its seeded initialisation (no
        # gradient through the long-term content path); W_v feeds nothing else, so every other parameter
        # (W_S, W_g, W_Q, W_K, W_V, W_omega, SAC) trains exactly as in A0 / A16. Only the requires_grad
        # flags change: no new state, so every other arm's checkpoint bytes are unchanged.
        self.frozen_content = bool(getattr(flags, "frozen_content", False))
        if self.frozen_content:
            if not self.delayed or self.delay < 1:
                raise ValueError("A16' (frozen_content) is defined only with delayed consolidation and Delta >= 1")
            for p in params.W_v.parameters():
                p.requires_grad_(False)
        self.key_tau = float(mc.get("key_temperature", KEY_TEMPERATURE_DEFAULT))
        if self.key_tau <= 0.0:
            raise ValueError("memory.key_temperature must be > 0")
        self.kd = key_dim(cfg)                            # stored key dimension (Z | d)
        self.rho = float(mc.rho_merge)
        self.addressing = str(mc.lt_addressing)          # "fixed_key" | "content" | "soft"
        if self.addressing not in ADDRESSING_MODES:
            raise ValueError(f"unknown memory.lt_addressing {self.addressing!r}")
        self.soft_tau = float(mc.get("soft_tau", SOFT_TAU_DEFAULT))
        self.key_centering = bool(mc.get("key_centering", KEY_CENTERING_DEFAULT))     # offset-free keys
        self.mu_horizon = int(mc.get("key_mean_horizon_slots", KEY_MEAN_HORIZON_DEFAULT))
        if self.mu_horizon < 1:
            raise ValueError("memory.key_mean_horizon_slots must be >= 1")
        self.mu_alpha = 1.0 / self.mu_horizon
        self.ms_max, self.ml_max = float(mc.m_max_s), float(mc.m_max_l)
        # A3: the flat store's Eq. 42 bound (memory.m_max_flat = sqrt(Ms^2 + Ml^2))
        self.mf_max = float(mc.get("m_max_flat", math.sqrt(self.ms_max ** 2 + self.ml_max ** 2)))
        P = int(cfg.workload.period_slots)
        self.a14_lo = int(round(float(mc.a14_lo_frac) * P))
        self.a14_hi = int(round(float(mc.a14_hi_frac) * P))
        self.a14_redraw = int(mc.a14_redraw_slots)
        self.a14_snap = int(mc.a14_snapshot_slots)
        self.rng_a14 = rng_a14
        if flags.displaced and rng_a14 is None:
            raise ValueError("A14 needs rng_a14 = streams.a14")
        self.store = None
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, t0: int | None = None) -> None:
        """Start of a memory segment or test run: everything empty."""
        dev = self.params.device
        z = lambda n: torch.zeros(n, self.d, dtype=torch.float32, device=dev)  # noqa: E731
        zk = lambda n: torch.zeros(n, self.kd, dtype=torch.float32, device=dev)  # noqa: E731
        self.S_raw, self.S_fill, self.S_ptr = z(self.KS), 0, 0
        self.S_label = np.full(self.KS, -1, np.int64)
        self.S_time = np.full(self.KS, -1, np.int64)
        self.S_key = zk(self.KS)                          # slot key of each M_S row ("arrivals")
        self.S_keyed = np.zeros(self.KS, bool)            # row written with a defined key
        self.L_raw = z(self.KL)
        self.L_key = zk(self.KL)                          # detached key of the slot's first write
        self.L_filled = np.zeros(self.KL, bool)
        self.L_label = np.full(self.KL, -1, np.int64)
        self.L_time = np.full(self.KL, -1, np.int64)      # latest write slot
        self.L_use = np.full(self.KL, -1, np.int64)       # LRU clock (write or top-1 use)
        self.F_raw, self.F_fill, self.F_ptr, self.F_count = z(self.KF), 0, 0, 0
        self.F_label = np.full(self.KF, -1, np.int64)
        self.F_time = np.full(self.KF, -1, np.int64)
        self.F_kind = np.full(self.KF, -1, np.int64)      # 0 = e_t, 1 = z(t)
        self.F_key = zk(self.KF)                          # e rows = e_t, z rows = k_t; "arrivals": slot key
        self.F_keyed = np.zeros(self.KF, bool)
        # Running mean of the candidates (offset-free keys): mu = mu_S / mu_c (bias-corrected EMA),
        # accumulated in float64; mu_n = number of candidates observed in the segment
        self.mu_S = torch.zeros(self.d, dtype=torch.float64, device=dev)
        self.mu_c = 0.0
        self.mu_n = 0
        self.t0, self.t_now = t0, None
        self.n_events = 0
        self.delta, self.delta_block = None, None         # A14
        self.snapshots: dict[int, dict] = {}
        self.pending: list[dict] = []                     # A16: (s, k_s, label, ...) awaiting s + Delta

    def attach_store(self, store) -> None:
        self.store = store

    def _tick(self, t: int) -> None:
        if self.t0 is None:
            self.t0 = int(t)
        self.t_now = int(t)

    # ---------------------------------------------------------------- visible
    def M_S(self) -> torch.Tensor:
        """Visible M_S after Eq. (42)."""
        return project_norm(self.S_raw, self.ms_max)

    def M_L(self) -> torch.Tensor:
        """Visible M_L after Eq. (42)."""
        return project_norm(self.L_raw, self.ml_max)

    def M_F(self) -> torch.Tensor:
        """Visible A3 flat store after Eq. (42)."""
        return project_norm(self.F_raw, self.mf_max)

    def mbar_short(self) -> np.ndarray:
        """mbar_S(t): mean of the filled rows of the visible M_S (A3: of the flat
        store), taken before this slot's write; zeros if empty or in A2. float64 (d,)."""
        if self.flags.flat:
            n, M = self.F_fill, self.M_F()
        elif self.flags.use_ms:
            n, M = self.S_fill, self.M_S()
        else:
            n, M = 0, None
        if n == 0:
            return np.zeros(self.d)
        return M.sum(dim=0).double().cpu().numpy() / n     # unfilled rows are zero

    def fro_norms(self) -> tuple[float, float]:
        """(||M_S||_F, ||M_L||_F) of the visible matrices, for J4 (Eq. 33).
        A3: (||flat||_F, 0); absent levels give 0."""
        if self.flags.flat:
            return float(torch.linalg.vector_norm(self.M_F(), dtype=torch.float64)), 0.0
        s = float(torch.linalg.vector_norm(self.M_S(), dtype=torch.float64)) if self.flags.use_ms else 0.0
        l_ = float(torch.linalg.vector_norm(self.M_L(), dtype=torch.float64)) if self.flags.use_ml else 0.0
        return s, l_

    # --------------------------------------------------- candidate mean / key
    @property
    def mu(self) -> torch.Tensor | None:
        """mu_t (d,) float64: bias-corrected EMA of the candidates seen so far (None if
        no candidate has been observed in the segment)."""
        if self.mu_n == 0:
            return None
        return self.mu_S / self.mu_c

    @torch.no_grad()
    def update_mu(self, z) -> None:
        """mu_t from mu_{t-1} and z_t: S <- (1-a) S + a z, c <- (1-a) c + a, mu = S / c,
        a = 1 / key_mean_horizon_slots; c starts at 0, so mu_1 = z_1 and mu_t is the
        (1-a)^(t-i)-weighted mean of z_1..z_t (an EMA of time constant H normalised by its
        total weight)."""
        z64 = self.params._t(z).reshape(self.d).detach().to(torch.float64)
        a = self.mu_alpha
        self.mu_S = (1.0 - a) * self.mu_S + a * z64
        self.mu_c = (1.0 - a) * self.mu_c + a
        self.mu_n += 1

    def observe_candidate(self, z) -> None:
        """Call once per slot in which z_t is defined, AFTER the slot's consolidate (the key
        of that write uses mu_{t-1}); updates mu with z_t. No-op when key_centering is off
        (mu is then never used), so that the off path is bit-identical to a memory without
        key centering, and under key_source = "arrivals" (the candidate mean is not a key
        there)."""
        if self.key_centering and not self.arrivals:
            self.update_mu(z)

    @torch.no_grad()
    def key_of(self, z) -> torch.Tensor:
        """Offset-free key: k_t = z_t - mu_{t-1} if key_centering and a previous candidate exists in the
        segment, else k_t = z_t. Detached float32 (d,)."""
        z = self.params._t(z).reshape(self.d).detach()
        if self.key_centering and self.mu_n > 0:
            return (z - self.mu.to(torch.float32)).detach()
        return z

    @torch.no_grad()
    def key_for_slot(self, t: int):
        """Under "arrivals": the arrival key k_t of slot t (float32 (Z,) on the params
        device) or None while undefined (fewer than w slots recorded up to t). Computed
        once per slot and cached in the attached SegmentStore (which must have recorded
        slot t already), so that merge, writes, the retrieval query and reconstruction
        all see one value. Raises under key_source = "candidate"."""
        if not self.arrivals:
            raise ValueError("key_for_slot is defined under memory.key_source = 'arrivals'")
        if self.store is None:
            raise ValueError("key_source='arrivals' needs an attached SegmentStore")
        k = self.store.arrival_key(int(t))
        if k is None:
            return None
        return torch.as_tensor(k, device=self.params.device)

    @torch.no_grad()
    def write_key_for_slot(self, t: int):
        """The WRITE key of slot t under "arrivals": k_t itself (A0 shape), or under
        A15 (flags.key_shuffled) k_{pi(t)} with pi(t) the logged shuffle slot of t, drawn
        here on the FIRST call in slot t (write_short in the slot order; consolidate
        then reads the log): pi(t) ~ U{t' : t0 + w - 1 <= t' < t} from rng_a15, one
        integers() draw per slot with a defined key, independent of the gate; at the first
        defined slot (empty range) pi(t) = t and nothing is drawn. None while k_t is
        undefined (no draw, no log). Raises under key_source = "candidate"."""
        k = self.key_for_slot(t)
        if k is None or not self.shuffled:
            return k
        t = int(t)
        pi = self.store.pi_of(t)
        if pi < 0:
            lo = self.store.t0 + self.w - 1                   # first defined slot
            pi = t if t <= lo else int(self.rng_a15.integers(lo, t))
            self.store.log_pi(t, pi)
        return torch.as_tensor(self.store.arrival_key(pi), device=self.params.device)

    def _slot_key(self, z, t: int):
        """The key of a write at slot t: write_key_for_slot(t) under "arrivals" (None while
        undefined; k_{pi(t)} under A15), key_of(z) under "candidate"."""
        if self.arrivals:
            return self.write_key_for_slot(t)
        return self.key_of(z)

    @torch.no_grad()
    def candidate_for(self, mepr):
        """z_t of this slot from the MEPR state (dispatcher on candidate_source), or None
        when it is undefined: no long-term candidate path (A1), no pattern yet, or
        ("novelty") no reference yet (warm-up)."""
        if not (self.flags.use_ml or self.flags.flat) or mepr.P_torch is None:
            return None
        if self.params.repr == "novelty" and mepr.P_ref_used is None:
            return None
        return self.params.candidate_from(mepr.P_torch, mepr.v_torch, mepr.P_ref_used,
                                          mepr.v_ref_used)

    # ----------------------------------------------------------------- writes
    def _fifo_write(self, row: torch.Tensor, label: int, t: int, kind: int,
                    key: torch.Tensor | None = None) -> int:
        pos = self.F_ptr
        self.F_raw[pos] = row
        if self.arrivals:
            self.F_key[pos] = 0.0 if key is None else key   # slot key (None: unkeyed row)
            self.F_keyed[pos] = key is not None
        else:
            self.F_key[pos] = row if key is None else key
            self.F_keyed[pos] = True
        self.F_label[pos], self.F_time[pos], self.F_kind[pos] = label, t, kind
        self.F_ptr = (pos + 1) % self.KF
        self.F_fill = min(self.F_fill + 1, self.KF)
        self.F_count += 1
        return pos

    @torch.no_grad()
    def write_short(self, e, label: int, t: int) -> None:
        """Eqs. (21)-(22): insert e_t into the K_S circular buffer (every slot);
        A3: into the single FIFO of K_S + K_L. A2: no M_S (no-op). Eq. (42) is applied
        to the visible matrix (see module docstring)."""
        self._tick(t)
        e = self.params._t(e).reshape(self.d).detach()
        if self.store is not None:
            self.store.log_e(t, e)
        # Slot key of the M_S / flat rows; under A15 k_{pi(t)}, pi(t) drawn HERE (first
        # call of the slot), whether or not a long-term write follows
        key = self.write_key_for_slot(t) if self.arrivals else None
        if self.flags.flat:
            self._fifo_write(e, label, t, 0, key)
        elif self.flags.use_ms:
            p = self.S_ptr
            self.S_raw[p] = e
            self.S_label[p], self.S_time[p] = label, t
            if self.arrivals:
                self.S_key[p] = 0.0 if key is None else key
                self.S_keyed[p] = key is not None
            self.S_ptr = (p + 1) % self.KS
            self.S_fill = min(self.S_fill + 1, self.KS)
        if self.delayed:                                  # A16: writes due at this slot
            self._flush_due(t)

    @torch.no_grad()
    def write_pattern(self, P, p_ref, label: int, t: int, v=None, v_ref=None):
        """Convenience: z = params.candidate_from(P, v, p_ref, v_ref), then
        consolidate(z, label, t, p_ref, v_ref=v_ref). Use this (or pass p_ref / v_ref
        explicitly) so that the event log carries the reference that reconstruction needs
        under candidate_repr = "novelty". v (mepr.v) is needed under candidate_source =
        "zone"; the event does not log v (replay recomputes it from the window)."""
        z = self.params.candidate_from(P, v, p_ref, v_ref)
        return self.consolidate(z, label, t, p_ref=p_ref, v_ref=v_ref)

    @torch.no_grad()
    def consolidate(self, z, label: int, t: int, p_ref=None, key=None, v_ref=None):
        """Eq. (45) write of the candidate z at slot t (see _consolidate_now for the rules).
        A16 (flags.delayed): nothing is written at t; the slot t (whose gate is open), its key
        k_t = write_key_for_slot(t) and the label are pushed onto the pending list and the
        writes due at t (slots s with s + Delta <= t) are flushed; returns the last event of
        the flush (None if none). Delta = 0 writes z at t itself (the A0 path)."""
        if not self.delayed:
            return self._consolidate_now(z, label, t, p_ref=p_ref, key=key, v_ref=v_ref)
        self._tick(t)
        k = self.write_key_for_slot(t) if key is None else self.params._t(key).reshape(self.kd).detach()
        if k is None:                                     # key undefined (first w - 1 slots): no entry
            return None
        self.pending.append(dict(s=int(t), key=k.clone(), label=int(label),
                                 z=None if z is None else self.params._t(z).reshape(self.d).detach().clone(),
                                 p_ref=None if p_ref is None else np.asarray(p_ref, np.float32),
                                 v_ref=None if v_ref is None else np.asarray(v_ref, np.float32)))
        evs = self._flush_due(t)
        return evs[-1] if evs else None

    @torch.no_grad()
    def delayed_content(self, s: int) -> torch.Tensor:
        """A16: the content written at s + Delta under key k_s,
        y_s = repr(W_v [ mean over slots s+1..s+Delta of za / zone_mean_{s+Delta} ; mean over the
        same slots of uav_occ ]) -- compute_v over the next-Delta rows of the attached store
        (zone mean of the write slot, as compute_v_batch takes the current slot's zone mean for
        every row), through A0's W_v and repr (candidate_zone). Raises if those rows are not
        recorded yet or Delta = 0 (then the content is the slot's own candidate)."""
        from .mepr import compute_v
        if self.delay <= 0:
            raise ValueError("delayed_content needs Delta >= 1 (Delta = 0 writes the slot's candidate)")
        st = self.store
        if st is None or st.t0 is None:
            raise ValueError("A16 needs an attached SegmentStore with recorded slots")
        lo, hi = int(s) - st.t0 + 1, int(s) - st.t0 + self.delay       # rows of slots s+1 .. s+Delta
        if lo < 0 or hi >= st.n:
            raise ValueError(f"delayed_content({s}): slots {s + 1}..{s + self.delay} not all recorded")
        v = compute_v(st.zone_arr[lo:hi + 1], st.zone_mean[hi], st.uav_occ[lo:hi + 1].mean(axis=0))
        return self.params.candidate_zone(v.to(self.params.device), None).reshape(self.d).detach()

    @torch.no_grad()
    def _flush_due(self, t: int) -> list:
        """A16: write every pending entry whose slot s has s + Delta <= t, in slot order, with
        content delayed_content(s) (z_s when Delta = 0), key k_s, the label of s; the event is
        logged at t with delay_src = s. Returns the events."""
        out = []
        while self.pending and self.pending[0]["s"] + self.delay <= int(t):
            p = self.pending.pop(0)
            y = p["z"] if self.delay == 0 else self.delayed_content(p["s"])
            if y is None:
                continue
            ev = self._consolidate_now(y, p["label"], t, p_ref=p["p_ref"], key=p["key"], v_ref=p["v_ref"],
                                       delay_src=-1 if self.delay == 0 else p["s"])
            if ev is not None:
                out.append(ev)
        return out

    @torch.no_grad()
    def _consolidate_now(self, z, label: int, t: int, p_ref=None, key=None, v_ref=None, delay_src: int = -1):
        """Eq. (45) (= Eqs. 23-24 with Phi = P) with merge / LRU addressing; then Eq. (42).

        key = the candidate's addressing key k_t (key_of(z) if None: z - mu_{t-1}
        under key_centering, z otherwise -- call consolidate BEFORE observe_candidate(z)).
        Under "arrivals": key = write_key_for_slot(t) if None (= key_for_slot(t); under A15
        k_{pi(t)}); while that key is undefined (first w - 1 slots of the segment)
        nothing is written and None is returned.
        - max_k cos(k_t, L_key[k]) >= rho_merge  -> gated update of that slot ("merge");
        - otherwise the first empty slot ("new"), else the LRU slot ("replace");
        - always row <- row * g + z * (1 - g), g = sigmoid(W_g [row; z]) (the FULL z).
        Addressing (memory.lt_addressing):
        - "fixed_key" (default): the cosine is taken against each filled slot's KEY,
          the detached k of the slot's first write. A merge updates the content only (key
          unchanged). A new or replaced slot starts from zero content (row_before = 0)
          and its key is set to k_t.
        - "content": the cosine is taken against the slot contents with z (the key is not
          used for addressing; the slot key is still recorded for the retrieval keys under
          key centering); a replaced LRU slot is gated from its old content.
        - "soft": keys as in fixed_key. If max_k cos(k_t, K_k) >= rho_merge, EVERY
          filled slot is updated with the soft-addressed gate (soft_update): w =
          softmax_k(cos(k_t, K_k) / soft_tau), g_k = 1 - w_k (1 - sigmoid(W_g [M_L[k]; z])).
          Otherwise a new slot is opened (empty-first, else LRU) with key k_t and a fresh
          gated write from zero content, exactly as in fixed_key. Label / time / LRU of a
          soft event follow the argmax-cos slot.
        A3: ungated FIFO write of z into the flat store, retrieval key k_t. A1: no M_L
        (returns None).
        Returns the event {t, k, mode, row_before, row_after, key, ks, rows_before,
        rows_after, keys, w, maxcos, p_ref, v_ref, cand_key} (float32 numpy; ks / rows_* /
        keys / w cover every touched slot, one row for the hard modes; key / keys = the
        slot key(s) after the event; cand_key = k_t; maxcos = max cosine of k_t to the
        addressed rows, -1 if none; p_ref / v_ref = the MEPR references used for z, or
        None)."""
        self._tick(t)
        key = self._slot_key(z, t) if key is None else self.params._t(key).reshape(self.kd).detach()
        if key is None:                                   # "arrivals": key undefined -> no write
            return None
        if self.flags.flat:
            z = self.params._t(z).reshape(self.d)
            k = self._fifo_write(z, label, t, 1, key)
            ev = dict(t=int(t), k=int(k), mode="fifo", key=_np(key),
                      row_before=np.zeros(self.d, np.float32), row_after=_np(z))
        elif self.flags.use_ml:
            z = self.params._t(z).reshape(self.d)
            filled = np.flatnonzero(self.L_filled)
            k, mode, maxcos = -1, "", -1.0
            if filled.size:
                content_addr = self.addressing == "content"
                rows = (self.L_raw if content_addr else self.L_key)[filled]
                cos = cos_rows(rows, z if content_addr else key)
                j = int(torch.argmax(cos))
                maxcos = float(cos[j])
                if float(cos[j]) >= self.rho:
                    k, mode = int(filled[j]), "soft" if self.addressing == "soft" else "merge"
            if k < 0:
                empty = np.flatnonzero(~self.L_filled)
                if empty.size:
                    k, mode = int(empty[0]), "new"
                else:
                    k, mode = int(np.argmin(self.L_use)), "replace"
            if mode == "soft":
                ks = filled                                     # every filled slot is touched
                before = self.L_raw[ks].clone()
                keys = self.L_key[ks].clone()
                mask = torch.ones(ks.size, dtype=torch.bool, device=z.device)
                after, w = soft_update(before, keys, z, mask, self.soft_tau, self.params, zq=key)
                self.L_raw[ks] = after
                # label / time / LRU follow the dominant (argmax-cos) slot
                self.L_label[k], self.L_time[k], self.L_use[k] = label, t, t
                jj = int(np.flatnonzero(ks == k)[0])
                ev = dict(t=int(t), k=k, mode=mode, row_before=_np(before[jj]),
                          row_after=_np(after[jj]), key=_np(keys[jj]), ks=ks.copy(),
                          rows_before=_np(before), rows_after=_np(after), keys=_np(keys),
                          w=_np(w))
            else:
                if mode != "merge":
                    self.L_key[k] = key                       # new / replaced slot: key = k_t
                if self.addressing != "content" and mode != "merge":
                    before = torch.zeros_like(z)              # fresh slot (fixed_key / soft)
                else:
                    before = self.L_raw[k].clone()
                g = self.params.gate(before, z)
                after = before * g + z * (1.0 - g)
                self.L_raw[k] = after
                self.L_filled[k] = True
                self.L_label[k], self.L_time[k], self.L_use[k] = label, t, t
                ev = dict(t=int(t), k=k, mode=mode, row_before=_np(before), row_after=_np(after),
                          key=_np(self.L_key[k]), ks=np.array([k]), rows_before=_np(before)[None],
                          rows_after=_np(after)[None], keys=_np(self.L_key[k])[None],
                          w=np.ones(1, np.float32))
        else:
            return None
        if "maxcos" not in ev:
            ev["maxcos"] = maxcos if self.flags.use_ml and not self.flags.flat else -1.0
        ev["p_ref"] = None if p_ref is None else np.asarray(p_ref, np.float32)
        ev["v_ref"] = None if v_ref is None else np.asarray(v_ref, np.float32)
        ev["cand_key"] = _np(key)
        ev["delay_src"] = int(delay_src)                  # A16: slot whose key/gate the write carries
        self.n_events += 1
        if self.store is not None:
            self.store.add_event(ev)
        return ev

    # -------------------------------------------------------------- retrieval
    def _snapshot(self) -> dict:
        return dict(S_raw=self.S_raw.clone(), S_fill=self.S_fill, S_label=self.S_label.copy(),
                    S_time=self.S_time.copy(), L_raw=self.L_raw.clone(),
                    L_key=self.L_key.clone(),                 # A14 reads the snapshot's keys
                    S_key=self.S_key.clone(), S_keyed=self.S_keyed.copy(),   # "arrivals" slot keys
                    L_filled=self.L_filled.copy(), L_label=self.L_label.copy(),
                    L_time=self.L_time.copy())

    def _levels(self, src: dict):
        """Visible levels (M, mask, K) (+ per-entry label/time/level tags) of a state dict.
        K = attention keys: None (= contents) for M_S and for M_L without key_centering;
        the unprojected, detached L_key for M_L under key_centering. Under
        "arrivals": S_key / L_key always, and only keyed M_S rows are attended."""
        dev = self.params.device
        levels, tags = [], []
        if self.flags.use_ms:
            mask = torch.zeros(1, self.KS, dtype=torch.bool, device=dev)
            mask[0, :src["S_fill"]] = True  # rows are filled from index 0; ptr wraps after full
            K = None
            if self.arrivals:
                mask &= torch.as_tensor(src["S_keyed"], device=dev)[None]
                K = src["S_key"][None]
            levels.append((project_norm(src["S_raw"], self.ms_max)[None], mask, K))
            tags.append(("S", src["S_label"], src["S_time"]))
        else:
            levels.append(None)
            tags.append(None)
        if self.flags.use_ml:
            mask = torch.as_tensor(src["L_filled"], device=dev)[None]
            K = src["L_key"][None] if (self.key_centering or self.arrivals) else None
            levels.append((project_norm(src["L_raw"], self.ml_max)[None], mask, K))
            tags.append(("L", src["L_label"], src["L_time"]))
        else:
            levels.append(None)
            tags.append(None)
        return levels, tags

    @torch.no_grad()
    def retrieve(self, qin, t: int | None = None, update_lru: bool = True,
                 min_age: int | None = None):
        """Eqs. (25), (26), (46): r(t) and the info dict.

        info: top_label, top_age (slots), top_level ('S' | 'L' | 'F' | None), used_slot
        (M_L slot of the top-1 entry or None), l1 = ||r||_1, empty (no filled entry),
        mem_time (slot whose memory was read; -1 if none), delta (A14 displacement).
        Top-1 = argmax of omega_l * mean_heads(alpha) over all entries; undefined
        (None / -1) under A4. The M_L LRU clock is updated with the top-1 slot
        unless update_lru=False, under A4, or under A14 (displaced reads).
        Under "arrivals": the query is key_for_slot(t), the TRUE k_t (also under A15:
        only the write keys are shuffled); while it is undefined r = 0 and info is empty
        (as during the MEPR warm-up). min_age (diagnostic, age-matched phase share): adds
        top_old_slot / top_old_age = the M_L entry (of the memory actually read, i.e. the
        snapshot under A14) with the highest key score among entries of age t - L_time >=
        min_age: cos(k_t, L_key) under "arrivals", the head-averaged M_L attention weight
        under "candidate"; None / -1 if none (no M_L, A4, empty). Never touches r or LRU."""
        if t is None:
            t = self.t_now
        self._tick(t)
        t = int(t)
        dev = self.params.device
        mem_time = t
        src = None
        if self.flags.displaced:
            rel = t - self.t0
            blk = rel // self.a14_redraw
            if blk != self.delta_block:                   # redraw every a14_redraw slots
                self.delta = int(self.rng_a14.integers(self.a14_lo, self.a14_hi, endpoint=True))
                self.delta_block = blk
            sblk = rel // self.a14_snap
            if not any((s - self.t0) // self.a14_snap == sblk for s in self.snapshots):
                self.snapshots[t] = self._snapshot()      # state after this slot's writes
            keep_from = t - self.a14_hi - self.a14_snap
            for s in [s for s in self.snapshots if s < keep_from]:
                del self.snapshots[s]
            tau = t - self.delta
            cands = [s for s in self.snapshots if s <= tau]
            if tau < self.t0 or not cands:
                mem_time = -1
            else:
                mem_time = max(cands)
                src = self.snapshots[mem_time]
        elif not self.flags.flat:
            src = dict(S_raw=self.S_raw, S_fill=self.S_fill, S_label=self.S_label,
                       S_time=self.S_time, S_key=self.S_key, S_keyed=self.S_keyed,
                       L_raw=self.L_raw, L_key=self.L_key,
                       L_filled=self.L_filled, L_label=self.L_label, L_time=self.L_time)
        if self.store is not None:
            self.store.log_mem_time(t, mem_time, -1 if self.delta is None else self.delta)

        info = dict(top_label=-1, top_age=-1, top_level=None, used_slot=None, l1=0.0,
                    empty=True, mem_time=mem_time, delta=self.delta)
        if min_age is not None:
            info.update(top_old_slot=None, top_old_age=-1)
        if mem_time < 0:
            return np.zeros(self.d), info
        qkey = None
        if self.arrivals:
            qkey = self.key_for_slot(t)                   # q_t = k_t, the slot's own key
            if qkey is None:
                return np.zeros(self.d), info             # key undefined: r = 0, info empty
            qkey = qkey[None]
        q = self.params._t(qin).reshape(1, -1)
        if self.flags.flat:
            mask = torch.zeros(1, self.KF, dtype=torch.bool, device=dev)
            mask[0, :self.F_fill] = True
            if self.arrivals:
                mask &= torch.as_tensor(self.F_keyed, device=dev)[None]
            Kf = self.F_key[None] if (self.key_centering or self.arrivals) else None
            levels = [(self.M_F()[None], mask, Kf)]
            tags = [("F", self.F_label, self.F_time)]
        else:
            levels, tags = self._levels(src)
        r, overall, present = attend(self.params, q, levels, self.flags.mean_pool,
                                     qkey=qkey, key_tau=self.key_tau)
        info["empty"] = not bool(present.any())
        if min_age is not None and not self.flags.flat and self.flags.use_ml \
                and not self.flags.mean_pool:
            ages = t - np.asarray(src["L_time"], np.int64)
            old = np.asarray(src["L_filled"], bool) & (ages >= int(min_age))
            if old.any():
                idx = np.flatnonzero(old)
                if self.arrivals:
                    sc = cos_rows(src["L_key"][idx], qkey[0]).double().cpu().numpy()
                else:
                    sc = overall[1][0].double().cpu().numpy()[idx]
                j = int(idx[int(np.argmax(sc))])
                info.update(top_old_slot=j, top_old_age=int(ages[j]))
        r_np = r[0].double().cpu().numpy()
        info["l1"] = float(np.abs(r_np).sum())
        if not info["empty"] and not self.flags.mean_pool:
            best, bl, bk = -1.0, -1, -1
            for li, w in enumerate(overall):
                if w is None or not bool(present[0, li]):
                    continue
                k = int(torch.argmax(w[0]))
                if float(w[0, k]) > best:
                    best, bl, bk = float(w[0, k]), li, k
            lvl, labels, times = tags[bl]
            info.update(top_label=int(labels[bk]), top_age=int(t - times[bk]), top_level=lvl)
            if lvl == "L":
                info["used_slot"] = bk
                if update_lru and not self.flags.displaced:
                    self.L_use[bk] = t
        return r_np, info

    # ------------------------------------------------------------- checkpoint
    def get_state(self) -> dict:
        snaps = {int(s): {k: (_np(v) if isinstance(v, torch.Tensor) else
                              (v.copy() if isinstance(v, np.ndarray) else v))
                          for k, v in d.items()} for s, d in self.snapshots.items()}
        return dict(
            S_raw=_np(self.S_raw), S_fill=self.S_fill, S_ptr=self.S_ptr,
            S_label=self.S_label.copy(), S_time=self.S_time.copy(),
            S_key=_np(self.S_key), S_keyed=self.S_keyed.copy(),           # "arrivals" slot keys
            L_raw=_np(self.L_raw), L_key=_np(self.L_key), L_filled=self.L_filled.copy(), L_label=self.L_label.copy(),
            L_time=self.L_time.copy(), L_use=self.L_use.copy(),
            F_raw=_np(self.F_raw), F_fill=self.F_fill, F_ptr=self.F_ptr, F_count=self.F_count,
            F_label=self.F_label.copy(), F_time=self.F_time.copy(), F_kind=self.F_kind.copy(),
            F_key=_np(self.F_key), F_keyed=self.F_keyed.copy(),
            mu_S=self.mu_S.detach().cpu().numpy().copy(), mu_c=float(self.mu_c), mu_n=int(self.mu_n),
            t0=self.t0, t_now=self.t_now, n_events=self.n_events,
            delta=self.delta, delta_block=self.delta_block, snapshots=snaps,
            **({"pending": [dict(s=p["s"], key=_np(p["key"]), label=p["label"],
                                z=None if p["z"] is None else _np(p["z"]), p_ref=p["p_ref"], v_ref=p["v_ref"])
                           for p in self.pending]} if self.delayed else {}))         # A16 only: other arms' state unchanged

    def set_state(self, s: dict) -> None:
        dev = self.params.device
        T = lambda a: torch.as_tensor(np.array(a, dtype=np.float32), device=dev)  # noqa: E731
        self.S_raw, self.S_fill, self.S_ptr = T(s["S_raw"]), int(s["S_fill"]), int(s["S_ptr"])
        self.S_label, self.S_time = s["S_label"].copy(), s["S_time"].copy()
        self.S_key, self.S_keyed = T(s["S_key"]), s["S_keyed"].copy()
        self.L_raw, self.L_filled = T(s["L_raw"]), s["L_filled"].copy()
        self.L_key = T(s["L_key"])
        self.L_label, self.L_time, self.L_use = s["L_label"].copy(), s["L_time"].copy(), s["L_use"].copy()
        self.F_raw, self.F_fill, self.F_ptr = T(s["F_raw"]), int(s["F_fill"]), int(s["F_ptr"])
        self.F_count = int(s["F_count"])
        self.F_label, self.F_time, self.F_kind = s["F_label"].copy(), s["F_time"].copy(), s["F_kind"].copy()
        self.F_key, self.F_keyed = T(s["F_key"]), s["F_keyed"].copy()
        self.mu_S = torch.as_tensor(np.array(s["mu_S"], dtype=np.float64), device=dev)
        self.mu_c, self.mu_n = float(s["mu_c"]), int(s["mu_n"])
        self.t0, self.t_now, self.n_events = s["t0"], s["t_now"], int(s["n_events"])
        self.delta, self.delta_block = s["delta"], s["delta_block"]
        self.pending = [dict(s=int(p["s"]), key=T(p["key"]), label=int(p["label"]),
                             z=None if p["z"] is None else T(p["z"]), p_ref=p["p_ref"], v_ref=p["v_ref"])
                        for p in s.get("pending", [])]                                 # A16
        self.snapshots = {}
        for t_, d in s["snapshots"].items():
            self.snapshots[int(t_)] = {k: (T(v) if k in ("S_raw", "L_raw", "L_key", "S_key") else
                                           (v.copy() if isinstance(v, np.ndarray) else v))
                                       for k, v in d.items()}
