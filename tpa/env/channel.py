"""Communication model, Eqs. (2) and (4)-(9). Pure functions on numpy float64.

Conventions
-----------
* h, active, rates are (N, M): device i, UAV j.
* `active[k, l]` is x_kl: device k has an upload IN PROGRESS to UAV l, i.e. (k, l) is in
  the interferer set the caller passes. Under cfg.channel.interference_set = "in_progress"
  the simulator passes the uploads in progress (for predictions, the set carried over from
  the previous slot); under "slot_assigned" (the default) it passes the uploads assigned in
  the current slot (for predictions, those assigned earlier in the slot).
* cfg.channel.access = "shared" (`rate`): all devices share the full band B (no OFDMA), Eq. 7
  literally; "fdma_per_uav" (the default, `rate_fdma`): each UAV cell has its own band B, split
  equally among its uploads.
"""
from __future__ import annotations

import numpy as np


def los_probability(elev_deg, d_ground, cfg):
    """Eq. (6): P_LoS = 1/(1 + a exp(-b(theta - a))) * exp(-omega d_ground).
    theta in degrees (Al-Hourani)."""
    c = cfg.channel
    sig = 1.0 / (1.0 + c.los_a * np.exp(-c.los_b * (np.asarray(elev_deg) - c.los_a)))
    return sig * np.exp(-c.omega_atten_per_m * np.asarray(d_ground))


def env_factor(p_los, cfg):
    """Eq. (5): Psi = P_LoS psi_LoS + (1 - P_LoS) psi_NLoS."""
    c = cfg.channel
    return p_los * c.psi_los + (1.0 - p_los) * c.psi_nlos


def geometry(dev_pos: np.ndarray, uav_pos: np.ndarray, cfg):
    """Ground distance (N,M), squared 3-D distance (N,M) and elevation (deg)."""
    H = cfg.scenario.hover_altitude_m
    d_ground = np.linalg.norm(dev_pos[:, None, :] - uav_pos[None, :, :], axis=-1)
    dist_sq = d_ground ** 2 + H ** 2
    elev_deg = np.degrees(np.arctan2(H, d_ground))
    return d_ground, dist_sq, elev_deg


def channel_gain(dev_pos: np.ndarray, uav_pos: np.ndarray, world, xi: np.ndarray, cfg) -> np.ndarray:
    """Eq. (4): h_ij = beta0 (kappa_ij / (||p_j - q_i||^2 + H^2))^(alpha/2) xi_ij Psi_ij.

    kappa_ij sits inside the path-loss ratio literally and is the same
    K-factor used for the Rician fading xi (passed in, see draw_fading).
    alpha is one per world; kappa per link, fixed per world."""
    d_ground, dist_sq, elev = geometry(dev_pos, uav_pos, cfg)
    psi = env_factor(los_probability(elev, d_ground, cfg), cfg)
    return cfg.channel.beta0 * (world.kappa / dist_sq) ** (world.alpha / 2.0) * xi * psi


def draw_fading(rng: np.random.Generator, kappa: np.ndarray) -> np.ndarray:
    """Unit-mean Rician power gain xi_ij with K-factor kappa_ij.

    Always draws the FULL (N, M) matrix: exactly 2*N*M standard normals per
    call, whatever the queue holds (common random numbers)."""
    w = rng.standard_normal((2,) + kappa.shape)
    s = np.sqrt(kappa / (kappa + 1.0))
    sigma = np.sqrt(0.5 / (kappa + 1.0))
    return (s + sigma * w[0]) ** 2 + (sigma * w[1]) ** 2


def system_noise(n_active: np.ndarray, eta: np.ndarray, cfg) -> np.ndarray:
    """Eq. (8): zeta_j = zeta0 (1 - eta_j) + zeta1 sum_i x_ij.  (M,)"""
    c = cfg.channel
    return c.zeta0 * (1.0 - np.asarray(eta)) + c.zeta1 * np.asarray(n_active, dtype=float)


def interference(h: np.ndarray, ptx: np.ndarray, active: np.ndarray, cfg) -> np.ndarray:
    """Eq. (9): I_ij = sum_{k != i} P_k h_kj x_kj + sum_{l != j} sum_k phi_jl P_k h_kl x_kl.

    x = uploads in progress. For a pair (i,j) that is not active the
    first sum is over all active co-channel uploads (k = i contributes nothing
    since x_ij = 0), i.e. the interference i WOULD see if it were added.
    phi_jl is a single coupling constant."""
    x = np.asarray(active, dtype=float)
    p_h_x = ptx[:, None] * h * x                    # (N,M)
    s_j = p_h_x.sum(axis=0)                         # (M,) received power per UAV
    co = s_j[None, :] - p_h_x                       # k != i
    cross = cfg.channel.phi_jl * (s_j.sum() - s_j)  # (M,) sum over l != j, all k
    return co + cross[None, :]


def rate(h: np.ndarray, ptx: np.ndarray, active: np.ndarray, eta: np.ndarray, cfg) -> np.ndarray:
    """Eq. (7): R_ij = eta_j B log2(1 + P_i h_ij / (N0 B + I_ij + zeta_j))  (bit/s, (N,M)).

    For active pairs zeta_j counts the n_j uploads in progress (Eq. 8). For a
    pair not in `active` the rate is the one it would get if added to the
    existing interferers, so its zeta_j counts n_j + 1."""
    c = cfg.channel
    act = np.asarray(active, dtype=bool)
    n_j = act.sum(axis=0)
    zeta_act = system_noise(n_j, eta, cfg)[None, :]
    zeta_add = system_noise(n_j + 1, eta, cfg)[None, :]
    zeta = np.where(act, zeta_act, zeta_add)
    denom = c.noise_psd_w_hz * c.bandwidth_hz + interference(h, ptx, act, cfg) + zeta
    sinr = ptx[:, None] * h / denom
    return np.asarray(eta)[None, :] * c.bandwidth_hz * np.log2(1.0 + sinr)


def rate_fdma(h: np.ndarray, ptx: np.ndarray, active: np.ndarray, eta: np.ndarray, cfg) -> np.ndarray:
    """Per-UAV FDMA rate, a MODEL CHANGE (cfg.channel.access = "fdma_per_uav"; not Eqs. 7/9 literally).

    Each UAV cell j has its own band B, split equally among its n_j uploads:
        R_ij = eta_j (B/n_j) log2(1 + P_i h_ij / (N0 B/n_j + I^cross_ij + zeta_j)),
        I^cross_ij = phi_jl sum_{l != j} sum_k P_k h_kj x_kl / n_l   (foreign sub-band overlap),
    with no intra-cell co-channel term; zeta_j is Eq. 8. As in `rate`, a pair
    (i,j) not in `active` gets the rate it would have if added: n_j -> n_j + 1
    (bandwidth share and zeta_j). Units bit/s, (N,M)."""
    c = cfg.channel
    act = np.asarray(active, dtype=bool)
    x = act.astype(float)
    n_l = act.sum(axis=0).astype(float)                      # uploads per cell
    n_eff = np.where(act, n_l[None, :], n_l[None, :] + 1.0)  # (N,M) share if (i,j) joins / is in
    inv_n = np.divide(1.0, n_l, out=np.zeros_like(n_l), where=n_l > 0)
    # power received at UAV j from uploads of other cells l, scaled by 1/n_l
    p_kj = ptx[:, None] * h                                   # (N,M): P_k h_kj
    w_kl = x * inv_n[None, :]                                 # (N,M): x_kl / n_l
    full = p_kj.T @ w_kl                                      # (M,M): [j,l] = sum_k P_k h_kj x_kl/n_l
    cross = c.phi_jl * (full.sum(axis=1) - np.diag(full))     # (M,) sum over l != j
    zeta = system_noise(n_eff, np.broadcast_to(eta, n_eff.shape), cfg)
    bw = c.bandwidth_hz / n_eff
    sinr = ptx[:, None] * h / (c.noise_psd_w_hz * bw + cross[None, :] + zeta)
    return np.asarray(eta)[None, :] * bw * np.log2(1.0 + sinr)


def rate_fdma_rows(h, ptx, active, eta, cfg, rows, p_kj=None):
    """`rate_fdma` evaluated for the device rows `rows` only, (len(rows), M).

    The interferer-dependent quantities that need every device (n_l, w_kl, the (M, M)
    cross-cell matrix `full` and `cross`) are computed exactly as in `rate_fdma` (same
    expressions, same BLAS call on the same shapes); the remaining operations are
    elementwise, so restricting them to `rows` gives the same float64 values as the
    full matrix's rows. `p_kj` = ptx[:, None] * h may be passed precomputed for the slot
    (h is fixed within a slot). Bit-identity with `rate_fdma` is asserted by
    tests/test_d8_equivalence.py."""
    c = cfg.channel
    act = np.asarray(active, dtype=bool)
    x = act.astype(float)
    n_l = act.sum(axis=0).astype(float)                      # uploads per cell
    inv_n = np.divide(1.0, n_l, out=np.zeros_like(n_l), where=n_l > 0)
    if p_kj is None:
        p_kj = ptx[:, None] * h                               # (N,M): P_k h_kj
    w_kl = x * inv_n[None, :]                                 # (N,M): x_kl / n_l
    full = p_kj.T @ w_kl                                      # (M,M): [j,l] = sum_k P_k h_kj x_kl/n_l
    cross = c.phi_jl * (full.sum(axis=1) - np.diag(full))     # (M,) sum over l != j
    rows = np.asarray(rows, dtype=np.intp)
    act_r = act[rows]
    n_eff = np.where(act_r, n_l[None, :], n_l[None, :] + 1.0)
    zeta = system_noise(n_eff, np.asarray(eta)[None, :], cfg)   # same elementwise values as the broadcast_to form
    bw = c.bandwidth_hz / n_eff
    sinr = ptx[rows][:, None] * h[rows] / (c.noise_psd_w_hz * bw + cross[None, :] + zeta)
    return np.asarray(eta)[None, :] * bw * np.log2(1.0 + sinr)


class FdmaScan:
    """Slot-assigned FDMA rate prediction for one interferer prefix
    `act` (N, M) and its single-candidate extensions act | {(d, j)}, bit-identical to
    `rate_fdma(h, ptx, act | {(d, j)}, eta, cfg)` at the entries read.

    The prefix quantities (x, n_l, inv_n, w_kl) are rate_fdma's own expressions, computed
    once per prefix. A candidate (d, j) changes n_l[j] (one more upload in cell j) and
    column j of w_kl only; `full` is then recomputed with the same (M, N) @ (N, M) dgemm
    call on a w_kl whose values are exactly those rate_fdma would form (the other columns
    are the prefix's own; column j = x_j with x[d, j] = 1 times 1/n_l[j]). With some BLAS
    builds a dgemm column depends on the matrix width, so one dgemm per candidate is the
    floor for bit-identity: nothing is batched across candidates. The remaining operations
    (cross, n_eff, zeta, bw, sinr, log2) are elementwise and are evaluated at the requested
    entries only, in rate_fdma's association order."""
    __slots__ = ("h", "ptx", "eta", "p_kj", "zeta0", "zeta1", "phi", "B", "N0",
                 "act", "x", "n_l", "inv_n", "w", "_full0")

    def __init__(self, h, ptx, act, eta, cfg, p_kj=None):
        c = cfg.channel
        self.h, self.ptx, self.eta = h, ptx, np.asarray(eta, dtype=np.float64)
        self.p_kj = ptx[:, None] * h if p_kj is None else p_kj
        self.zeta0, self.zeta1 = float(c.zeta0), float(c.zeta1)
        self.phi, self.B, self.N0 = float(c.phi_jl), float(c.bandwidth_hz), float(c.noise_psd_w_hz)
        act = np.asarray(act, dtype=bool)
        self.act = act
        self.x = act.astype(float)
        self.n_l = act.sum(axis=0).astype(float)                      # uploads per cell
        self.inv_n = np.divide(1.0, self.n_l, out=np.zeros_like(self.n_l), where=self.n_l > 0)
        self.w = self.x * self.inv_n[None, :]                         # (N,M): x_kl / n_l
        self._full0 = None

    def full0(self) -> np.ndarray:
        if self._full0 is None:
            self._full0 = self.p_kj.T @ self.w                        # (M,M), prefix itself
        return self._full0

    def entries(self, d: int, j: int, devs, uavs) -> np.ndarray:
        """Rates (len(devs),) at entries (devs[k], uavs[k]) under the prefix with (d, j) added."""
        devs = np.asarray(devs, dtype=np.intp)
        uavs = np.asarray(uavs, dtype=np.intp)
        if self.act[d, j]:                                            # already an interferer: the prefix
            full, n_l = self.full0(), self.n_l
            act_e = self.act[devs, uavs]
        else:
            n_l = self.n_l.copy()
            n_l[j] = self.n_l[j] + 1.0
            inv_j = 1.0 / n_l[j]
            xc = self.x[:, j].copy()
            xc[d] = 1.0
            w = self.w.copy()
            w[:, j] = xc * inv_j
            full = self.p_kj.T @ w
            act_e = self.act[devs, uavs] | ((devs == d) & (uavs == j))
        cross = self.phi * (full.sum(axis=1) - np.diag(full))         # (M,) sum over l != j
        n_eff = np.where(act_e, n_l[uavs], n_l[uavs] + 1.0)
        zeta = self.zeta0 * (1.0 - self.eta[uavs]) + self.zeta1 * n_eff          # system_noise
        bw = self.B / n_eff
        sinr = self.ptx[devs] * self.h[devs, uavs] / (self.N0 * bw + cross[uavs] + zeta)
        return self.eta[uavs] * bw * np.log2(1.0 + sinr)

    def rows(self, rows) -> np.ndarray:
        """rate_fdma rows of the prefix itself, (len(rows), M)."""
        rows = np.asarray(rows, dtype=np.intp)
        full = self.full0()
        cross = self.phi * (full.sum(axis=1) - np.diag(full))
        act_r = self.act[rows]
        n_eff = np.where(act_r, self.n_l[None, :], self.n_l[None, :] + 1.0)
        zeta = self.zeta0 * (1.0 - self.eta)[None, :] + self.zeta1 * n_eff
        bw = self.B / n_eff
        sinr = self.ptx[rows][:, None] * self.h[rows] / (self.N0 * bw + cross[None, :] + zeta)
        return self.eta[None, :] * bw * np.log2(1.0 + sinr)


def rate_access(h, ptx, active, eta, cfg):
    """Dispatch on cfg.channel.access: "shared" -> Eq. 7 (`rate`); "fdma_per_uav" -> `rate_fdma`."""
    acc = str(cfg.channel.access)
    if acc == "shared":
        return rate(h, ptx, active, eta, cfg)
    if acc == "fdma_per_uav":
        return rate_fdma(h, ptx, active, eta, cfg)
    raise ValueError(f"channel.access must be 'shared' or 'fdma_per_uav', got {acc!r}")


def h_min_value(world, cfg) -> float:
    """h_min of Eq. (2) per cfg.channel.h_min_rule (derived once per world):
    * "fixed":      cfg.channel.h_min
    * "from_r_min": h_min = (2^(R_min/B) - 1) N0 B / P_max, the gain at which the
                    world's max-power device reaches R_min with no interference."""
    c = cfg.channel
    rule = str(c.h_min_rule)
    if rule == "fixed":
        return float(c.h_min)
    if rule == "from_r_min":
        r_min = float(c.r_min_mbps) * 1e6                  # Mb/s -> bit/s
        return float((2.0 ** (r_min / c.bandwidth_hz) - 1.0) * c.noise_psd_w_hz * c.bandwidth_hz
                     / float(np.max(world.device_ptx)))
    raise ValueError(f"channel.h_min_rule must be 'fixed' or 'from_r_min', got {rule!r}")


def viable(h: np.ndarray, cfg, h_min: float | None = None) -> np.ndarray:
    """Eq. (2): link (i,j) is in E(t) iff h_ij >= h_min.  bool (N,M).
    Pass the world's h_min (Simulator.h_min / h_min_value) under "from_r_min";
    the default is cfg.channel.h_min (the "fixed" rule)."""
    return h >= (cfg.channel.h_min if h_min is None else h_min)


# alias named after the viable link set E(t) of Eq. (2)
viable_links = viable


def r_ref(world, cfg) -> float:
    """R_ref: per-world median over devices of the best-link rate
    without fading (xi = 1) and without interference, from the initial geometry."""
    n, m = world.kappa.shape
    h = channel_gain(world.device_pos, world.uav_pos, world, np.ones((n, m)), cfg)
    r = rate_access(h, world.device_ptx, np.zeros((n, m), dtype=bool), world.eta, cfg)
    return float(np.median(r.max(axis=1)))
