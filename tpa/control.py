"""TPA controller: the per-slot pipeline (order below)
for arms A0-A9, A13-A16 and A16P (the memory-free arm A9 shares it with memory switched off).

Per slot (act):  S(t) -> x_t -> e_t -> write M_S -> MEPR (P, v) -> candidate z_t / key
                 -> consolidate (if gate open) -> mu update -> retrieve r(t)
                 -> priorities -> top-32 window -> actor -> DRM (+ burst for t+1)
                 (the store also records the UAV occupancy o_t = obs.uav_occupancy; zeros if absent)
Per slot (observe): J(t) -> normalise -> Eq. 59 weights -> reward -> replay -> SAC update.

A transition (s_t, a_t, r_t, s_{t+1}) is pushed when s_{t+1} has been assembled in the
next act(); the last transition of an episode is not stored (see replay.py).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from .arms import arm
from .drm.drm import DRM
from .learner.networks import Actor, TwinCritic
from .learner.objectives import AdaptiveWeights, Normalizer, reward, slot_objectives
from .learner.replay import Replay
from .learner.sac import SAC
from .memory import MEPR, MemoryParams, SegmentStore, TwoLevelMemory, flags_for_arm
from .memory.recon import reconstruct
from .run.timing import ModuleTimer


def task_rows(obs, idx, cfg, r_ref: float) -> np.ndarray:
    """Task features for queue indices `idx`, shape (len(idx), 5 + Z + M + 1):
    [size/5MB, cycles/1e9, slack/T_max, rho, phi/1.5, zone one-hot, R_pred/R_ref (M), mask]."""
    Z, M = cfg.scenario.n_zones, cfg.scenario.n_uav
    out = np.zeros((len(idx), 5 + Z + M + 1))
    rr = r_ref if r_ref > 0 else 1.0
    for row, i in enumerate(idx):
        tk = obs.queue[i]
        slack = tk.deadline_s - (obs.t - tk.t_arr) * cfg.scenario.slot_s
        out[row, 0] = tk.size_bits / (8e6 * cfg.task.size_mb[1])
        out[row, 1] = tk.cycles / 1e9
        out[row, 2] = slack / tk.deadline_s
        out[row, 3] = tk.priority
        out[row, 4] = tk.complexity / cfg.task.complexity[1]
        out[row, 5 + tk.zone] = 1.0
        out[row, 5 + Z:5 + Z + M] = obs.rates_pred[i] / rr
        out[row, -1] = 1.0
    return out


def task_features(obs, idx, cfg, r_ref: float) -> np.ndarray:
    """Actor window tau(t): rows for the top-q_max tasks `idx`, zero-padded to q_max."""
    q = cfg.task.q_max
    tf = np.zeros((q, 5 + cfg.scenario.n_zones + cfg.scenario.n_uav + 1))
    rows = task_rows(obs, list(idx)[:q], cfg, r_ref)
    tf[: len(rows)] = rows
    return tf


def queue_summary(obs, cfg, r_ref: float) -> np.ndarray:
    """tau_bar(t): mean and max of the task features over the whole queue."""
    d = 5 + cfg.scenario.n_zones + cfg.scenario.n_uav + 1
    if not obs.queue:
        return np.zeros(2 * d)
    rows = task_rows(obs, list(range(len(obs.queue))), cfg, r_ref)
    return np.concatenate([rows.mean(axis=0), rows.max(axis=0)])


class TPAController:
    """Controller protocol: reset_segment, start_episode, act, observe, eval_clone,
    get_state / set_state."""

    def __init__(self, cfg, arm_key: str, streams, r_ref: float, h_min: float,
                 device: str = "cpu", timer: ModuleTimer | None = None, _shared=None):
        self.cfg, self.arm_key, self.streams = cfg, arm_key, streams
        self.flags = arm(arm_key)
        if self.flags.controller != "tpa":
            raise ValueError(f"{arm_key} is not a TPA-pipeline arm")
        self.mflags = flags_for_arm(arm_key)
        self.has_mem = self.mflags is not None
        self.r_ref, self.device = float(r_ref), device
        self.timer = timer or ModuleTimer()
        L = cfg.learner
        if _shared is None:                                   # networks: seeded, no global RNG
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(streams.torch_seed())
                actor = Actor(cfg, self.has_mem)
                critics = TwinCritic(cfg, self.has_mem)
            # Updates may run on the GPU (device="cuda"); per-slot decisions always run on
            # a CPU mirror (single-sample GPU inference is launch-bound).
            self.params_learn = MemoryParams(cfg, seed=streams.torch_seed(), device=device) \
                if self.has_mem else None
            if self.has_mem and str(device) != "cpu":
                self.params = MemoryParams(cfg, seed=streams.torch_seed(), device="cpu")
                self.params.load_state_dict({k: v.cpu() for k, v in
                                             self.params_learn.state_dict().items()})
            else:
                self.params = self.params_learn
            self.sac = SAC(cfg, actor, critics, self.params_learn, device=device, rng=streams.policy)
            self.replay = Replay(L.replay, streams.replay)
            self.train_mode = True
        else:                                                 # evaluation clone
            self.sac, self.params = _shared
            self.params_learn = self.params
            self.replay = None
            self.train_mode = False
        drm_flags = self.flags if self.flags.beta_c is not None else             replace(self.flags, beta_c=float(cfg.drm.beta_c))     # None = use config value
        self.drm = DRM(cfg, drm_flags, arm_key)
        self.drm.h_min = float(h_min)                         # same Eq. 2 threshold as the simulator
        self.norm = Normalizer(cfg)
        self.weights = AdaptiveWeights(cfg, fixed=not self.flags.adaptive_w)
        if self.has_mem:
            self.mem = TwoLevelMemory(cfg, self.mflags, self.params,
                                      rng_a14=streams.a14 if self.mflags.displaced else None,
                                      rng_a15=streams.a15 if self.mflags.key_shuffled else None)
            self.mepr = MEPR(cfg, self.mflags)
            if getattr(self.mflags, "frozen_content", False) and self.params_learn is not self.params:
                for p in self.params_learn.W_v.parameters():      # A16': freeze the learner copy too
                    p.requires_grad_(False)
        else:
            self.mem = self.mepr = None
        self.stores: dict[int, SegmentStore] = {}
        self.r_log: dict[int, dict[int, np.ndarray]] = {}   # rollout r(t) per segment (recon_neighbors = "stored")
        self.seg_id = -1
        self.store = None
        self.slots_seen = 0                                  # global training slot counter
        self._pending = None
        self._cur = None
        self.r_prev = None
        self.last = {}

    # ------------------------------------------------------------ lifecycle
    def reset_segment(self, t0: int) -> None:
        """Start of a memory segment / test run: stores, P_ref and MEPR state empty."""
        self.seg_id += 1
        self.drm.reset()
        if self.has_mem:
            self.mem.reset(t0)
            self.mepr.reset()
            self.store = SegmentStore(self.cfg, self.mflags)
            self.mem.attach_store(self.store)
            self.stores[self.seg_id] = self.store
            self.r_log[self.seg_id] = {}
            if self.replay is not None:
                live = self.replay.live_segments() | {self.seg_id}
                for k in [k for k in self.stores if k not in live]:
                    del self.stores[k]
                for k in [k for k in self.r_log if k not in live]:
                    del self.r_log[k]
            else:
                self.stores = {self.seg_id: self.store}
                self.r_log = {self.seg_id: self.r_log[self.seg_id]}
        self.r_prev = None
        self._pending = None

    def start_episode(self) -> None:
        """Episode boundary inside a segment: queue flushed by the simulator; the last
        transition of the previous episode is dropped; burst state restarts."""
        self._pending = None
        self.drm.burst.reset()

    # ------------------------------------------------------------ act
    def act(self, obs, label: int, train: bool):
        cfg, T = self.cfg, self.timer
        t = obs.t
        S = obs.S_sys
        za, zm = obs.zone_arrivals, obs.zone_mean
        r = None
        sigma = 0.0
        P = None
        cons = False
        rinfo = {}
        if self.has_mem:
            qin = np.concatenate([S, queue_summary(obs, cfg, self.r_ref)])
            uo = getattr(obs, "uav_occupancy", None)              # UAV occupancy o_t (M,) = in-flight / q_max
            uo = np.zeros(cfg.scenario.n_uav) if uo is None else np.asarray(uo, dtype=np.float64)
            with T("mepr"):
                x = np.concatenate([qin, self.mem.mbar_short()])
                self.store.record(t, x, qin, za, zm, uav_occ=uo)
                with torch.no_grad():
                    e = self.params.encode(x)
                self.mem.write_short(e, label, t)
                P, sigma, cons = self.mepr.step_buffered(e, za, zm, uav_occ=uo)
                # Ordering: z_t every slot it is defined -> key k_t = z_t - mu_{t-1}
                # (inside consolidate) -> Eq. 45 write if the gate is open -> mu_t.
                # The event log carries P_ref / v_ref for the "novelty" candidate.
                # key_source = "arrivals": the slot key is mem.key_for_slot(t), computed
                # once from the za / zone_mean recorded above (store.record precedes every
                # write) and used by write_short, consolidate and retrieve; same ordering.
                with torch.no_grad():
                    z = self.mem.candidate_for(self.mepr)
                    if cons:
                        self.mem.consolidate(z, label, t, p_ref=self.mepr.P_ref_used,
                                             v_ref=self.mepr.v_ref_used)
                    if z is not None:
                        self.mem.observe_candidate(z)
            with T("retrieval"):
                r, rinfo = self.mem.retrieve(qin, t)
                r = np.asarray(r, dtype=np.float64)
                self.r_log[self.seg_id][int(t)] = r.astype(np.float32)
            P = np.asarray(P, dtype=np.float64) if P is not None else None
        with T("drm"):
            pi = self.drm.priority(obs, P, sigma)
            win = self.drm.select_window(pi, obs)
            tf = task_features(obs, win, cfg, self.r_ref)
        ref = (self.seg_id, int(t))
        # complete the previous transition now that s_{t+1} is known
        if self._pending is not None and train and self.replay is not None:
            p = self._pending
            self.replay.push(dict(S_sys=p["S"], tf=p["tf"], a=p["a"], reward=p["reward"],
                                  sigma=p["sigma"], ref=p["ref"],
                                  S_sys2=S, tf2=tf, sigma2=sigma, ref2=ref))
        self._pending = None
        with T("actor"):
            if train and self.slots_seen < cfg.learner.warmup_slots:
                a = self.sac.random_action()
            else:
                a = self.sac.act(S, tf, r, np.array([sigma]) if self.has_mem else None,
                                 deterministic=not train)
        with T("drm"):
            P_prev = self.mepr.P_prev if self.has_mem else None
            P_ref = self.mepr.P_ref if self.has_mem else None
            dec = self.drm.run(obs, win, a, P, sigma, P_prev=P_prev, P_ref=P_ref)
            dec.burst_budget = np.asarray(dec.util_terms.get("burst_budget_frac",
                                                             np.zeros(cfg.scenario.n_uav)))
        l1 = float(np.abs(r).sum()) if r is not None else 0.0
        mem_energy = cfg.uav.kappa_m * l1
        if self.has_mem:
            fs, fl = self.mem.fro_norms()
            dr = float(np.linalg.norm(r - self.r_prev)) if self.r_prev is not None else 0.0
            mem_stats = dict(l1=l1, fro_s=fs, fro_l=fl, dr_norm=dr)
            self.r_prev = r.copy()
        else:
            mem_stats = None
        self._cur = dict(S=S, tf=tf, a=a, sigma=sigma, ref=ref, mem_stats=mem_stats)
        self.last = dict(rinfo=rinfo, sigma=sigma, cons=bool(self.has_mem and cons),
                         P=P, n_window=len(win))
        return dec, mem_energy

    # ------------------------------------------------------------ observe
    def observe(self, obs, dec, outcome, train: bool):
        cfg = self.cfg
        cur = self._cur
        J = slot_objectives(dec, outcome, cur["mem_stats"], cfg)
        if train:
            self.norm.observe(J)
        Jn = self.norm.normalize(J)
        w = self.weights.step(Jn, cur["sigma"])
        rew = reward(Jn, w)
        info = dict(J=J, reward=rew)
        if train:
            self._pending = dict(cur, reward=rew)
            self.slots_seen += 1
            U = int(cfg.learner.update_every)
            if (self.slots_seen >= cfg.learner.warmup_slots and self.slots_seen % U == 0
                    and len(self.replay) >= cfg.learner.batch):
                with self.timer("learner"):
                    batch = self.replay.sample(cfg.learner.batch, device=self.device)
                    info["update"] = self.sac.update(batch, self.recon_fn if self.has_mem else None)
                    if self.has_mem and self.params is not self.params_learn:
                        self.params.load_state_dict({k: v.detach().cpu() for k, v in
                                                     self.params_learn.state_dict().items()})
        return info

    def mem_fill(self) -> float:
        """Filled memory slots (S + L, or the flat store under A3); 0 without memory."""
        if not self.has_mem:
            return 0.0
        m = self.mem
        if self.mflags.flat:
            return float(m.F_fill)
        return float(m.S_fill + int(m.L_filled.sum()))

    def lt_occupancy(self) -> int:
        return int(self.mem.L_filled.sum()) if self.has_mem and not self.mflags.flat else 0

    # ------------------------------------------------------------ replay reconstruction
    def recon_fn(self, refs, grad: bool):
        """Rebuild r(t) for (segment_id, t) refs with the current memory params.
        t before a segment's first slot (the t-1 of L_mem) is clamped to that first slot."""
        mapped = []
        for s, t in refs:
            st = self.stores[s]
            mapped.append((st, max(int(t), int(st.t0))))
        if not grad and str(self.cfg.learner.recon_neighbors) == "stored":
            # r(t+1) (TD target) and r(t-1) (L_mem) from the stored rollout retrievals
            rs = [self.r_log[s][t] for (s, _), (_, t) in zip(refs, mapped)]
            return torch.from_numpy(np.stack(rs)).to(self.params_learn.device)
        return reconstruct(self.params_learn, self.cfg, self.mflags, mapped, grad=grad)

    # ------------------------------------------------------------ evaluation
    def eval_clone(self, streams) -> "TPAController":
        """Frozen-policy controller sharing networks and memory params; own memory,
        MEPR, DRM, frozen normaliser copy, fresh Eq. 59 EWM; no replay, no updates."""
        c = TPAController(self.cfg, self.arm_key, streams, self.r_ref, self.drm.h_min,
                          device=self.device, timer=ModuleTimer(),
                          _shared=(self.sac, self.params))
        c.norm.set_state(self.norm.get_state())
        return c

    def learned_state_keys(self) -> tuple[str, ...]:
        """The get_state() keys eval_clone takes from the trained controller
        (networks, memory parameters when the arm has memory, frozen normaliser); the best and
        final checkpoints hold exactly these (tpa.run.trainer._net_state)."""
        return ("sac", "params", "norm") if self.has_mem else ("sac", "norm")

    # ------------------------------------------------------------ state
    def get_state(self) -> dict:
        st = dict(sac=self.sac.get_state(), replay=self.replay.get_state() if self.replay else None,
                  drm=self.drm.get_state(), norm=self.norm.get_state(),
                  weights=self.weights.get_state(), seg_id=self.seg_id,
                  slots_seen=self.slots_seen, pending=self._pending,
                  r_prev=None if self.r_prev is None else self.r_prev.copy())
        if self.has_mem:
            st.update(params=self.params_learn.state_dict(), r_log=self.r_log,
                      mem=self.mem.get_state(),
                      mepr=self.mepr.get_state(),
                      stores={k: v.get_state() for k, v in self.stores.items()})
        return st

    def set_state(self, st: dict) -> None:
        self.sac.set_state(st["sac"])
        if self.replay is not None and st["replay"] is not None:
            self.replay.set_state(st["replay"])
        self.drm.set_state(st["drm"])
        self.norm.set_state(st["norm"])
        self.weights.set_state(st["weights"])
        self.seg_id = st["seg_id"]
        self.slots_seen = st["slots_seen"]
        self.r_prev = st["r_prev"]
        self._pending = st.get("pending")
        if self.has_mem:
            self.params_learn.load_state_dict(st["params"])
            if self.params is not self.params_learn:
                self.params.load_state_dict({k: v.cpu() for k, v in st["params"].items()})
            self.r_log = {k: dict(v) for k, v in st.get("r_log", {}).items()}
            self.stores = {}
            for k, s in st["stores"].items():
                store = SegmentStore(self.cfg, self.mflags)
                store.set_state(s)
                self.stores[k] = store
            self.store = self.stores.get(self.seg_id)
            self.mem.set_state(st["mem"])
            if self.store is not None:
                self.mem.attach_store(self.store)
            self.mepr.set_state(st["mepr"])
