"""Shared helpers for the environment tests."""
from __future__ import annotations

import hashlib

import numpy as np

from tpa.config import load_config
from tpa.env.simulator import Simulator
from tpa.env.world import build_world
from tpa.rng import Streams
from tpa.types import Assignment, Decision


def make_sim(seed=1, block="train", cfg=None, **kw):
    cfg = cfg or load_config()
    st = Streams(seed, block)
    world = build_world(cfg, st.world)
    return Simulator(cfg, world, st, **kw), cfg, world


def greedy_decision(obs, frac=0.25, max_assign=None):
    """Simple rate-greedy policy: each queued task to its best-rate UAV with
    f = min(residual, frac * cap_bound)."""
    res = obs.residual_f.copy()
    out = []
    for i in range(len(obs.queue)):
        if max_assign is not None and len(out) >= max_assign:
            break
        j = int(np.argmax(obs.rates_pred[i]))
        f = min(res[j], frac * obs.cap_bound[j])
        if f <= 0:
            continue
        res[j] -= f
        out.append(Assignment(task_index=i, uav=j, f=f, t_total_pred=0.0))
    return Decision(assignments=out)


def hash_slot(sim, h):
    h.update(np.ascontiguousarray(sim.xi).tobytes())
    for k in sim.last_arrivals:
        h.update(np.array([k.tid, k.device, k.zone, k.t_arr, k.size_bits, k.cycles, k.deadline_s,
                           k.priority, k.complexity], dtype=np.float64).tobytes())
        h.update(k.loss_u.tobytes())


def run_hash(sim, policy, T, t0=0):
    h_env = hashlib.sha256()
    for t in range(t0, t0 + T):
        obs = sim.begin_slot(t)
        hash_slot(sim, h_env)
        sim.end_slot(policy(obs), 0.0)
    return h_env.hexdigest()
