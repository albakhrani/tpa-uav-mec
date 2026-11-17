"""Training run entry point: tpa.run.trainer.run(row, cfg) (called by scripts/run_queue.py).

Protocol:
  * cfg.run.episodes episodes of cfg.run.t_ep slots on one continuous training trace.
  * Memory segments of cfg.run.segment_slots: at each segment start the controller's
    memory/MEPR/DRM state is reset (as at the start of a test run) and the load is
    constant within the segment, cycling light -> medium -> heavy.
  * Episode boundary: queue flushed, batteries reset, last transition dropped.
  * Every cfg.run.val_every_episodes episodes: frozen evaluation on the same seed's validation block;
    the best checkpoint by validation completion is kept (selection never sees test).
  * Every cfg.run.ckpt_every_episodes episodes: atomic rolling checkpoint with ALL state (networks,
    optimisers, replay, memory stores, simulator, every RNG stream, logs) so an
    interrupted run resumes with bitwise-identical results on CPU.
  * At the end: final + best checkpoints, then test evaluation of the best checkpoint.

row keys used: run_id, arm, seed, workload, run_dir, db_path (optional), device,
and optional 'env' (dict of Simulator kwargs: mobility, patrol, packet_loss),
'evaluate' (bool, default True), 'episodes' (override for pilots/tests).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..env.simulator import Simulator
from ..env.world import build_world
from ..rng import Streams
from . import checkpoint as ck
from .evaluator import evaluate, validate
from .logio import ParquetLog
from .manifest import write_manifest
from .metrics import SlotAccumulator
from .timing import ModuleTimer


_CUR_EP = [-1]          # episode currently being simulated (-1 = validation/test); test aid


class LostClaim(RuntimeError):
    """This worker no longer holds the run's claim; another worker may own the directory."""


class SimulatedInterrupt(RuntimeError):
    """Raised by the resume test hook right after a rolling checkpoint."""


def make_controller(cfg, arm_key, streams, r_ref, h_min, device="cpu", timer=None):
    try:
        from ..factory import make_controller as mk      # controllers of all arms
    except ImportError:
        mk = None
    if mk is not None:
        return mk(cfg, arm_key, streams, r_ref, h_min, device=device, timer=timer)
    from ..control import TPAController
    return TPAController(cfg, arm_key, streams, r_ref, h_min, device=device, timer=timer)


def parse_workload(spec: str) -> tuple[str, dict]:
    """'kind|key=value|...' -> (kind, Simulator kwargs). Keys: mobility, patrol, packet_loss
    (robustness runs are enqueued through the workload column)."""
    parts = str(spec).split("|")
    kw = {}
    for p in parts[1:]:
        k, v = p.split("=", 1)
        if k == "mobility":
            kw[k] = v
        elif k == "patrol":
            kw[k] = v.lower() in ("1", "true", "yes")
        elif k == "packet_loss":
            kw[k] = float(v)
        else:
            raise ValueError(f"unknown workload option {k!r} in {spec!r}")
    return parts[0], kw


def _segment_of(slot: int, cfg) -> tuple[int, int, float]:
    L = int(cfg.run.segment_slots)
    seg = slot // L
    loads = list(cfg.workload.loads)
    return seg, seg * L, float(loads[seg % len(loads)])


def _ep_row(acc_row: dict, ep: int, ctrl, wall_s: float, upd: list, timer: ModuleTimer) -> dict:
    r = dict(acc_row)
    r.update(episode=ep, wall_s=wall_s, n_updates=len(upd),
             alpha=float(np.mean([u["alpha"] for u in upd])) if upd else float("nan"),
             critic_loss=float(np.mean([u["critic_loss"] for u in upd])) if upd else float("nan"),
             actor_loss=float(np.mean([u["actor_loss"] for u in upd])) if upd else float("nan"),
             l_mem=float(np.mean([u.get("l_mem", 0.0) for u in upd])) if upd else float("nan"),
             lt_occupancy=ctrl.lt_occupancy() if hasattr(ctrl, "lt_occupancy") else 0)
    for k, v in timer.summary().items():
        r[f"t_{k}_ms"] = v["mean_ms"]
    return r


def run(row: dict, cfg) -> dict:
    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    arm_key, seed = row["arm"], int(row["seed"])
    workload, env_kw = parse_workload(row.get("workload", "sinusoidal"))
    env_kw.update(row.get("env") or {})
    device = row.get("device", "cpu")
    R = cfg.run
    E = int(row.get("episodes") or R.episodes)
    T_ep = int(R.t_ep)
    torch.set_num_threads(1)
    if device == "cuda":
        torch.use_deterministic_algorithms(True)
    write_manifest(run_dir, cfg, extra=dict(row={k: v for k, v in row.items()
                                                 if isinstance(v, (str, int, float, bool, list, dict))}))

    streams = Streams(seed, "train")
    world = build_world(cfg, streams.world)
    timer = ModuleTimer()
    # the simulator of segment 0 supplies r_ref / h_min (world-level, identical for all segments)
    _, s0, load0 = _segment_of(0, cfg)
    sim = Simulator(cfg, world, streams, arm_key=arm_key, workload_kind=workload, load=load0,
                    t0=s0, horizon=int(R.segment_slots), **env_kw)
    ctrl = make_controller(cfg, arm_key, streams, sim.r_ref, sim.h_min, device=device, timer=timer)
    log_rows_dir = run_dir / "train"
    state = dict(ep=0, seg=-1, best=-np.inf, best_ep=-1, val=[])
    plog: ParquetLog | None = None
    pair_snap_resumed = None

    # ---------------- resume
    rp = ck.resume_path(run_dir)
    if rp is not None:
        s = ck.load(rp)
        state = s["state"]
        streams.set_state(s["streams"])
        seg, seg_start, load = _segment_of(state["ep"] * T_ep, cfg)
        sim = Simulator(cfg, world, streams, arm_key=arm_key, workload_kind=workload, load=load,
                        t0=seg_start, horizon=int(R.segment_slots), **env_kw)
        sim.set_state(s["sim"])
        streams.set_state(s["streams"])          # Simulator construction must not move streams
        ctrl.set_state(s["ctrl"])
        pair_snap_resumed = state.get("pair_snap")
        if s.get("log") is not None:
            plog = ParquetLog(log_rows_dir, schema=s["log"]["schema"])
            plog.set_state(s["log"]["state"])

    t_start = time.perf_counter()
    paired = bool(getattr(ctrl, "paired_episodes", False))
    pair_snap = pair_snap_resumed
    while state["ep"] < E:
        ep = state["ep"]
        slot0 = ep * T_ep
        if paired and ep % 2 == 1:
            slot0 = (ep - 1) * T_ep                             # replay the pair's trace window
        seg, seg_start, load = _segment_of(slot0, cfg)
        if seg != state["seg"]:                                   # new memory segment
            sim = Simulator(cfg, world, streams, arm_key=arm_key, workload_kind=workload,
                            load=load, t0=seg_start, horizon=int(R.segment_slots), **env_kw)
            ctrl.reset_segment(seg_start)
            state["seg"] = seg
        if paired:
            if ep % 2 == 0:                                     # snapshot env for the pair member
                if sim.next_t is not None and sim.next_t < slot0:
                    # the pair before this one replayed its even window, so the Simulator
                    # still expects the odd window's first slot; that window is skipped by
                    # design (12 episodes = one segment), so move on to this one
                    sim.next_t = slot0
                pair_snap = dict(sim=sim.get_state(), streams={k: v for k, v in
                                 streams.get_state().items() if k in _ENV_STREAMS})
            elif pair_snap is not None:
                sim.set_state(pair_snap["sim"])
                streams.set_state(pair_snap["streams"])
        sim.flush()
        sim.reset_battery()
        ctrl.start_episode()
        _CUR_EP[0] = ep
        acc = SlotAccumulator(cfg, getattr(ctrl, "has_mem", False), False)
        cap_nom = world.f_max * world.eta
        upd = []
        w0 = time.perf_counter()
        timer.reset()
        for t in range(slot0, slot0 + T_ep):
            with timer("env"):
                obs = sim.begin_slot(t)
            tb = sim.truth_phase_bin(t)
            dec, me = ctrl.act(obs, tb, train=True)
            with timer("env"):
                out = sim.end_slot(dec, mem_energy_j=me)
            info = ctrl.observe(obs, dec, out, train=True)
            if "update" in info:
                upd.append(info["update"])
            acc.add(t, obs, dec, out, getattr(ctrl, "last", {}), tb, info,
                    ctrl.mem_fill() if hasattr(ctrl, "mem_fill") else 0.0, cap_nom)
        er = _ep_row(acc.summary(), ep, ctrl, time.perf_counter() - w0, upd, timer)
        er.update(arm=arm_key, seed=seed, workload=workload, load=load, segment=seg)
        if plog is None:
            import pyarrow as pa
            plog = ParquetLog(log_rows_dir, schema=pa.Table.from_pylist([er]).schema)
        plog.append(er)
        state["ep"] = ep + 1
        _CUR_EP[0] = -1

        # ---------------- validation-based checkpoint selection
        if state["ep"] % int(R.val_every_episodes) == 0 or state["ep"] == E:
            v = validate(ctrl, cfg, seed, arm_key)
            state["val"].append((state["ep"], v))
            if v > state["best"]:
                state["best"], state["best_ep"] = v, state["ep"]
                ck.save_best(run_dir, dict(ctrl=_net_state(ctrl), ep=state["ep"], val=v))
        # ---------------- rolling checkpoint (episode boundary)
        if state["ep"] % int(R.ckpt_every_episodes) == 0 or state["ep"] == E:
            plog.flush()
            state["pair_snap"] = pair_snap
            ck.save_rolling(run_dir, dict(state=state, streams=streams.get_state(),
                                          sim=sim.get_state(), ctrl=ctrl.get_state(),
                                          log=dict(schema=plog.schema, state=plog.get_state())))
            _heartbeat(row, state["ep"])
            if row.get("_interrupt_after") == state["ep"]:           # test hook (resume test)
                raise SimulatedInterrupt(f"interrupted after episode {state['ep']}")

    plog.flush()
    ck.save_final(run_dir, dict(ctrl=_net_state(ctrl), ep=state["ep"]))
    result = dict(run_id=row.get("run_id"), episodes=state["ep"], best_ep=state["best_ep"],
                  best_val=state["best"], val_curve=state["val"],
                  train_wall_s=time.perf_counter() - t_start)
    if row.get("evaluate", True):
        best = ck.load(ck.best_path(run_dir))
        ctrl.set_state(_merge_net_state(ctrl.get_state(), best["ctrl"]))
        # pilot / tuning runs evaluate on the validation block so test traces stay untouched
        block = row.get("eval_block") or ("val_full" if row.get("phase") in ("pilot", "tune") else "test")
        rows = evaluate(ctrl, cfg, seed, arm_key, workloads=(workload,), env_kw=env_kw, block=block)
        import pyarrow as pa
        keys = sorted({k for r in rows for k in r})
        rows = [{k: r.get(k, float("nan")) for k in keys} for r in rows]
        tl = ParquetLog(run_dir / block, schema=pa.Table.from_pylist(rows).schema)
        tl.extend(rows)
        tl.flush()
        result["test_rows"] = len(rows)
    (run_dir / "RESULT.json").write_text(json.dumps(result, default=float, indent=1))
    ck.prune(run_dir)
    return result


_ENV_STREAMS = ("world", "arrivals", "channel", "mobility", "loss")


def _net_state(ctrl) -> dict:
    """Learned state for the final/best checkpoints: exactly the get_state() keys the
    controller declares in learned_state_keys(), i.e. what its eval_clone takes from the
    trained controller (networks, memory parameters, DDPG learner, ES parameters, frozen
    normaliser). A fixed key list such as ("sac", "params", "norm") would leave the B_HUANG
    learner and the B_SONG parameters out of the checkpoints."""
    st = ctrl.get_state()
    keys = tuple(ctrl.learned_state_keys())
    missing = [k for k in keys if k not in st]
    if missing:
        raise KeyError(f"learned_state_keys {missing} not in {type(ctrl).__name__}.get_state()")
    keep = {k: st[k] for k in keys}
    if "sac" in keep and isinstance(keep["sac"], dict):
        keep["sac"] = {k: v for k, v in keep["sac"].items()}
    return keep


def _merge_net_state(full: dict, nets: dict) -> dict:
    out = dict(full)
    out.update(nets)
    return out


def _heartbeat(row: dict, ep: int) -> None:
    db = row.get("db_path")
    if not db:
        return
    try:
        from .queue import RunQueue
        q = RunQueue(db)
        ok = q.heartbeat(row["run_id"], pid=os.getpid(), episodes_done=ep,
                         last_ckpt=str(ck.rolling_path(row["run_dir"])))
        q.close()
    except Exception:          # heartbeat is best-effort; the worker thread also beats
        return
    if ok is False:            # claim lost (requeued as stale) -> stop writing
        raise LostClaim(f"run {row['run_id']} no longer held by this worker")
