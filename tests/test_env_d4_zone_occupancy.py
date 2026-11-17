"""Environment observations: SlotObs.zone_arrivals equals the per-zone count of the tasks the
workload generator produced in that slot; SlotObs.uav_occupancy equals the per-UAV
in-flight count at slot start divided by q_max. Seed 1, train block, as the other env tests (no test-block traces)."""
import numpy as np

from _env_helpers import greedy_decision, make_sim
from tpa.types import Decision

NONE = Decision()
T = 200


def test_zone_arrivals_match_workload_generator():
    sim, cfg, w = make_sim(1)
    Z = w.n_zones
    assert float(cfg.workload.zone_ewm) >= 1.0          # cumulative mean since start
    cum = np.zeros(Z)
    for t in range(T):
        obs = sim.begin_slot(t)
        arr = sim.last_arrivals                          # what the generator produced in slot t
        expect = np.bincount(np.array([k.zone for k in arr], dtype=int), minlength=Z)
        assert obs.zone_arrivals.shape == (Z,)
        assert np.array_equal(obs.zone_arrivals, expect)
        assert obs.zone_arrivals.sum() == len(arr)
        assert all(k.t_arr == t for k in arr)
        cum += obs.zone_arrivals
        assert np.allclose(obs.zone_mean, cum / (t + 1))
        sim.end_slot(NONE if t % 2 else greedy_decision(obs, frac=0.1), 0.0)
    assert cum.sum() > 0                                 # the run actually produced arrivals


def test_uav_occupancy_matches_inflight():
    sim, cfg, w = make_sim(1)
    M, q_max, dt = w.n_uav, sim.q_max, sim.dt
    assert q_max == int(cfg.task.q_max)
    ledger = {}                                          # tid -> (uav, t_arr, deadline_s), in flight
    first_assign_slot = None
    seen_nonzero = False
    for t in range(T):
        obs = sim.begin_slot(t)
        # step 5b of begin_slot: in-flight tasks past their deadline are aborted at slot start
        if sim.abort_at_deadline:
            for tid in [k for k, (_, ta, dl) in ledger.items() if (t - ta) * dt > dl]:
                del ledger[tid]
        expect = np.bincount(np.array([v[0] for v in ledger.values()], dtype=int), minlength=M)
        assert obs.uav_occupancy is not None and obs.uav_occupancy.shape == (M,)
        assert obs.uav_occupancy.dtype == np.float64
        got = obs.uav_occupancy * q_max
        assert np.allclose(got, expect), (t, got, expect)
        # direct check against the simulator's own in-flight list at slot start
        direct = np.bincount(np.array([k.uav for k in sim.inflight], dtype=int), minlength=M)
        assert np.array_equal(np.rint(got).astype(int), direct)
        if first_assign_slot is None:
            assert np.all(obs.uav_occupancy == 0.0)
        seen_nonzero |= bool(got.sum() > 0)
        # assign every third slot so the ledger goes up and down
        dec = greedy_decision(obs, frac=0.2) if (t % 3 == 0 and len(obs.queue)) else NONE
        if dec.assignments and first_assign_slot is None:
            first_assign_slot = t
        for a in dec.assignments:
            k = obs.queue[a.task_index]
            ledger[k.tid] = (int(a.uav), k.t_arr, k.deadline_s)
        out = sim.end_slot(dec, 0.0)
        for k in out.completed + out.late + out.failed:
            ledger.pop(k.tid, None)                      # failed also holds queue expiries (not in ledger)
    assert first_assign_slot is not None and seen_nonzero
