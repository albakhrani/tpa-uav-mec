"""Environment switches: channel.h_min_rule, channel.interference_set, uav.abort_at_deadline;
flush() burst reset; SlotOutcome.info."""
import itertools
import math

import numpy as np
import pytest

from _env_helpers import greedy_decision, make_sim, run_hash
from tpa.config import load_config
from tpa.env import channel as ch
from tpa.types import Assignment, Decision, Task
from test_env_simulator import _outcome_digest, mk_task

BASE = load_config()


def cfg_with(**kw):
    m = {"h_min_rule": "channel.h_min_rule", "iset": "channel.interference_set",
         "abort": "uav.abort_at_deadline", "access": "channel.access"}
    return BASE.with_overrides({m[k]: v for k, v in kw.items()})


def test_h_min_rules():
    sim, cfg, w = make_sim(1, cfg=cfg_with(h_min_rule="fixed"))
    assert sim.h_min == 1e-11
    sim, cfg, w = make_sim(1, cfg=cfg_with(h_min_rule="from_r_min"))
    hand = (2 ** (0.1e6 / 10e6) - 1) * 4e-21 * 10e6 / w.device_ptx.max()
    assert sim.h_min == pytest.approx(hand, rel=1e-12)
    # at h_min the max-power device reaches R_min with no interference/system noise
    i = int(np.argmax(w.device_ptx))
    r = 10e6 * math.log2(1 + w.device_ptx[i] * sim.h_min / (4e-21 * 10e6))
    assert r == pytest.approx(0.1e6, rel=1e-12)
    h = np.array([[2 * hand, 0.5 * hand]])
    assert ch.viable(h, cfg, sim.h_min).tolist() == [[True, False]]
    with pytest.raises(ValueError):
        make_sim(1, cfg=cfg_with(h_min_rule="bogus"))


def test_slot_assigned_fixed_rates_and_interferers():
    # checks the literal shared-band Eqs. 7-9 (the adopted default is fdma_per_uav, tested in test_env_d1)
    sim, cfg, w = make_sim(1, load=0.0, cfg=cfg_with(iset="slot_assigned", access="shared"))
    sim.queue += [mk_task(0, device=0, size_bits=4e7, deadline=100.0),
                  mk_task(1, device=1, size_bits=4e7, deadline=100.0),
                  mk_task(2, device=2, size_bits=4e7, deadline=100.0)]
    obs = sim.begin_slot(0)
    # prediction at the start of slot t: nothing assigned yet -> no interferers
    R0 = ch.rate(sim.h_now, w.device_ptx, np.zeros((200, 5), bool), w.eta, cfg)
    assert np.allclose(obs.rates_pred, R0[[0, 1, 2]], rtol=1e-12)
    # helper: prediction with the set already assigned earlier in the slot
    act = sim.active_from_assignments([(0, 0)])
    assert act.sum() == 1 and act[0, 0]
    assert np.allclose(sim.predict_rates(act), ch.rate(sim.h_now, w.device_ptx, act, w.eta, cfg)[[0, 1, 2]])
    sim.end_slot(Decision([Assignment(0, 0, 1e9, 0.0), Assignment(1, 0, 1e9, 0.0)]), 0.0)
    act01 = np.zeros((200, 5), bool); act01[0, 0] = act01[1, 0] = True
    R01 = ch.rate(sim.h_now, w.device_ptx, act01, w.eta, cfg)
    assert sim.fixed_rate[0] == pytest.approx(R01[0, 0]) and sim.fixed_rate[1] == pytest.approx(R01[1, 0])
    rate0 = sim.fixed_rate[0]
    # slot 1: a new upload to UAV 0 does not see task 0/1 as interferers, and does not change their rate
    obs = sim.begin_slot(1)
    sim.end_slot(Decision([Assignment(0, 0, 1e9, 0.0)]), 0.0)
    act2 = np.zeros((200, 5), bool); act2[2, 0] = True
    R2 = ch.rate(sim.h_now, w.device_ptx, act2, w.eta, cfg)
    assert sim.fixed_rate[2] == pytest.approx(R2[2, 0])
    done = []
    t = 2
    while sim.inflight and t < 2000:
        sim.begin_slot(t); out = sim.end_slot(Decision(), 0.0); done += out.completed + out.late; t += 1
    up = {k.tid: k.t_upload_done for k in done}
    bits = 4e7 * 1.1
    assert up[0] == pytest.approx(bits / rate0, rel=1e-9)             # whole T_trans at the fixed rate
    assert up[2] == pytest.approx(1.0 + bits / R2[2, 0], rel=1e-9)


@pytest.mark.parametrize("abort", [True, False])
def test_abort_at_deadline_frees_capacity(abort):
    sim, cfg, w = make_sim(1, load=0.0, cfg=cfg_with(abort=abort))
    sim._rates = lambda h, active: np.full((sim.N, sim.M), 1e6)   # 8.8 s upload
    sim.queue.append(mk_task(0, deadline=1.5))
    obs = sim.begin_slot(0)
    cap = obs.cap_bound[0]
    sim.end_slot(Decision([Assignment(0, 0, 0.5 * cap, 0.0)]), 0.0)
    obs = sim.begin_slot(1)                       # 1 > 1.5 false: still in flight
    assert obs.residual_f[0] == pytest.approx(0.5 * cap)
    sim.end_slot(Decision(), 0.0)
    obs = sim.begin_slot(2)                       # 2 > 1.5: aborted now (if enabled)
    out = sim.end_slot(Decision(), 0.0)
    if abort:
        assert obs.residual_f[0] == pytest.approx(cap)
        assert [k.tid for k in out.failed] == [0] and out.info["n_aborted"] == 1
        assert not sim.inflight
    else:
        assert obs.residual_f[0] == pytest.approx(0.5 * cap) and not out.failed
        t, late = 3, []
        while sim.inflight:
            sim.begin_slot(t); late += sim.end_slot(Decision(), 0.0).late; t += 1
        assert [k.tid for k in late] == [0]


MODES = list(itertools.product(["fixed", "from_r_min"], ["in_progress", "slot_assigned"], [True, False]))


@pytest.mark.parametrize("rule,iset,abort", MODES)
def test_crn_each_mode(rule, iset, abort):
    c = cfg_with(h_min_rule=rule, iset=iset, abort=abort)
    s1, _, _ = make_sim(5, cfg=c, packet_loss=0.05)
    s2, _, _ = make_sim(5, cfg=c, packet_loss=0.05)
    assert run_hash(s1, lambda o: Decision(), 200) == run_hash(s2, lambda o: greedy_decision(o, frac=0.1), 200)
    assert len(s2.inflight) > 0


@pytest.mark.parametrize("iset", ["in_progress", "slot_assigned"])
def test_round_trip_each_interference_mode(iset):
    c = cfg_with(iset=iset)
    sim, _, _ = make_sim(2, cfg=c, packet_loss=0.1)
    pol = lambda o: greedy_decision(o, frac=0.2)
    for t in range(60):
        sim.end_slot(pol(sim.begin_slot(t)), 0.0)
    st = sim.get_state()
    d1 = _outcome_digest(sim, 40, pol)
    fresh, _, _ = make_sim(2, cfg=c, packet_loss=0.1)
    fresh.set_state(st)
    assert _outcome_digest(fresh, 40, pol) == d1


def test_flush_resets_burst_keeps_ewms():
    sim, cfg, w = make_sim(1)
    for t in range(20):
        obs = sim.begin_slot(t)
        d = greedy_decision(obs)
        d.burst_next = np.ones(5); d.burst_budget = np.full(5, 0.5)
        sim.end_slot(d, 0.0)
    lam, cyc = sim.lam_eff.copy(), sim.ewm_cycles.copy()
    sim.flush()
    assert not sim.burst.any() and not sim.burst_budget.any()
    assert np.array_equal(sim.lam_eff, lam) and np.array_equal(sim.ewm_cycles, cyc)
    obs = sim.begin_slot(20)
    assert obs.cap_bound == pytest.approx(w.f_max * w.eta)
    out = sim.end_slot(Decision(), 0.0)
    assert isinstance(out.info, dict) and "n_expired" in out.info
