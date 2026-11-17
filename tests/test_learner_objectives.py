"""Objectives J1-J4 (Eqs. 30-33), Normalizer (Eq. 34),
adaptive weights (Eq. 59), reward."""
import math

import numpy as np
import pytest

from tpa.config import load_config
from tpa.learner.objectives import AdaptiveWeights, Normalizer, reward, slot_objectives
from tpa.types import Assignment, Decision, SlotOutcome

CFG = load_config()
M = CFG.scenario.n_uav


def mk_outcome(comp, mem, burst):
    return SlotOutcome(t=0, completed=[], failed=[], late=[], comp_energy_j=np.asarray(comp, float),
                       mem_energy_j=mem, burst_energy_j=np.asarray(burst, float),
                       uplink_energy_j=0.0, downlink_energy_j=0.0, hover_energy_j=0.0,
                       cpu_util=np.zeros(M), bw_util=0.0, arrivals=0)


def test_slot_objectives_hand():
    dec = Decision(assignments=[Assignment(0, 0, 1e9, 1.5), Assignment(1, 1, 1e9, 2.5)],
                   util_terms={"cpu_frac": np.array([0.5, 0, 0, 0, 0]),
                               "rate_frac": np.array([1.0, 0, 0, 0, 0])})
    out = mk_outcome([1.0, 2.0, 0, 0, 0], 0.01, [100.0] * M)
    ms = {"l1": 60.0, "fro_s": 10.0, "fro_l": 20.0, "dr_norm": 0.3}
    J = slot_objectives(dec, out, ms, CFG)
    assert J[0] == pytest.approx(4.0)                                        # Eq. 30
    assert J[1] == pytest.approx(0.5 * 0.5 + 0.3 * 1.0 + M * 0.2 * 60 / 600)  # Eq. 31
    assert J[2] == pytest.approx(3.0 + M * 0.01)                             # Eq. 32, no burst
    assert J[3] == pytest.approx(0.5 * 10 + 0.5 * 20 + 1.0 * 0.3)            # Eq. 33
    J0 = slot_objectives(dec, mk_outcome([1.0, 2.0, 0, 0, 0], 0.0, [0.0] * M), None, CFG)
    assert J0[3] == 0.0 and J0[2] == pytest.approx(3.0)                       # memoryless


def test_normalizer_running_then_frozen_percentiles():
    cfg = CFG.with_overrides({"learner.norm_freeze_slots": 100})
    nz = Normalizer(cfg)
    out = nz([1.0, 5.0, 2.0, 0.0])
    assert np.all(out == 0.0)                                  # single point: degenerate
    nz([3.0, 7.0, 2.0, 0.0])
    assert nz.normalize([2.0, 6.0, 2.0, 0.0]) == pytest.approx([0.5, 0.5, 0.0, 0.0])
    for k in range(98):
        nz([float(k), 5.0, 2.0, 0.0])
    assert nz.frozen and nz.count == 100
    vals = np.array([1.0, 3.0] + [float(k) for k in range(98)])
    lo, hi = np.percentile(vals, 1.0), np.percentile(vals, 99.0)
    assert nz.lo[0] == pytest.approx(lo) and nz.hi[0] == pytest.approx(hi)
    assert nz.normalize([50.0, 0, 0, 0])[0] == pytest.approx((50 - lo) / (hi - lo))
    # frozen: further observations do not move lo/hi; clip to [0, 1]
    nz([1e6, 1e6, 1e6, 1e6])
    assert nz.hi[0] == pytest.approx(hi)
    assert nz.normalize([1e6, 0, 0, 0])[0] == 1.0 and nz.normalize([-5.0, 0, 0, 0])[0] == 0.0
    # column 2 (constant 2.0) and 3 (J4 = 0) are degenerate -> 0
    assert nz.normalize([0, 0, 2.0, 0.0])[2:] == pytest.approx([0.0, 0.0])
    nz2 = Normalizer(cfg)
    nz2.set_state(nz.get_state())
    assert np.array_equal(nz2.normalize([50, 6, 2, 0]), nz.normalize([50, 6, 2, 0]))


def test_eq59_adaptive_weights_hand():
    aw = AdaptiveWeights(CFG)
    w1 = aw.step([0.2, 0.5, 0.1, 0.0], 1.0)
    assert w1 == pytest.approx(np.array([0.4, 0.2, 0.2, 0.2]) * math.exp(0.1) * 0.25)
    w2 = aw.step([0.4, 0.3, 0.1, 0.0], 0.5)
    # c1 = (0.2,-0.5,0.1,0), c2 = (0.4,-0.3,0.1,0); dJ = EWM(t-1) - c2 = (-0.2,-0.2,0,0)
    e = [math.exp(0.2), math.exp(0.2), 1.0, 1.0]
    g = np.array(e) / sum(e)
    hand = np.array([0.4, 0.2, 0.2, 0.2]) * math.exp(0.1 * 0.5) * g
    assert w2 == pytest.approx(hand, rel=1e-12)
    assert w2[0] == pytest.approx(0.4 * 1.0512710964 * 0.2749170, rel=1e-6)
    assert w2.sum() != pytest.approx(1.0)                        # no renormalisation
    # EWM update: 0.95 c1 + 0.05 c2
    assert aw.ewm == pytest.approx(0.95 * np.array([0.2, -0.5, 0.1, 0]) + 0.05 * np.array([0.4, -0.3, 0.1, 0]))
    # worsening objective (J1 cost up) gained weight relative to unchanged J3
    assert g[0] > g[2]


def test_fixed_weights_a7():
    aw = AdaptiveWeights(CFG, fixed=True)
    for J, s in (([0.1, 0.9, 0.3, 0.2], 2.0), ([0.9, 0.1, 0.3, 0.7], 0.0)):
        assert aw.step(J, s) == pytest.approx([0.1, 0.05, 0.05, 0.05])


def test_reward_sign():
    assert reward([0.5, 0.5, 0.5, 0.5], [0.1] * 4) == pytest.approx(-0.1)
    assert reward([0, 1.0, 0, 0], [1, 1, 1, 1]) == pytest.approx(1.0)   # utilisation raises reward
