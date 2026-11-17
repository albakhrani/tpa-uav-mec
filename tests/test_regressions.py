"""Regression tests: segment zone mean, disturbance metrics, workload options, B_SONG antithetic pairs."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from _sys_helpers import row, tiny_cfg
from tpa.env.simulator import Simulator
from tpa.env.world import build_world
from tpa.rng import Streams
from tpa.run import trainer as TR
from tpa.run.evaluator import disturbance_metrics
from tpa.types import Decision


def test_zone_mean_is_cumulative_segment_mean():
    cfg = tiny_cfg()
    st = Streams(1)
    w = build_world(cfg, st.world)
    sim = Simulator(cfg, w, st, load=0.55, t0=0, horizon=200)
    tot = np.zeros(w.n_zones)
    for t in range(200):
        obs = sim.begin_slot(t)
        tot += obs.zone_arrivals
        assert np.allclose(obs.zone_mean, tot / (t + 1))
        sim.end_slot(Decision(burst_next=np.zeros(w.n_uav)))


def test_disturbance_metrics_synthetic():
    n = 1000
    arr = np.full(n, 10.0)
    done = np.full(n, 9.0)
    burst = np.zeros(n, bool)
    burst[500:560] = True
    done[500:600] = 5.0                                     # completion drops after onset
    detect = np.zeros(n, bool)
    detect[503] = True
    act = np.full(n, 2.0)
    act[503] = 7.5
    m = disturbance_metrics(act, arr, done, detect, burst)
    assert m["n_disturbances"] == 1
    assert m["recovery_ms"] == pytest.approx(7.5)
    assert m["detect_delay_slots"] == 3
    assert m["reaction_ms_p50"] == pytest.approx(2.0)
    assert 90 <= m["service_recovery_slots"] <= 100          # window clears the dip at ~k=100
    m0 = disturbance_metrics(act, arr, done, detect, np.zeros(n, bool))
    assert m0["n_disturbances"] == 0 and math.isnan(m0["recovery_ms"])


def test_parse_workload_options():
    k, kw = TR.parse_workload("sinusoidal|mobility=gauss_markov|packet_loss=0.05|patrol=1")
    assert k == "sinusoidal" and kw == dict(mobility="gauss_markov", packet_loss=0.05, patrol=True)
    assert TR.parse_workload("mmpp") == ("mmpp", {})
    with pytest.raises(ValueError):
        TR.parse_workload("sinusoidal|foo=1")


def test_song_pairs_share_trace(tmp_path, monkeypatch):
    """Episodes 2p and 2p+1 (an antithetic pair) must see identical arrivals."""
    seen = {}
    orig = Simulator.begin_slot

    def spy(self, t):
        obs = orig(self, t)
        seen.setdefault(TR._CUR_EP[0], hashlib.sha256())
        for tk in self.last_arrivals:
            seen[TR._CUR_EP[0]].update(np.array([tk.t_arr, tk.device, tk.size_bits]).tobytes())
        return obs

    monkeypatch.setattr(Simulator, "begin_slot", spy)
    cfg = tiny_cfg(**{"baselines.es_pairs": 2})
    TR.run(row(tmp_path, "B_SONG", evaluate=False), cfg)
    h = {k: v.hexdigest() for k, v in seen.items() if k >= 0}
    assert h[0] == h[1] and h[2] == h[3] and h[0] != h[2]
