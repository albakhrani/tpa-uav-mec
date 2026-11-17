"""The adopted memory defaults, the arrival key shared by every memory arm (A0-A8, A13,
A14) and A3's flat store, and the mechanism-only arm A15 kept out of tuning / campaign
sets (tests/test_tuning_arms.py covers the scripts)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tpa.config import load_config
from tpa.memory import flags_for_arm
from test_memory_driver import Driver, make_cfg

MEMORY_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A13", "A14"]


def test_default_config_has_adopted_memory_defaults():
    cfg = load_config()
    m = cfg.memory
    assert m.rho_merge == 0.95 and m.theta_p == 0.35
    assert m.key_source == "arrivals" and m.candidate_source == "zone"
    assert m.candidate_repr == "layernorm" and m.key_centering is True
    assert m.lt_addressing == "fixed_key" and m.key_temperature == 0.1
    assert m.key_mean_horizon_slots == 2 * cfg.workload.period_slots


def _run(arm, T=160, seed=0, period=40):
    cfg = make_cfg()                      # the adopted defaults; k_long default, driver-sized inputs
    return Driver(cfg, flags_for_arm(arm), seed=seed, period=period).run(T)


@pytest.mark.parametrize("arm", MEMORY_ARMS)
def test_arrival_key_identical_for_every_memory_arm(arm):
    """The per-slot arrival key depends only on the logged arrivals / zone means, so
    every memory arm sees the same k_t sequence as A0 on the same inputs; the keys of the
    entries each arm stores are rows of that sequence (A3: the flat store's F_key)."""
    T = 160
    d0, d = _run("A0", T), _run(arm, T)
    k0 = d0.store.arrival_keys_upto(T)
    k = d.store.arrival_keys_upto(T)
    assert np.array_equal(k0, k)
    assert d.mem.arrivals and d.mem.key_source == "arrivals"
    w = int(d.cfg.memory.window_w)
    rows = {tuple(np.round(k[i], 6)) for i in range(w - 1, T)}
    mem = d.mem
    if mem.flags.flat:                                           # A3: single FIFO, keyed rows
        keyed = np.asarray(mem.F_keyed, bool)[: mem.F_fill]
        assert keyed.any()
        F_key = mem.F_key[: mem.F_fill].cpu().numpy()[keyed]
        F_time = np.asarray(mem.F_time)[: mem.F_fill][keyed]
        for key, t in zip(F_key, F_time):
            assert np.allclose(key, k[int(t)], atol=1e-6)     # F_key row = arrival key of its write slot
    else:
        if mem.flags.use_ml:
            filled = np.asarray(mem.L_filled, bool)
            assert filled.any()
            for key in mem.L_key[torch.as_tensor(filled)].cpu().numpy():
                assert tuple(np.round(key, 6)) in rows         # L_key = some slot's arrival key
        if mem.flags.use_ms:
            keyed = np.asarray(mem.S_keyed, bool)
            S_key = mem.S_key.cpu().numpy()[keyed]
            S_time = np.asarray(mem.S_time)[keyed]
            for key, t in zip(S_key, S_time):
                assert np.allclose(key, k[int(t)], atol=1e-6)


def test_a0_vs_a3_differ_only_in_store_shape():
    """A0 and A3 share inputs, keys and (here) gate decisions; the only other route by
    which they differ is the store layout itself: x_t carries the arm's own short-term mean
    m_bar (M_S for A0, the flat FIFO for A3), so e_t and sigma_p can differ at float level
    through that term alone (A3 = capacity-matched single store)."""
    T = 160
    d0, d3 = _run("A0", T), _run("A3", T)
    assert d0.cons == d3.cons
    assert np.array_equal(d0.store.arrival_keys_upto(T), d3.store.arrival_keys_upto(T))
    sig0 = np.array([d0.sigma[t] for t in range(T)]); sig3 = np.array([d3.sigma[t] for t in range(T)])
    assert np.allclose(sig0, sig3, atol=0.05)
    assert np.array_equal(d0.store._za.a[:T], d3.store._za.a[:T])  # same logged arrivals


def test_a15_not_a_default_memory_arm():
    f = flags_for_arm("A15")
    assert f.key_shuffled and not flags_for_arm("A0").key_shuffled
    for arm in MEMORY_ARMS:
        assert not flags_for_arm(arm).key_shuffled
