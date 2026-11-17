"""Arm registry checks (tpa/arms.py): every ablation changes exactly the fields listed for it
below relative to A0, the memory flags of an arm agree with its ArmFlags,
and the key-shuffled control A15 (= A0 + key_shuffled) is a mechanism-check arm:
scripts/enqueue.py, config/default.toml and tpa/run/*.py never name it (the main campaign
runs it at tier C only, through scripts/enqueue_main_extra.py)."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, fields

import pytest

from tpa.arms import ARMS, ArmFlags, arm
from tpa.memory import MemoryFlags, flags_for_arm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MEMLESS = dict(memory=False, use_ms=False, use_ml=False, pattern_terms=False)
_BASE = dict(adaptive_w=False, burst_on=False, **_MEMLESS)

# The fields each arm changes relative to A0 (and nothing else)
SPEC_DIFF = {
    "A0": {},
    "A1": dict(use_ml=False),
    "A2": dict(use_ms=False),
    "A3": dict(flat=True),
    "A4": dict(mean_pool=True),
    "A5": dict(gate_on=False),
    "A6": dict(pattern_terms=False),
    "A7": dict(adaptive_w=False),
    "A8": dict(burst_on=False),
    "A9": dict(**_MEMLESS),
    "A10": dict(controller="lstm", **_MEMLESS),
    "A11": dict(controller="framestack", **_MEMLESS),
    "A12": dict(controller="lookup", memory=False, use_ms=False, use_ml=False, learned=False),
    "A13": dict(beta_c=0.25),
    "A14": dict(displaced=True),
    "A15": dict(key_shuffled=True),            # key-shuffled control (mechanism check)
    "A16": dict(delayed=True),                 # delayed consolidation (forecasting write rule)
    "A16P": dict(delayed=True, frozen_content=True),   # A16' = A16 with W_v frozen
    "B_ZHOU": dict(controller="zhou", learned=False, **_BASE),
    "B_QIN": dict(controller="qin", learned=False, **_BASE),
    "B_HUANG": dict(controller="huang", **_BASE),
    "B_SONG": dict(controller="song", **_BASE),
}

MEMORY_FLAG_FIELDS = [f.name for f in fields(MemoryFlags)]


def test_registry_matches_spec_exactly():
    assert set(ARMS) == set(SPEC_DIFF)
    a0 = asdict(ARMS["A0"])
    assert a0 == asdict(ArmFlags())
    for name, want in SPEC_DIFF.items():
        got = {k: v for k, v in asdict(ARMS[name]).items() if v != a0[k]}
        assert got == want, (name, got, want)


def test_a0_defaults():
    a0 = ARMS["A0"]
    assert a0.controller == "tpa" and a0.memory and a0.use_ms and a0.use_ml and not a0.flat
    assert not a0.mean_pool and a0.gate_on and a0.pattern_terms and a0.adaptive_w and a0.burst_on
    assert a0.beta_c is None and not a0.displaced and not a0.key_shuffled and a0.learned


def test_memory_flags_agree_with_arm_flags():
    assert MEMORY_FLAG_FIELDS == ["use_ms", "use_ml", "flat", "mean_pool", "gate_on", "displaced",
                                  "key_shuffled", "delayed", "frozen_content"]   # + A16: delayed, A16P: frozen_content
    for name, f in ARMS.items():
        mf = flags_for_arm(name)
        if not f.memory:
            assert mf is None, name
            continue
        assert mf is not None, name
        for k in MEMORY_FLAG_FIELDS:
            assert getattr(mf, k) == getattr(f, k), (name, k)
    assert flags_for_arm("A15") == MemoryFlags(key_shuffled=True)
    assert flags_for_arm("A14") == MemoryFlags(displaced=True)
    assert MemoryFlags().key_shuffled is False and MemoryFlags().to_dict()["key_shuffled"] is False


def test_arm_lookup():
    assert arm("A15") is ARMS["A15"] and arm("A15").key_shuffled and arm("A15").memory
    assert arm("A15") == ArmFlags(key_shuffled=True)
    with pytest.raises(KeyError):
        arm("A99")


def test_a15_not_in_campaign_arm_lists():
    """A15 is a mechanism-check arm. scripts/enqueue.py, config/default.toml and tpa/run/*.py never
    name it (its tier-C main rows come from scripts/enqueue_main_extra.py); scripts/tuning.py excludes
    it through MECHANISM_ONLY (import-based check in tests/test_tuning_arms.py, since tuning.py carries
    the literal name there)."""
    files = [os.path.join(ROOT, "scripts", "enqueue.py"), os.path.join(ROOT, "config", "default.toml")]
    run_dir = os.path.join(ROOT, "tpa", "run")
    files += [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith(".py")]
    for path in files:
        with open(path, encoding="utf-8") as fh:
            assert not re.search(r"\bA15\b", fh.read()), path
