"""Arm registry. Each arm is a set of flags read by the controller,
memory, MEPR, DRM and objectives. A0 is the full framework; every ablation
changes exactly the flags given in its ARMS entry below (tests/test_arms.py
compares each arm's difference from A0 with an expected table).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ArmFlags:
    controller: str = "tpa"     # tpa | lstm (A10) | framestack (A11) | lookup (A12)
                                # | zhou | qin | huang | song
    memory: bool = True         # explicit memory + MEPR + retrieval present
    use_ms: bool = True         # A2 False: no short-term buffer
    use_ml: bool = True         # A1 False: no long-term store / consolidation
    flat: bool = False          # A3: single capacity-matched FIFO store
    mean_pool: bool = False     # A4: uniform attention
    gate_on: bool = True        # A5 False: theta_p = -inf (consolidate every slot)
    pattern_terms: bool = True  # A6 False (and all memoryless arms): reactive DRM
    adaptive_w: bool = True     # A7 False: fixed w_base/4
    burst_on: bool = True       # A8 False
    beta_c: float | None = None # A13: link term weight (None = cfg.drm.beta_c = 0)
    displaced: bool = False     # A14: time-displaced retrieval
    key_shuffled: bool = False  # A15: write keys of a random earlier slot; query unchanged
    delayed: bool = False       # A16: delayed consolidation, content of (s, s+Delta] under key k_s
    frozen_content: bool = False  # A16' = A16P: W_v of the long-term content frozen at its seeded init
    learned: bool = True        # False for zhou, qin, lookup


_A0 = ArmFlags()
_MEMLESS = dict(memory=False, use_ms=False, use_ml=False, pattern_terms=False)

ARMS: dict[str, ArmFlags] = {
    "A0": _A0,
    "A1": replace(_A0, use_ml=False),
    "A2": replace(_A0, use_ms=False),
    "A3": replace(_A0, flat=True),
    "A4": replace(_A0, mean_pool=True),
    "A5": replace(_A0, gate_on=False),
    "A6": replace(_A0, pattern_terms=False),
    "A7": replace(_A0, adaptive_w=False),
    "A8": replace(_A0, burst_on=False),
    "A9": replace(_A0, **_MEMLESS),
    "A10": replace(_A0, controller="lstm", **_MEMLESS),
    "A11": replace(_A0, controller="framestack", **_MEMLESS),
    "A12": replace(_A0, controller="lookup", memory=False, use_ms=False, use_ml=False,
                   learned=False),
    "A13": replace(_A0, beta_c=0.25),          # tuned in {0.1, 0.25, 0.5}
    "A14": replace(_A0, displaced=True),
    "A15": replace(_A0, key_shuffled=True),    # key-shuffled control (mechanism checks; tier-C main rows)
    "A16": replace(_A0, delayed=True),         # forecasting write rule (delayed consolidation)
    "A16P": replace(_A0, delayed=True, frozen_content=True),   # A16' = A16 with W_v frozen
    # baselines: own rule through the shared pattern-free executor; no burst mode
    # (burst escalation is a TPA DRM mechanism, not part of the published methods)
    "B_ZHOU": replace(_A0, controller="zhou", adaptive_w=False, learned=False, burst_on=False, **_MEMLESS),
    "B_QIN": replace(_A0, controller="qin", adaptive_w=False, learned=False, burst_on=False, **_MEMLESS),
    "B_HUANG": replace(_A0, controller="huang", adaptive_w=False, burst_on=False, **_MEMLESS),
    "B_SONG": replace(_A0, controller="song", adaptive_w=False, burst_on=False, **_MEMLESS),
}
# A12 keeps pattern_terms=True: its lookup prediction plays the role of P(t).


def arm(name: str) -> ArmFlags:
    try:
        return ARMS[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}") from None
