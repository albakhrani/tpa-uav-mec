"""Two-level memory, MEPR and replay reconstruction (Eqs. 19-26, 42-47, Alg. 1;
arms A1-A5, A14; offset-free keys, zone-resolved candidate, phase-consistency
criterion; arrival keys, memory.key_source = "arrivals"; key-shuffled control A15).

    params = MemoryParams(cfg)
    mem    = TwoLevelMemory(cfg, flags_for_arm("A0"), params, rng_a14=streams.a14)
    mepr   = MEPR(cfg, flags)
    store  = SegmentStore(cfg, flags); mem.attach_store(store)

Dimensions are derived from the config (no literals): S_sys = 7M + 2Z + 2 (= 49),
task features = 5 + Z + M + 1 (= 17), x = S_sys + 2*task + d (= 147),
q = S_sys + 2*task (= 83).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

EPS = 1e-8      # epsilon guard (||P_ref||, zone means, cosine denominators)


@dataclass(frozen=True)
class MemoryFlags:
    use_ms: bool = True       # A2 = False
    use_ml: bool = True       # A1 = False
    flat: bool = False        # A3
    mean_pool: bool = False   # A4
    gate_on: bool = True      # A5 = False (MEPR theta_p = -inf after warm-up)
    displaced: bool = False   # A14
    key_shuffled: bool = False   # A15 (key-shuffled control): key_source = "arrivals" only
    delayed: bool = False        # A16: delayed consolidation; key_source = "arrivals" only
    frozen_content: bool = False  # A16': W_v frozen at its seeded init (needs delayed, Delta >= 1)

    def to_dict(self) -> dict:
        return asdict(self)


_ARM_FLAGS = {
    "A1": dict(use_ml=False),
    "A2": dict(use_ms=False),
    "A3": dict(flat=True),
    "A4": dict(mean_pool=True),
    "A5": dict(gate_on=False),
    "A14": dict(displaced=True),
    "A15": dict(key_shuffled=True),
    "A16": dict(delayed=True),
    "A16P": dict(delayed=True, frozen_content=True),
}
_NO_MEMORY = {"A9", "A10", "A11", "A12"}


def flags_for_arm(arm_key: str) -> MemoryFlags | None:
    """Memory flags of an arm. Returns None for memoryless arms / baselines."""
    if arm_key in _NO_MEMORY or arm_key.startswith("B_"):
        return None
    return MemoryFlags(**_ARM_FLAGS.get(arm_key, {}))


def s_sys_dim(cfg) -> int:
    M, Z = cfg.scenario.n_uav, cfg.scenario.n_zones
    return 6 * M + 2 * Z + 2 + M          # 30 + 12 + 2 + 5 = 49


def task_feat_dim(cfg) -> int:
    M, Z = cfg.scenario.n_uav, cfg.scenario.n_zones
    return 5 + Z + M + 1                   # 5 scalars + zone one-hot + rates + mask = 17


def q_dim(cfg) -> int:
    return s_sys_dim(cfg) + 2 * task_feat_dim(cfg)          # [S_sys; tau_bar(mean, max)] = 83


def x_dim(cfg) -> int:
    return q_dim(cfg) + cfg.memory.d                        # + mbar_S (detached) = 147


from .stores import MemoryParams, TwoLevelMemory, key_dim, project_norm  # noqa: E402
from .mepr import MEPR, compute_P, compute_P_batch, compute_v, compute_v_batch  # noqa: E402
from .retrieval import attend  # noqa: E402
from .recon import SegmentStore, reconstruct, reconstruct_P  # noqa: E402
from .diag import (age_matched_phase_share, greedy_key_groups, isolated_key_count,  # noqa: E402
                   phase_consistency, phase_coverage, phase_hit, run_diagnostic)

__all__ = [
    "EPS", "MemoryFlags", "flags_for_arm", "s_sys_dim", "task_feat_dim", "q_dim", "x_dim",
    "MemoryParams", "TwoLevelMemory", "key_dim", "project_norm", "MEPR", "compute_P",
    "compute_P_batch", "compute_v", "compute_v_batch", "attend", "SegmentStore", "reconstruct",
    "reconstruct_P", "run_diagnostic", "phase_consistency", "phase_hit",
    "greedy_key_groups", "isolated_key_count", "age_matched_phase_share", "phase_coverage",
]
