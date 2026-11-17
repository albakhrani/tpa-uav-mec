"""Controller factory: TPAController for the TPA-pipeline arms, the tpa.baselines controllers otherwise."""
from __future__ import annotations

from .arms import arm


def make_controller(cfg, arm_key: str, streams, r_ref: float, h_min: float,
                    device: str = "cpu", timer=None):
    kind = arm(arm_key).controller
    if kind == "tpa":
        from .control import TPAController
        return TPAController(cfg, arm_key, streams, r_ref, h_min, device=device, timer=timer)
    from .baselines import CONTROLLERS
    try:
        cls = CONTROLLERS[kind]
    except KeyError:
        raise KeyError(f"arm {arm_key!r}: unknown controller kind {kind!r}") from None
    return cls(cfg, arm_key, streams, r_ref, h_min, device=device, timer=timer)
