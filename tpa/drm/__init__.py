"""Dynamic resource management (Alg. 3) and burst mode (Eqs. 57-58)."""
from .drm import DRM, Admission, device_busy, finalize_predictions, slot_admissible
from .burst import Burst, load_pressure, spectral_norm

__all__ = ["DRM", "Admission", "slot_admissible", "device_busy", "finalize_predictions", "Burst", "load_pressure", "spectral_norm"]
