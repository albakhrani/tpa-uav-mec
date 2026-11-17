"""Statistics for the campaign: see :mod:`tpa.stats.analysis`."""
from .analysis import cohen_d_pooled, contrast, holm, holm_reject, paired, trial_table, welch

__all__ = ["paired", "welch", "holm", "holm_reject", "cohen_d_pooled", "trial_table", "contrast"]
