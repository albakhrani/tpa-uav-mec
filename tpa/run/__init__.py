"""Run infrastructure: checkpoints, run queue, manifests, timing, Parquet logs.

Submodules are imported explicitly (``from tpa.run import checkpoint``); this
package deliberately imports nothing heavy at package import time, so worker
processes can set thread environment variables before numpy/torch load.
The package also holds the training loop (``trainer.py``) and the frozen-policy
evaluation (``evaluator.py``).
"""
