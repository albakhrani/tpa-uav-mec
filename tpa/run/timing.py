"""Low-overhead per-module wall-clock timer.

    timer = ModuleTimer()
    with timer("mepr"):
        ...
    timer.summary()  # {key: {n, total_ms, mean_ms, p25, p50, p75, p90, p99}}  (percentiles in ms)

Each key owns one reusable context object (no allocation per call); time is
``time.perf_counter_ns``. Percentiles come from a fixed-size reservoir
(Vitter's Algorithm R) whose replacement draws come from a private, seeded
``np.random.Generator`` (pre-drawn in blocks), so no global RNG is touched and
the reservoir is reproducible. Count and total are exact.

Discard warm-up by calling ``reset()`` after it. A key must not be
re-entered while it is already open (nesting *different* keys is fine).
"""
from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter_ns

import numpy as np

KEYS = ("mepr", "retrieval", "actor", "drm", "learner", "env")
_PCTS = (25, 50, 75, 90, 99)
_UBLOCK = 4096


class _Slot:
    __slots__ = ("key", "owner", "n", "total", "res", "t0", "_open")

    def __init__(self, key: str, owner: "ModuleTimer"):
        self.key, self.owner = key, owner
        self.n = 0
        self.total = 0
        self.res = np.empty(owner.reservoir, dtype=np.int64)
        self.t0 = 0
        self._open = False

    def __enter__(self):
        if self._open:
            raise RuntimeError(f"timer key {self.key!r} re-entered while open")
        self._open = True
        self.t0 = perf_counter_ns()
        return self

    def __exit__(self, *exc):
        dt = perf_counter_ns() - self.t0
        self._open = False
        self.owner._record(self, dt)
        return False


class ModuleTimer:
    def __init__(self, reservoir: int = 8192, seed: int = 0, enabled: bool = True):
        self.reservoir = int(reservoir)
        self.enabled = enabled
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._u = self._rng.random(_UBLOCK)
        self._ui = 0
        self._slots: dict[str, _Slot] = {}
        self._null = nullcontext()

    def __call__(self, key: str):
        if not self.enabled:
            return self._null
        s = self._slots.get(key)
        if s is None:
            s = self._slots[key] = _Slot(key, self)
        return s

    def add(self, key: str, ns: int) -> None:
        """Record an externally measured duration (nanoseconds)."""
        s = self._slots.get(key)
        if s is None:
            s = self._slots[key] = _Slot(key, self)
        self._record(s, int(ns))

    def _record(self, s: _Slot, dt: int) -> None:
        n = s.n
        if n < self.reservoir:
            s.res[n] = dt
        else:
            if self._ui == _UBLOCK:
                self._u = self._rng.random(_UBLOCK)
                self._ui = 0
            j = int(self._u[self._ui] * (n + 1))
            self._ui += 1
            if j < self.reservoir:
                s.res[j] = dt
        s.n = n + 1
        s.total += dt

    def reset(self) -> None:
        self._slots.clear()

    def keys(self) -> list[str]:
        return list(self._slots)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, s in self._slots.items():
            if s.n == 0:
                continue
            sample = s.res[:min(s.n, self.reservoir)] / 1e6
            pct = np.percentile(sample, _PCTS)
            d = {"n": s.n, "total_ms": s.total / 1e6, "mean_ms": s.total / 1e6 / s.n}
            for p, v in zip(_PCTS, pct):
                d[f"p{p}"] = float(v)
            d["iqr_ms"] = d["p75"] - d["p25"]
            out[k] = d
        return out
