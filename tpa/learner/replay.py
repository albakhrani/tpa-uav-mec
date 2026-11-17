"""Replay buffer with compact numpy storage.

A transition stores the observed parts of s_t and s_{t+1} (S_sys, task window,
sigma_p) plus references (segment_id, t) into the memory segment store; r(t) and
r(t+1) are reconstructed at update time with the current memory parameters,
so no memory snapshot is stored. The final transition of an episode is not
pushed, so every stored transition bootstraps.

Next state by index. Row i keeps S_sys, tf, a, reward, sigma, ref.
Its next state (S_sys2, tf2, sigma2, ref2) is row (i+1)'s (S_sys, tf, sigma, ref) when
``_nx[i]`` is set, otherwise a side copy ``_side[i]``. ``_nx[i]`` is set only when the
push that writes row i+1 brings a current state whose float32/int64 bytes equal row i's
next state, so every sampled value is byte-identical to storing the copy. Rows without a
stored successor: the newest row, and the last pushed transition of an episode (the
final transition is dropped). ``sample`` draws idx with the same rng call and returns
the same keys in the same order.
"""
from __future__ import annotations

import numpy as np
import torch

_ARRAY_KEYS = ("S_sys", "tf", "a", "reward", "S_sys2", "tf2", "sigma", "sigma2")
_REF_KEYS = ("ref", "ref2")
_STORED = ("S_sys", "tf", "a", "reward", "sigma")             # float32 rows kept per row
_NEXT = (("S_sys2", "S_sys"), ("tf2", "tf"), ("sigma2", "sigma"), ("ref2", "ref"))
FORMAT = "nx1"


def _f32(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def _i64(v) -> np.ndarray:
    return np.asarray(v, dtype=np.int64)


def _bits(a: np.ndarray) -> np.ndarray:
    """Rows of ``a`` as raw bytes (n, nbytes) for an exact bitwise comparison."""
    return np.ascontiguousarray(a).reshape(len(a), -1).view(np.uint8)


class Replay:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self.size = 0
        self.ptr = 0
        self._buf: dict[str, np.ndarray] | None = None
        self._nx: np.ndarray | None = None
        self._side: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.size

    def _alloc(self, tr: dict) -> None:
        buf = {}
        for k in _STORED:
            v = _f32(tr[k])
            buf[k] = np.zeros((self.capacity,) + v.shape, dtype=np.float32)
        buf["ref"] = np.zeros((self.capacity, 2), dtype=np.int64)
        self._buf = buf
        self._nx = np.zeros(self.capacity, dtype=bool)
        self._side = {}

    def push(self, tr: dict) -> None:
        """Keys: S_sys, tf, a, reward, S_sys2, tf2, sigma, sigma2, ref, ref2
        (ref = (segment_id, t)). sigma may be 0 for memory-free arms."""
        if self._buf is None:
            self._alloc(tr)
        i = self.ptr
        if self.size > 0:                                   # link the previous push (row j)
            j = (i - 1) % self.capacity
            sd = self._side.get(j)
            if sd is not None and \
                    sd["S_sys2"].tobytes() == _f32(tr["S_sys"]).tobytes() and \
                    sd["tf2"].tobytes() == _f32(tr["tf"]).tobytes() and \
                    sd["sigma2"].tobytes() == _f32(tr["sigma"]).tobytes() and \
                    sd["ref2"].tobytes() == _i64(tr["ref"]).tobytes():
                self._nx[j] = True
                del self._side[j]
        self._side.pop(i, None)                             # row i is overwritten
        self._nx[i] = False
        for k in _STORED:
            self._buf[k][i] = _f32(tr[k])
        self._buf["ref"][i] = _i64(tr["ref"])
        self._side[i] = dict(S_sys2=_f32(tr["S_sys2"]).copy(), tf2=_f32(tr["tf2"]).copy(),
                             sigma2=_f32(tr["sigma2"]).copy(), ref2=_i64(tr["ref2"]).copy())
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _next(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        nxt = (idx + 1) % self.capacity
        out = {k2: self._buf[k][nxt] for k2, k in _NEXT}
        for p in np.nonzero(~self._nx[idx])[0]:
            sd = self._side[int(idx[p])]
            for k2, _ in _NEXT:
                out[k2][p] = sd[k2]
        return out

    def sample(self, B: int, device="cpu") -> dict:
        """Uniform with replacement. Returns float32 tensors (reward (B,),
        sigma (B,1)) plus `refs`, `refs2`: lists of (segment_id, t) tuples and `idx`."""
        if self.size == 0:
            raise ValueError("empty replay")
        idx = self.rng.integers(0, self.size, size=int(B))
        nx = self._next(idx)
        out = {}
        for k in _ARRAY_KEYS:
            v = nx[k] if k in nx else self._buf[k][idx]
            if k in ("sigma", "sigma2"):
                v = v.reshape(len(idx), 1)
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).to(device)
        out["reward"] = out["reward"].reshape(-1)
        out["refs"] = [(int(s), int(t)) for s, t in self._buf["ref"][idx]]
        out["refs2"] = [(int(s), int(t)) for s, t in nx["ref2"]]
        out["idx"] = idx
        return out

    def live_segments(self) -> set[int]:
        """Segment ids still referenced by stored transitions (so the caller can
        free SegmentStores no longer needed). A linked row's ref2 is
        its successor's ref, so only the side copies add segment ids."""
        if self.size == 0:
            return set()
        return set(np.unique(self._buf["ref"][: self.size, 0]).tolist()) | \
            {int(sd["ref2"][0]) for sd in self._side.values()}

    def arrays(self) -> dict[str, np.ndarray]:
        """All ten keys for rows [0, size) in the copy layout (every next state stored per row)
        (diagnostics and equivalence checks; allocates full copies)."""
        if self._buf is None:
            return {}
        idx = np.arange(self.size)
        nx = self._next(idx)
        out = {k: (nx[k] if k in nx else self._buf[k][: self.size].copy()) for k in _ARRAY_KEYS}
        out["ref"] = self._buf["ref"][: self.size].copy()
        out["ref2"] = nx["ref2"]
        return out

    # ---- checkpoint ----
    def get_state(self) -> dict:
        st = {"capacity": self.capacity, "size": self.size, "ptr": self.ptr,
              "rng": self.rng.bit_generator.state, "format": FORMAT}
        if self._buf is not None:
            st["arrays"] = {k: v[: self.size] for k, v in self._buf.items()}
            st["shapes"] = {k: v.shape[1:] for k, v in self._buf.items()}
            st["nx"] = self._nx[: self.size]
            st["side"] = dict(self._side)
        return st

    def set_state(self, st: dict) -> None:
        self.capacity = int(st["capacity"])
        self.size = int(st["size"])
        self.ptr = int(st["ptr"])
        self.rng.bit_generator.state = st["rng"]
        self._nx, self._side = None, {}
        if "arrays" not in st:
            self._buf = None
            return
        n, cap = self.size, self.capacity
        self._buf = {}
        for k in _STORED + ("ref",):
            dt = np.int64 if k in _REF_KEYS else np.float32
            a = np.zeros((cap,) + tuple(st["shapes"][k]), dtype=dt)
            a[:n] = st["arrays"][k]
            self._buf[k] = a
        self._nx = np.zeros(cap, dtype=bool)
        if st.get("format") == FORMAT:
            self._nx[:n] = st["nx"]
            self._side = {int(j): {k: np.array(v) for k, v in sd.items()}
                          for j, sd in st["side"].items()}
            return
        # checkpoint in the copy layout (next state stored as a copy): link row j to row
        # q = j+1 where q holds a later push and the bytes of all four next-state keys
        # equal q's current state; every other row keeps its next state as a side copy.
        old = st["arrays"]
        j = np.arange(n)
        q = (j + 1) % cap
        ok = (q < n) & (q != self.ptr)
        for k2, k in _NEXT:
            a2, a1 = _bits(np.asarray(old[k2][:n])), _bits(np.asarray(old[k][:n]))
            ok &= np.all(a2 == a1[np.minimum(q, n - 1)], axis=1)
        self._nx[:n] = ok
        for r in np.nonzero(~ok)[0]:
            self._side[int(r)] = {k2: np.array(old[k2][r], dtype=np.int64 if k2 == "ref2"
                                               else np.float32) for k2, _ in _NEXT}
