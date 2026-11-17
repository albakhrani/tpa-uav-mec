"""Replay buffer with the next state stored by index, and streamed zstd checkpoints.

The reference is the earlier Replay implementation (next state stored as a copy), copied
verbatim below. The new buffer must give byte-identical samples, rng state and live segments,
accept the old checkpoint format, and the loader must read all three checkpoint formats and
apply map_location to each.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpa.learner import replay as rp  # noqa: E402
from tpa.run import checkpoint as ck  # noqa: E402

_ARRAY_KEYS = ("S_sys", "tf", "a", "reward", "S_sys2", "tf2", "sigma", "sigma2")
_REF_KEYS = ("ref", "ref2")


class _OldReplay:                                   # the earlier Replay, verbatim
    def __init__(self, capacity, rng):
        self.capacity = int(capacity)
        self.rng = rng
        self.size = 0
        self.ptr = 0
        self._buf = None

    def _alloc(self, tr):
        buf = {}
        for k in _ARRAY_KEYS:
            v = np.asarray(tr[k], dtype=np.float32)
            buf[k] = np.zeros((self.capacity,) + v.shape, dtype=np.float32)
        for k in _REF_KEYS:
            buf[k] = np.zeros((self.capacity, 2), dtype=np.int64)
        self._buf = buf

    def push(self, tr):
        if self._buf is None:
            self._alloc(tr)
        i = self.ptr
        for k in _ARRAY_KEYS:
            self._buf[k][i] = np.asarray(tr[k], dtype=np.float32)
        for k in _REF_KEYS:
            self._buf[k][i] = np.asarray(tr[k], dtype=np.int64)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, B, device="cpu"):
        idx = self.rng.integers(0, self.size, size=int(B))
        out = {}
        for k in _ARRAY_KEYS:
            v = self._buf[k][idx]
            if k in ("sigma", "sigma2"):
                v = v.reshape(len(idx), 1)
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).to(device)
        out["reward"] = out["reward"].reshape(-1)
        for k, name in (("ref", "refs"), ("ref2", "refs2")):
            out[name] = [(int(s), int(t)) for s, t in self._buf[k][idx]]
        out["idx"] = idx
        return out

    def live_segments(self):
        if self.size == 0:
            return set()
        return set(np.unique(self._buf["ref"][: self.size, 0]).tolist()) | \
            set(np.unique(self._buf["ref2"][: self.size, 0]).tolist())

    def get_state(self):
        st = {"capacity": self.capacity, "size": self.size, "ptr": self.ptr,
              "rng": self.rng.bit_generator.state}
        if self._buf is not None:
            st["arrays"] = {k: v[: self.size] for k, v in self._buf.items()}
            st["shapes"] = {k: v.shape[1:] for k, v in self._buf.items()}
        return st


def _same(o, n, B):
    so, sn = o.sample(B), n.sample(B)
    assert list(so) == list(sn)
    for k in so:
        x, y = so[k], sn[k]
        if torch.is_tensor(x):
            assert x.dtype == y.dtype and x.shape == y.shape and x.numpy().tobytes() == y.numpy().tobytes(), k
        elif isinstance(x, np.ndarray):
            assert x.tobytes() == y.tobytes(), k
        else:
            assert x == y, k
    assert o.rng.bit_generator.state == n.rng.bit_generator.state
    assert o.live_segments() == n.live_segments()
    A = n.arrays()
    for k in A:
        assert A[k].tobytes() == o._buf[k][: o.size].tobytes(), k


def _chain(cap, n_push, ep_len, seed, scalar_sigma=False):
    """Transitions chained like the controller's: s'_i = s_{i+1}, broken at episode ends
    (the final transition of an episode is not pushed), several wrap-arounds."""
    g = np.random.default_rng(seed)
    o, n = _OldReplay(cap, np.random.default_rng(3)), rp.Replay(cap, np.random.default_rng(3))

    def st(t):
        return dict(S_sys=g.standard_normal(49), tf=g.standard_normal((8, 5)),
                    sigma=0.0 if scalar_sigma else g.standard_normal(), ref=(t // 100, t))
    cur = st(0)
    for i in range(n_push):
        if i % ep_len == 0:
            cur = st(i)
        nxt = st(i + 1)
        tr = dict(S_sys=cur["S_sys"], tf=cur["tf"], a=g.standard_normal(4), reward=g.standard_normal(),
                  sigma=cur["sigma"], ref=cur["ref"], S_sys2=nxt["S_sys"], tf2=nxt["tf"],
                  sigma2=nxt["sigma"], ref2=nxt["ref"])
        o.push(tr)
        n.push(tr)
        cur = nxt
        if i % 53 == 0:
            _same(o, n, 32)
    return o, n


def test_next_state_by_index_is_byte_identical():
    for cap, n_push, ep_len, sc in [(200, 1500, 37, False), (300, 1200, 90, True), (64, 400, 1, False)]:
        o, n = _chain(cap, n_push, ep_len, cap, sc)
        _same(o, n, 256)
        assert n._nx[: n.size].sum() > 0 or ep_len == 1
        assert len(n._side) <= n.size


def test_new_format_roundtrip_and_old_format_conversion():
    o, n = _chain(150, 700, 41, 5)
    n2 = rp.Replay(150, np.random.default_rng(0))
    n2.set_state(n.get_state())                     # new format
    o_ref = _OldReplay(150, np.random.default_rng(0))
    o_ref.__dict__.update(capacity=o.capacity, size=o.size, ptr=o.ptr,
                          _buf={k: v.copy() for k, v in o._buf.items()})
    o_ref.rng.bit_generator.state = o.rng.bit_generator.state
    _same(o_ref, n2, 128)
    c = rp.Replay(150, np.random.default_rng(0))
    c.set_state(o.get_state())                      # old format (next state stored as copies)
    _same(o, c, 128)
    # after the conversion both keep accepting pushes identically
    x = dict(S_sys=np.ones(49), tf=np.ones((8, 5)), a=np.ones(4), reward=1.0, sigma=0.5, ref=(9, 9),
             S_sys2=np.zeros(49), tf2=np.zeros((8, 5)), sigma2=0.25, ref2=(9, 10))
    o.push(x)
    c.push(x)
    _same(o, c, 64)


def test_checkpoint_formats_and_map_location(tmp_path):
    st = {"ep": 3, "arr": np.arange(10, dtype=np.float32), "t": torch.arange(4.0),
          "nested": [{"w": torch.ones(2)}], "rng": np.random.default_rng(1).bit_generator.state}
    p = tmp_path / "new.pt"
    ck.save(p, st)
    assert p.read_bytes()[:8] == ck._ZSTD_MAGIC
    s = ck.load(p)
    assert s["ep"] == 3 and s["arr"].tobytes() == st["arr"].tobytes() and torch.equal(s["t"], st["t"])
    assert s["rng"] == st["rng"]
    m = ck.load(p, map_location="meta")             # map_location reaches nested tensors
    assert m["t"].device.type == "meta" and m["nested"][0]["w"].device.type == "meta"
    q = tmp_path / "old_torch.pt"                   # older format: torch.save zip
    with open(q, "wb") as fh:
        torch.save(st, fh)
    s = ck.load(q)
    assert s["arr"].tobytes() == st["arr"].tobytes() and torch.equal(s["t"], st["t"])
    r = tmp_path / "plain.pt"                       # torch-free fallback format
    with open(r, "wb") as fh:
        fh.write(ck._PICKLE_MAGIC)
        pickle.dump(st, fh, protocol=5)
    s = ck.load(r, map_location="meta")
    assert s["t"].device.type == "meta"
    assert len(ck._ZSTD_MAGIC) == len(ck._PICKLE_MAGIC)
