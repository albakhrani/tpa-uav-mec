"""Replay buffer: push/sample, wrap-around, state round-trip."""
import numpy as np
import torch

from tpa.learner.replay import Replay


def tr(k):
    return {"S_sys": np.full(49, k, float), "tf": np.full((32, 17), k, float),
            "a": np.full((32, 10), k / 10.0), "reward": float(-k),
            "S_sys2": np.full(49, k + 0.5), "tf2": np.full((32, 17), k + 0.5),
            "sigma": 0.1 * k, "sigma2": 0.1 * k + 0.05, "ref": (2, k), "ref2": (2, k + 1)}


def test_push_sample_wrap():
    rb = Replay(3, np.random.Generator(np.random.PCG64(0)))
    for k in range(5):
        rb.push(tr(k))
    assert len(rb) == 3
    b = rb.sample(16)
    ks = b["S_sys"][:, 0].numpy()
    assert set(ks.tolist()) <= {2.0, 3.0, 4.0}
    for i in range(16):
        k = int(ks[i])
        assert b["refs"][i] == (2, k) and b["refs2"][i] == (2, k + 1)
        assert float(b["reward"][i]) == -k
        assert torch.allclose(b["S_sys2"][i], torch.full((49,), k + 0.5))
        assert b["sigma"].shape == (16, 1)
    assert b["tf"].dtype == torch.float32 and b["tf"].shape == (16, 32, 17)
    assert rb.live_segments() == {2}


def test_state_roundtrip():
    rb = Replay(10, np.random.Generator(np.random.PCG64(1)))
    for k in range(7):
        rb.push(tr(k))
    st = rb.get_state()
    rb2 = Replay(10, np.random.Generator(np.random.PCG64(99)))
    rb2.set_state(st)
    b1, b2 = rb.sample(5), rb2.sample(5)
    for key in ("S_sys", "tf", "a", "reward", "S_sys2", "tf2", "sigma", "sigma2"):
        assert torch.equal(b1[key], b2[key])
    assert b1["refs"] == b2["refs"]
    rb.push(tr(7))
    rb2.push(tr(7))
    assert rb.ptr == rb2.ptr and len(rb) == len(rb2) == 8
