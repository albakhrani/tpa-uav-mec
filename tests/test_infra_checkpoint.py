import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpa.run import checkpoint as ck  # noqa: E402


def _state(v):
    return {"episode": v, "arr": np.arange(5) * v, "rng": np.random.Generator(np.random.PCG64(v)).bit_generator.state}


def test_roundtrip_and_paths(tmp_path):
    p = ck.rolling_path(tmp_path)
    ck.save(p, _state(3))
    s = ck.load(p)
    assert s["episode"] == 3 and np.array_equal(s["arr"], np.arange(5) * 3)
    assert s["rng"] == _state(3)["rng"]
    assert ck.final_path(tmp_path).name != ck.best_path(tmp_path).name != p.name
    assert not p.with_name(p.name + ".tmp").exists()


def test_crash_before_replace_keeps_old_file(tmp_path, monkeypatch):
    p = ck.rolling_path(tmp_path)
    ck.save(p, _state(1))
    before = p.read_bytes()

    def boom(src, dst):
        raise OSError("simulated crash before os.replace")
    monkeypatch.setattr(ck, "_replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        ck.save(p, _state(2))
    # old checkpoint intact and loadable; the half-finished tmp is what got written
    assert p.read_bytes() == before
    assert ck.load(p)["episode"] == 1
    tmp = p.with_name(p.name + ".tmp")
    assert tmp.exists()
    monkeypatch.undo()
    # next save cleans up the stale tmp and succeeds
    ck.save(p, _state(2))
    assert ck.load(p)["episode"] == 2 and not tmp.exists()


def test_crash_mid_write_keeps_old_file(tmp_path, monkeypatch):
    p = ck.final_path(tmp_path)
    ck.save(p, _state(7))

    def half_dump(state, fh):
        fh.write(b"\x80garbage-partial")
        raise KeyboardInterrupt  # killed mid-write
    monkeypatch.setattr(ck, "_dump", half_dump)
    with pytest.raises(KeyboardInterrupt):
        ck.save(p, _state(8))
    assert ck.load(p)["episode"] == 7


def test_prune(tmp_path):
    ck.save_rolling(tmp_path, _state(1))
    ck.save_final(tmp_path, _state(2))
    ck.save_best(tmp_path, _state(3))
    (tmp_path / "x.pt.tmp").write_bytes(b"junk")
    assert ck.resume_path(tmp_path) is not None
    freed = ck.prune(tmp_path)
    assert freed > 0
    assert ck.resume_path(tmp_path) is None
    assert not (tmp_path / "x.pt.tmp").exists()
    assert ck.load(ck.final_path(tmp_path))["episode"] == 2
    assert ck.load(ck.best_path(tmp_path))["episode"] == 3


def test_torch_tensors_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    p = tmp_path / "t.pt"
    lin = torch.nn.Linear(3, 2)
    ck.save(p, {"model": lin.state_dict(), "cpu_rng": torch.get_rng_state()})
    s = ck.load(p)
    for k, v in lin.state_dict().items():
        assert torch.equal(v, s["model"][k])
    assert torch.equal(s["cpu_rng"], torch.get_rng_state())
