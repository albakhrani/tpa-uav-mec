import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpa.run.logio import ParquetLog, read_all  # noqa: E402
from tpa.run.manifest import code_files, code_hash, write_manifest  # noqa: E402
from tpa.run.timing import ModuleTimer  # noqa: E402

SCHEMA = {"episode": "int64", "reward": "float64", "arm": "string"}


def _ep(log, e):
    for k in range(3):
        log.append({"episode": e, "reward": e + k / 10, "arm": "A0"})
    return log.flush()


# ------------------------------------------------------------------ ParquetLog
def test_parquet_append_and_read(tmp_path):
    log = ParquetLog(tmp_path / "log", SCHEMA)
    p1 = _ep(log, 0)
    p2 = _ep(log, 1)
    assert p1.name == "part-00001.parquet" and p2.name == "part-00002.parquet"
    assert log.flush() is None
    import pyarrow.parquet as pq
    assert pq.ParquetFile(p1).metadata.row_group(0).column(0).compression.lower() == "zstd"
    df = read_all(tmp_path / "log")
    assert list(df["episode"]) == [0, 0, 0, 1, 1, 1]


def test_parquet_resume_rollback_identical(tmp_path):
    # uninterrupted: 6 episodes
    ref = ParquetLog(tmp_path / "ref", SCHEMA)
    for e in range(6):
        _ep(ref, e)
    # interrupted: checkpoint after episode 2, run to 4 + partial buffer, "crash", resume
    log = ParquetLog(tmp_path / "res", SCHEMA)
    for e in range(3):
        _ep(log, e)
    state = log.get_state()
    _ep(log, 3)
    _ep(log, 4)
    log.append({"episode": 5, "reward": 0.0, "arm": "A0"})   # lost on crash
    (tmp_path / "res" / "part-00006.parquet.tmp").write_bytes(b"half")  # crash mid-write
    del log
    log2 = ParquetLog(tmp_path / "res", SCHEMA)               # reopen: tmp removed
    assert not (tmp_path / "res" / "part-00006.parquet.tmp").exists()
    log2.set_state(state)                                     # drop parts after the checkpoint
    for e in range(3, 6):
        _ep(log2, e)
    a, b = read_all(tmp_path / "ref"), read_all(tmp_path / "res")
    assert a.equals(b)


def test_parquet_reopen_appends_next_part(tmp_path):
    log = ParquetLog(tmp_path / "l", SCHEMA)
    _ep(log, 0)
    log = ParquetLog(tmp_path / "l", SCHEMA)
    assert log.next_part == 2
    with pytest.raises(RuntimeError):
        log.append({"episode": 1, "reward": 0.0, "arm": "A0"})
        log.get_state()


# ------------------------------------------------------------------ ModuleTimer
def test_timer_counts_and_percentiles():
    t = ModuleTimer(reservoir=1000)
    for v in range(1, 101):                    # 1..100 ms
        t.add("drm", v * 1_000_000)
    s = t.summary()["drm"]
    assert s["n"] == 100
    assert s["total_ms"] == pytest.approx(5050.0)
    assert s["mean_ms"] == pytest.approx(50.5)
    assert s["p50"] == pytest.approx(np.percentile(np.arange(1, 101), 50))
    assert s["p90"] == pytest.approx(np.percentile(np.arange(1, 101), 90))
    assert s["iqr_ms"] == pytest.approx(s["p75"] - s["p25"])


def test_timer_context_and_reservoir_bounded():
    t = ModuleTimer(reservoir=64, seed=1)
    for _ in range(5000):
        with t("actor"):
            pass
    with t("mepr"):
        with t("retrieval"):
            pass
    s = t.summary()
    assert s["actor"]["n"] == 5000 and s["mepr"]["n"] == 1 and s["retrieval"]["n"] == 1
    assert s["actor"]["total_ms"] >= 0 and s["actor"]["p50"] <= s["actor"]["p90"]
    assert t._slots["actor"].res.size == 64
    # reservoir is an unbiased sample: uniform 0..9999 ns -> median near 5000
    u = ModuleTimer(reservoir=2000, seed=3)
    for v in range(100_000):
        u.add("env", v % 10_000)
    assert abs(u.summary()["env"]["p50"] * 1e6 - 5000) < 500
    with pytest.raises(RuntimeError):
        with t("drm"):
            with t("drm"):
                pass


def test_timer_overhead_small():
    from time import perf_counter_ns
    t = ModuleTimer()
    n = 20000
    t0 = perf_counter_ns()
    for _ in range(n):
        with t("env"):
            pass
    per_call_us = (perf_counter_ns() - t0) / n / 1000
    assert per_call_us < 20, per_call_us     # typically ~1 us; loose bound for a busy machine


def test_timer_disabled():
    t = ModuleTimer(enabled=False)
    with t("env"):
        pass
    assert t.summary() == {}


# ------------------------------------------------------------------ manifest
def test_code_hash_stable_and_sensitive(tmp_path):
    assert code_hash() == code_hash()
    (tmp_path / "tpa" / "sub").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "tpa" / "a.py").write_text("x = 1\n")
    (tmp_path / "tpa" / "sub" / "b.py").write_text("y = 2\n")
    (tmp_path / "config" / "default.toml").write_text("[run]\n")
    (tmp_path / "tpa" / "notes.txt").write_text("ignored")
    h1 = code_hash(tmp_path)
    assert h1 == code_hash(tmp_path)
    assert [p.name for p in code_files(tmp_path)] == ["default.toml", "a.py", "b.py"]
    (tmp_path / "tpa" / "notes.txt").write_text("changed")      # not covered
    assert code_hash(tmp_path) == h1
    (tmp_path / "config" / "default.toml").write_text("[run]\n#\n")
    h2 = code_hash(tmp_path)
    assert h2 != h1
    (tmp_path / "tpa" / "sub" / "b.py").rename(tmp_path / "tpa" / "sub" / "c.py")  # rename only
    assert code_hash(tmp_path) != h2


def test_write_manifest(tmp_path):
    from tpa.config import load_config
    cfg = load_config()
    man = write_manifest(tmp_path, cfg, extra={"run_id": "x"})
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    for k in ("python", "numpy", "torch", "nvidia_smi", "cpu_model", "cpu_logical", "affinity",
              "thread_env", "os", "code_hash", "config_hash", "default_toml_text", "created_utc"):
        assert k in on_disk, k
    assert on_disk["config_hash"] == cfg.hash() and man["extra"]["run_id"] == "x"
    assert on_disk["code_hash"] == code_hash()
    assert "OMP_NUM_THREADS" in on_disk["thread_env"]
