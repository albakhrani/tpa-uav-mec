"""Atomic checkpoints.

    save(path, state)   -> stream to <path>.tmp, flush + fsync, os.replace(tmp, path)
    load(path)          -> dict
    rolling_path(run_dir), final_path(run_dir), best_path(run_dir)
    prune(run_dir)      -> delete the rolling resume checkpoint (+ stray .tmp) once a run is done

A crash at any point leaves either the previous complete file or the new
complete file at ``path``; a half-written ``.tmp`` is never read by ``load``.

Serialisation: pickle protocol 5 streamed through a zstd stream (pyarrow) straight
into the file, so no in-memory copy of the state is made (``torch.save`` pickles the
whole state into a buffer first: +14.7 GB at a full replay buffer). Without pyarrow
the pickle is written uncompressed. ``load`` reads all three formats (zstd pickle,
plain pickle, legacy ``torch.save`` zip) and applies ``map_location`` to every
tensor in each. Pickled tensors are rebuilt on the device they were saved from
before being mapped, so a checkpoint holding CUDA tensors needs CUDA to load.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

ROLLING_NAME = "ckpt_rolling.pt"
FINAL_NAME = "ckpt_final.pt"
BEST_NAME = "ckpt_best.pt"
TMP_SUFFIX = ".tmp"
_PICKLE_MAGIC = b"TPAPKL1\n"
_ZSTD_MAGIC = b"TPAPKZ1\n"          # same length as _PICKLE_MAGIC
_REPLACE_RETRIES = 50          # Windows: target may be briefly locked (AV scanner, indexer)
_REPLACE_BACKOFF_S = 0.02


def rolling_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / ROLLING_NAME


def final_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / FINAL_NAME


def best_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / BEST_NAME


def _tmp_of(path: Path) -> Path:
    return path.with_name(path.name + TMP_SUFFIX)


class _KeepOpen:
    """File proxy whose close() only flushes: closing the zstd stream must not close
    the file that save() still fsyncs."""
    closed = False

    def __init__(self, fh):
        self._fh = fh

    def write(self, b):
        return self._fh.write(b)

    def flush(self):
        self._fh.flush()

    def tell(self):
        return self._fh.tell()

    def close(self):
        self._fh.flush()


def _dump(state: dict, fh) -> None:
    try:
        import pyarrow as pa
    except ImportError:
        fh.write(_PICKLE_MAGIC)
        pickle.dump(state, fh, protocol=5)
        return
    fh.write(_ZSTD_MAGIC)
    zs = pa.CompressedOutputStream(pa.PythonFile(_KeepOpen(fh), mode="w"), "zstd")
    pickle.dump(state, zs, protocol=5)
    zs.close()


def _map_location(o, loc):
    """Move every tensor of a freshly unpickled state to ``loc``; dicts and lists are
    updated in place so container types (OrderedDict state dicts) are kept."""
    import torch
    if isinstance(o, torch.Tensor):
        return o.to(loc)
    if isinstance(o, dict):
        for k, v in o.items():
            nv = _map_location(v, loc)
            if nv is not v:
                o[k] = nv
        return o
    if isinstance(o, list):
        for i, v in enumerate(o):
            nv = _map_location(v, loc)
            if nv is not v:
                o[i] = nv
        return o
    if isinstance(o, tuple):
        nv = [_map_location(v, loc) for v in o]
        if any(a is not b for a, b in zip(nv, o)):
            return type(o)(*nv) if hasattr(o, "_fields") else tuple(nv)
    return o


def _replace(src: Path, dst: Path) -> None:
    """os.replace with a short retry loop for transient Windows sharing violations."""
    for i in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S)


def _fsync_dir(d: Path) -> None:
    """fsync the directory entry (POSIX). Windows cannot open directories; NTFS
    journals the rename, so this is a no-op there."""
    if os.name == "nt":
        return
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path: str | Path, state: dict) -> Path:
    """Atomically write ``state`` to ``path`` (tmp + fsync + os.replace)."""
    if not isinstance(state, dict):
        raise TypeError("checkpoint state must be a dict")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_of(path)
    if tmp.exists():                       # leftover from an earlier crash
        tmp.unlink()
    with open(tmp, "wb") as fh:
        _dump(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    _replace(tmp, path)
    _fsync_dir(path.parent)
    return path


def load(path: str | Path, map_location: str | None = "cpu") -> dict:
    """Load a checkpoint written by :func:`save` (never reads the ``.tmp``)."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(len(_PICKLE_MAGIC))
        if head in (_ZSTD_MAGIC, _PICKLE_MAGIC):
            if head == _ZSTD_MAGIC:
                import pyarrow as pa
                st = pickle.load(pa.CompressedInputStream(pa.PythonFile(fh, mode="r"), "zstd"))
            else:
                st = pickle.load(fh)
            return _map_location(st, map_location) if map_location is not None else st
        fh.seek(0)
        import torch
        # weights_only=False: checkpoints hold numpy RNG states and plain dicts
        # written by this project only (trusted source).
        return torch.load(fh, map_location=map_location, weights_only=False)


def exists(path: str | Path) -> bool:
    return Path(path).is_file()


def resume_path(run_dir: str | Path) -> Path | None:
    """The rolling checkpoint to resume from, or None if the run has none."""
    p = rolling_path(run_dir)
    return p if p.is_file() else None


def save_rolling(run_dir: str | Path, state: dict) -> Path:
    return save(rolling_path(run_dir), state)


def save_final(run_dir: str | Path, state: dict) -> Path:
    return save(final_path(run_dir), state)


def save_best(run_dir: str | Path, state: dict) -> Path:
    return save(best_path(run_dir), state)


def prune(run_dir: str | Path) -> int:
    """Delete the rolling checkpoint and any stray ``*.tmp`` in ``run_dir``.

    Call only after the run is complete (final/best are kept). Returns bytes freed.
    """
    run_dir = Path(run_dir)
    freed = 0
    victims = [rolling_path(run_dir)] + list(run_dir.glob("*" + TMP_SUFFIX))
    for p in victims:
        if p.is_file():
            freed += p.stat().st_size
            p.unlink()
    return freed
