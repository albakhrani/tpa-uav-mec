"""Resume-safe Parquet logs.

    log = ParquetLog(run_dir / "train_log", schema)   # dir of part-00001.parquet, ...
    log.append({...}); log.extend([...])
    log.flush()                                        # one zstd part per call (per episode)
    state = log.get_state()                            # store in the checkpoint
    ...crash...
    log = ParquetLog(same_dir, schema); log.set_state(state)  # drops parts written after the checkpoint
    df = read_all(same_dir)

Each part is written to ``part-NNNNN.parquet.tmp`` and moved into place with
``os.replace``, so a crash never leaves a truncated part. Rows still in the
buffer at a crash are lost by design; after ``set_state`` the trainer replays
those episodes from the checkpoint and re-logs them, giving logs identical to
an uninterrupted run.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

_PART_RE = re.compile(r"^part-(\d{5,})\.parquet$")
_TYPES = {
    "float64": pa.float64(), "float32": pa.float32(), "int64": pa.int64(),
    "int32": pa.int32(), "bool": pa.bool_(), "string": pa.string(), "str": pa.string(),
}


def make_schema(schema: pa.Schema | Mapping[str, str | pa.DataType]) -> pa.Schema:
    if isinstance(schema, pa.Schema):
        return schema
    fields = []
    for name, t in schema.items():
        fields.append(pa.field(name, _TYPES[t] if isinstance(t, str) else t))
    return pa.schema(fields)


def _parts(d: Path) -> list[tuple[int, Path]]:
    out = []
    if d.is_dir():
        for p in d.iterdir():
            m = _PART_RE.match(p.name)
            if m:
                out.append((int(m.group(1)), p))
    return sorted(out)


class ParquetLog:
    def __init__(self, path: str | Path, schema: pa.Schema | Mapping[str, str | pa.DataType],
                 compression: str = "zstd", compression_level: int | None = None):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.schema = make_schema(schema)
        self.compression = compression
        self.compression_level = compression_level
        self._buf: list[dict] = []
        for p in self.dir.glob("*.tmp"):        # half-written part from a crash
            p.unlink()
        parts = _parts(self.dir)
        self._next = (parts[-1][0] + 1) if parts else 1

    # ------------------------------------------------------------------ write
    def append(self, row: Mapping) -> None:
        self._buf.append(dict(row))

    def extend(self, rows: Iterable[Mapping]) -> None:
        self._buf.extend(dict(r) for r in rows)

    @property
    def buffered(self) -> int:
        return len(self._buf)

    @property
    def next_part(self) -> int:
        return self._next

    def flush(self) -> Path | None:
        """Write buffered rows as the next part file. No-op (None) if empty."""
        if not self._buf:
            return None
        table = pa.Table.from_pylist(self._buf, schema=self.schema)
        final = self.dir / f"part-{self._next:05d}.parquet"
        tmp = final.with_name(final.name + ".tmp")
        pq.write_table(table, tmp, compression=self.compression,
                       compression_level=self.compression_level)
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, final)
        self._buf.clear()
        self._next += 1
        return final

    # ------------------------------------------------------------------ resume
    def get_state(self) -> dict:
        """Checkpoint state. Call right after ``flush()`` at a checkpoint boundary."""
        if self._buf:
            raise RuntimeError("ParquetLog.get_state() with unflushed rows; flush first")
        return {"next_part": self._next}

    def set_state(self, state: dict) -> None:
        """Roll back to a checkpoint: delete parts numbered >= state['next_part']."""
        nxt = int(state["next_part"])
        for idx, p in _parts(self.dir):
            if idx >= nxt:
                p.unlink()
        self._buf.clear()
        self._next = nxt


def read_all(path: str | Path, recursive: bool = False):
    """Concatenate all part files under ``path`` (in part order) into a DataFrame."""
    import pandas as pd
    path = Path(path)
    if recursive:
        files = sorted(p for p in path.rglob("part-*.parquet") if _PART_RE.match(p.name))
    else:
        files = [p for _, p in _parts(path)]
    if not files:
        return pd.DataFrame()
    tables = [pq.read_table(p) for p in files]
    return pa.concat_tables(tables, promote_options="default").to_pandas()
