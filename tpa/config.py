"""Config loader: one TOML file (config/default.toml) is the single source of truth.

    cfg = load_config()                                  # defaults
    cfg = load_config(overrides={"memory.theta_p": 0.25}) # dotted overrides
    cfg.memory.theta_p, cfg.hash()

Config objects are immutable; `with_overrides` returns a new one.
"""
from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "default.toml"


_RESERVED = frozenset(("get", "to_dict", "hash", "with_overrides"))


class Section:
    """Read-only attribute view of one TOML table.

    Every key is also bound as a plain instance attribute at construction
    (nested tables as Section objects), so `cfg.channel.phi_jl` is two dictionary lookups
    instead of two `__getattr__` calls; the decision path reads ~8,000 config values per
    slot. Values are the same objects; `__getattr__` remains the fallback (unknown key ->
    AttributeError); the view stays immutable (`__setattr__` raises)."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)
        for k, v in data.items():
            if isinstance(k, str) and k.isidentifier() and not k.startswith("_") and k not in _RESERVED:
                object.__setattr__(self, k, Section(v) if isinstance(v, dict) else v)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):                      # keeps copy/pickle from recursing
            raise AttributeError(key)
        try:
            v = self._data[key]
        except KeyError:
            raise AttributeError(f"config has no key {key!r}") from None
        return Section(v) if isinstance(v, dict) else v

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __setattr__(self, key, value):
        raise AttributeError("config is immutable; use with_overrides")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)


    def __getstate__(self):
        return self._data

    def __setstate__(self, data):
        self.__init__(data)


class Config(Section):
    def hash(self) -> str:
        payload = json.dumps(self._data, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        data = copy.deepcopy(self._data)
        for dotted, value in overrides.items():
            node = data
            *path, leaf = dotted.split(".")
            for k in path:
                node = node.setdefault(k, {})
            if leaf not in node and not dotted.startswith("t_mem_ms.") and dotted not in OVERRIDE_ONLY_KEYS:
                raise KeyError(f"unknown config key {dotted!r}")
            node[leaf] = value
        return Config(data)


# Keys that exist only as row overrides (not in config/default.toml, so that the
# config hash of every other arm's rows is unchanged): A16's consolidation delay Delta (slots).
OVERRIDE_ONLY_KEYS = {"memory.delay_slots"}


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> Config:
    with open(path or DEFAULT_PATH, "rb") as fh:
        cfg = Config(tomllib.load(fh))
    return cfg.with_overrides(overrides) if overrides else cfg
