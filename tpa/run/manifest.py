"""Per-run manifest.

    write_manifest(run_dir, cfg, extra=None) -> dict   # also writes <run_dir>/manifest.json
    code_hash(root=None) -> str                        # sha256 over sorted tpa/**/*.py + config/default.toml

Every probe degrades gracefully: a missing torch, CUDA or nvidia-smi is
recorded as ``null`` (with the reason), never an exception.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "CUDA_VISIBLE_DEVICES",
                   "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
_NVSMI_TIMEOUT_S = 15


def code_files(root: str | Path | None = None) -> list[Path]:
    """The files covered by :func:`code_hash`, in hashing order."""
    root = Path(root) if root is not None else PROJECT_ROOT
    files = [p for p in (root / "tpa").rglob("*.py") if "__pycache__" not in p.parts]
    cfg = root / "config" / "default.toml"
    if cfg.is_file():
        files.append(cfg)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def code_hash(root: str | Path | None = None) -> str:
    """SHA-256 over every ``tpa/**/*.py`` plus ``config/default.toml``, sorted by
    relative POSIX path. Each file contributes ``path NUL size NUL bytes`` so a
    rename or a moved byte changes the hash. Raw bytes are hashed (no newline
    normalisation): the hash identifies the exact tree on disk, line endings included."""
    root = Path(root) if root is not None else PROJECT_ROOT
    h = hashlib.sha256()
    for p in code_files(root):
        data = p.read_bytes()
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(str(len(data)).encode())
        h.update(b"\0")
        h.update(data)
    return h.hexdigest()


def _cpu_model() -> str:
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _nvidia_smi() -> dict:
    q = "driver_version,name,memory.total,power.limit"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=_NVSMI_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    if out.returncode != 0:
        return {"available": False, "error": out.stderr.strip()[:500]}
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [s.strip() for s in line.split(",")]
        gpus.append(dict(zip(q.split(","), parts)))
    return {"available": True, "gpus": gpus,
            "driver_version": gpus[0].get("driver_version") if gpus else None}


def _torch_info() -> dict:
    try:
        import torch
    except ImportError as e:
        return {"version": None, "error": str(e)}
    info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception as e:  # noqa: BLE001 - probe only
        info["cudnn"] = None
        info["cudnn_error"] = str(e)
    if info["cuda_available"]:
        try:
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception as e:  # noqa: BLE001
            info["devices_error"] = str(e)
    return info


def _affinity() -> list[int] | None:
    try:
        import psutil
        return sorted(psutil.Process().cpu_affinity())
    except Exception:  # noqa: BLE001 - psutil missing or unsupported OS
        return None


def collect(cfg=None, extra: dict | None = None, root: str | Path | None = None) -> dict:
    root = Path(root) if root is not None else PROJECT_ROOT
    import numpy as np
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
    except ImportError:
        phys = None
    toml_path = root / "config" / "default.toml"
    man: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": _torch_info(),
        "nvidia_smi": _nvidia_smi(),
        "cpu_model": _cpu_model(),
        "cpu_logical": os.cpu_count(),
        "cpu_physical": phys,
        "affinity": _affinity(),
        "pid": os.getpid(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "os": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "code_hash": code_hash(root),
        "code_files": len(code_files(root)),
        "default_toml_text": toml_path.read_text(encoding="utf-8") if toml_path.is_file() else None,
    }
    if cfg is not None:
        man["config_hash"] = cfg.hash()
        man["config_effective"] = cfg.to_dict()   # default.toml + overrides actually used
    if extra:
        man["extra"] = extra
    return man


def write_manifest(run_dir: str | Path, cfg, extra: dict | None = None,
                   name: str = "manifest.json") -> dict:
    """Collect the manifest and write it atomically to ``<run_dir>/<name>``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    man = collect(cfg, extra)
    path = run_dir / name
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return man
