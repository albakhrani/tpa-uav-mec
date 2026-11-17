"""Identify P-cores and E-cores from the OS, not by assumption.

Reads ``GetSystemCpuSetInformation`` (kernel32) through ctypes. For every
logical CPU it records Group, LogicalProcessorIndex, CoreIndex (logical CPUs
with the same CoreIndex share a physical core) and EfficiencyClass (higher =
more performant; on hybrid Intel parts P-cores have the higher class).

Writes ``runs/cpu_topology.json`` with:
  p_core_logical      one logical CPU per physical P-core (lowest index of the core)
  p_core_all_logical  every P-core logical CPU (incl. hyper-thread siblings)
  e_core_logical      every E-core logical CPU (one per physical E-core)
  cores               per physical core: class, logical CPUs
  cpus                per logical CPU: the raw fields

Usage:  python scripts/cpu_topology.py [--out runs/cpu_topology.json]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "cpu_topology.json"

_CPU_SET_INFORMATION = 0           # CpuSetInformation (CPU_SET_INFORMATION_TYPE)
_ERROR_INSUFFICIENT_BUFFER = 122
_GROUP_SIZE = 64                   # logical CPUs per processor group (psutil index = g*64 + idx)

# SYSTEM_CPU_SET_INFORMATION layout (winnt.h):
#   DWORD Size; DWORD Type; DWORD Id; WORD Group; BYTE LogicalProcessorIndex;
#   BYTE CoreIndex; BYTE LastLevelCacheIndex; BYTE NumaNodeIndex; BYTE EfficiencyClass;
#   BYTE AllFlags; DWORD Reserved|SchedulingClass; DWORD64 AllocationTag
_FMT = "<IIIHBBBBBBIQ"
_FMT_SIZE = struct.calcsize(_FMT)  # 32


def read_cpu_sets() -> list[dict]:
    if os.name != "nt":
        raise OSError("GetSystemCpuSetInformation is a Windows API")
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = k32.GetSystemCpuSetInformation
    fn.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                   ctypes.c_void_p, ctypes.c_ulong]
    fn.restype = ctypes.c_int
    need = ctypes.c_ulong(0)
    fn(None, 0, ctypes.byref(need), None, 0)
    err = ctypes.get_last_error()
    if need.value == 0:
        raise OSError(f"GetSystemCpuSetInformation size query failed (error {err})")
    buf = ctypes.create_string_buffer(need.value)
    if not fn(buf, need, ctypes.byref(need), None, 0):
        raise OSError(f"GetSystemCpuSetInformation failed (error {ctypes.get_last_error()})")
    raw = buf.raw[:need.value]
    cpus, off = [], 0
    while off < len(raw):
        size = struct.unpack_from("<I", raw, off)[0]
        if size == 0:
            break
        if size >= _FMT_SIZE:
            (sz, typ, cid, group, lpi, core, llc, numa, eff, flags, sched,
             tag) = struct.unpack_from(_FMT, raw, off)
            if typ == _CPU_SET_INFORMATION:
                cpus.append({
                    "logical": group * _GROUP_SIZE + lpi, "cpu_set_id": cid, "group": group,
                    "logical_index": lpi, "core_index": core, "llc_index": llc,
                    "numa_node": numa, "efficiency_class": eff,
                    "parked": bool(flags & 0x1), "allocated": bool(flags & 0x2),
                    "realtime": bool(flags & 0x8), "scheduling_class": sched,
                })
        off += size
    return sorted(cpus, key=lambda c: c["logical"])


def classify(cpus: list[dict]) -> dict:
    classes = sorted({c["efficiency_class"] for c in cpus})
    p_class = max(classes)
    hybrid = len(classes) > 1
    cores: dict[tuple[int, int], dict] = {}
    for c in cpus:
        key = (c["group"], c["core_index"])
        e = cores.setdefault(key, {"group": c["group"], "core_index": c["core_index"],
                                   "efficiency_class": c["efficiency_class"], "logical": []})
        e["logical"].append(c["logical"])
    core_list = sorted(cores.values(), key=lambda e: min(e["logical"]))
    for e in core_list:
        e["logical"].sort()
        e["kind"] = "P" if (e["efficiency_class"] == p_class) else "E"
    p_cores = [e for e in core_list if e["kind"] == "P"]
    e_cores = [e for e in core_list if e["kind"] == "E"]
    return {
        "hybrid": hybrid,
        "efficiency_classes": classes,
        "p_efficiency_class": p_class,
        "n_logical": len(cpus),
        "n_physical": len(core_list),
        "n_p_cores": len(p_cores),
        "n_e_cores": len(e_cores),
        "p_core_logical": [e["logical"][0] for e in p_cores],
        "p_core_all_logical": sorted(x for e in p_cores for x in e["logical"]),
        "e_core_logical": sorted(x for e in e_cores for x in e["logical"]),
        "e_core_one_per_physical": [e["logical"][0] for e in e_cores],
        "cores": core_list,
    }


def build(out: Path | None = DEFAULT_OUT) -> dict:
    cpus = read_cpu_sets()
    topo = classify(cpus)
    cpu_model = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu_model = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except OSError:
        pass
    topo.update({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "cpu_model": cpu_model, "os": platform.platform(),
                 "source": "GetSystemCpuSetInformation", "cpus": cpus})
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(topo, indent=2), encoding="utf-8")
        os.replace(tmp, out)
    return topo


def load_or_build(path: Path = DEFAULT_OUT) -> dict:
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return build(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    topo = build(args.out)
    print(f"CPU: {topo['cpu_model']}")
    print(f"{'logical':>7} {'group':>5} {'core':>5} {'effcls':>6} {'kind':>4} {'parked':>6}")
    kind_of = {x: e["kind"] for e in topo["cores"] for x in e["logical"]}
    for c in topo["cpus"]:
        print(f"{c['logical']:>7} {c['group']:>5} {c['core_index']:>5} "
              f"{c['efficiency_class']:>6} {kind_of[c['logical']]:>4} {str(c['parked']):>6}")
    print(f"logical={topo['n_logical']} physical={topo['n_physical']} "
          f"P-cores={topo['n_p_cores']} E-cores={topo['n_e_cores']} hybrid={topo['hybrid']}")
    print(f"p_core_logical (one per physical P-core): {topo['p_core_logical']}")
    print(f"p_core_all_logical: {topo['p_core_all_logical']}")
    print(f"e_core_logical: {topo['e_core_logical']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
