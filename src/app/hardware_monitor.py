from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any


def _read_proc(path: str) -> dict[str, Any]:
    """Parse key: value lines from /proc files."""
    result: dict[str, Any] = {}
    try:
        with open(path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip()] = v.strip()
    except OSError:
        pass
    return result


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5).decode()
    except Exception:
        return ""


def snapshot() -> dict[str, Any]:
    """Capture a point-in-time snapshot of GPU, CPU, RAM metrics.

    Returns an empty dict on non-Linux or if nothing is readable
    (e.g. during local tests or non-Modal environments).
    """
    info: dict[str, Any] = {}

    # --- GPU via nvidia-smi ---
    raw = _run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    if raw:
        gpus: list[dict[str, Any]] = []
        for line in raw.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                    "name": parts[1] if len(parts) > 1 else "",
                    "gpu_util_pct": _float_or(parts[2]),
                    "mem_used_mib": _float_or(parts[3]),
                    "mem_total_mib": _float_or(parts[4]),
                    "temp_c": _float_or(parts[5]) if len(parts) > 5 else None,
                    "power_w": _float_or(parts[6]) if len(parts) > 6 else None,
                })
        if gpus:
            info["gpu"] = gpus[0] if len(gpus) == 1 else gpus

    # --- RAM via /proc/meminfo ---
    mem = _read_proc("/proc/meminfo")
    if mem:
        info["ram"] = {
            "total_kb": _int_or(mem.get("MemTotal", "").replace(" kB", "")),
            "available_kb": _int_or(mem.get("MemAvailable", "").replace(" kB", "")),
            "free_kb": _int_or(mem.get("MemFree", "").replace(" kB", "")),
        }
        if info["ram"].get("total_kb") and info["ram"].get("available_kb"):
            used = info["ram"]["total_kb"] - info["ram"]["available_kb"]
            info["ram"]["used_kb"] = used
            info["ram"]["used_pct"] = round(used / info["ram"]["total_kb"] * 100, 1)

    # --- CPU via /proc/stat ---
    stat = _read_proc("/proc/stat")
    cpu_line = stat.get("cpu", "")
    if cpu_line:
        fields = cpu_line.split()
        if len(fields) >= 5:
            try:
                user = int(fields[1])
                nice = int(fields[2])
                system = int(fields[3])
                idle = int(fields[4])
                total = user + nice + system + idle
                # Store twice for delta calculation, or just report idle ratio
                cpu_snapshot = _CPU_SNAPSHOT
                if cpu_snapshot is None:
                    _set_cpu_snapshot(user, nice, system, idle, total)
                    info["cpu"] = {"idle_pct": 100.0, "user_pct": 0.0, "system_pct": 0.0}
                else:
                    d_total = total - cpu_snapshot["total"]
                    d_idle = idle - cpu_snapshot["idle"]
                    d_user = user - cpu_snapshot["user"]
                    d_system = system - cpu_snapshot["system"]
                    if d_total > 0:
                        info["cpu"] = {
                            "user_pct": round(d_user / d_total * 100, 1),
                            "system_pct": round(d_system / d_total * 100, 1),
                            "idle_pct": round(d_idle / d_total * 100, 1),
                        }
                    _set_cpu_snapshot(user, nice, system, idle, total)
            except (ValueError, IndexError):
                pass

    # --- vLLM process info ---
    info["vllm_procs"] = _count_vllm_procs()

    return info


def reset_cpu_snapshot() -> None:
    _CPU_SNAPSHOT.clear()


# --- Internal helpers ---

_CPU_SNAPSHOT: dict[str, int] = {}


def _set_cpu_snapshot(user: int, nice: int, system: int, idle: int, total: int) -> None:
    _CPU_SNAPSHOT["user"] = user
    _CPU_SNAPSHOT["nice"] = nice
    _CPU_SNAPSHOT["system"] = system
    _CPU_SNAPSHOT["idle"] = idle
    _CPU_SNAPSHOT["total"] = total


def _count_vllm_procs() -> int:
    try:
        out = subprocess.check_output(["pgrep", "-af", "vllm"], stderr=subprocess.DEVNULL, timeout=3).decode()
        return len(out.strip().splitlines())
    except Exception:
        return 0


def _float_or(s: str) -> float | None:
    s = s.strip().strip('"')
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _int_or(s: str) -> int | None:
    s = s.strip().strip('"')
    try:
        return int(s)
    except (ValueError, TypeError):
        return None
