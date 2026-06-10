"""Cross-platform hardware detection and local-model recommendation.

This module lets the system stay model-agnostic: instead of hardcoding a single
Ollama model, it detects how much RAM the host has (on macOS, Windows, or Linux)
and recommends sensible "fast" and "synthesis" models that the machine can run
comfortably. Everything here is pure standard library so it imports cheaply and
has no third-party dependencies.

Example:
    from investment_monitor.analysis.hardware import recommend_models

    rec = recommend_models()
    print(rec.fast)        # e.g. "qwen2.5:7b"
    print(rec.synthesis)   # e.g. "qwen2.5:32b"
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

from loguru import logger

# Bytes-per-GiB, used consistently so "48 GB" of RAM reports as ~48.0.
_BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class ModelRecommendation:
    """A recommended pair of local Ollama models for a given machine.

    Attributes:
        fast: Small/quick model for high-frequency tasks (news relevance,
            sentiment, factor scoring). Optimized for speed.
        synthesis: Larger model for low-frequency reasoning tasks (weekly
            synthesis, research reports). Optimized for quality.
        ram_gb: Detected total system RAM in GiB (best-effort).
        tier: Human-readable tier label (e.g. "high", "minimal").
    """

    fast: str
    synthesis: str
    ram_gb: float
    tier: str


# Tiers ordered from most to least capable. The first tier whose ``min_ram_gb``
# is satisfied wins. Model names are chosen to be widely available on the Ollama
# registry; users can always override via configuration.
#   (min_ram_gb, fast_model, synthesis_model, tier_name)
_TIERS: tuple[tuple[float, str, str, str], ...] = (
    (60.0, "qwen2.5:7b", "qwen2.5:72b", "workstation"),
    (28.0, "qwen2.5:7b", "qwen2.5:32b", "high"),
    (14.0, "qwen2.5:7b", "qwen2.5:14b", "medium"),
    (7.0, "llama3.1:8b", "qwen2.5:7b", "low"),
    (0.0, "phi3:mini", "phi3:mini", "minimal"),
)

# Used when RAM can't be detected at all - stay conservative.
_FALLBACK_RAM_GB = 8.0


def total_ram_gb() -> float | None:
    """Return total physical RAM in GiB, or None if it can't be determined.

    Tries, in order: psutil (if installed), the platform-native method
    (``/proc/meminfo`` on Linux, ``sysctl hw.memsize`` on macOS,
    ``GlobalMemoryStatusEx`` on Windows), then a POSIX ``sysconf`` fallback.
    """
    # 1) psutil is the most portable if it happens to be installed.
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / _BYTES_PER_GIB
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # MemTotal is reported in kB.
                        kb = int(line.split()[1])
                        return kb * 1024 / _BYTES_PER_GIB
        elif system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
            return int(out) / _BYTES_PER_GIB
        elif system == "Windows":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / _BYTES_PER_GIB
    except Exception as e:
        logger.debug(f"Platform RAM detection failed: {e}")

    # 3) POSIX fallback (works on many Linux/macOS configs).
    try:
        return (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _BYTES_PER_GIB
        )
    except (ValueError, OSError, AttributeError):
        return None


def recommend_models(ram_gb: float | None = None) -> ModelRecommendation:
    """Recommend fast/synthesis models based on available RAM.

    Args:
        ram_gb: Override the detected RAM (mainly for testing). When None, RAM
            is auto-detected; if detection fails, a conservative fallback is used.

    Returns:
        A ModelRecommendation for the closest matching capability tier.
    """
    detected = ram_gb if ram_gb is not None else total_ram_gb()
    effective = detected if detected is not None else _FALLBACK_RAM_GB

    for min_ram, fast, synthesis, tier in _TIERS:
        if effective >= min_ram:
            return ModelRecommendation(
                fast=fast,
                synthesis=synthesis,
                ram_gb=round(effective, 1),
                tier=tier,
            )

    # _TIERS ends with a 0.0 floor, so this is unreachable in practice.
    min_ram, fast, synthesis, tier = _TIERS[-1]
    return ModelRecommendation(
        fast=fast, synthesis=synthesis, ram_gb=round(effective, 1), tier=tier
    )
