"""Container-aware runtime introspection helpers.

The Python stdlib's :func:`os.cpu_count` reports the host's CPU count, which
is wrong inside a container with cgroup CPU limits — Java's
``Runtime.availableProcessors()`` has handled this correctly since Java 10,
but Python has not caught up. This module reads cgroup CPU and memory
limits directly so HoloDeck can size in-process concurrency to whatever the
container was actually allocated.

Used by the serve layer to derive per-replica session caps (spec 034 P1a).
The Claude session cap is derived from **memory**, not CPU: Azure Container
Apps (and some other managed runtimes) do not expose CPU limits via
``/sys/fs/cgroup/cpu.max``, but the kernel enforces memory limits so
``memory.max`` is reliably populated. The thing we're protecting against
(SDK subprocess OOM) is a memory constraint anyway.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")

_CGROUP_V2_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MEMORY_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")

# Per-active-turn and serve-baseline footprints calibrated against the spec
# 034 P4 cloud validation (2 GiB ACA replica OOMed at 4 concurrent turns).
# The active-turn estimate has to cover the SDK subprocess's steady-state
# resident set (~300 MiB Node CLI) plus the simultaneous-startup spike and
# parent-side transient work (hybrid search, rerank, context generation).
# 500 MiB lands cap=3 on a 2 GiB replica — the empirically safe ceiling.
DEFAULT_BASELINE_BYTES = 400 * 1024 * 1024  # 400 MiB: server + tools + OTEL
DEFAULT_PER_SESSION_BYTES = 500 * 1024 * 1024  # 500 MiB: one concurrent active turn


def cpu_quota() -> float:
    """Return the CPU quota available to this process, in fractional cores.

    Reads the cgroup CPU quota directly. Returns ``os.cpu_count()`` (or
    ``1.0`` if even that is unavailable) when no cgroup limit is set or
    cgroup files cannot be read — covers local dev on macOS/Windows and
    unconstrained Linux processes.

    Returns:
        Fractional CPU count (e.g. ``0.5`` for half a core, ``2.0`` for two
        full cores). Always positive.
    """
    quota = _read_cgroup_v2_quota()
    if quota is not None:
        return quota
    quota = _read_cgroup_v1_quota()
    if quota is not None:
        return quota
    return float(os.cpu_count() or 1)


def _read_cgroup_v2_quota() -> float | None:
    """Read /sys/fs/cgroup/cpu.max (cgroups v2).

    Format: ``"<quota> <period>"`` where quota is an integer or the literal
    ``"max"`` (unconstrained). Returns ``quota / period``, or ``None`` when
    the file is missing or unparseable.
    """
    try:
        raw = _CGROUP_V2_CPU_MAX.read_text().strip()
    except OSError:
        return None
    try:
        quota_str, period_str = raw.split()
        if quota_str == "max":
            return None
        return int(quota_str) / int(period_str)
    except (ValueError, ZeroDivisionError):
        logger.debug("Unparseable cgroups v2 cpu.max: %r", raw)
        return None


def _read_cgroup_v1_quota() -> float | None:
    """Read cgroups v1 cpu.cfs_quota_us / cpu.cfs_period_us.

    A negative quota means unconstrained.
    """
    try:
        quota = int(_CGROUP_V1_QUOTA.read_text().strip())
        period = int(_CGROUP_V1_PERIOD.read_text().strip())
    except (OSError, ValueError):
        return None
    if quota < 0 or period <= 0:
        return None
    return quota / period


def derived_session_cap(cpu_cores: float, multiplier: int = 2) -> int:
    """Derive a CPU-based session cap (legacy).

    Kept for compatibility with the original spec 034 P1a derivation;
    superseded by :func:`derived_session_cap_from_memory` which maps more
    directly to the OOM constraint. Not used in the default serve path.
    """
    return max(1, math.floor(cpu_cores * multiplier))


def memory_limit_bytes() -> int | None:
    """Return the memory limit (bytes) available to this process.

    Reads the cgroup memory limit directly. Returns ``None`` when no
    cgroup limit is set or cgroup files cannot be read — caller should
    fall back to a conservative default.

    Returns:
        Memory limit in bytes, or ``None`` when unbounded/unavailable.
    """
    limit = _read_cgroup_v2_memory()
    if limit is not None:
        return limit
    return _read_cgroup_v1_memory()


def _read_cgroup_v2_memory() -> int | None:
    """Read /sys/fs/cgroup/memory.max (cgroups v2).

    Contains an integer byte count or the literal ``"max"`` (unconstrained).
    """
    try:
        raw = _CGROUP_V2_MEMORY_MAX.read_text().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.debug("Unparseable cgroups v2 memory.max: %r", raw)
        return None


def _read_cgroup_v1_memory() -> int | None:
    """Read /sys/fs/cgroup/memory/memory.limit_in_bytes (cgroups v1).

    Kernels report an effectively-unlimited sentinel (very large int) when
    no limit is set; we treat anything ≥ 1 PiB as unbounded.
    """
    try:
        raw = int(_CGROUP_V1_MEMORY_LIMIT.read_text().strip())
    except (OSError, ValueError):
        return None
    if raw >= 1 << 50:  # ≥ 1 PiB → effectively unbounded
        return None
    return raw


def derived_session_cap_from_memory(
    memory_bytes: int,
    baseline_bytes: int = DEFAULT_BASELINE_BYTES,
    per_session_bytes: int = DEFAULT_PER_SESSION_BYTES,
) -> int:
    """Derive the per-replica Claude session cap from a memory budget.

    The active-turn footprint is the binding constraint for concurrency.
    Each concurrent turn spawns a fresh Node CLI subprocess (~300 MiB
    steady state) and incurs parent-side transient work (hybrid search,
    rerank, context generation). After reserving a baseline for the
    serve process, tools and OTEL exporter, the remaining headroom
    divided by the per-turn footprint gives the cap.

    Args:
        memory_bytes: Total memory limit available to the replica.
        baseline_bytes: Bytes reserved for non-session overhead.
        per_session_bytes: Estimated bytes per active SDK subprocess.

    Returns:
        Integer session cap, always ≥ 1.
    """
    usable = max(0, memory_bytes - baseline_bytes)
    return max(1, usable // per_session_bytes)
