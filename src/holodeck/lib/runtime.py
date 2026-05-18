"""Container-aware runtime introspection helpers.

The Python stdlib's :func:`os.cpu_count` reports the host's CPU count, which
is wrong inside a container with cgroup CPU limits — Java's
``Runtime.availableProcessors()`` has handled this correctly since Java 10,
but Python has not caught up. This module reads cgroup CPU quotas directly
so HoloDeck can size in-process concurrency to whatever the container was
actually allocated.

Used by the serve layer to derive per-replica session caps (spec 034 P1a).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


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
    """Derive the default per-replica Claude session cap from CPU quota.

    Calibrated against the ~300 MiB resident-per-subprocess footprint
    observed in the spec 034 P1 investigation: one SDK subprocess plus
    headroom for the serve process fits comfortably in 1 GiB per core when
    capped at two sessions per core.

    Args:
        cpu_cores: Fractional CPU quota for this replica.
        multiplier: Sessions allowed per CPU core (default 2).

    Returns:
        Integer session cap, always ≥ 1.
    """
    import math

    return max(1, math.floor(cpu_cores * multiplier))
