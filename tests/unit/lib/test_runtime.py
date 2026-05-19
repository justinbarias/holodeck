"""Unit tests for holodeck.lib.runtime (spec 034 P1a)."""

from __future__ import annotations

from pathlib import Path

import pytest

from holodeck.lib import runtime


@pytest.mark.unit
class TestCpuQuota:
    """cgroup-aware CPU quota detection."""

    def test_cgroups_v2_quota_returns_fractional_cores(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """cgroups v2 cpu.max with quota and period yields quota/period."""
        v2 = tmp_path / "cpu.max"
        v2.write_text("50000 100000\n")  # 0.5 CPU
        monkeypatch.setattr(runtime, "_CGROUP_V2_CPU_MAX", v2)

        assert runtime.cpu_quota() == 0.5

    def test_cgroups_v2_quota_max_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """cgroups v2 cpu.max == "max <period>" means unconstrained → fallback."""
        v2 = tmp_path / "cpu.max"
        v2.write_text("max 100000\n")
        monkeypatch.setattr(runtime, "_CGROUP_V2_CPU_MAX", v2)
        # Block v1 fallback too
        monkeypatch.setattr(runtime, "_CGROUP_V1_QUOTA", tmp_path / "missing-quota")
        monkeypatch.setattr(runtime, "_CGROUP_V1_PERIOD", tmp_path / "missing-period")

        # Falls back to os.cpu_count() (or 1), which is always ≥ 1
        assert runtime.cpu_quota() >= 1.0

    def test_cgroups_v1_quota_used_when_v2_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """cgroups v1 quota/period read when v2 file is missing."""
        monkeypatch.setattr(runtime, "_CGROUP_V2_CPU_MAX", tmp_path / "missing-v2")
        v1_quota = tmp_path / "cpu.cfs_quota_us"
        v1_period = tmp_path / "cpu.cfs_period_us"
        v1_quota.write_text("200000\n")  # 2.0 CPU
        v1_period.write_text("100000\n")
        monkeypatch.setattr(runtime, "_CGROUP_V1_QUOTA", v1_quota)
        monkeypatch.setattr(runtime, "_CGROUP_V1_PERIOD", v1_period)

        assert runtime.cpu_quota() == 2.0

    def test_cgroups_v1_negative_quota_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """cgroups v1 quota=-1 means unconstrained → fallback."""
        monkeypatch.setattr(runtime, "_CGROUP_V2_CPU_MAX", tmp_path / "missing-v2")
        v1_quota = tmp_path / "cpu.cfs_quota_us"
        v1_period = tmp_path / "cpu.cfs_period_us"
        v1_quota.write_text("-1\n")
        v1_period.write_text("100000\n")
        monkeypatch.setattr(runtime, "_CGROUP_V1_QUOTA", v1_quota)
        monkeypatch.setattr(runtime, "_CGROUP_V1_PERIOD", v1_period)

        assert runtime.cpu_quota() >= 1.0

    def test_no_cgroup_files_falls_back_to_os_cpu_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Missing cgroup files → os.cpu_count() (covers macOS/Windows dev)."""
        monkeypatch.setattr(runtime, "_CGROUP_V2_CPU_MAX", tmp_path / "nope-v2")
        monkeypatch.setattr(runtime, "_CGROUP_V1_QUOTA", tmp_path / "nope-v1q")
        monkeypatch.setattr(runtime, "_CGROUP_V1_PERIOD", tmp_path / "nope-v1p")

        result = runtime.cpu_quota()
        assert result >= 1.0


@pytest.mark.unit
class TestDerivedSessionCap:
    """Legacy CPU-based derivation, kept for compatibility."""

    @pytest.mark.parametrize(
        ("cores", "expected"),
        [
            (0.25, 1),  # floors to 1 even on tiny replicas
            (0.5, 1),
            (1.0, 2),
            (1.5, 3),
            (2.0, 4),
            (4.0, 8),
        ],
    )
    def test_derivation_floors_at_one(self, cores: float, expected: int) -> None:
        assert runtime.derived_session_cap(cores) == expected

    def test_custom_multiplier(self) -> None:
        """Multiplier is overridable for future tuning."""
        assert runtime.derived_session_cap(1.0, multiplier=4) == 4


_MIB = 1024 * 1024


@pytest.mark.unit
class TestMemoryLimitBytes:
    """cgroup-aware memory limit detection."""

    def test_cgroups_v2_memory_max_returns_bytes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        v2 = tmp_path / "memory.max"
        v2.write_text(f"{2 * 1024 * _MIB}\n")  # 2 GiB
        monkeypatch.setattr(runtime, "_CGROUP_V2_MEMORY_MAX", v2)

        assert runtime.memory_limit_bytes() == 2 * 1024 * _MIB

    def test_cgroups_v2_memory_max_literal_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        v2 = tmp_path / "memory.max"
        v2.write_text("max\n")
        monkeypatch.setattr(runtime, "_CGROUP_V2_MEMORY_MAX", v2)
        monkeypatch.setattr(runtime, "_CGROUP_V1_MEMORY_LIMIT", tmp_path / "missing-v1")

        assert runtime.memory_limit_bytes() is None

    def test_cgroups_v1_memory_used_when_v2_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(runtime, "_CGROUP_V2_MEMORY_MAX", tmp_path / "missing-v2")
        v1 = tmp_path / "memory.limit_in_bytes"
        v1.write_text(f"{512 * _MIB}\n")
        monkeypatch.setattr(runtime, "_CGROUP_V1_MEMORY_LIMIT", v1)

        assert runtime.memory_limit_bytes() == 512 * _MIB

    def test_cgroups_v1_unbounded_sentinel_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """v1 reports a near-INT64-max sentinel when unlimited; treat as None."""
        monkeypatch.setattr(runtime, "_CGROUP_V2_MEMORY_MAX", tmp_path / "missing-v2")
        v1 = tmp_path / "memory.limit_in_bytes"
        v1.write_text("9223372036854771712\n")  # kernel's "unlimited" sentinel
        monkeypatch.setattr(runtime, "_CGROUP_V1_MEMORY_LIMIT", v1)

        assert runtime.memory_limit_bytes() is None

    def test_no_cgroup_files_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(runtime, "_CGROUP_V2_MEMORY_MAX", tmp_path / "nope-v2")
        monkeypatch.setattr(runtime, "_CGROUP_V1_MEMORY_LIMIT", tmp_path / "nope-v1")

        assert runtime.memory_limit_bytes() is None


@pytest.mark.unit
class TestDerivedSessionCapFromMemory:
    """Session cap derivation from memory budget."""

    @pytest.mark.parametrize(
        ("mem_mib", "per_session_mib", "expected"),
        [
            # Default 400 MiB baseline + 200 MiB/session
            (1024, 200, 3),  # (1024-400)/200 = 3
            (2048, 200, 8),  # (2048-400)/200 = 8
            (4096, 200, 18),  # (4096-400)/200 = 18
            (512, 200, 1),  # tiny replica — floors to 1
            (200, 200, 1),  # below baseline — floors to 1
        ],
    )
    def test_derivation_uses_baseline_and_per_session(
        self, mem_mib: int, per_session_mib: int, expected: int
    ) -> None:
        assert (
            runtime.derived_session_cap_from_memory(
                mem_mib * _MIB, per_session_bytes=per_session_mib * _MIB
            )
            == expected
        )

    def test_custom_baseline(self) -> None:
        """Baseline reservation is overridable."""
        # 2 GiB - 1 GiB baseline = 1 GiB / 200 MiB = 5
        cap = runtime.derived_session_cap_from_memory(
            2048 * _MIB,
            baseline_bytes=1024 * _MIB,
            per_session_bytes=200 * _MIB,
        )
        assert cap == 5

    def test_floors_at_one_even_with_zero_headroom(self) -> None:
        """Below-baseline memory still yields a cap of 1, not 0."""
        cap = runtime.derived_session_cap_from_memory(
            100 * _MIB,
            baseline_bytes=400 * _MIB,
            per_session_bytes=200 * _MIB,
        )
        assert cap == 1
