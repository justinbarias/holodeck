"""Unit tests for the atomic EvalRun writer (T016).

Asserts:
(a) happy path writes final file
(b) os.replace failure leaves no .tmp dangling, no partial target
(c) fsync invoked on temp fd before replace
(d) collision on pre-existing target appends 4-hex suffix
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from holodeck.lib.eval_run.writer import write_eval_run
from holodeck.models.agent import Agent
from holodeck.models.eval_run import EvalRun, EvalRunMetadata, PromptVersion
from holodeck.models.test_result import ReportSummary, TestReport


def _make_run(timestamp: str = "2026-04-18T14:22:09.812Z") -> EvalRun:
    agent = Agent(
        name="my-agent",
        model={"provider": "openai", "name": "gpt-4o"},
        instructions={"inline": "hi"},
    )
    report = TestReport(
        agent_name="my-agent",
        agent_config_path="agent.yaml",
        results=[],
        summary=ReportSummary(
            total_tests=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            total_duration_ms=0,
        ),
        timestamp=timestamp,
        holodeck_version="0.1.0",
    )
    metadata = EvalRunMetadata(
        agent_config=agent,
        prompt_version=PromptVersion(
            version="auto-12345678", source="inline", body_hash="0" * 64
        ),
        holodeck_version="0.1.0",
        cli_args=[],
        git_commit=None,
    )
    return EvalRun(report=report, metadata=metadata)


@pytest.mark.unit
class TestWriterHappyPath:
    def test_writes_file_under_results_slug(self, tmp_path: Path):
        run = _make_run()
        result = write_eval_run(run, agent_base_dir=tmp_path)
        assert result.exists()
        assert result.parent == tmp_path / "results" / "my-agent"
        assert result.suffix == ".json"
        # Filename contains hyphen-normalised timestamp.
        assert ":" not in result.name
        assert "2026-04-18T14-22-09" in result.name

    def test_creates_parent_directory(self, tmp_path: Path):
        run = _make_run()
        target_dir = tmp_path / "results" / "my-agent"
        assert not target_dir.exists()
        write_eval_run(run, agent_base_dir=tmp_path)
        assert target_dir.is_dir()


@pytest.mark.unit
class TestWriterAtomicity:
    def test_replace_failure_leaves_no_tmp(self, tmp_path: Path):
        run = _make_run()

        def _boom(*args, **kwargs):
            raise OSError("simulated mid-write failure")

        with (
            patch("holodeck.lib.eval_run.writer.os.replace", side_effect=_boom),
            pytest.raises(OSError),
        ):
            write_eval_run(run, agent_base_dir=tmp_path)

        target_dir = tmp_path / "results" / "my-agent"
        # Directory may have been created, but no .tmp residues nor target file.
        residues = list(target_dir.glob("*.tmp")) if target_dir.exists() else []
        assert residues == []
        json_files = list(target_dir.glob("*.json")) if target_dir.exists() else []
        assert json_files == []

    def test_fsync_called_before_replace(self, tmp_path: Path):
        run = _make_run()
        call_order: list[str] = []

        real_fsync = os.fsync
        real_replace = os.replace

        def tracked_fsync(fd):
            call_order.append("fsync")
            return real_fsync(fd)

        def tracked_replace(src, dst):
            call_order.append("replace")
            return real_replace(src, dst)

        with (
            patch("holodeck.lib.eval_run.writer.os.fsync", side_effect=tracked_fsync),
            patch(
                "holodeck.lib.eval_run.writer.os.replace", side_effect=tracked_replace
            ),
        ):
            write_eval_run(run, agent_base_dir=tmp_path)

        assert call_order == ["fsync", "replace"]


@pytest.mark.unit
class TestWriterCollision:
    def test_collision_appends_hex_suffix(self, tmp_path: Path):
        run = _make_run()
        first = write_eval_run(run, agent_base_dir=tmp_path)
        # Second write of an EvalRun with the same timestamp must not overwrite.
        second = write_eval_run(run, agent_base_dir=tmp_path)
        assert first.exists()
        assert second.exists()
        assert first != second
        # Collision suffix is 4 hex chars before .json.
        stem_diff = second.stem.replace(first.stem, "")
        assert stem_diff.startswith("-")
        assert len(stem_diff) == 5  # "-" + 4 hex
        for ch in stem_diff[1:]:
            assert ch in "0123456789abcdef"
