"""Test that the writer resolves `results/` against agent_base_dir, not CWD (T018)."""

from __future__ import annotations

from pathlib import Path

import pytest

from holodeck.lib.eval_run.writer import write_eval_run
from holodeck.models.agent import Agent
from holodeck.models.eval_run import EvalRun, EvalRunMetadata, PromptVersion
from holodeck.models.test_result import ReportSummary, TestReport


def _make_run() -> EvalRun:
    agent = Agent(
        name="path-agent",
        model={"provider": "openai", "name": "gpt-4o"},
        instructions={"inline": "hi"},
    )
    report = TestReport(
        agent_name="path-agent",
        agent_config_path="agent.yaml",
        results=[],
        summary=ReportSummary(
            total_tests=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            total_duration_ms=0,
        ),
        timestamp="2026-04-18T14:22:09.812Z",
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
class TestWriterPathResolution:
    def test_resolves_against_agent_base_dir_not_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent_dir = tmp_path / "subdir"
        agent_dir.mkdir()

        # Change CWD elsewhere to prove agent_base_dir wins.
        other_cwd = tmp_path / "elsewhere"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        target = write_eval_run(_make_run(), agent_base_dir=agent_dir)

        assert target.is_relative_to(agent_dir)
        assert target.parent == agent_dir / "results" / "path-agent"
        # The CWD must NOT contain a results/ directory.
        assert not (other_cwd / "results").exists()
