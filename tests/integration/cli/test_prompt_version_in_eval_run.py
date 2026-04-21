"""Integration test: frontmatter-annotated instructions land on EvalRun.

T113 (031-eval-runs-dashboard US2). Runs ``holodeck test`` against a fixture
agent whose ``instructions.file`` carries frontmatter; asserts the persisted
``EvalRun.metadata.prompt_version`` reflects the frontmatter values.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.test import test
from holodeck.models.eval_run import EvalRun
from holodeck.models.test_result import ReportSummary, TestReport, TestResult


def _make_report() -> TestReport:
    return TestReport(
        agent_name="my-agent",
        agent_config_path="agent.yaml",
        results=[
            TestResult(
                test_name="t0",
                test_input="hi",
                agent_response="hi back",
                metric_results=[],
                passed=True,
                execution_time_ms=10,
                timestamp="2026-04-18T14:22:09.812Z",
            )
        ],
        summary=ReportSummary(
            total_tests=1,
            passed=1,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=10,
        ),
        timestamp="2026-04-18T14:22:09.812Z",
        holodeck_version="0.1.0",
    )


@pytest.fixture
def patched_executor():
    with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor:
        instance = MagicMock()
        instance.execute_tests = AsyncMock(return_value=_make_report())
        instance.shutdown = AsyncMock()
        mock_executor.return_value = instance
        yield instance


@pytest.mark.integration
def test_frontmatter_version_and_tags_persisted_on_eval_run(
    tmp_path: Path, patched_executor, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-real")

    instructions = dedent("""\
        ---
        version: "1.2"
        author: jane
        description: Customer support prompt
        tags:
          - support
          - v1
        ---
        You are a helpful support agent.
        """)
    (tmp_path / "instructions.md").write_text(instructions)

    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(dedent("""\
            name: my-agent
            model:
              provider: openai
              name: gpt-4o
              api_key: ${OPENAI_API_KEY}
            instructions:
              file: instructions.md
            test_cases:
              - name: greeting
                input: "Say hi"
            """))

    runner = CliRunner()
    result = runner.invoke(test, [str(agent_yaml)])
    assert result.exit_code == 0, result.output

    files = list((tmp_path / "results" / "my-agent").glob("*.json"))
    assert len(files) == 1
    run = EvalRun.model_validate_json(files[0].read_text())

    pv = run.metadata.prompt_version
    assert pv.version == "1.2"
    assert pv.author == "jane"
    assert pv.description == "Customer support prompt"
    assert pv.tags == ["support", "v1"]
    assert pv.source == "file"
    assert pv.file_path is not None
    assert pv.file_path.endswith("instructions.md")
    # Guard against US1 stub sentinels.
    assert pv.body_hash != "0" * 64
    assert pv.version != "auto-00000000"
