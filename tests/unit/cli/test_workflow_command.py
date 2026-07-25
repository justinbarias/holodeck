"""Unit tests for the ``holodeck workflow`` CLI command group (036, T6).

Drives the real CLI through ``CliRunner`` with a fake backend injected at the
``BackendSelector`` boundary, plus an autouse guard that fails the run if any
concrete backend is constructed. Covers US1 scenarios 1-2 end to end and the
error -> exit-code mapping: 2 = the workflow is misauthored or the input is
invalid, 3 = valid policy could not decide, 4 = the gate rejected the model's
output, 5 = the invocation failed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from holodeck.cli.main import main as cli
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.backends.claude_backend import ClaudeBackend
from holodeck.lib.backends.openai_agents_backend import OpenAIAgentsBackend
from holodeck.lib.workflow import edge

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "workflow" / "single_gate"

VALID_EVIDENCE: dict[str, Any] = {"net_income": 5000, "expenses": 3000}


class _FakeBackend:
    """Stands in for an ``AgentBackend`` — never talks to a model."""

    def __init__(self, result: ExecutionResult) -> None:
        self.result = result
        self.messages: list[str] = []

    async def invoke_once(
        self, message: str, context: list[dict[str, Any]] | None = None
    ) -> ExecutionResult:
        self.messages.append(message)
        return self.result

    async def teardown(self) -> None:
        return None


class _RecordingSelector:
    """Stands in for ``BackendSelector``; records every selection request."""

    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend
        self.calls: list[Any] = []

    async def select(
        self,
        agent: Any,
        tool_instances: dict[str, Any] | None = None,
        mode: str = "test",
    ) -> _FakeBackend:
        self.calls.append(agent)
        return self.backend


@pytest.fixture(autouse=True)
def forbid_real_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blow up if any concrete backend is constructed (proves zero LLM calls)."""

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "a real backend was constructed — this test must never reach an LLM"
        )

    monkeypatch.setattr(ClaudeBackend, "__init__", _boom)
    monkeypatch.setattr(OpenAIAgentsBackend, "__init__", _boom)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def workflow_dir(tmp_path: Path) -> Path:
    """A writable copy of the single-gate fixture workflow."""
    shutil.copytree(FIXTURE_DIR, tmp_path / "wf")
    return tmp_path / "wf"


def _install_result(
    monkeypatch: pytest.MonkeyPatch, result: ExecutionResult
) -> _RecordingSelector:
    """Inject a fake backend returning ``result``."""
    selector = _RecordingSelector(_FakeBackend(result))
    monkeypatch.setattr(edge, "BackendSelector", selector)
    return selector


def _install_backend(
    monkeypatch: pytest.MonkeyPatch, structured_output: dict[str, Any] | None
) -> _RecordingSelector:
    """Inject a fake backend returning ``structured_output``."""
    return _install_result(
        monkeypatch,
        ExecutionResult(response="done", structured_output=structured_output),
    )


@pytest.mark.unit
def test_workflow_run_help_is_available(runner: CliRunner) -> None:
    """`holodeck workflow run --help` documents the command."""
    # Act
    result = runner.invoke(cli, ["workflow", "run", "--help"])

    # Assert
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "--input" in result.output


@pytest.mark.unit
def test_workflow_group_help_lists_run(runner: CliRunner) -> None:
    """`holodeck workflow` with no subcommand shows its help."""
    # Act
    result = runner.invoke(cli, ["workflow"])

    # Assert
    assert result.exit_code == 0
    assert "run" in result.output


@pytest.mark.unit
def test_valid_run_echoes_the_verdict(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, workflow_dir: Path
) -> None:
    """US1-1: a gate-valid extraction produces the table's verdict on stdout."""
    # Arrange
    _install_backend(monkeypatch, dict(VALID_EVIDENCE))

    # Act
    result = runner.invoke(
        cli,
        [
            "workflow",
            "run",
            str(workflow_dir / "workflow.yaml"),
            "--input",
            str(workflow_dir / "input.json"),
        ],
    )

    # Assert
    assert result.exit_code == 0, result.output
    assert "hardship-single-gate" in result.output
    assert "affordable" in result.output
    assert "2026-06-01.1" in result.output
    assert "rule 1" in result.output


@pytest.mark.unit
def test_gate_rejection_exits_four_with_a_gate_error(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, workflow_dir: Path
) -> None:
    """US1-2: free text is rejected at the gate; exit 4, no verdict."""
    # Arrange
    _install_backend(monkeypatch, None)

    # Act
    result = runner.invoke(
        cli,
        [
            "workflow",
            "run",
            str(workflow_dir / "workflow.yaml"),
            "--input",
            str(workflow_dir / "input.json"),
        ],
    )

    # Assert — 4, not 3: this is evidence about the model, not about policy.
    assert result.exit_code == 4
    assert "gate rejected output of node 'evidence'" in result.output
    assert "affordable" not in result.output


@pytest.mark.unit
def test_failed_invocation_exits_five(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, workflow_dir: Path
) -> None:
    """An edge agent that fails without output is an invocation failure, not a
    gate rejection: ExecutionError is not a WorkflowError and must still map."""
    # Arrange
    _install_result(
        monkeypatch,
        ExecutionResult(
            response="",
            structured_output=None,
            is_error=True,
            error_reason="connection reset by peer",
        ),
    )

    # Act
    result = runner.invoke(
        cli,
        [
            "workflow",
            "run",
            str(workflow_dir / "workflow.yaml"),
            "--input",
            str(workflow_dir / "input.json"),
        ],
    )

    # Assert
    assert result.exit_code == 5
    assert "connection reset by peer" in result.output


@pytest.mark.unit
def test_undecidable_table_exits_three(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, workflow_dir: Path
) -> None:
    """Valid policy that cannot decide is exit 3, distinct from a gate error."""
    # Arrange — drop every rule but one that cannot match, and no default.
    _install_backend(monkeypatch, dict(VALID_EVIDENCE))
    table_path = workflow_dir / "tables" / "affordability.dmn.yaml"
    table = yaml.safe_load(table_path.read_text(encoding="utf-8"))
    table["rules"] = [
        {
            "when": {"residency_status": '"unverified"'},
            "then": {"affordability": "unaffordable"},
        }
    ]
    table_path.write_text(yaml.safe_dump(table), encoding="utf-8")

    # Act
    result = runner.invoke(
        cli,
        [
            "workflow",
            "run",
            str(workflow_dir / "workflow.yaml"),
            "--input",
            str(workflow_dir / "input.json"),
        ],
    )

    # Assert
    assert result.exit_code == 3
    assert "no rule matched" in result.output


@pytest.mark.unit
def test_cycle_exits_two_without_invoking_an_agent(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cyclic workflow is refused at load with exit 2 and zero agent calls."""
    # Arrange
    selector = _install_backend(monkeypatch, dict(VALID_EVIDENCE))
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump(
            {
                "name": "cyclic",
                "version": "1.0.0",
                "nodes": [
                    {"id": "a", "decision": "a.dmn.yaml", "inputs": ["b"]},
                    {"id": "b", "decision": "b.dmn.yaml", "inputs": ["a"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    # Act
    result = runner.invoke(cli, ["workflow", "run", str(workflow_path)])

    # Assert
    assert result.exit_code == 2
    assert "cycle" in result.output
    assert selector.calls == []


@pytest.mark.unit
def test_unreviewed_generated_table_exits_two_naming_the_table(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, workflow_dir: Path
) -> None:
    """SC-009: a generated, unreviewed table is refused before any agent runs."""
    # Arrange
    selector = _install_backend(monkeypatch, dict(VALID_EVIDENCE))
    table_path = workflow_dir / "tables" / "affordability.dmn.yaml"
    table = yaml.safe_load(table_path.read_text(encoding="utf-8"))
    table["provenance"] = {"generated_by": "claude-sonnet-4-20250514"}
    table_path.write_text(yaml.safe_dump(table), encoding="utf-8")

    # Act
    result = runner.invoke(
        cli,
        [
            "workflow",
            "run",
            str(workflow_dir / "workflow.yaml"),
            "--input",
            str(workflow_dir / "input.json"),
        ],
    )

    # Assert
    assert result.exit_code == 2
    assert "refusing to run table 'affordability'" in result.output
    assert selector.calls == []


@pytest.mark.unit
def test_missing_input_payload_exits_two(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, workflow_dir: Path
) -> None:
    """FR-025: omitting a declared fact of record fails before any agent runs."""
    # Arrange
    selector = _install_backend(monkeypatch, dict(VALID_EVIDENCE))

    # Act — no --input at all, but the workflow declares 'applicant'.
    result = runner.invoke(
        cli, ["workflow", "run", str(workflow_dir / "workflow.yaml")]
    )

    # Assert
    assert result.exit_code == 2
    assert "input_data 'applicant'" in result.output
    assert selector.calls == []


@pytest.mark.unit
def test_non_object_input_payload_exits_two(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, workflow_dir: Path
) -> None:
    """An --input file that is not a JSON object is rejected."""
    # Arrange
    _install_backend(monkeypatch, dict(VALID_EVIDENCE))
    payload = workflow_dir / "bad.json"
    payload.write_text("[1, 2, 3]", encoding="utf-8")

    # Act
    result = runner.invoke(
        cli,
        [
            "workflow",
            "run",
            str(workflow_dir / "workflow.yaml"),
            "--input",
            str(payload),
        ],
    )

    # Assert
    assert result.exit_code == 2
    assert "must be a JSON object" in result.output


@pytest.mark.unit
def test_missing_workflow_file_exits_two(runner: CliRunner, tmp_path: Path) -> None:
    """Click rejects a workflow path that does not exist."""
    # Act
    result = runner.invoke(cli, ["workflow", "run", str(tmp_path / "nope.yaml")])

    # Assert
    assert result.exit_code == 2
