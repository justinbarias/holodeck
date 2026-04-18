"""Unit tests for EvalRun, EvalRunMetadata, and PromptVersion models.

Covers Phase 3 tasks T011–T013 of the 031-eval-runs-dashboard feature.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from holodeck.models.agent import Agent
from holodeck.models.eval_run import EvalRun, EvalRunMetadata, PromptVersion
from holodeck.models.test_result import ReportSummary, TestReport


def _make_agent(name: str = "test-agent") -> Agent:
    return Agent(
        name=name,
        model={"provider": "openai", "name": "gpt-4o"},
        instructions={"inline": "You are a test agent."},
    )


def _make_report(agent_name: str = "test-agent") -> TestReport:
    return TestReport(
        agent_name=agent_name,
        agent_config_path="agent.yaml",
        results=[],
        summary=ReportSummary(
            total_tests=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            total_duration_ms=0,
            metrics_evaluated={},
            average_scores={},
        ),
        timestamp="2026-04-18T14:22:09.812Z",
        holodeck_version="0.1.0",
        environment={},
    )


def _make_prompt_version() -> PromptVersion:
    return PromptVersion(
        version="auto-12345678",
        source="inline",
        body_hash="0" * 64,
    )


def _make_metadata(agent_name: str = "test-agent") -> EvalRunMetadata:
    return EvalRunMetadata(
        agent_config=_make_agent(agent_name),
        prompt_version=_make_prompt_version(),
        holodeck_version="0.1.0",
        cli_args=["test", "agent.yaml"],
        git_commit=None,
    )


# T011 — EvalRun structural assertions.


@pytest.mark.unit
class TestEvalRunStructure:
    def test_required_fields(self):
        run = EvalRun(report=_make_report(), metadata=_make_metadata())
        assert run.report.agent_name == "test-agent"
        assert run.metadata.agent_config.name == "test-agent"

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            EvalRun(
                report=_make_report(),
                metadata=_make_metadata(),
                extra_field="nope",  # type: ignore[call-arg]
            )

    def test_consistency_mismatch_raises(self):
        with pytest.raises(ValidationError) as exc:
            EvalRun(
                report=_make_report(agent_name="agent-A"),
                metadata=_make_metadata(agent_name="agent-B"),
            )
        msg = str(exc.value)
        assert "agent_name" in msg or "consistency" in msg.lower()


# T012 — round-trip.


@pytest.mark.unit
class TestEvalRunRoundTrip:
    def test_dump_and_validate_round_trip(self):
        run = EvalRun(report=_make_report(), metadata=_make_metadata())
        dump = run.model_dump_json()
        rehydrated = EvalRun.model_validate_json(dump)
        assert rehydrated == run


# T013 — EvalRunMetadata required fields and no created_at.


@pytest.mark.unit
class TestEvalRunMetadataShape:
    def test_required_fields_present(self):
        meta = _make_metadata()
        assert meta.agent_config.name == "test-agent"
        assert meta.prompt_version.version == "auto-12345678"
        assert meta.holodeck_version == "0.1.0"
        assert meta.cli_args == ["test", "agent.yaml"]
        assert meta.git_commit is None

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            EvalRunMetadata(
                agent_config=_make_agent(),
                prompt_version=_make_prompt_version(),
                holodeck_version="0.1.0",
                cli_args=[],
                git_commit=None,
                created_at="2026-04-18T00:00:00Z",  # type: ignore[call-arg]
            )

    def test_no_created_at_field(self):
        # Timestamp source of truth is report.timestamp, not metadata.created_at.
        assert "created_at" not in EvalRunMetadata.model_fields

    def test_required_field_set(self):
        # Ensure each documented required field rejects omission.
        required = (
            "agent_config",
            "prompt_version",
            "holodeck_version",
            "cli_args",
        )
        for missing in required:
            kwargs: dict[str, object] = {
                "agent_config": _make_agent(),
                "prompt_version": _make_prompt_version(),
                "holodeck_version": "0.1.0",
                "cli_args": [],
            }
            kwargs.pop(missing)
            with pytest.raises(ValidationError):
                EvalRunMetadata(**kwargs)  # type: ignore[arg-type]
