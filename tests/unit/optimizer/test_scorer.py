"""Unit tests for scorer.score wrapping TestExecutor (T3)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_result import (
    MetricResult,
    ReportSummary,
    TestReport,
    TestResult,
)
from holodeck.optimizer.loss import scalarize
from holodeck.optimizer.scorer import score


def _agent() -> Agent:
    return Agent(
        name="opt-agent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="You are helpful."),
    )


def _report(average_scores: dict[str, float]) -> TestReport:
    results = [
        TestResult(
            test_name="t0",
            test_input="q",
            agent_response="a",
            passed=True,
            execution_time_ms=1,
            timestamp="2026-05-31T00:00:00Z",
            metric_results=[
                MetricResult(metric_name=name, kind="standard", score=score)
                for name, score in average_scores.items()
            ],
        )
    ]
    summary = ReportSummary(
        total_tests=1,
        passed=1,
        failed=0,
        pass_rate=100.0,
        total_duration_ms=1,
        metrics_evaluated=dict.fromkeys(average_scores, 1),
        average_scores=average_scores,
    )
    return TestReport(
        agent_name="opt-agent",
        agent_config_path="agent.yaml",
        results=results,
        summary=summary,
        timestamp="2026-05-31T00:00:00Z",
        holodeck_version="test",
    )


class TestScore:
    """score() runs the executor and scalarizes its report."""

    @pytest.mark.asyncio
    async def test_returns_scalarized_score_and_report(self) -> None:
        report = _report({"groundedness": 0.8, "relevance": 0.6})
        weights = {"groundedness": 1.0, "relevance": 3.0}
        agent = _agent()

        with patch("holodeck.optimizer.scorer.TestExecutor") as mock_executor:
            instance = mock_executor.return_value
            instance.execute_tests = AsyncMock(return_value=report)

            value, returned_report = await score(agent, "agent.yaml", weights)

        assert returned_report is report
        assert value == pytest.approx(scalarize(report, weights))

    @pytest.mark.asyncio
    async def test_uses_force_ingest_false(self) -> None:
        report = _report({"groundedness": 0.5})
        agent = _agent()

        with patch("holodeck.optimizer.scorer.TestExecutor") as mock_executor:
            instance = mock_executor.return_value
            instance.execute_tests = AsyncMock(return_value=report)

            await score(agent, "path/to/agent.yaml", {"groundedness": 1.0})

        _, kwargs = mock_executor.call_args
        assert kwargs["force_ingest"] is False
        assert kwargs["agent_config"] is agent
        assert kwargs["agent_config_path"] == "path/to/agent.yaml"

    @pytest.mark.asyncio
    async def test_passes_injected_backend_through(self) -> None:
        report = _report({"groundedness": 0.5})
        backend = MagicMock()

        with patch("holodeck.optimizer.scorer.TestExecutor") as mock_executor:
            instance = mock_executor.return_value
            instance.execute_tests = AsyncMock(return_value=report)

            await score(_agent(), "agent.yaml", {"groundedness": 1.0}, backend=backend)

        _, kwargs = mock_executor.call_args
        assert kwargs["backend"] is backend
