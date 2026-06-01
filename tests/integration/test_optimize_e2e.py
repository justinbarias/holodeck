"""End-to-end integration test for `holodeck test optimize` (T9).

Drives the real TestExecutor + equality evaluator through the optimizer loop
with a stub backend (no network), and asserts the run improves and writes its
three artifacts. The backend returns the correct answer only when the candidate
temperature crosses a threshold, so the numeric proposer has a real gradient to
climb.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from holodeck.lib.backends.base import ExecutionResult
from holodeck.models.agent import Agent, Instructions
from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import TestCaseModel
from holodeck.optimizer.config import (
    AxesConfig,
    NumericAxis,
    OptimizerConfig,
    PhaseConfig,
)
from holodeck.optimizer.loop import OptimizerLoop
from holodeck.optimizer.output import write_outputs
from holodeck.optimizer.proposers.numeric import NumericProposer
from holodeck.optimizer.scorer import score


def _agent(temperature: float) -> Agent:
    return Agent(
        name="e2e-opt-agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o-mini",
            temperature=temperature,
            api_key="test-key",
        ),
        instructions=Instructions(inline="Answer the question."),
        test_cases=[
            TestCaseModel(name="greet", input="say hello", ground_truth="hello")
        ],
        evaluations=EvaluationConfig(metrics=[EvaluationMetric(metric="equality")]),
    )


def _stub_backend(response: str) -> Mock:
    backend = Mock()
    backend.initialize = AsyncMock()
    backend.teardown = AsyncMock()
    backend.invoke_once = AsyncMock(return_value=ExecutionResult(response=response))
    backend.create_session = AsyncMock()
    backend.__class__.__name__ = "_StubBackend"
    return backend


@pytest.mark.integration
@pytest.mark.asyncio
async def test_optimize_improves_and_writes_artifacts(tmp_path: Path) -> None:
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text("name: e2e-opt-agent\n")  # path only used for resolution

    # The agent answers correctly only once temperature >= 0.5.
    async def scorer(candidate: Agent) -> tuple[float, object]:
        temp = candidate.model.temperature or 0.0
        response = "hello" if temp >= 0.5 else "wrong"
        return await score(
            candidate,
            str(agent_path),
            {"equality": 1.0},
            backend=_stub_backend(response),
        )

    config = OptimizerConfig(
        loss={"equality": 1.0},
        axes=AxesConfig(
            numeric=[
                NumericAxis(path="model.temperature", type="float", range=[0.0, 1.0])
            ]
        ),
        min_delta=0.0,
        max_cycles=1,
        numeric_phase=PhaseConfig(max_trials=20, patience=20),
        textual_phase=PhaseConfig(max_trials=1, patience=1),
        seed=3,
    )

    loop = OptimizerLoop(
        original_agent=_agent(0.1),
        scorer=scorer,  # type: ignore[arg-type]
        config=config,
        numeric_proposer=NumericProposer(axes=config.axes.numeric, seed=config.seed),
        run_id="e2e-run",
    )

    result = await loop.run()

    # Baseline (temp 0.1) answers wrong → loss 1.0; optimum answers right → loss 0.0.
    assert result.baseline_loss == pytest.approx(1.0)
    assert result.best_loss <= result.baseline_loss
    assert result.best_loss == pytest.approx(0.0)
    assert result.accepted_count >= 1

    run_dir = write_outputs(result, tmp_path / "results")
    assert (run_dir / "best.yaml").exists()
    assert (run_dir / "trials.jsonl").exists()
    assert (run_dir / "report.md").exists()

    # The original agent on disk is untouched.
    assert agent_path.read_text() == "name: e2e-opt-agent\n"
