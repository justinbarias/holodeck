"""End-to-end integration test for `holodeck test optimize` (T9).

Drives the real TestExecutor + equality evaluator through the optimizer loop
with a stub backend (no network), and asserts the run improves and writes its
three artifacts. The backend returns the correct answer only when the candidate
temperature crosses a threshold, so the numeric proposer has a real gradient to
climb.
"""

import json
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
    TextualAxis,
)
from holodeck.optimizer.loop import OptimizerLoop
from holodeck.optimizer.output import write_outputs
from holodeck.optimizer.proposers.numeric import NumericProposer
from holodeck.optimizer.proposers.textual import TextualProposer
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_progress_stream_validates_against_schema(tmp_path: Path) -> None:
    """A real run with a JsonlEmitter yields a schema-valid, ordered NDJSON stream
    whose trial events match trials.jsonl field-for-field (FR-003, FR-004)."""
    import io

    from jsonschema import Draft202012Validator

    from holodeck.optimizer.progress import JsonlEmitter, progress_json_schema

    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text("name: e2e-opt-agent\n")

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

    buf = io.StringIO()
    loop = OptimizerLoop(
        original_agent=_agent(0.1),
        scorer=scorer,  # type: ignore[arg-type]
        config=config,
        numeric_proposer=NumericProposer(axes=config.axes.numeric, seed=config.seed),
        run_id="e2e-progress",
        emitter=JsonlEmitter(buf),
    )

    result = await loop.run()
    run_dir = write_outputs(result, tmp_path / "results")

    validator = Draft202012Validator(progress_json_schema())
    events = [json.loads(line) for line in buf.getvalue().splitlines()]
    for event in events:
        assert not list(validator.iter_errors(event)), event

    # Loop emits run_started first and cycle_completed last (run_completed is CLI-only).
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "cycle_completed", (
        "the loop must NOT emit run_completed; the CLI owns it (artifact paths are "
        "only known after write_outputs). A refactor that moves it here would break "
        "the documented loop/CLI emission split."
    )

    # Trial events equal the persisted trials.jsonl rows plus best_loss (FR-004).
    rows = [
        json.loads(line) for line in (run_dir / "trials.jsonl").read_text().splitlines()
    ]
    trial_events = [e for e in events if e["event"] == "trial"]
    assert len(trial_events) == len(rows)
    for event, row in zip(trial_events, rows, strict=True):
        stripped = {
            k: v for k, v in event.items() if k not in ("schema", "event", "best_loss")
        }
        assert stripped == row

    # Each scored trial is announced by a trial_started carrying the same id.
    started_ids = [e["trial_id"] for e in events if e["event"] == "trial_started"]
    assert started_ids == [e["trial_id"] for e in trial_events]


_TEXTUAL_GROUND_TRUTHS = ["a1", "a2", "a3"]


def _textual_agent() -> Agent:
    """An agent with three equality test cases and one instruction axis."""
    return Agent(
        name="e2e-textual-agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4o-mini", api_key="test-key"
        ),
        instructions=Instructions(inline="You are helpful."),
        test_cases=[
            TestCaseModel(name=f"c{i}", input=f"q{i}", ground_truth=gt)
            for i, gt in enumerate(_TEXTUAL_GROUND_TRUTHS)
        ],
        evaluations=EvaluationConfig(metrics=[EvaluationMetric(metric="equality")]),
    )


def _seq_backend(responses: list[str]) -> Mock:
    """Stub backend whose successive invoke_once calls return ``responses``."""
    backend = _stub_backend("")
    backend.invoke_once = AsyncMock(
        side_effect=[ExecutionResult(response=r) for r in responses]
    )
    return backend


def _subagent(name: str) -> Agent:
    return Agent(
        name=name,
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="subagent prompt"),
    )


class _CountingInvoker:
    """Applier emits a fresh ``rewrite-N`` each call; Critic is constant."""

    def __init__(self) -> None:
        self.n = 0

    async def __call__(self, agent: Agent, prompt: str) -> ExecutionResult:
        if "critic" in agent.name:
            return ExecutionResult(response='{"gradient": "improve"}')
        self.n += 1
        return ExecutionResult(
            response=json.dumps({"new_text": f"rewrite-{self.n}", "summary": "s"})
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_iterative_textual_reduces_loss(tmp_path: Path) -> None:
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text("name: e2e-textual-agent\n")

    # Rewrite-N makes the first N of the three cases answer correctly, so loss
    # falls 1.0 → 0.67 → 0.33 → 0.0 across successive refinement steps.
    async def scorer(candidate: Agent) -> tuple[float, object]:
        text = candidate.instructions.inline or ""
        n = int(text.split("-")[1]) if text.startswith("rewrite-") else 0
        responses = [
            gt if i < n else "wrong" for i, gt in enumerate(_TEXTUAL_GROUND_TRUTHS)
        ]
        return await score(
            candidate,
            str(agent_path),
            {"equality": 1.0},
            backend=_seq_backend(responses),
        )

    config = OptimizerConfig(
        loss={"equality": 1.0},
        axes=AxesConfig(textual=[TextualAxis(path="instructions.inline")]),
        min_delta=0.0,
        max_cycles=1,
        numeric_phase=PhaseConfig(max_trials=1, patience=1),
        textual_phase=PhaseConfig(max_trials=3, patience=3),
    )
    proposer = TextualProposer(
        axes=config.axes.textual,
        critic_agent=_subagent("optimizer-critic"),
        applier_agent=_subagent("optimizer-applier"),
        invoker=_CountingInvoker(),
    )
    loop = OptimizerLoop(
        original_agent=_textual_agent(),
        scorer=scorer,  # type: ignore[arg-type]
        config=config,
        textual_proposer=proposer,
        run_id="e2e-textual-run",
    )

    result = await loop.run()

    textual_losses = [t.loss for t in result.trials if t.phase == "textual"]
    assert len(textual_losses) == 3
    # Loss strictly decreases across the iterative steps.
    assert textual_losses == sorted(textual_losses, reverse=True)
    assert textual_losses[0] < result.baseline_loss
    assert result.best_loss == pytest.approx(0.0)
    assert result.best_agent.instructions.inline == "rewrite-3"

    run_dir = write_outputs(result, tmp_path / "results")
    assert (run_dir / "best.yaml").exists()
    assert (run_dir / "report.md").exists()
