"""E2E integration for US2 — 4-turn per-turn ground truth + expected_tools (T026).

Uses the ConvFinQA-shaped fixture extended with per-turn ground truths and
expected tool names. A stubbed backend scripts the agent's response and tool
calls for each turn so the test is fast and deterministic. Verifies:

- per-turn `TurnResult.metric_results` is populated,
- per-turn `tools_matched=True` when scoped calls match,
- rolled-up `TestResult.passed=True`,
- the markdown report renders each turn block.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.lib.test_runner.reporter import generate_markdown_report
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import TestCaseModel, Turn
from holodeck.models.token_usage import TokenUsage


@pytest.mark.integration
@pytest.mark.asyncio
async def test_4turn_per_turn_ground_truth_tools() -> None:
    test_case = TestCaseModel(
        name="convfinqa-us2",
        turns=[
            Turn(
                input="what is the net cash from operating activities in 2009?",
                ground_truth="206588",
                expected_tools=["lookup"],
            ),
            Turn(
                input="what about in 2008?",
                ground_truth="181001",
                expected_tools=["lookup"],
            ),
            Turn(
                input="what is the difference?",
                ground_truth="25587",
                expected_tools=["subtract"],
            ),
            Turn(
                input="what percentage change does this represent?",
                ground_truth="0.14136",
                expected_tools=["divide"],
            ),
        ],
    )

    evaluations = EvaluationConfig(
        metrics=[EvaluationMetric(metric="bleu", threshold=0.5)],
    )
    agent = Agent(
        name="convfinqa-agent",
        description="ConvFinQA-shaped agent for US2 E2E",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4", api_key="test-key"
        ),
        instructions=Instructions(inline="answer numerically"),
        test_cases=[test_case],
        evaluations=evaluations,
        execution=None,
    )

    # Scripted per-turn (response, tool_calls).
    script = [
        ("206588", [{"name": "Finance-lookup", "arguments": {}}]),
        ("181001", [{"name": "Finance-lookup", "arguments": {}}]),
        ("25587", [{"name": "Math-subtract", "arguments": {}}]),
        ("0.14136", [{"name": "Math-divide", "arguments": {}}]),
    ]

    async def send(_msg: str) -> ExecutionResult:
        resp, tool_calls = script.pop(0)
        return ExecutionResult(
            response=resp,
            tool_calls=tool_calls,
            tool_results=[],
            token_usage=TokenUsage(
                prompt_tokens=2, completion_tokens=1, total_tokens=3
            ),
        )

    session = Mock()
    session.send = AsyncMock(side_effect=send)
    session.close = AsyncMock()

    backend = Mock()
    backend.initialize = AsyncMock()
    backend.teardown = AsyncMock()
    backend.invoke_once = AsyncMock()
    backend.create_session = AsyncMock(return_value=session)
    backend.__class__.__name__ = "_StubBackend"

    loader = Mock(spec=ConfigLoader)
    loader.load_agent_yaml.return_value = agent
    loader.resolve_execution_config.return_value = ExecutionConfig(llm_timeout=60)

    bleu = Mock()
    bleu.name = "bleu"
    bleu.get_param_spec = Mock(
        return_value=ParamSpec(
            required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
        )
    )
    # Script exact matches against ground truths → score 1.0 on all 4 turns.
    bleu.evaluate = AsyncMock(return_value={"bleu": 1.0})

    executor = TestExecutor(
        agent_config_path="tests/fixtures/multi_turn/convfinqa_sample.yaml",
        config_loader=loader,
        file_processor=Mock(spec=FileProcessor),
        backend=backend,
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(llm_timeout=60),
        evaluators={"bleu": bleu},
    )

    report = await executor.execute_tests()
    result = report.results[0]

    # Rolled-up assertions.
    assert result.passed is True
    assert result.turns is not None and len(result.turns) == 4
    assert result.tools_matched is True
    bleu_rollup = [m for m in result.metric_results if m.metric_name == "bleu"]
    assert len(bleu_rollup) == 1
    assert bleu_rollup[0].score == pytest.approx(1.0)

    # Per-turn assertions.
    for idx, turn in enumerate(result.turns):
        assert turn.passed is True, f"turn {idx} should pass"
        assert turn.tools_matched is True, f"turn {idx} tools should match"
        names = [m.metric_name for m in turn.metric_results]
        assert names == ["bleu"], f"turn {idx} should have bleu result, got {names}"

    # Markdown breakdown contains each turn.
    md = generate_markdown_report(report)
    assert "##### Turn 0" in md
    assert "##### Turn 3" in md
    assert "bleu" in md
    # Tool-match glyph line is rendered per turn.
    assert md.count("**Tool Match:**") == 4
