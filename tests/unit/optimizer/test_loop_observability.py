"""In-memory OTel assertions for the optimizer span tree.

These tests spin a real (in-process) TracerProvider with an
``InMemorySpanExporter`` so they can assert true span names and parent/child
relationships — something mock-based tests cannot prove. The telemetry module's
``get_tracer`` / ``get_observability_context`` are monkeypatched onto the local
provider, so no global OTel state or real exporter endpoint is needed.

Covers T2 (baseline + cycle spans). T3 (phase/trial/propose) and T4 (metrics)
extend this module.
"""

from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_result import ReportSummary, TestReport, TestResult
from holodeck.optimizer.config import (
    AxesConfig,
    NumericAxis,
    OptimizerConfig,
    PhaseConfig,
)
from holodeck.optimizer.loop import OptimizerLoop
from holodeck.optimizer.proposers.base import Proposal

pytestmark = pytest.mark.unit


def _agent(temperature: float = 0.3) -> Agent:
    return Agent(
        name="opt-agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4o-mini", temperature=temperature
        ),
        instructions=Instructions(inline="You are helpful."),
    )


def _dummy_report() -> TestReport:
    return TestReport(
        agent_name="opt-agent",
        agent_config_path="agent.yaml",
        results=[
            TestResult(
                test_name="t0",
                test_input="q",
                agent_response="a",
                passed=True,
                execution_time_ms=1,
                timestamp="2026-06-06T00:00:00Z",
            )
        ],
        summary=ReportSummary(
            total_tests=1,
            passed=1,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=1,
            metrics_evaluated={"groundedness": 1},
            average_scores={"groundedness": 0.5},
        ),
        timestamp="2026-06-06T00:00:00Z",
        holodeck_version="test",
    )


class _StubNumericProposer:
    """Yields a preset list of numeric proposals, resetting each phase."""

    phase = "numeric"

    def __init__(self, param_dicts: list[dict]) -> None:
        self._param_dicts = param_dicts
        self._index = 0

    def begin(self, best_agent: Agent, best_report: TestReport | None) -> None:
        self._index = 0

    async def ask(self) -> Proposal | None:
        if self._index >= len(self._param_dicts):
            return None
        params = self._param_dicts[self._index]
        self._index += 1
        return Proposal(params=params)

    def tell(self, proposal, score, accepted, report=None) -> None:  # type: ignore[no-untyped-def]
        pass


def _config() -> OptimizerConfig:
    return OptimizerConfig(
        loss={"groundedness": 1.0},
        axes=AxesConfig(
            numeric=[
                NumericAxis(path="model.temperature", type="float", range=[0.0, 1.0])
            ]
        ),
        min_delta=0.01,
        max_cycles=3,
        numeric_phase=PhaseConfig(max_trials=10, patience=3),
        textual_phase=PhaseConfig(max_trials=5, patience=3),
    )


def _flat_scorer(loss: float = 0.50):
    """Constant loss → no accepts → loop stops after one cycle."""

    async def scorer(agent: Agent) -> tuple[float, TestReport]:
        return loss, _dummy_report()

    return scorer


@pytest.fixture
def otel(monkeypatch) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "holodeck.optimizer.telemetry.get_tracer",
        lambda name: provider.get_tracer(name),
    )
    monkeypatch.setattr(
        "holodeck.optimizer.telemetry.get_observability_context",
        lambda: object(),
    )
    return SimpleNamespace(exporter=exporter, provider=provider)


def _loop() -> OptimizerLoop:
    return OptimizerLoop(
        original_agent=_agent(0.3),
        scorer=_flat_scorer(),
        config=_config(),
        numeric_proposer=_StubNumericProposer([{"model.temperature": 0.5}]),
    )


@pytest.mark.asyncio
async def test_baseline_and_cycle_spans_emitted(otel: SimpleNamespace) -> None:
    await _loop().run()

    names = [s.name for s in otel.exporter.get_finished_spans()]
    assert "holodeck.optimize.baseline" in names
    # Flat loss → single dry cycle → exactly one cycle span.
    assert names.count("holodeck.optimize.cycle") == 1


@pytest.mark.asyncio
async def test_baseline_and_cycle_nest_under_root(otel: SimpleNamespace) -> None:
    root_tracer = otel.provider.get_tracer("test.root")
    with root_tracer.start_as_current_span("holodeck.optimize"):
        await _loop().run()

    finished = {s.name: s for s in otel.exporter.get_finished_spans()}
    root = finished["holodeck.optimize"]
    assert finished["holodeck.optimize.baseline"].parent.span_id == root.context.span_id
    assert finished["holodeck.optimize.cycle"].parent.span_id == root.context.span_id


@pytest.mark.asyncio
async def test_no_spans_when_disabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "holodeck.optimizer.telemetry.get_observability_context",
        lambda: None,
    )
    loop = _loop()
    result = await loop.run()

    assert loop._telemetry.enabled is False
    # Behavior is unchanged: a single dry cycle still runs to completion.
    assert result.cycles_run == 1
