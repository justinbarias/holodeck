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
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
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


class _StubTextualProposer:
    """Yields preset textual proposals (or an error proposal)."""

    phase = "textual"

    def __init__(self, proposals: list[Proposal]) -> None:
        self._proposals = proposals
        self._index = 0

    def begin(self, best_agent: Agent, best_report: TestReport | None) -> None:
        self._index = 0

    async def ask(self) -> Proposal | None:
        if self._index >= len(self._proposals):
            return None
        proposal = self._proposals[self._index]
        self._index += 1
        return proposal

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


# --- T3: phase / trial / propose spans ----------------------------------------


@pytest.mark.asyncio
async def test_phase_and_trial_spans_emitted(otel: SimpleNamespace) -> None:
    await _loop().run()

    by_name = {s.name: s for s in otel.exporter.get_finished_spans()}
    assert "holodeck.optimize.phase" in by_name
    assert "holodeck.optimize.trial" in by_name

    phase = by_name["holodeck.optimize.phase"]
    assert phase.attributes["holodeck.optimize.phase"] == "numeric"

    trial = by_name["holodeck.optimize.trial"]
    assert trial.attributes["holodeck.optimize.trial_id"] == 1
    assert trial.attributes["holodeck.optimize.loss"] == pytest.approx(0.50)
    assert trial.attributes["holodeck.optimize.accepted"] is False
    # Numeric params are recorded as a single JSON string (build decision).
    assert trial.attributes["holodeck.optimize.params"] == '{"model.temperature": 0.5}'


@pytest.mark.asyncio
async def test_trial_nests_under_phase_nests_under_cycle(otel: SimpleNamespace) -> None:
    await _loop().run()

    by_name = {s.name: s for s in otel.exporter.get_finished_spans()}
    cycle = by_name["holodeck.optimize.cycle"]
    phase = by_name["holodeck.optimize.phase"]
    trial = by_name["holodeck.optimize.trial"]
    assert phase.parent.span_id == cycle.context.span_id
    assert trial.parent.span_id == phase.context.span_id


@pytest.mark.asyncio
async def test_genai_child_nests_under_trial(otel: SimpleNamespace) -> None:
    # A stub eval that emits its own "genai" span while the trial span is active.
    genai_tracer = otel.provider.get_tracer("genai.test")

    async def scorer(agent: Agent) -> tuple[float, TestReport]:
        with genai_tracer.start_as_current_span("gen_ai.chat"):
            pass
        return 0.50, _dummy_report()

    loop = OptimizerLoop(
        original_agent=_agent(0.3),
        scorer=scorer,
        config=_config(),
        numeric_proposer=_StubNumericProposer([{"model.temperature": 0.5}]),
    )
    await loop.run()

    by_name = {s.name: s for s in otel.exporter.get_finished_spans()}
    trial = by_name["holodeck.optimize.trial"]
    genai = by_name["gen_ai.chat"]
    assert genai.parent.span_id == trial.context.span_id


@pytest.mark.asyncio
async def test_no_instruction_text_in_attributes(otel: SimpleNamespace) -> None:
    await _loop().run()

    for span in otel.exporter.get_finished_spans():
        for value in span.attributes.values():
            if isinstance(value, str):
                assert "You are helpful." not in value


@pytest.mark.asyncio
async def test_skipped_trial_span_carries_error(otel: SimpleNamespace) -> None:
    loop = OptimizerLoop(
        original_agent=_agent(0.3),
        scorer=_flat_scorer(),
        config=_config(),
        textual_proposer=_StubTextualProposer(
            [Proposal(textual_axis="instructions.inline", error="critic failed")]
        ),
    )
    await loop.run()

    trials = [
        s
        for s in otel.exporter.get_finished_spans()
        if s.name == "holodeck.optimize.trial"
    ]
    assert any(
        t.attributes.get("holodeck.optimize.error") == "critic failed" for t in trials
    )


@pytest.mark.asyncio
async def test_propose_span_for_textual_only(otel: SimpleNamespace) -> None:
    loop = OptimizerLoop(
        original_agent=_agent(0.3),
        scorer=_flat_scorer(),
        config=_config(),
        textual_proposer=_StubTextualProposer(
            [
                Proposal(
                    textual_axis="instructions.inline",
                    new_text="Be concise.",
                    edit_summary="tightened wording",
                )
            ]
        ),
    )
    await loop.run()

    names = [s.name for s in otel.exporter.get_finished_spans()]
    assert "holodeck.optimize.propose" in names


# --- T4: metrics --------------------------------------------------------------


@pytest.fixture
def meter(monkeypatch) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(
        "holodeck.optimizer.telemetry.get_meter",
        lambda name: provider.get_meter(name),
    )
    monkeypatch.setattr(
        "holodeck.optimizer.telemetry.get_observability_context",
        lambda: object(),
    )
    return SimpleNamespace(reader=reader, provider=provider)


def _points(reader: InMemoryMetricReader) -> dict:
    """Collect ``{metric_name: [data_points]}`` from the in-memory reader."""
    out: dict = {}
    data = reader.get_metrics_data()
    if data is None:  # No instrument ever recorded (e.g. observability disabled).
        return out
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                out[metric.name] = list(metric.data.data_points)
    return out


def _improving_loop() -> OptimizerLoop:
    # 0.3 (baseline) -> 0.60, 0.5 -> 0.40: one accepted improvement.
    table = {0.3: 0.60, 0.5: 0.40}

    async def scorer(agent: Agent) -> tuple[float, TestReport]:
        return table[agent.model.temperature], _dummy_report()

    return OptimizerLoop(
        original_agent=_agent(0.3),
        scorer=scorer,
        config=_config(),
        numeric_proposer=_StubNumericProposer([{"model.temperature": 0.5}]),
    )


@pytest.mark.asyncio
async def test_trial_and_cycle_metrics_emitted(meter: SimpleNamespace) -> None:
    await _loop().run()  # flat loss → one rejected trial, one cycle

    points = _points(meter.reader)
    assert "holodeck.optimize.trials" in points
    assert "holodeck.optimize.trial.loss" in points
    assert "holodeck.optimize.trial.duration" in points
    assert "holodeck.optimize.cycles" in points
    assert "holodeck.optimize.improvement" in points

    trial = points["holodeck.optimize.trials"][0]
    assert trial.value == 1
    assert trial.attributes["holodeck.optimize.phase"] == "numeric"
    assert trial.attributes["holodeck.optimize.accepted"] is False
    assert points["holodeck.optimize.cycles"][0].value == 1


@pytest.mark.asyncio
async def test_best_loss_metric_on_accept(meter: SimpleNamespace) -> None:
    await _improving_loop().run()

    points = _points(meter.reader)
    assert "holodeck.optimize.best_loss" in points
    best = points["holodeck.optimize.best_loss"][0]
    assert best.attributes["holodeck.optimize.phase"] == "numeric"
    # Improvement histogram captured baseline(0.60) - best(0.40) = 0.20.
    improvement = points["holodeck.optimize.improvement"][0]
    assert improvement.sum == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_skipped_trial_metric(meter: SimpleNamespace) -> None:
    loop = OptimizerLoop(
        original_agent=_agent(0.3),
        scorer=_flat_scorer(),
        config=_config(),
        textual_proposer=_StubTextualProposer(
            [Proposal(textual_axis="instructions.inline", error="critic failed")]
        ),
    )
    await loop.run()

    points = _points(meter.reader)
    assert "holodeck.optimize.trials.skipped" in points
    skipped = points["holodeck.optimize.trials.skipped"][0]
    assert skipped.attributes["holodeck.optimize.phase"] == "textual"


@pytest.mark.asyncio
async def test_no_metrics_when_disabled(meter: SimpleNamespace, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "holodeck.optimizer.telemetry.get_observability_context",
        lambda: None,
    )
    await _loop().run()

    points = _points(meter.reader)
    assert not any(name.startswith("holodeck.optimize") for name in points)
