"""Structured, versioned progress events for ``holodeck test optimize``.

A third sink alongside ``trials.jsonl`` and OpenTelemetry: the same per-trial
``TrialRecord`` plus run/cycle/phase lifecycle events, streamed live as NDJSON so a
subprocess (HoloDeck Studio's Optimizer tab, CI dashboards, notebooks) can render a
*live* run without scraping human logs. Every line carries a ``schema`` version tag;
breaking changes bump the version and consumers branch on it.

The emitter is modeled on :class:`~holodeck.optimizer.telemetry.OptimizerTelemetry`:
:class:`NullEmitter` is the default strict no-op, and :class:`JsonlEmitter` writes one
event per line to a stream. :class:`~holodeck.optimizer.loop.OptimizerLoop` emits the
run/cycle/phase/trial events; the CLI emits ``run_completed`` (the artifact paths are
only known once the artifacts are written) and a terminal ``error``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

#: Version tag stamped on every event. Bump on a breaking change to the schema.
SCHEMA_VERSION: Literal["holodeck.optimize.progress/v1"] = (
    "holodeck.optimize.progress/v1"
)


class ProgressEvent(BaseModel):
    """Base for every progress event; stamps the ``schema`` version centrally."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # ``schema`` shadows a BaseModel attribute, so the field is ``schema_`` with a
    # ``schema`` alias; serialize with ``by_alias=True`` to emit the wire key.
    schema_: Literal["holodeck.optimize.progress/v1"] = Field(
        default=SCHEMA_VERSION, alias="schema"
    )


class NumericAxisInfo(BaseModel):
    """A declared numeric axis, as advertised in ``run_started``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    type: str
    range: list[Any]


class TextualAxisInfo(BaseModel):
    """A declared textual axis, as advertised in ``run_started``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    max_chars: int


class RunAxes(BaseModel):
    """The numeric and textual axes the run will sweep."""

    model_config = ConfigDict(extra="forbid")

    numeric: list[NumericAxisInfo] = Field(default_factory=list)
    textual: list[TextualAxisInfo] = Field(default_factory=list)


class RunStarted(ProgressEvent):
    """Opens the stream: identifies the run and its search space."""

    event: Literal["run_started"] = "run_started"
    run_id: str
    agent: str
    max_cycles: int
    axes: RunAxes
    loss_weights: dict[str, float]
    started_at: str


class Baseline(ProgressEvent):
    """The original agent's scalarized loss (the bar every trial must beat)."""

    event: Literal["baseline"] = "baseline"
    loss: float


class CycleStarted(ProgressEvent):
    """Top of a coordinate-descent cycle; supplies the ``cycle x/of`` counter."""

    event: Literal["cycle_started"] = "cycle_started"
    cycle: int
    of: int


class PhaseStarted(ProgressEvent):
    """Start of a numeric or textual phase within a cycle."""

    event: Literal["phase_started"] = "phase_started"
    cycle: int
    phase: Literal["numeric", "textual"]


class TrialStarted(ProgressEvent):
    """Opens a scored trial: the candidate being scored, before its loss is known.

    Emitted right after a proposal is applied and before scoring begins, so a live
    consumer can show *what* a trial is proposing during the (often minutes-long)
    scoring window. Correlates with the terminal :class:`Trial` event via
    ``trial_id``. Skipped/proposer-error trials have no scoring window and emit no
    ``trial_started``. Deliberately omits ``loss``/``accepted``/``best_loss`` —
    those are unknown until scoring completes.
    """

    event: Literal["trial_started"] = "trial_started"
    trial_id: int
    cycle: int
    phase: Literal["numeric", "textual"]
    baseline_loss: float
    params: dict[str, Any] | None = None
    textual_axis: str | None = None
    edit_summary: str | None = None


class Trial(ProgressEvent):
    """One scored (or skipped) trial.

    Carries every :class:`~holodeck.optimizer.models.TrialRecord` field verbatim plus
    a running ``best_loss`` (the best loss *after* this trial's accept/reject decision).
    Built from the same record appended to ``trials.jsonl`` so the two never diverge.
    """

    event: Literal["trial"] = "trial"
    trial_id: int
    cycle: int
    phase: Literal["numeric", "textual"]
    loss: float
    baseline_loss: float
    best_loss: float
    accepted: bool
    params: dict[str, Any] | None = None
    textual_axis: str | None = None
    edit_summary: str | None = None
    excluded_metrics: list[str] = Field(default_factory=list)
    error: str | None = None


class PhaseCompleted(ProgressEvent):
    """End of a phase: how many trials it ran and how many it accepted."""

    event: Literal["phase_completed"] = "phase_completed"
    cycle: int
    phase: Literal["numeric", "textual"]
    trials: int
    accepted: int


class CycleCompleted(ProgressEvent):
    """End of a cycle: accepts this cycle, the running best, and why it stopped."""

    event: Literal["cycle_completed"] = "cycle_completed"
    cycle: int
    accepted: int
    best_loss: float
    stop_reason: Literal["no_accepts"] | None = None


class RunArtifacts(BaseModel):
    """Paths to the three run artifacts, as written by ``write_outputs``."""

    model_config = ConfigDict(extra="forbid")

    best_yaml: str
    trials_jsonl: str
    report_md: str


class RunCompleted(ProgressEvent):
    """Closes the stream: final outcome plus the on-disk artifact paths."""

    event: Literal["run_completed"] = "run_completed"
    run_id: str
    baseline_loss: float
    best_loss: float
    accepted: int
    cycles: int
    artifacts: RunArtifacts


class ErrorEvent(ProgressEvent):
    """A recoverable or fatal error. Recoverable trial errors ride a ``trial`` event;
    this terminal event signals a run-ending failure (``fatal=True``)."""

    event: Literal["error"] = "error"
    message: str
    fatal: bool


#: Discriminated union of every event, keyed on ``event``.
ProgressEventUnion = Annotated[
    RunStarted
    | Baseline
    | CycleStarted
    | PhaseStarted
    | TrialStarted
    | Trial
    | PhaseCompleted
    | CycleCompleted
    | RunCompleted
    | ErrorEvent,
    Field(discriminator="event"),
]

_ADAPTER: TypeAdapter[Any] = TypeAdapter(ProgressEventUnion)


class ProgressEmitter(Protocol):
    """Sink for progress events. Injected into the loop like the telemetry seam."""

    def emit(self, event: ProgressEvent) -> None:
        """Emit one event (no-op for the disabled path)."""
        ...


class NullEmitter:
    """Default no-op emitter; the loop behaves identically when progress is off."""

    def emit(self, event: ProgressEvent) -> None:
        """Discard the event."""
        return None


class JsonlEmitter:
    """Writes one NDJSON event per line to a text stream, flushed per event."""

    def __init__(self, stream: TextIO) -> None:
        """Bind the emitter to a writable text stream (e.g. ``sys.stdout``)."""
        self._stream = stream

    def emit(self, event: ProgressEvent) -> None:
        """Serialize ``event`` to one JSON line and flush so live consumers see it."""
        self._stream.write(event.model_dump_json(by_alias=True) + "\n")
        self._stream.flush()


def progress_json_schema() -> dict[str, Any]:
    """Return the published JSON Schema for the progress event union."""
    schema: dict[str, Any] = _ADAPTER.json_schema(by_alias=True)
    return schema


def parse_event(obj: dict[str, Any]) -> ProgressEvent:
    """Validate a decoded JSON object into the matching event model."""
    event: ProgressEvent = _ADAPTER.validate_python(obj)
    return event
