"""Unit tests for the progress event contract and emitters (038 T1).

Covers the versioned event models, the published JSON Schema (validation + drift),
and the ``NullEmitter``/``JsonlEmitter`` seam. The loop-level wiring (event ordering,
trial/best_loss semantics) is exercised in ``test_progress_loop.py``.
"""

import io
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from holodeck.optimizer.models import TrialRecord
from holodeck.optimizer.progress import (
    SCHEMA_VERSION,
    Baseline,
    CycleCompleted,
    CycleStarted,
    ErrorEvent,
    JsonlEmitter,
    NullEmitter,
    PhaseCompleted,
    PhaseStarted,
    RunArtifacts,
    RunAxes,
    RunCompleted,
    RunStarted,
    Trial,
    parse_event,
    progress_json_schema,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_PATH = _REPO_ROOT / "schemas" / "optimize-progress.schema.json"
_WORKED_EXAMPLE = (
    _REPO_ROOT / "tests" / "fixtures" / "optimizer" / "worked_example.jsonl"
)


def _one_of_each() -> list:
    """A representative instance of every event type."""
    return [
        RunStarted(
            run_id="r",
            agent="a",
            max_cycles=2,
            axes=RunAxes(),
            loss_weights={"groundedness": 1.0},
            started_at="2026-06-18T11:50:36Z",
        ),
        Baseline(loss=0.5),
        CycleStarted(cycle=0, of=2),
        PhaseStarted(cycle=0, phase="numeric"),
        Trial(
            trial_id=1,
            cycle=0,
            phase="numeric",
            loss=0.4,
            baseline_loss=0.5,
            best_loss=0.4,
            accepted=True,
            params={"model.temperature": 0.7},
        ),
        PhaseCompleted(cycle=0, phase="numeric", trials=1, accepted=1),
        CycleCompleted(cycle=0, accepted=1, best_loss=0.4, stop_reason=None),
        RunCompleted(
            run_id="r",
            baseline_loss=0.5,
            best_loss=0.4,
            accepted=1,
            cycles=1,
            artifacts=RunArtifacts(
                best_yaml="o/r/best.yaml",
                trials_jsonl="o/r/trials.jsonl",
                report_md="o/r/report.md",
            ),
        ),
    ]


class TestEventModels:
    """Every event serializes to a single JSON object tagged with the schema."""

    def test_every_event_carries_schema_and_event(self) -> None:
        for event in _one_of_each():
            obj = json.loads(event.model_dump_json(by_alias=True))
            assert obj["schema"] == SCHEMA_VERSION
            assert "event" in obj

    def test_trial_keeps_nulls_and_empty_lists(self) -> None:
        # A textual trial leaves params null and excluded_metrics empty — both must
        # survive serialization so consumers can rely on a stable shape.
        trial = Trial(
            trial_id=2,
            cycle=0,
            phase="textual",
            loss=0.5,
            baseline_loss=0.5,
            best_loss=0.5,
            accepted=False,
            textual_axis="instructions.inline",
            edit_summary="rewrote intro",
        )
        obj = json.loads(trial.model_dump_json(by_alias=True))
        assert obj["params"] is None
        assert obj["excluded_metrics"] == []
        assert obj["error"] is None


class TestSourceOfTruthParity:
    """A Trial event is field-equal to its TrialRecord plus best_loss (FR-004)."""

    def test_trial_matches_record_field_for_field(self) -> None:
        record = TrialRecord(
            trial_id=1,
            cycle=0,
            phase="numeric",
            loss=0.4,
            baseline_loss=0.5,
            accepted=True,
            params={"model.temperature": 0.7},
            excluded_metrics=["relevance"],
        )
        trial = Trial(best_loss=0.4, **record.model_dump())
        as_dict = trial.model_dump()
        # Drop the three event-only keys; the rest must equal the record exactly.
        for key in ("schema_", "event", "best_loss"):
            as_dict.pop(key)
        assert as_dict == record.model_dump()


class TestEmitters:
    """NullEmitter is a no-op; JsonlEmitter writes one flushed line per event."""

    def test_null_emitter_writes_nothing(self) -> None:
        # A NullEmitter must accept any event and produce no output.
        NullEmitter().emit(Baseline(loss=0.5))  # no stream, no error

    def test_jsonl_emitter_one_line_per_event(self) -> None:
        buf = io.StringIO()
        emitter = JsonlEmitter(buf)
        for event in _one_of_each():
            emitter.emit(event)
        lines = buf.getvalue().splitlines()
        assert len(lines) == len(_one_of_each())
        for line in lines:
            json.loads(line)  # each line is a standalone JSON object

    def test_jsonl_emitter_flushes_each_event(self) -> None:
        class _CountingStream(io.StringIO):
            flushes = 0

            def flush(self) -> None:
                self.flushes += 1

        stream = _CountingStream()
        emitter = JsonlEmitter(stream)
        emitter.emit(Baseline(loss=0.5))
        emitter.emit(Baseline(loss=0.4))
        assert stream.flushes == 2


class TestParseEvent:
    """parse_event round-trips an emitted line back to the original model."""

    def test_round_trip(self) -> None:
        for event in (
            *_one_of_each(),
            ErrorEvent(message="boom", fatal=True),
        ):
            decoded = json.loads(event.model_dump_json(by_alias=True))
            assert parse_event(decoded) == event


class TestPublishedSchema:
    """The published JSON Schema is current and accepts the worked example."""

    def test_committed_schema_matches_generated(self) -> None:
        committed = json.loads(_SCHEMA_PATH.read_text())
        assert (
            committed == progress_json_schema()
        ), "schemas/optimize-progress.schema.json is stale — regenerate it."

    def test_worked_example_validates_and_parses(self) -> None:
        validator = Draft202012Validator(json.loads(_SCHEMA_PATH.read_text()))
        lines = _WORKED_EXAMPLE.read_text().splitlines()
        assert lines, "worked example fixture is empty"
        for i, line in enumerate(lines):
            obj = json.loads(line)
            errors = sorted(validator.iter_errors(obj), key=str)
            assert not errors, f"line {i} ({obj.get('event')}): {errors}"
            parse_event(obj)  # also round-trips through the discriminated union

    def test_worked_example_opens_and_closes_correctly(self) -> None:
        lines = [json.loads(x) for x in _WORKED_EXAMPLE.read_text().splitlines()]
        assert lines[0]["event"] == "run_started"
        assert lines[-1]["event"] == "run_completed"
