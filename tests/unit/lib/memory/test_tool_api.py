"""Tests for the memory tool argument and result models (spec 043, design 6)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from holodeck.lib.memory.tool_api import (
    MEMORY_FUNCTION_NAMES,
    MEMORY_TOOL_MODELS,
    MemoryHitOut,
    MemoryProvenanceArgs,
    MemoryProvenanceResult,
    MemorySearchArgs,
    MemorySearchResult,
    MemoryTimelineArgs,
    MemoryToolError,
)
from holodeck.lib.memory.types import EpisodeRef, MemoryHit, Provenance

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


def _load_generator():
    path = ROOT / "scripts" / "generate_memory_schemas.py"
    spec = importlib.util.spec_from_file_location("generate_memory_schemas", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fact_hit(**kw) -> MemoryHit:
    return MemoryHit(
        kind="fact",
        text="Rate is 5%",
        score=0.9,
        source_tool="docs",
        chunk_id="c1",
        source_uri="a.md",
        fact_id="f1",
        valid_from=datetime(2020, 7, 1, tzinfo=UTC),
        valid_to=datetime(2022, 7, 1, tzinfo=UTC),
        **kw,
    )


def _chunk_hit(**kw) -> MemoryHit:
    return MemoryHit(
        kind="chunk",
        text="passage",
        score=0.1,
        source_tool="docs",
        chunk_id="c2",
        source_uri="b.md",
        **kw,
    )


def test_hst_01_function_names() -> None:
    assert MEMORY_FUNCTION_NAMES == (
        "memory_search",
        "memory_timeline",
        "memory_provenance",
    )
    assert tuple(MEMORY_TOOL_MODELS) == MEMORY_FUNCTION_NAMES


def test_hst_02_dates_resolve_and_default_to_now() -> None:
    args = MemorySearchArgs(query="q", valid_at="2022-07-01")
    assert args.resolved_valid_at(NOW) == datetime(2022, 7, 1, tzinfo=UTC)
    assert args.resolved_known_at(NOW) == NOW
    args = MemorySearchArgs(query="q", known_at="2022-07-01T09:30:00Z")
    assert args.resolved_known_at(NOW) == datetime(2022, 7, 1, 9, 30, tzinfo=UTC)
    assert args.resolved_valid_at(NOW) == NOW


def test_bad_date_fails_validation() -> None:
    with pytest.raises(ValidationError, match="valid_at is not a date"):
        MemorySearchArgs(query="q", valid_at="last July")
    with pytest.raises(ValidationError, match="known_at is not a date"):
        MemorySearchArgs(query="q", known_at="yesterday-ish")


@pytest.mark.parametrize("hops", [0, 3])
def test_ret_04_hops_out_of_range_rejected(hops: int) -> None:
    with pytest.raises(ValidationError):
        MemorySearchArgs(query="q", hops=hops)
    assert MemorySearchArgs(query="q", hops=2).hops == 2


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_arguments_rejected(blank: str) -> None:
    with pytest.raises(ValidationError):
        MemorySearchArgs(query=blank)
    with pytest.raises(ValidationError):
        MemoryTimelineArgs(entity=blank)
    with pytest.raises(ValidationError):
        MemoryProvenanceArgs(fact_id=blank)


def test_unknown_argument_rejected() -> None:
    with pytest.raises(ValidationError):
        MemorySearchArgs.model_validate({"query": "q", "limit": 5})
    with pytest.raises(ValidationError):
        MemoryTimelineArgs.model_validate({"entity": "e", "x": 1})


def test_hst_02_fact_hit_serializes_instants_with_z() -> None:
    out = MemoryHitOut.from_hit(_fact_hit())
    assert out.valid_from == "2020-07-01T00:00:00Z"
    assert out.valid_to == "2022-07-01T00:00:00Z"
    assert out.fact_id == "f1"


def test_chunk_hit_has_null_fact_fields() -> None:
    out = MemoryHitOut.from_hit(_chunk_hit())
    assert out.kind == "chunk"
    assert out.fact_id is None
    assert out.valid_from is None
    assert "degraded" not in out.model_dump()


def test_ret_07_search_result_degraded_and_instants() -> None:
    ok = MemorySearchResult.build([_fact_hit(), _chunk_hit()], NOW, NOW)
    assert ok.degraded is False
    assert ok.valid_at == "2026-11-11T12:00:00Z"
    assert ok.known_at == "2026-11-11T12:00:00Z"
    bad = MemorySearchResult.build([_fact_hit(), _chunk_hit(degraded=True)], NOW, NOW)
    assert bad.degraded is True
    assert len(bad.hits) == 2


def test_provenance_none_is_not_found() -> None:
    result = MemoryProvenanceResult.build(None)
    assert result.found is False
    assert result.fact is None
    assert result.sources == []


def test_ret_06_provenance_keeps_source_text() -> None:
    ref = EpisodeRef(
        episode_id=uuid4(),
        source_tool="docs",
        chunk_id="c1",
        source_uri="a.md",
        reference_time=datetime(2020, 7, 1, tzinfo=UTC),
        reference_time_source="front_matter",
        text="the chunk text",
    )
    prov = Provenance(
        fact_id="f1",
        text="Rate is 5%",
        valid_from=None,
        valid_to=None,
        recorded_at=NOW,
        expired_at=None,
        sources=(ref,),
    )
    result = MemoryProvenanceResult.build(prov)
    assert result.found is True
    assert result.fact is not None
    assert result.fact.recorded_at == "2026-11-11T12:00:00Z"
    assert result.fact.valid_from is None
    assert result.sources[0].text == "the chunk text"
    assert result.sources[0].reference_time == "2020-07-01T00:00:00Z"


def test_error_model_codes() -> None:
    err = MemoryToolError.model_validate(
        {"error": {"code": "not_supported", "message": "m"}}
    )
    assert err.error.code == "not_supported"
    with pytest.raises(ValidationError):
        MemoryToolError.model_validate({"error": {"code": "other", "message": "m"}})


def test_hst_01_schema_check_passes_on_committed_files() -> None:
    gen = _load_generator()
    assert gen.stale_files(gen.SCHEMA_DIR) == []
    assert gen.main(["--check"]) == 0


def test_hst_01_schema_check_fails_on_altered_copy(tmp_path: Path) -> None:
    gen = _load_generator()
    for name, text in gen.render_schemas().items():
        (tmp_path / name).write_text(text)
    assert gen.stale_files(tmp_path) == []
    altered = tmp_path / "memory_search.args.schema.json"
    altered.write_text(altered.read_text().replace("query", "qwery", 1))
    (tmp_path / "error.schema.json").unlink()
    stale = {p.name for p in gen.stale_files(tmp_path)}
    assert stale == {"memory_search.args.schema.json", "error.schema.json"}


def test_hst_01_schema_files_are_object_schemas() -> None:
    gen = _load_generator()
    files = sorted(gen.SCHEMA_DIR.glob("*.json"))
    assert len(files) == 7
    for path in files:
        schema = json.loads(path.read_text())
        assert schema["type"] == "object"
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"]


def test_ret_07_an_empty_degraded_result_stays_degraded() -> None:
    assert MemorySearchResult.build([], NOW, NOW, degraded=True).degraded is True
    assert MemorySearchResult.build([], NOW, NOW).degraded is False


def test_ret_04_a_boolean_is_not_a_hop_count() -> None:
    with pytest.raises(ValidationError):
        MemorySearchArgs.model_validate({"query": "rate", "hops": True})
    assert MemorySearchArgs.model_validate({"query": "rate", "hops": 2}).hops == 2
