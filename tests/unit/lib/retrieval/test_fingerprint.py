"""Pipeline fingerprint: stable hash, and which inputs change it (FR-023, AC-07)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from holodeck.lib.retrieval.fingerprint import (
    PipelineInputs,
    diff_inputs,
    format_diff,
    is_legacy,
    legacy_inputs_json,
    parse_inputs,
    record_layout_id,
)
from holodeck.lib.retrieval.schemas import document_schema, hierarchical_schema
from holodeck.lib.text_chunker import LEGACY_SPLITTER_POLICY, SPLITTER_POLICY


def _inputs(**changes: Any) -> PipelineInputs:
    base = PipelineInputs(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        parser_policy="markitdown/1",
        chunker_policy=SPLITTER_POLICY,
        chunker_settings=(("chunk_size", 512), ("chunk_overlap", 50)),
        record_layout=record_layout_id(document_schema(1536)),
        retrieval_settings=(),
    )
    return dataclasses.replace(base, **changes)


def test_fingerprint_is_stable_and_hex() -> None:
    first = _inputs().fingerprint
    assert first == _inputs().fingerprint
    assert len(first) == 64
    int(first, 16)


def test_canonical_form_is_frozen() -> None:
    """A change here reports every existing collection as incompatible."""
    inputs = PipelineInputs(
        embedding_provider="openai",
        embedding_model="m",
        embedding_dimensions=8,
        parser_policy="p/1",
        chunker_policy="c/1",
        chunker_settings=(("b", 2), ("a", 1)),
        record_layout="layout/x",
    )
    assert inputs.canonical_json() == (
        '{"chunker.a":1,"chunker.b":2,"chunker.policy":"c/1",'
        '"embedding.dimensions":8,"embedding.model":"m",'
        '"embedding.provider":"openai","parser.policy":"p/1",'
        '"record.layout":"layout/x"}'
    )


def test_settings_order_does_not_change_the_fingerprint() -> None:
    reordered = _inputs(chunker_settings=(("chunk_overlap", 50), ("chunk_size", 512)))
    assert reordered.fingerprint == _inputs().fingerprint


@pytest.mark.parametrize(
    "changes",
    [
        {"embedding_provider": "azure_openai"},
        {"embedding_model": "text-embedding-3-large"},
        {"embedding_dimensions": 768},
        {"parser_policy": "markitdown/2"},
        {"chunker_policy": LEGACY_SPLITTER_POLICY},
        {"chunker_settings": (("chunk_size", 256), ("chunk_overlap", 50))},
        {"chunker_settings": (("chunk_size", 512), ("chunk_overlap", 0))},
        {"record_layout": record_layout_id(hierarchical_schema(1536, "t"))},
        {"retrieval_settings": (("contextual_embeddings", True),)},
    ],
)
def test_each_input_changes_the_fingerprint(changes: dict[str, Any]) -> None:
    assert _inputs(**changes).fingerprint != _inputs().fingerprint


def test_layout_ignores_the_collection_name_but_not_the_fields() -> None:
    plain = document_schema(8)
    renamed = dataclasses.replace(plain, name="documents_dim8__abcd1234")
    assert record_layout_id(plain) == record_layout_id(renamed)
    assert record_layout_id(plain) != record_layout_id(document_schema(16))
    assert record_layout_id(plain) != record_layout_id(hierarchical_schema(8, "t"))


def test_diff_lists_changed_inputs_and_ignores_the_legacy_mark() -> None:
    old = parse_inputs(legacy_inputs_json(_inputs()))
    new = _inputs(chunker_settings=(("chunk_size", 256), ("chunk_overlap", 50)))
    differences = diff_inputs(old, new.as_dict())
    assert differences == [("chunker.chunk_size", 512, 256)]
    assert format_diff(differences) == "  chunker.chunk_size: 512 -> 256"


def test_legacy_mark_round_trips() -> None:
    assert is_legacy(legacy_inputs_json(_inputs()))
    assert not is_legacy(_inputs().canonical_json())
    assert not is_legacy("not json")
    assert parse_inputs("[1, 2]") == {}
