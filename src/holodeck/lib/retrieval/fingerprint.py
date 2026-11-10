"""Pipeline fingerprint: what produced the records of a collection (FR-023).

A collection's records depend on more than its schema: the embedding model,
the parser, the chunker and its settings, and the retrieval settings that
change stored text. ``PipelineInputs`` lists those inputs and
``PipelineInputs.fingerprint`` hashes them. Two pipelines with the same
fingerprint write interchangeable records; a different fingerprint means the
records cannot be mixed in one collection.

Policy IDs (``parser_policy``, ``chunker_policy``) are version strings defined
next to the code they describe, for example
``holodeck.lib.text_chunker.SPLITTER_POLICY``. A change of behavior in that
code bumps its constant, which changes the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from holodeck.lib.retrieval.types import CollectionSchema

SettingValue = str | int | float | bool | None
"""A chunker or retrieval setting value."""

LEGACY_KEY = "legacy"
"""Key set in stored inputs of a collection created before fingerprints."""


def record_layout_id(schema: CollectionSchema) -> str:
    """Return a short stable ID of the stored layout of ``schema``.

    It covers field names, types and index flags, the key and vector field
    names, the dimensions and the distance. The collection name is not part
    of the layout: a migrated collection has a new name and the same layout.
    """
    layout = {
        "key_field": schema.key_field,
        "vector_field": schema.vector_field,
        "dimensions": schema.dimensions,
        "distance": schema.distance.value,
        "fields": [
            [spec.name, spec.type.value, spec.indexed, spec.full_text]
            for spec in schema.fields
        ],
    }
    return "layout/" + _sha256(_canonical(layout))[:16]


@dataclass(frozen=True, slots=True)
class PipelineInputs:
    """Everything that decides the content of stored records.

    Attributes:
        embedding_provider: Embedding provider name (``openai``, ``ollama``).
        embedding_model: Embedding model name.
        embedding_dimensions: Vector dimensions.
        parser_policy: Version string of the file-to-text conversion.
        chunker_policy: Version string of the splitter or structured chunker.
        chunker_settings: Chunker settings, as ``(name, value)`` pairs.
        record_layout: ``record_layout_id`` of the collection schema.
        retrieval_settings: Settings that change stored text, as pairs.
    """

    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    parser_policy: str
    chunker_policy: str
    chunker_settings: tuple[tuple[str, SettingValue], ...]
    record_layout: str
    retrieval_settings: tuple[tuple[str, SettingValue], ...] = ()

    def as_dict(self) -> dict[str, SettingValue]:
        """Return the inputs as one flat mapping, settings prefixed by group."""
        flat: dict[str, SettingValue] = {
            "embedding.provider": self.embedding_provider,
            "embedding.model": self.embedding_model,
            "embedding.dimensions": self.embedding_dimensions,
            "parser.policy": self.parser_policy,
            "chunker.policy": self.chunker_policy,
            "record.layout": self.record_layout,
        }
        for name, value in sorted(self.chunker_settings):
            flat[f"chunker.{name}"] = value
        for name, value in sorted(self.retrieval_settings):
            flat[f"retrieval.{name}"] = value
        return flat

    def canonical_json(self) -> str:
        """Return the inputs as canonical JSON (sorted keys, no spaces)."""
        return _canonical(self.as_dict())

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 hex digest of ``canonical_json``."""
        return _sha256(self.canonical_json())


def parse_inputs(text: str) -> dict[str, SettingValue]:
    """Decode stored canonical inputs. Malformed text decodes as empty."""
    try:
        loaded = json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {
        str(key): value
        for key, value in loaded.items()
        if value is None or isinstance(value, str | int | float | bool)
    }


def legacy_inputs_json(inputs: PipelineInputs) -> str:
    """Return the canonical inputs of ``inputs`` marked as a legacy adoption."""
    marked = dict(inputs.as_dict())
    marked[LEGACY_KEY] = True
    return _canonical(marked)


def is_legacy(inputs_json: str) -> bool:
    """Return True when stored inputs carry the legacy mark."""
    return parse_inputs(inputs_json).get(LEGACY_KEY) is True


def diff_inputs(
    stored: Mapping[str, SettingValue], current: Mapping[str, SettingValue]
) -> list[tuple[str, SettingValue, SettingValue]]:
    """Return ``(name, stored, current)`` for every input that differs.

    The legacy mark is not an input and is ignored.
    """
    names = sorted((set(stored) | set(current)) - {LEGACY_KEY})
    return [
        (name, stored.get(name), current.get(name))
        for name in names
        if stored.get(name) != current.get(name)
    ]


def format_diff(differences: list[tuple[str, SettingValue, SettingValue]]) -> str:
    """Return ``name: old -> new`` lines for ``diff_inputs`` output."""
    return "\n".join(
        f"  {name}: {_show(old)} -> {_show(new)}" for name, old, new in differences
    )


def _show(value: SettingValue) -> str:
    return "(none)" if value is None else json.dumps(value)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
