"""JSON-schema-driven structured output for the OpenAI Agents backend (FR-004).

HoloDeck's ``Agent.response_format`` is a JSON-schema *dict* (or a path to a JSON
file holding one), not a Python type. The SDK's only concrete output schema —
``AgentOutputSchema`` — requires a Python ``type`` to build its schema from. This
module provides :func:`build_output_schema`, a factory returning a
``JSONSchemaOutputSchema`` that implements the SDK's ``AgentOutputSchemaBase``
abstract surface directly from a JSON-schema dict, validating the model's JSON
output with the ``jsonschema`` library.

It also exposes :func:`load_response_format_schema`, which resolves the three
``response_format`` shapes (inline dict / file path / ``None``) to a schema dict,
matching the Claude backend's str-path resolution semantics.

Every ``import agents`` is performed lazily inside the factory so importing this
module never pulls the SDK (SC-005). The ``JSONSchemaOutputSchema`` subclass is
defined inside :func:`build_output_schema`, which imports the SDK base class at
runtime, mirroring the pattern used in ``openai_agents_cost.py`` /
``openai_agents_fallback.py``.

Strictness
----------
OpenAI strict structured outputs constrain the JSON schema (every object must set
``additionalProperties: false`` and list all properties as ``required``). We do
**not** rewrite the operator's schema to satisfy that. Strictness therefore
defaults to ``False`` unless the schema already qualifies — :func:`
schema_qualifies_for_strict` checks the top-level object — in which case strict
mode is requested so the provider guarantees conforming JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.config.context import agent_base_dir

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from agents.agent_output import AgentOutputSchemaBase

logger = logging.getLogger(__name__)

_DEFAULT_SCHEMA_NAME = "holodeck_response"


def load_response_format_schema(
    response_format: dict[str, Any] | str | None,
    base_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Resolve ``Agent.response_format`` to a JSON-schema dict, or ``None``.

    Mirrors the Claude backend's resolution semantics: an inline dict is returned
    as-is; a string is treated as a path to a JSON file and loaded; ``None``
    yields ``None`` (no structured output). Relative paths resolve against
    *base_dir* (falling back to the ``agent_base_dir`` context variable, then
    CWD), consistent with the rest of this backend's path handling.

    Args:
        response_format: A JSON-schema dict, a path to a JSON schema file, or
            ``None``.
        base_dir: Directory for resolving a relative schema path. Falls back to
            the ``agent_base_dir`` context variable, then CWD.

    Returns:
        The JSON-schema dict, or ``None`` when no ``response_format`` is set.

    Raises:
        ValueError: If the path does not exist or does not hold a JSON object.
    """
    if response_format is None:
        return None
    if isinstance(response_format, dict):
        return response_format

    resolved_base = base_dir
    if resolved_base is None:
        ctx = agent_base_dir.get()
        resolved_base = Path(ctx) if ctx else None
    schema_path = (
        resolved_base / response_format if resolved_base else Path(response_format)
    )
    if not schema_path.exists():
        raise ValueError(f"response_format schema file not found: {schema_path}")
    parsed = json.loads(schema_path.read_text())
    if not isinstance(parsed, dict):
        raise ValueError(
            f"response_format schema file must hold a JSON object: {schema_path}"
        )
    return parsed


def schema_qualifies_for_strict(schema: dict[str, Any]) -> bool:
    """Return whether *schema* already satisfies OpenAI strict-mode constraints.

    Strict mode requires the top-level object to declare
    ``additionalProperties: false`` and to mark every declared property as
    ``required``. We only enable strict mode when the operator's schema already
    meets this — we never rewrite their schema.

    Args:
        schema: The JSON-schema dict.

    Returns:
        ``True`` when the schema qualifies for strict mode, ``False`` otherwise.
    """
    if schema.get("type") != "object":
        return False
    if schema.get("additionalProperties") is not False:
        return False
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return False
    required = schema.get("required")
    if not isinstance(required, list):
        return False
    return set(properties) == set(required)


def build_output_schema(
    schema: dict[str, Any], *, name: str = _DEFAULT_SCHEMA_NAME
) -> AgentOutputSchemaBase:
    """Build an SDK ``AgentOutputSchemaBase`` from a JSON-schema *dict*.

    The returned object drives the SDK's structured-output path from a JSON
    schema rather than a Python type. ``validate_json`` parses the model's JSON
    output and validates it against *schema* via the ``jsonschema`` library,
    returning the parsed object; an invalid payload raises ``ModelBehaviorError``
    so the SDK surfaces it consistently with its own schema.

    Strict mode is enabled only when :func:`schema_qualifies_for_strict` returns
    ``True`` for *schema* (default ``False``); the operator's schema is never
    rewritten.

    Args:
        schema: The JSON-schema dict describing the expected output object.
        name: The schema name surfaced to the provider.

    Returns:
        An ``AgentOutputSchemaBase`` usable as ``Agent(output_type=...)``.
    """
    from agents.agent_output import AgentOutputSchemaBase as SDKBase
    from agents.exceptions import ModelBehaviorError

    strict = schema_qualifies_for_strict(schema)

    class JSONSchemaOutputSchema(SDKBase):
        """``AgentOutputSchemaBase`` driven by a JSON-schema dict."""

        def __init__(self, schema: dict[str, Any], name: str, *, strict: bool) -> None:
            self._schema = schema
            self._name = name
            self._strict = strict

        def is_plain_text(self) -> bool:
            """Structured output is always a JSON object, never plain text."""
            return False

        def name(self) -> str:
            """The schema name surfaced to the provider."""
            return self._name

        def json_schema(self) -> dict[str, Any]:
            """The JSON schema of the output object."""
            return self._schema

        def is_strict_json_schema(self) -> bool:
            """Whether strict mode is requested for this schema."""
            return self._strict

        def validate_json(self, json_str: str) -> Any:
            """Parse and validate *json_str* against the JSON schema.

            Args:
                json_str: The raw JSON string produced by the model.

            Returns:
                The parsed object on success.

            Raises:
                ModelBehaviorError: If the JSON is malformed or fails schema
                    validation.
            """
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as exc:
                raise ModelBehaviorError(f"Output is not valid JSON: {exc}") from exc

            import jsonschema

            try:
                jsonschema.validate(instance=parsed, schema=self._schema)
            except jsonschema.ValidationError as exc:
                raise ModelBehaviorError(
                    f"Output does not match response_format schema: {exc.message}"
                ) from exc
            return parsed

    return JSONSchemaOutputSchema(schema, name, strict=strict)
