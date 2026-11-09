"""OTel span processor that redacts credential-shaped trace attributes.

Sits *before* exporting span processors on the tracer provider so any
exporter (OTLP, Console, Azure Monitor) sees scrubbed attributes. Runs
independently of ``claude.disable_default_hooks`` — operators cannot
accidentally disable trace redaction by disabling user-facing hooks.

Spec 034 P2b §"OTel attribute redaction (independent of hooks)".
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import Any

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Event, ReadableSpan, Span, SpanProcessor
from opentelemetry.util.types import AnyValue

from holodeck.lib.backends.claude_hooks import redact_credentials

logger = logging.getLogger(__name__)

# One-shot guard: emit the SDK-shape-drift warning at most once per process.
_warned_missing_attributes_attr = False

# Span attribute name prefixes whose values are scrubbed. Anything with a
# different prefix is left alone — these namespaces are the ones the GenAI
# instrumentors populate with prompt/completion/tool I/O content:
# `otel-instrumentation-claude-agent-sdk` uses the tool.* / gen_ai.tool.*
# and legacy gen_ai.prompt / gen_ai.completion names; LiteLLM's OTel
# callback emits message content under the current GenAI semconv names
# gen_ai.input.messages / gen_ai.output.messages / gen_ai.system_instructions.
_REDACTED_PREFIXES: tuple[str, ...] = (
    "tool.input",
    "tool.output",
    "gen_ai.tool.input",
    "gen_ai.tool.output",
    # Current GenAI semconv tool-call content, emitted by
    # otel-instrumentation-claude-agent-sdk on `execute_tool` spans.
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "gen_ai.prompt",
    "gen_ai.completion",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    # LiteLLM's raw-response child span (emitted only when content capture is
    # on) carries the provider payload verbatim.
    "llm.openai.stringified_raw_response",
    # Failure telemetry is written regardless of content capture: LiteLLM's
    # error.* attributes and the OTel ``exception`` event echo the provider
    # error, which can quote request content.
    "error.",
    "exception.",
)


def _should_redact(attribute_key: str) -> bool:
    return any(attribute_key.startswith(prefix) for prefix in _REDACTED_PREFIXES)


# Exported in place of a value whose redaction raised: fail closed, never
# export the original.
_REDACTION_FAILED = "[REDACTED:redaction-failed]"


def _decode(value: AnyValue) -> AnyValue:
    """Decode bytes (alone or inside a sequence) so the scrubber can see them.

    OTel SDK 1.45+ keeps ``bytes`` attribute values as bytes, and
    ``redact_credentials`` only scans text.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)) and any(isinstance(v, bytes) for v in value):
        decoded = [
            v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
            for v in value
        ]
        return tuple(decoded) if isinstance(value, tuple) else decoded
    return value


def _redacted_value(key: str, value: AnyValue, what: str) -> AnyValue:
    try:
        redacted: AnyValue = redact_credentials(_decode(value))
        return redacted
    except Exception:  # noqa: BLE001 — never break tracing on redact failure
        logger.warning(
            "RedactingSpanProcessor: failed to redact %s %s; "
            "exporting a placeholder instead",
            what,
            key,
        )
        return _REDACTION_FAILED


def _redact_mapping(attributes: Mapping[str, Any], what: str) -> Mapping[str, Any]:
    """Scrub every redacted-prefix key of *attributes*.

    The SDK freezes attribute mappings (event attributes at ``add_event``,
    span attributes before ``on_end`` since OTel SDK 1.45). A frozen
    ``BoundedAttributes`` is unfrozen for the edit and frozen again, so the
    scrub lands in the one mapping the ``Span`` and its ``ReadableSpan``
    snapshot share: a processor holding the original span sees it too.
    Returns the mapping to keep, which is a new copy only for a read-only
    mapping of another type.
    """
    keys = [key for key in attributes if _should_redact(key)]
    # Keep a typed handle: where opentelemetry is untyped (e.g. the pre-commit
    # mypy env), isinstance() narrows *attributes* to Any.
    original = attributes
    if not keys:
        return original
    if isinstance(attributes, BoundedAttributes):
        frozen = attributes._immutable
        attributes._immutable = False
        try:
            for key in keys:
                attributes[key] = _redacted_value(key, attributes[key], what)
        finally:
            attributes._immutable = frozen
        return original
    if isinstance(attributes, MutableMapping):
        for key in keys:
            attributes[key] = _redacted_value(key, attributes[key], what)
        return original
    scrubbed = dict(attributes)
    for key in keys:
        scrubbed[key] = _redacted_value(key, scrubbed[key], what)
    rebuilt: Mapping[str, Any] = BoundedAttributes(None, scrubbed, immutable=True)
    return rebuilt


def _redact_event(event: Event) -> None:
    """Scrub an event's attributes (exporters read ``event.attributes``)."""
    attributes = event.attributes
    if not attributes:
        return
    scrubbed = _redact_mapping(attributes, f"event {event.name!r} attribute")
    if scrubbed is not attributes:
        event._attributes = scrubbed


class RedactingSpanProcessor(SpanProcessor):
    """SpanProcessor that scrubs credential-shaped strings on span end.

    ``Span.end`` builds one ``ReadableSpan`` snapshot that shares the span's
    attribute mapping and hands it to every processor in registration order.
    The mapping is frozen by then (OTel SDK 1.45+), so it is unfrozen for the
    scrub and frozen again (see ``_redact_mapping``). Span events
    (``record_exception`` and any content events) are scrubbed the same way.
    A value whose redaction raises is exported as a placeholder, never as the
    original. Exporters registered AFTER this processor see the redacted
    payload.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        global _warned_missing_attributes_attr
        attributes = getattr(span, "_attributes", None)
        if attributes is None:
            if not _warned_missing_attributes_attr:
                logger.error(
                    "RedactingSpanProcessor: span has no `_attributes` attr "
                    "— OTel SDK may have changed shape. Trace redaction is "
                    "NOT running. Audit the OTel SDK version."
                )
                _warned_missing_attributes_attr = True
            return
        if attributes:
            scrubbed = _redact_mapping(attributes, "attribute")
            if scrubbed is not attributes:
                span._attributes = scrubbed
        for event in span.events:
            _redact_event(event)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True
