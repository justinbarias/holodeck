"""OTel span processor that redacts credential-shaped trace attributes.

Sits *before* exporting span processors on the tracer provider so any
exporter (OTLP, Console, Azure Monitor) sees scrubbed attributes. Runs
independently of ``claude.disable_default_hooks`` — operators cannot
accidentally disable trace redaction by disabling user-facing hooks.

Spec 034 P2b §"OTel attribute redaction (independent of hooks)".
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Event, ReadableSpan, Span, SpanProcessor

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


def _redact_mapping(attributes: MutableMapping[str, Any], what: str) -> None:
    """Scrub every redacted-prefix key of *attributes* in place."""
    for key in list(attributes.keys()):
        if not _should_redact(key):
            continue
        try:
            attributes[key] = redact_credentials(attributes[key])
        except Exception:  # noqa: BLE001 — never break tracing on redact failure
            logger.warning(
                "RedactingSpanProcessor: failed to redact %s %s; "
                "leaving original value",
                what,
                key,
            )


def _redact_event(event: Event) -> None:
    """Replace an event's attribute mapping with a scrubbed copy.

    The SDK wraps event attributes in an immutable ``BoundedAttributes`` at
    ``add_event`` time, so the mapping is rebuilt (same limits, still
    immutable) rather than edited in place. Exporters read ``event.attributes``,
    which returns the replaced mapping.
    """
    attributes = event.attributes
    if not attributes:
        return
    keys = [key for key in attributes if _should_redact(key)]
    if not keys:
        return
    try:
        scrubbed = dict(attributes)
        for key in keys:
            scrubbed[key] = redact_credentials(scrubbed[key])
        maxlen = getattr(attributes, "maxlen", None)
        max_value_len = getattr(attributes, "max_value_len", None)
        event._attributes = BoundedAttributes(
            maxlen, scrubbed, immutable=True, max_value_len=max_value_len
        )
    except Exception:  # noqa: BLE001 — never break tracing on redact failure
        logger.warning(
            "RedactingSpanProcessor: failed to redact event %r attributes; "
            "leaving original values",
            event.name,
        )


class RedactingSpanProcessor(SpanProcessor):
    """SpanProcessor that scrubs credential-shaped strings on span end.

    The OTel SDK exposes span attributes on ``ReadableSpan`` via the
    ``_attributes`` dict, which the SDK mutates in place during the span's
    lifetime. Mutating it on ``on_end`` is the documented mechanism used by
    e.g. the Baggage span processor. Span events (``record_exception`` and
    any content events) are scrubbed the same way through their mutable
    attribute mapping. Exporters registered AFTER this processor see the
    redacted payload.
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
            _redact_mapping(attributes, "attribute")
        for event in span.events:
            _redact_event(event)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True
