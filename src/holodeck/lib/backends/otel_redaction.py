"""OTel span processor that redacts credential-shaped trace attributes.

Sits *before* exporting span processors on the tracer provider so any
exporter (OTLP, Console, Azure Monitor) sees scrubbed attributes. Runs
independently of ``claude.disable_default_hooks`` — operators cannot
accidentally disable trace redaction by disabling user-facing hooks.

Spec 034 P2b §"OTel attribute redaction (independent of hooks)".
"""

from __future__ import annotations

import logging

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from holodeck.lib.backends.claude_hooks import redact_credentials

logger = logging.getLogger(__name__)

# Span attribute name prefixes whose values are scrubbed. Anything with a
# different prefix is left alone — these namespaces are the ones the GenAI
# instrumentor (`otel-instrumentation-claude-agent-sdk`) populates with
# tool I/O content.
_REDACTED_PREFIXES: tuple[str, ...] = (
    "tool.input",
    "tool.output",
    "gen_ai.tool.input",
    "gen_ai.tool.output",
    "gen_ai.prompt",
    "gen_ai.completion",
)


def _should_redact(attribute_key: str) -> bool:
    return any(attribute_key.startswith(prefix) for prefix in _REDACTED_PREFIXES)


class RedactingSpanProcessor(SpanProcessor):
    """SpanProcessor that scrubs credential-shaped strings on span end.

    The OTel SDK exposes span attributes on ``ReadableSpan`` via the
    ``_attributes`` dict, which the SDK mutates in place during the span's
    lifetime. Mutating it on ``on_end`` is the documented mechanism used by
    e.g. the Baggage span processor. Exporters registered AFTER this
    processor see the redacted payload.
    """

    def on_start(  # type: ignore[override]
        self, span: Span, parent_context: Context | None = None
    ) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        attributes = getattr(span, "_attributes", None)
        if not attributes:
            return
        for key in list(attributes.keys()):
            if not _should_redact(key):
                continue
            try:
                attributes[key] = redact_credentials(attributes[key])
            except Exception:  # noqa: BLE001 — never break tracing on redact failure
                logger.warning(
                    "RedactingSpanProcessor: failed to redact attribute %s; "
                    "leaving original value",
                    key,
                )

    def shutdown(self) -> None:  # type: ignore[override]
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # type: ignore[override]
        return True
