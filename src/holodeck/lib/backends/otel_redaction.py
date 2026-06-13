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

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True
