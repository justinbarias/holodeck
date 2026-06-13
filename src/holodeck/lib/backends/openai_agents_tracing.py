"""OTel-mirroring ``TracingProcessor`` for the OpenAI Agents backend (H1).

The OpenAI Agents SDK runs its own tracing pipeline: every agent / generation /
function / handoff step opens an SDK ``Span`` and emits start/end events to the
registered ``TracingProcessor`` list. HoloDeck's observability stack is OTel,
so this module bridges the two: :func:`build_tracing_mirror` returns a
``TracingProcessor`` that, on each SDK span *end*, reconstructs an OTel span via
HoloDeck's global ``TracerProvider`` — the same provider that carries the
``RedactingSpanProcessor`` and the configured exporters. Mirrored spans
therefore (a) flow to whatever exporter the agent configured (OTLP, console,
Azure Monitor) and (b) have their credential-shaped ``tool.input`` / ``tool.output``
attributes scrubbed by ``RedactingSpanProcessor`` before export (FR-088).

Why mirror on *end* only
------------------------
SDK spans carry explicit ISO-8601 ``started_at`` / ``ended_at`` timestamps, so
the OTel span can be created after the fact with an explicit start/end window.
Mirroring on ``on_span_end`` (rather than juggling live OTel spans across the
start/end pair) is the simplest correct approach and keeps the processor
stateless and thread-safe.

Hierarchy reconstruction (best-effort)
--------------------------------------
The SDK exposes string ``trace_id`` / ``span_id`` / ``parent_id`` values that do
**not** map onto OTel's 128/64-bit trace/span ids, and OTel ids cannot be forced
onto a span. Rather than fabricate a parent ``Context`` (which would require
holding live OTel spans and risk leaking them), the SDK ids are attached as
plain attributes (``gen_ai.openai.trace_id`` / ``.span_id`` / ``.parent_id``) so
the hierarchy is reconstructable downstream from the attribute values. Each
mirrored span is otherwise a root OTel span.

SDK-free import (SC-005)
------------------------
Every ``import agents`` is performed lazily inside :func:`build_tracing_mirror`,
so importing this module never pulls the SDK. The ``TracingProcessor`` subclass
is defined *inside* the factory (mirroring ``openai_agents_cost.py`` /
``openai_agents_fallback.py``) because its base class is the SDK ABC, which is
only available at runtime.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from holodeck.lib.observability import get_tracer

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from agents.tracing import Span, Trace, TracingProcessor

logger = logging.getLogger(__name__)

_TRACER_NAME = "holodeck.openai_agents"


def _iso_to_unix_nanos(value: str | None) -> int | None:
    """Convert an ISO-8601 timestamp string to epoch nanoseconds, or ``None``.

    The SDK emits ``started_at`` / ``ended_at`` as ISO-8601 strings (e.g.
    ``2026-06-13T01:00:00.000000+00:00``). OTel spans want epoch nanoseconds.

    Args:
        value: An ISO-8601 timestamp string, or ``None``.

    Returns:
        The timestamp in epoch nanoseconds, or ``None`` when *value* is missing
        or unparseable.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return int(parsed.timestamp() * 1_000_000_000)


def _usage_attributes(usage: Any) -> dict[str, int]:
    """Extract token-usage attributes from an SDK span's ``usage`` mapping.

    The SDK records usage as a plain dict (``{"input_tokens": .., ...}``) on
    generation / response spans. Only integer-valued token fields are mapped, to
    the GenAI ``gen_ai.usage.*`` semantic-convention keys.

    Args:
        usage: The span data's ``usage`` value (a dict, or ``None``).

    Returns:
        A mapping of OTel attribute keys to token counts (empty when no usage).
    """
    if not isinstance(usage, dict):
        return {}
    attributes: dict[str, int] = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        raw = usage.get(field)
        if isinstance(raw, int):
            attributes[f"gen_ai.usage.{field}"] = raw
    return attributes


def build_tracing_mirror(agent_name: str) -> TracingProcessor:
    """Build an OTel-mirroring SDK ``TracingProcessor`` for *agent_name*.

    The returned processor mirrors each finished SDK span into an OTel span on
    HoloDeck's global ``TracerProvider`` (carrying its ``RedactingSpanProcessor``
    and configured exporters). The SDK ``TracingProcessor`` base class is
    imported here (not at module import) so this module stays SDK-free (SC-005),
    and the subclass is defined inside this factory for the same reason.

    Args:
        agent_name: The HoloDeck agent name, recorded on every mirrored span as
            ``gen_ai.agent.name`` so spans correlate to the agent.

    Returns:
        A ``TracingProcessor`` ready to pass to ``agents.add_trace_processor`` /
        ``agents.set_trace_processors``.
    """
    from agents.tracing import TracingProcessor as SDKTracingProcessor

    class _OTelTracingMirror(SDKTracingProcessor):
        """Mirrors SDK trace spans into HoloDeck's OTel pipeline on span end."""

        def __init__(self, agent_name: str) -> None:
            self._agent_name = agent_name

        def on_trace_start(self, trace: Trace) -> None:
            """No-op: traces have no OTel analogue; spans carry the hierarchy."""
            del trace

        def on_trace_end(self, trace: Trace) -> None:
            """No-op: the OTel span window is taken from each span's own times."""
            del trace

        def on_span_start(self, span: Span[Any]) -> None:
            """No-op: spans are mirrored after the fact in :meth:`on_span_end`."""
            del span

        def on_span_end(self, span: Span[Any]) -> None:
            """Mirror a finished SDK span into an OTel span.

            Reconstructs an OTel span with the SDK span's explicit start/end
            timestamps, maps the span-type-specific attributes, and ends it so
            the global ``RedactingSpanProcessor`` and exporters run. Any failure
            is swallowed and logged: tracing must never break the agent run.

            Args:
                span: The finished SDK ``Span``.
            """
            try:
                self._mirror(span)
            except Exception:  # noqa: BLE001 - never break the run on a trace error
                logger.debug(
                    "openai_agents tracing mirror: failed to mirror span",
                    exc_info=True,
                )

        def _mirror(self, span: Span[Any]) -> None:
            """Create and end the OTel span for *span* (see :meth:`on_span_end`)."""
            span_data = span.span_data
            span_type = str(getattr(span_data, "type", "span"))
            attributes = self._attributes(span, span_type, span_data)

            start_ns = _iso_to_unix_nanos(getattr(span, "started_at", None))
            end_ns = _iso_to_unix_nanos(getattr(span, "ended_at", None))

            tracer = get_tracer(_TRACER_NAME)
            otel_span = tracer.start_span(
                self._span_name(span_type, span_data),
                start_time=start_ns,
                attributes=attributes,
            )
            error = getattr(span, "error", None)
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    otel_span.set_status(_error_status(str(message)))
            otel_span.end(end_time=end_ns)

        def _span_name(self, span_type: str, span_data: Any) -> str:
            """Build the OTel span name for an SDK span.

            Args:
                span_type: The SDK span type (``"function"``, ``"generation"``…).
                span_data: The SDK span data carrying an optional ``name``.

            Returns:
                A ``openai_agents.<type>[ <name>]`` span name.
            """
            name = getattr(span_data, "name", None)
            base = f"openai_agents.{span_type}"
            return f"{base} {name}" if isinstance(name, str) and name else base

        def _attributes(
            self, span: Span[Any], span_type: str, span_data: Any
        ) -> dict[str, Any]:
            """Map an SDK span onto OTel attributes (redaction-safe keys).

            Function-span input/output use the ``tool.input`` / ``tool.output``
            keys so the global ``RedactingSpanProcessor`` scrubs credentials from
            them before export (FR-088).

            Args:
                span: The finished SDK span (source of the hierarchy ids).
                span_type: The SDK span type string.
                span_data: The SDK span data.

            Returns:
                The OTel attribute mapping for this span.
            """
            attributes: dict[str, Any] = {
                "gen_ai.system": "openai",
                "gen_ai.agent.name": self._agent_name,
                "gen_ai.openai.span_type": span_type,
            }
            trace_id = getattr(span, "trace_id", None)
            span_id = getattr(span, "span_id", None)
            parent_id = getattr(span, "parent_id", None)
            if trace_id:
                attributes["gen_ai.openai.trace_id"] = str(trace_id)
            if span_id:
                attributes["gen_ai.openai.span_id"] = str(span_id)
            if parent_id:
                attributes["gen_ai.openai.parent_id"] = str(parent_id)

            model = getattr(span_data, "model", None)
            if isinstance(model, str) and model:
                attributes["gen_ai.request.model"] = model
            attributes.update(_usage_attributes(getattr(span_data, "usage", None)))

            if span_type == "function":
                name = getattr(span_data, "name", None)
                if isinstance(name, str) and name:
                    attributes["gen_ai.tool.name"] = name
                tool_input = getattr(span_data, "input", None)
                if tool_input is not None:
                    attributes["tool.input"] = str(tool_input)
                tool_output = getattr(span_data, "output", None)
                if tool_output is not None:
                    attributes["tool.output"] = str(tool_output)
            elif span_type == "handoff":
                from_agent = getattr(span_data, "from_agent", None)
                to_agent = getattr(span_data, "to_agent", None)
                if from_agent:
                    attributes["gen_ai.openai.handoff.from"] = str(from_agent)
                if to_agent:
                    attributes["gen_ai.openai.handoff.to"] = str(to_agent)

            return attributes

        def shutdown(self) -> None:
            """No-op: the OTel provider owns exporter flushing / shutdown."""

        def force_flush(self) -> None:
            """No-op: spans are exported synchronously on each ``on_span_end``."""

    return _OTelTracingMirror(agent_name)


def _error_status(message: str) -> Any:
    """Build an OTel ``ERROR`` ``Status`` carrying *message*.

    Imported lazily so this module's import cost stays minimal; the OTel SDK is
    always present (it is a core HoloDeck dependency).

    Args:
        message: The SDK span's error message.

    Returns:
        An OTel ``Status`` with ``StatusCode.ERROR``.
    """
    from opentelemetry.trace import Status, StatusCode

    return Status(StatusCode.ERROR, message)
