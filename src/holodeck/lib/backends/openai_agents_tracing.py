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

Process-global processor ownership (FR-100–FR-102, D13)
------------------------------------------------------
The SDK keeps **one** process-global trace-processor list, but a single worker
may initialize several backends with different upload policies (an OpenAI agent
that uploads to platform.openai.com next to an Azure agent that must not). A
single processor list cannot express both, so HoloDeck owns that list outright:
:func:`register_tracing_policy` installs exactly one HoloDeck *router* via
``set_trace_processors`` (replacing the SDK default exporter) and records a
per-backend :class:`TracingPolicy`. Each run tags its trace with the policy id
(``RunConfig.trace_metadata[TRACE_POLICY_METADATA_KEY]``); the router resolves
the policy on ``on_trace_start`` and forwards every trace/span of that trace to
the provider exporter only when ``policy.upload`` is true and to the policy's
OTel mirror only when one is configured. Provider upload therefore never
depends on which backend initialized first, on repeated initialization, or on
whether observability is enabled. Spans prefer the policy of the HoloDeck run
starting them (:func:`active_tracing_policy`), pinned per span id until the
span ends, so a run nested in a caller-owned ``trace()`` still follows its
backend and a span finished outside its scope or after its trace ended never
falls back to upload. Traces
that carry no HoloDeck identity at all (non-HoloDeck SDK usage) keep the SDK
default behaviour. Policies are resolved per event, so a withdrawn id drops
its events immediately (fail closed). :func:`unregister_tracing_policy`
removes a backend's policy at teardown.

SDK-free import (SC-005)
------------------------
Every ``import agents`` is performed lazily inside :func:`build_tracing_mirror`
and :func:`register_tracing_policy`, so importing this module never pulls the
SDK. The ``TracingProcessor`` subclasses are defined *inside* those factories
(mirroring ``openai_agents_cost.py`` / ``openai_agents_fallback.py``) because
their base class is the SDK ABC, which is only available at runtime.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from holodeck.lib.observability import get_tracer

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from agents.tracing import Span, Trace, TracingProcessor

logger = logging.getLogger(__name__)

_TRACER_NAME = "holodeck.openai_agents"

#: ``RunConfig.trace_metadata`` key carrying the backend's tracing-policy id.
TRACE_POLICY_METADATA_KEY = "holodeck.tracing_policy"

#: Policy id for the run currently executing on this task (fallback identity
#: for spans whose trace was opened outside HoloDeck's ``RunConfig``).
_active_policy_id: ContextVar[str | None] = ContextVar(
    "holodeck_openai_agents_tracing_policy", default=None
)


@dataclass(frozen=True)
class TracingPolicy:
    """Per-backend routing decision for SDK trace output.

    Attributes:
        upload: Forward the backend's traces/spans to the SDK's provider
            exporter (platform.openai.com). ``False`` for Azure and for
            ``observability.disable_provider_tracing: true``.
        mirror: The OTel-mirroring ``TracingProcessor`` receiving the backend's
            spans, or ``None`` when OTel tracing is disabled for the agent.
    """

    upload: bool
    mirror: TracingProcessor | None = None


_registry_lock = threading.Lock()
_registry: dict[str, TracingPolicy] = {}
_router: Any | None = None


@contextmanager
def active_tracing_policy(policy_id: str | None) -> Iterator[None]:
    """Mark *policy_id* as the tracing policy for runs started in this scope.

    The router prefers the trace's own metadata tag; this context variable is
    the fallback for spans emitted under a trace HoloDeck did not open (a
    caller's own ``trace()``), so such spans still follow the backend's policy.
    Tasks created inside the scope (``Runner.run_streamed``) inherit the value.

    Args:
        policy_id: The backend's registered policy id, or ``None`` for no-op.
    """
    if policy_id is None:
        yield
        return
    token = _active_policy_id.set(policy_id)
    try:
        yield
    finally:
        _active_policy_id.reset(token)


def register_tracing_policy(policy_id: str, policy: TracingPolicy) -> None:
    """Register *policy* for *policy_id*, installing the router once per process.

    The first registration replaces the SDK's process-global processor list
    with HoloDeck's router (``set_trace_processors``); later registrations only
    update the registry. Registering an existing id replaces its policy.

    Args:
        policy_id: The backend instance's unique policy id (also tagged onto
            each run's trace metadata under :data:`TRACE_POLICY_METADATA_KEY`).
        policy: The routing decision for that backend's traces.
    """
    global _router
    with _registry_lock:
        _registry[policy_id] = policy
        if _router is not None:
            return
        # Install under the lock so a concurrent second registration cannot
        # observe the router as installed before the SDK list is replaced.
        import agents

        router = _build_router()
        agents.set_trace_processors([router])
        _router = router


def unregister_tracing_policy(policy_id: str) -> None:
    """Remove the policy for *policy_id*; unknown ids are ignored.

    Takes effect immediately: every later event resolved to this id — from
    new or in-flight traces — is dropped rather than uploaded, so a torn-down
    Azure backend can never leak to the provider.

    Args:
        policy_id: The id previously passed to :func:`register_tracing_policy`.
    """
    with _registry_lock:
        _registry.pop(policy_id, None)


def _reset_tracing_router() -> None:
    """Forget the installed router and every policy (test isolation only).

    Does not touch the SDK processor list; callers restore that themselves.
    """
    global _router
    with _registry_lock:
        _registry.clear()
        _router = None


def _policy_id_from_trace(trace: Any) -> str | None:
    """Return the HoloDeck policy id tagged on *trace* metadata, if any."""
    metadata = getattr(trace, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(TRACE_POLICY_METADATA_KEY)
    return value if isinstance(value, str) and value else None


def _build_router() -> Any:
    """Build the process-global HoloDeck trace router (SDK imported lazily)."""
    from agents.tracing import TracingProcessor as SDKTracingProcessor

    class _HoloDeckTraceRouter(SDKTracingProcessor):
        """Route each trace's events by the registered :class:`TracingPolicy`.

        Identity resolution:

        * **Trace events** use the ``holodeck.tracing_policy`` metadata tag,
          then the :func:`active_tracing_policy` context variable, else the
          trace is *untagged* (SDK default: provider upload only).
        * **Span events** resolve identity once, at ``on_span_start``: the
          :func:`active_tracing_policy` of the task starting the span — the
          HoloDeck run that produced it — else the span's trace identity, else
          untagged. That identity is pinned to the span id until
          ``on_span_end`` consumes it, so a span finished outside its scope,
          after its trace ended, or after the bounded trace map evicted its
          trace still follows the policy it started under. A run nested inside
          a caller-owned or differently tagged outer ``trace()`` therefore
          follows its own backend's policy.

        Policies are looked up in the registry on every event, so
        :func:`unregister_tracing_policy` takes effect immediately: events for
        a withdrawn id are dropped (fail closed).
        """

        _UNTAGGED = object()
        _MAX_TRACES = 4096

        def __init__(self) -> None:
            self._lock = threading.Lock()
            # trace_id -> policy id (or _UNTAGGED); retained after trace end,
            # oldest evicted beyond _MAX_TRACES.
            self._by_trace: OrderedDict[str, Any] = OrderedDict()
            # span_id -> identity pinned at on_span_start; popped at on_span_end.
            self._by_span: dict[str, Any] = {}
            self._upload: TracingProcessor | None = None

        # -- resolution ---------------------------------------------------

        def _upload_processor(self) -> TracingProcessor:
            """Return the SDK's default provider exporter (created lazily)."""
            if self._upload is None:
                from agents.tracing.processors import default_processor

                self._upload = default_processor()
            return self._upload

        def _targets(self, identity: Any) -> list[TracingProcessor]:
            """Return the processors receiving events for *identity*.

            *identity* is a policy id, or the untagged marker. A policy id that
            is no longer registered yields no targets (fail closed).
            """
            if identity is self._UNTAGGED:
                return [self._upload_processor()]
            with _registry_lock:
                policy = _registry.get(identity)
            if policy is None:
                logger.debug(
                    "openai_agents tracing router: dropping event for "
                    "unregistered policy %s",
                    identity,
                )
                return []
            targets: list[TracingProcessor] = []
            if policy.upload:
                targets.append(self._upload_processor())
            if policy.mirror is not None:
                targets.append(policy.mirror)
            return targets

        def _trace_identity(self, trace: Trace) -> Any:
            """Identity for a trace: metadata tag, else active scope, else untagged."""
            policy_id = _policy_id_from_trace(trace) or _active_policy_id.get()
            return self._UNTAGGED if policy_id is None else policy_id

        def _span_identity(self, span: Span[Any]) -> Any:
            """Identity for a span: active scope, else its trace, else untagged."""
            policy_id = _active_policy_id.get()
            if policy_id is not None:
                return policy_id
            trace_id = getattr(span, "trace_id", None)
            if not isinstance(trace_id, str):
                return self._UNTAGGED
            with self._lock:
                return self._by_trace.get(trace_id, self._UNTAGGED)

        def _remember(self, trace_id: str, identity: Any) -> None:
            with self._lock:
                self._by_trace[trace_id] = identity
                self._by_trace.move_to_end(trace_id)
                while len(self._by_trace) > self._MAX_TRACES:
                    self._by_trace.popitem(last=False)

        @staticmethod
        def _forward(
            targets: list[TracingProcessor], method: str, payload: Any
        ) -> None:
            """Call *method* on each target, isolating individual failures."""
            for target in targets:
                try:
                    getattr(target, method)(payload)
                except Exception:  # noqa: BLE001 - one processor must not block others
                    logger.debug(
                        "openai_agents tracing router: %s failed in %s",
                        method,
                        type(target).__name__,
                        exc_info=True,
                    )

        # -- TracingProcessor ---------------------------------------------

        def on_trace_start(self, trace: Trace) -> None:
            identity = self._trace_identity(trace)
            self._remember(trace.trace_id, identity)
            self._forward(self._targets(identity), "on_trace_start", trace)

        def on_trace_end(self, trace: Trace) -> None:
            with self._lock:
                identity = self._by_trace.get(trace.trace_id, self._UNTAGGED)
            self._forward(self._targets(identity), "on_trace_end", trace)

        def on_span_start(self, span: Span[Any]) -> None:
            identity = self._span_identity(span)
            span_id = getattr(span, "span_id", None)
            if isinstance(span_id, str):
                with self._lock:
                    self._by_span[span_id] = identity
            self._forward(self._targets(identity), "on_span_start", span)

        def on_span_end(self, span: Span[Any]) -> None:
            span_id = getattr(span, "span_id", None)
            with self._lock:
                identity = self._by_span.pop(span_id, None) if span_id else None
            if identity is None:
                identity = self._span_identity(span)
            self._forward(self._targets(identity), "on_span_end", span)

        def _all_processors(self) -> list[TracingProcessor]:
            """Every processor the router has handed events to."""
            with _registry_lock:
                mirrors = [p.mirror for p in _registry.values() if p.mirror]
            processors: list[TracingProcessor] = list(mirrors)
            if self._upload is not None:
                processors.append(self._upload)
            return processors

        def shutdown(self, timeout: float | None = None) -> None:
            for processor in self._all_processors():
                try:
                    if timeout is not None and processor is self._upload:
                        processor.shutdown(timeout=timeout)  # type: ignore[call-arg]
                    else:
                        processor.shutdown()
                except Exception:  # noqa: BLE001 - best-effort shutdown
                    logger.debug(
                        "openai_agents tracing router: shutdown failed", exc_info=True
                    )

        def force_flush(self) -> None:
            for processor in self._all_processors():
                try:
                    processor.force_flush()
                except Exception:  # noqa: BLE001 - best-effort flush
                    logger.debug(
                        "openai_agents tracing router: flush failed", exc_info=True
                    )

    return _HoloDeckTraceRouter()


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
