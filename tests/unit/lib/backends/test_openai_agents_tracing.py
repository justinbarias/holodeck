"""Unit tests for holodeck.lib.backends.openai_agents_tracing.

The OpenAI Agents SDK trace ``TracingProcessor`` mirror is exercised with fake
SDK span objects (no SDK run is driven). OTel spans are captured with an
in-memory ``TracerProvider`` so the mirror's attribute mapping — and the
existing ``RedactingSpanProcessor`` scrubbing of credential-shaped tool output —
are asserted end-to-end.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from holodeck.lib.backends.openai_agents_tracing import build_tracing_mirror

# ---------------------------------------------------------------------------
# Fakes mirroring the SDK Span / SpanData shapes
# ---------------------------------------------------------------------------


def _fake_span(
    span_data: Any,
    *,
    trace_id: str = "trace_abc",
    span_id: str = "span_1",
    parent_id: str | None = None,
    started_at: str | None = "2026-06-13T01:00:00.000000+00:00",
    ended_at: str | None = "2026-06-13T01:00:01.000000+00:00",
    error: dict[str, Any] | None = None,
) -> Any:
    """Build a duck-typed stand-in for an SDK ``Span``."""
    return SimpleNamespace(
        span_data=span_data,
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id,
        started_at=started_at,
        ended_at=ended_at,
        error=error,
    )


class _FunctionSpanData:
    """Stand-in for ``agents.tracing.span_data.FunctionSpanData``."""

    def __init__(self, name: str, input: str | None, output: Any | None) -> None:
        self.name = name
        self.input = input
        self.output = output
        self.mcp_data: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        return "function"


class _GenerationSpanData:
    """Stand-in for ``agents.tracing.span_data.GenerationSpanData``."""

    def __init__(self, model: str | None, usage: dict[str, Any] | None) -> None:
        self.model = model
        self.usage = usage
        self.input = None
        self.output = None
        self.model_config: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        return "generation"


class _AgentSpanData:
    """Stand-in for ``agents.tracing.span_data.AgentSpanData``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.handoffs: list[str] | None = None
        self.tools: list[str] | None = None
        self.output_type: str | None = None

    @property
    def type(self) -> str:
        return "agent"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> InMemorySpanExporter:
    """Install an isolated TracerProvider with redaction + in-memory export.

    The mirror resolves its tracer via ``holodeck...get_tracer`` (which reads the
    global OTel provider). Point that provider at an in-memory exporter behind
    the real ``RedactingSpanProcessor`` so scrubbing is exercised on export.
    """
    from holodeck.lib.backends.otel_redaction import RedactingSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(RedactingSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # The mirror calls get_tracer(name) -> trace.get_tracer(name); patch the
    # module-global provider lookup used by that helper.
    monkeypatch.setattr(
        "holodeck.lib.backends.openai_agents_tracing.get_tracer",
        lambda name: provider.get_tracer(name),
    )
    return exporter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTracingMirror:
    """The mirror creates OTel spans from SDK span-end events."""

    def test_function_span_maps_to_otel_span(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        mirror = build_tracing_mirror("test-agent")
        span = _fake_span(
            _FunctionSpanData(
                name="search_docs", input='{"q": "x"}', output="found 3 rows"
            )
        )
        mirror.on_span_start(span)
        mirror.on_span_end(span)
        mirror.force_flush()

        spans = in_memory_tracing.get_finished_spans()
        assert len(spans) == 1
        otel = spans[0]
        assert otel.name == "openai_agents.function search_docs"
        attrs = otel.attributes or {}
        assert attrs["gen_ai.openai.span_type"] == "function"
        assert attrs["gen_ai.tool.name"] == "search_docs"
        assert attrs["tool.input"] == '{"q": "x"}'
        assert attrs["tool.output"] == "found 3 rows"

    def test_generation_span_records_model_and_usage(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        mirror = build_tracing_mirror("test-agent")
        span = _fake_span(
            _GenerationSpanData(
                model="gpt-4o-mini",
                usage={"input_tokens": 12, "output_tokens": 5},
            ),
            span_id="span_gen",
        )
        mirror.on_span_end(span)
        mirror.force_flush()

        otel = in_memory_tracing.get_finished_spans()[0]
        attrs = otel.attributes or {}
        assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
        assert attrs["gen_ai.usage.input_tokens"] == 12
        assert attrs["gen_ai.usage.output_tokens"] == 5

    def test_credential_tool_output_is_redacted(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        """A credential-shaped tool.output is scrubbed before export (FR-088)."""
        mirror = build_tracing_mirror("test-agent")
        key = "sk-ant-api03-" + "A" * 95
        leaked = f"token {key} done"
        span = _fake_span(
            _FunctionSpanData(name="leaky", input=None, output=leaked),
            span_id="span_leak",
        )
        mirror.on_span_end(span)
        mirror.force_flush()

        otel = in_memory_tracing.get_finished_spans()[0]
        output_attr = (otel.attributes or {})["tool.output"]
        assert key not in output_attr
        assert "REDACTED" in output_attr

    def test_hierarchy_ids_recorded_as_attributes(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        """SDK trace/span/parent ids are attached as attributes (best-effort)."""
        mirror = build_tracing_mirror("test-agent")
        span = _fake_span(
            _AgentSpanData(name="root"),
            trace_id="trace_xyz",
            span_id="span_root",
            parent_id="span_parent",
        )
        mirror.on_span_end(span)
        mirror.force_flush()

        attrs = in_memory_tracing.get_finished_spans()[0].attributes or {}
        assert attrs["gen_ai.openai.trace_id"] == "trace_xyz"
        assert attrs["gen_ai.openai.span_id"] == "span_root"
        assert attrs["gen_ai.openai.parent_id"] == "span_parent"

    def test_uses_sdk_timestamps(self, in_memory_tracing: InMemorySpanExporter) -> None:
        """Explicit SDK start/end ISO timestamps drive the OTel span window."""
        mirror = build_tracing_mirror("test-agent")
        span = _fake_span(
            _AgentSpanData(name="root"),
            started_at="2026-06-13T01:00:00.000000+00:00",
            ended_at="2026-06-13T01:00:02.500000+00:00",
        )
        mirror.on_span_end(span)
        mirror.force_flush()

        otel = in_memory_tracing.get_finished_spans()[0]
        # 2.5 seconds in nanoseconds.
        assert otel.end_time - otel.start_time == 2_500_000_000

    def test_on_span_end_never_raises_on_malformed_span(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        """A malformed span must not break the agent run (errors swallowed)."""
        mirror = build_tracing_mirror("test-agent")
        broken = SimpleNamespace()  # missing every expected attribute
        # Must not raise.
        mirror.on_span_end(broken)
        mirror.force_flush()

    def test_trace_lifecycle_methods_are_noops(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        """Trace start/end + shutdown must not emit spans or raise."""
        mirror = build_tracing_mirror("test-agent")
        fake_trace = SimpleNamespace(trace_id="t", name="wf")
        mirror.on_trace_start(fake_trace)
        mirror.on_trace_end(fake_trace)
        mirror.shutdown()
        assert in_memory_tracing.get_finished_spans() == ()


# ---------------------------------------------------------------------------
# T1 / D13 — process-global trace router + per-backend policies
# ---------------------------------------------------------------------------

from holodeck.lib.backends import openai_agents_tracing as tracing_module  # noqa: E402
from holodeck.lib.backends.openai_agents_tracing import (  # noqa: E402
    TRACE_POLICY_METADATA_KEY,
    TracingPolicy,
    active_tracing_policy,
    register_tracing_policy,
    unregister_tracing_policy,
)


def _trace(trace_id: str, policy_id: str | None) -> Any:
    """Duck-typed SDK ``Trace`` carrying HoloDeck's policy tag in metadata."""
    metadata = {TRACE_POLICY_METADATA_KEY: policy_id} if policy_id else {"x": "y"}
    return SimpleNamespace(trace_id=trace_id, name="wf", metadata=metadata)


def _span(trace_id: str, span_id: str = "span_1") -> Any:
    """Duck-typed SDK ``Span`` belonging to *trace_id*."""
    return SimpleNamespace(trace_id=trace_id, span_id=span_id, span_data=None)


@pytest.fixture
def router_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Isolate the module-global router and stub the SDK's install + exporter."""
    from unittest.mock import MagicMock

    tracing_module._reset_tracing_router()
    upload = MagicMock(name="upload_processor")
    set_procs = MagicMock(name="set_trace_processors")
    monkeypatch.setattr("agents.set_trace_processors", set_procs)
    monkeypatch.setattr("agents.tracing.processors.default_processor", lambda: upload)
    yield {"upload": upload, "set_procs": set_procs}
    tracing_module._reset_tracing_router()


def _mirror() -> Any:
    from unittest.mock import MagicMock

    return MagicMock(name="mirror")


def _drive(router: Any, trace: Any, *spans: Any) -> None:
    """Emit a full trace lifecycle (start, spans, end) through *router*."""
    router.on_trace_start(trace)
    for span in spans:
        router.on_span_start(span)
        router.on_span_end(span)
    router.on_trace_end(trace)


@pytest.mark.unit
class TestTraceRouterInstall:
    """Exactly one router owns the SDK's process-global processor list."""

    def test_first_registration_installs_router_once(
        self, router_env: dict[str, Any]
    ) -> None:
        register_tracing_policy("a", TracingPolicy(upload=True))
        register_tracing_policy("b", TracingPolicy(upload=False))
        register_tracing_policy("a", TracingPolicy(upload=False))
        router_env["set_procs"].assert_called_once()
        (installed,) = router_env["set_procs"].call_args.args[0]
        assert installed is tracing_module._router

    def test_register_replaces_policy_for_same_id(
        self, router_env: dict[str, Any]
    ) -> None:
        mirror = _mirror()
        register_tracing_policy("a", TracingPolicy(upload=True))
        register_tracing_policy("a", TracingPolicy(upload=False, mirror=mirror))
        router = tracing_module._router
        _drive(router, _trace("t1", "a"), _span("t1"))
        router_env["upload"].on_trace_start.assert_not_called()
        mirror.on_span_end.assert_called_once()

    def test_unregister_unknown_id_is_noop(self, router_env: dict[str, Any]) -> None:
        unregister_tracing_policy("missing")
        assert tracing_module._router is None


@pytest.mark.unit
class TestTraceRouterRouting:
    """Per-trace routing by policy (FR-100 / FR-101 / FR-102)."""

    def test_openai_policy_uploads_and_mirrors(
        self, router_env: dict[str, Any]
    ) -> None:
        mirror = _mirror()
        register_tracing_policy("oa", TracingPolicy(upload=True, mirror=mirror))
        router = tracing_module._router
        trace, span = _trace("t1", "oa"), _span("t1")
        _drive(router, trace, span)

        upload = router_env["upload"]
        upload.on_trace_start.assert_called_once_with(trace)
        upload.on_span_start.assert_called_once_with(span)
        upload.on_span_end.assert_called_once_with(span)
        upload.on_trace_end.assert_called_once_with(trace)
        mirror.on_span_end.assert_called_once_with(span)
        mirror.on_trace_end.assert_called_once_with(trace)

    def test_azure_policy_mirrors_without_upload(
        self, router_env: dict[str, Any]
    ) -> None:
        mirror = _mirror()
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=mirror))
        router = tracing_module._router
        span = _span("t1")
        _drive(router, _trace("t1", "az"), span)

        upload = router_env["upload"]
        assert upload.method_calls == []
        mirror.on_span_end.assert_called_once_with(span)

    def test_provider_override_suppresses_upload_keeps_mirror(
        self, router_env: dict[str, Any]
    ) -> None:
        mirror = _mirror()
        register_tracing_policy("ov", TracingPolicy(upload=False, mirror=mirror))
        _drive(tracing_module._router, _trace("t1", "ov"), _span("t1"))
        assert router_env["upload"].method_calls == []
        assert mirror.on_span_end.call_count == 1

    def test_no_upload_no_mirror_drops_everything(
        self, router_env: dict[str, Any]
    ) -> None:
        # Azure with observability disabled: nothing leaves the process.
        register_tracing_policy("az-off", TracingPolicy(upload=False, mirror=None))
        _drive(tracing_module._router, _trace("t1", "az-off"), _span("t1"))
        assert router_env["upload"].method_calls == []

    @pytest.mark.parametrize("first", ["openai", "azure"])
    def test_mixed_providers_route_per_trace_in_either_order(
        self, router_env: dict[str, Any], first: str
    ) -> None:
        oa_mirror, az_mirror = _mirror(), _mirror()
        policies = {
            "openai": ("oa", TracingPolicy(upload=True, mirror=oa_mirror)),
            "azure": ("az", TracingPolicy(upload=False, mirror=az_mirror)),
        }
        second = "azure" if first == "openai" else "openai"
        for key in (first, second):
            register_tracing_policy(*policies[key])
        router = tracing_module._router

        oa_span, az_span = _span("t-oa", "s-oa"), _span("t-az", "s-az")
        _drive(router, _trace("t-oa", "oa"), oa_span)
        _drive(router, _trace("t-az", "az"), az_span)

        upload = router_env["upload"]
        uploaded_spans = [c.args[0] for c in upload.on_span_end.call_args_list]
        assert uploaded_spans == [oa_span]
        assert [c.args[0] for c in oa_mirror.on_span_end.call_args_list] == [oa_span]
        assert [c.args[0] for c in az_mirror.on_span_end.call_args_list] == [az_span]

    def test_untagged_trace_keeps_sdk_default_upload(
        self, router_env: dict[str, Any]
    ) -> None:
        mirror = _mirror()
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=mirror))
        span = _span("ext")
        _drive(tracing_module._router, _trace("ext", None), span)
        router_env["upload"].on_span_end.assert_called_once_with(span)
        mirror.on_span_end.assert_not_called()

    def test_tagged_but_unregistered_trace_is_dropped(
        self, router_env: dict[str, Any]
    ) -> None:
        register_tracing_policy("az", TracingPolicy(upload=False))
        unregister_tracing_policy("az")
        _drive(tracing_module._router, _trace("t1", "az"), _span("t1"))
        assert router_env["upload"].method_calls == []

    def test_untagged_trace_inside_active_scope_uses_scope_policy(
        self, router_env: dict[str, Any]
    ) -> None:
        mirror = _mirror()
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=mirror))
        span = _span("ext")
        with active_tracing_policy("az"):
            _drive(tracing_module._router, _trace("ext", None), span)
        assert router_env["upload"].method_calls == []
        mirror.on_span_end.assert_called_once_with(span)

    def test_span_of_untracked_trace_inside_scope_uses_scope_policy(
        self, router_env: dict[str, Any]
    ) -> None:
        mirror = _mirror()
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=mirror))
        router = tracing_module._router
        with active_tracing_policy("az"):
            router.on_span_end(_span("never-started"))
        assert router_env["upload"].method_calls == []
        assert mirror.on_span_end.call_count == 1

    def test_span_under_outer_trace_started_before_scope_follows_scope(
        self, router_env: dict[str, Any]
    ) -> None:
        # Codex finding: caller-owned trace() opened before the backend's run.
        mirror = _mirror()
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=mirror))
        router = tracing_module._router
        outer, span = _trace("outer", None), _span("outer")
        router.on_trace_start(outer)
        with active_tracing_policy("az"):
            router.on_span_start(span)
            router.on_span_end(span)
        router.on_trace_end(outer)

        upload = router_env["upload"]
        upload.on_span_end.assert_not_called()
        mirror.on_span_end.assert_called_once_with(span)
        # The caller's own trace record keeps SDK default behaviour.
        upload.on_trace_start.assert_called_once_with(outer)

    def test_openai_run_inside_azure_tagged_outer_trace_still_uploads(
        self, router_env: dict[str, Any]
    ) -> None:
        oa_mirror, az_mirror = _mirror(), _mirror()
        register_tracing_policy("oa", TracingPolicy(upload=True, mirror=oa_mirror))
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=az_mirror))
        router = tracing_module._router
        outer, span = _trace("outer", "az"), _span("outer")
        router.on_trace_start(outer)
        with active_tracing_policy("oa"):
            router.on_span_end(span)
        router.on_trace_end(outer)

        router_env["upload"].on_span_end.assert_called_once_with(span)
        oa_mirror.on_span_end.assert_called_once_with(span)
        az_mirror.on_span_end.assert_not_called()

    def test_late_span_after_trace_end_keeps_trace_policy(
        self, router_env: dict[str, Any]
    ) -> None:
        # Codex finding: span finishing after its trace, outside any scope.
        mirror = _mirror()
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=mirror))
        router = tracing_module._router
        trace, span = _trace("t1", "az"), _span("t1")
        router.on_trace_start(trace)
        router.on_span_start(span)
        router.on_trace_end(trace)
        router.on_span_end(span)

        assert router_env["upload"].method_calls == []
        mirror.on_span_end.assert_called_once_with(span)

    def test_unregister_drops_in_flight_trace_events_immediately(
        self, router_env: dict[str, Any]
    ) -> None:
        register_tracing_policy("oa", TracingPolicy(upload=True))
        router = tracing_module._router
        trace = _trace("t1", "oa")
        router.on_trace_start(trace)
        router.on_span_end(_span("t1", "s1"))
        unregister_tracing_policy("oa")
        router.on_span_end(_span("t1", "s2"))
        router.on_trace_end(trace)

        upload = router_env["upload"]
        assert [c.args[0].span_id for c in upload.on_span_end.call_args_list] == ["s1"]
        upload.on_trace_end.assert_not_called()

    def test_span_started_in_scope_finished_outside_keeps_policy(
        self, router_env: dict[str, Any]
    ) -> None:
        # Codex finding: identity must be pinned at span start, not re-resolved.
        mirror = _mirror()
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=mirror))
        router = tracing_module._router
        outer, span = _trace("outer", None), _span("outer")
        router.on_trace_start(outer)
        with active_tracing_policy("az"):
            router.on_span_start(span)
        router.on_span_end(span)
        router.on_trace_end(outer)

        router_env["upload"].on_span_end.assert_not_called()
        mirror.on_span_end.assert_called_once_with(span)
        assert router._by_span == {}

    def test_span_survives_trace_map_eviction(self, router_env: dict[str, Any]) -> None:
        # Codex finding: eviction of the trace identity must not fail open.
        mirror = _mirror()
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=mirror))
        router = tracing_module._router
        trace, span = _trace("t-late", "az"), _span("t-late")
        router.on_trace_start(trace)
        router.on_span_start(span)
        router.on_trace_end(trace)
        for index in range(router._MAX_TRACES + 10):
            _drive(router, _trace(f"churn{index}", "az"))
        assert "t-late" not in router._by_trace
        router.on_span_end(span)

        router_env["upload"].on_span_end.assert_not_called()
        mirror.on_span_end.assert_called_once_with(span)

    def test_late_span_on_evicted_restricted_trace_is_dropped(
        self, router_env: dict[str, Any]
    ) -> None:
        # Codex finding (stack review): a span that *starts* after its trace
        # identity was evicted must not fall back to the SDK default upload.
        mirror = _mirror()
        register_tracing_policy("az", TracingPolicy(upload=False, mirror=mirror))
        router = tracing_module._router
        trace = _trace("t-evicted", "az")
        router.on_trace_start(trace)
        for index in range(router._MAX_TRACES + 10):
            _drive(router, _trace(f"churn{index}", "az"))
        assert "t-evicted" not in router._by_trace
        late = _span("t-evicted", "late")
        router.on_span_start(late)
        router.on_span_end(late)
        router.on_trace_end(trace)

        assert router_env["upload"].method_calls == []
        assert late not in [c.args[0] for c in mirror.on_span_end.call_args_list]

    def test_span_of_never_seen_trace_is_dropped(
        self, router_env: dict[str, Any]
    ) -> None:
        register_tracing_policy("az", TracingPolicy(upload=False))
        span = _span("never-started")
        tracing_module._router.on_span_start(span)
        tracing_module._router.on_span_end(span)
        assert router_env["upload"].method_calls == []

    def test_trace_map_is_bounded(self, router_env: dict[str, Any]) -> None:
        register_tracing_policy("oa", TracingPolicy(upload=True))
        router = tracing_module._router
        for index in range(router._MAX_TRACES + 10):
            _drive(router, _trace(f"t{index}", "oa"))
        assert len(router._by_trace) == router._MAX_TRACES
        assert "t0" not in router._by_trace

    def test_failing_mirror_does_not_block_upload(
        self, router_env: dict[str, Any]
    ) -> None:
        mirror = _mirror()
        mirror.on_span_end.side_effect = RuntimeError("boom")
        register_tracing_policy("oa", TracingPolicy(upload=True, mirror=mirror))
        span = _span("t1")
        _drive(tracing_module._router, _trace("t1", "oa"), span)
        router_env["upload"].on_span_end.assert_called_once_with(span)


@pytest.mark.unit
class TestTraceRouterLifecycle:
    """shutdown / force_flush reach every processor the router feeds."""

    def test_flush_and_shutdown_forward_to_mirrors_and_upload(
        self, router_env: dict[str, Any]
    ) -> None:
        mirror = _mirror()
        register_tracing_policy("oa", TracingPolicy(upload=True, mirror=mirror))
        router = tracing_module._router
        _drive(router, _trace("t1", "oa"), _span("t1"))  # materializes upload
        router.force_flush()
        router.shutdown(timeout=1.0)
        router_env["upload"].force_flush.assert_called_once()
        router_env["upload"].shutdown.assert_called_once_with(timeout=1.0)
        mirror.force_flush.assert_called_once()
        mirror.shutdown.assert_called_once_with()

    def test_shutdown_skips_upload_when_never_used(
        self, router_env: dict[str, Any]
    ) -> None:
        register_tracing_policy("az", TracingPolicy(upload=False))
        tracing_module._router.shutdown()
        router_env["upload"].shutdown.assert_not_called()

    def test_import_stays_sdk_free(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys; import holodeck.lib.backends.openai_agents_tracing; "
            "sys.exit(1 if 'agents' in sys.modules else 0)"
        )
        assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
