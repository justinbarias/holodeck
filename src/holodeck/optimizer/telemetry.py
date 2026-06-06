"""Guarded OpenTelemetry instrumentation for the optimizer loop.

A single :class:`OptimizerTelemetry` instance per run owns the optimizer's
tracer (and, from Phase 3, its metric instruments). Every method is a strict
no-op when the observability context is not initialized, so
:class:`~holodeck.optimizer.loop.OptimizerLoop` behaves identically whether or
not the operator enabled observability — the disabled path never touches OTel.

Span tree (the ``holodeck.optimize`` root is opened by the CLI; see
``cli/commands/optimize.py``)::

    holodeck.optimize                 (root, CLI-owned)
    ├── holodeck.optimize.baseline
    └── holodeck.optimize.cycle
        └── …                         (phase/propose/trial spans — Phase 2 T3)
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from holodeck.lib.observability import get_observability_context, get_tracer


class OptimizerTelemetry:
    """Per-run holder for the optimizer's tracer; no-op when disabled.

    Enablement is decided once at construction from the presence of an
    initialized observability context. The CLI initializes that context before
    building the loop, so the loop's telemetry reflects the operator's choice.
    """

    def __init__(self) -> None:
        self._enabled = get_observability_context() is not None
        self._tracer = get_tracer(__name__) if self._enabled else None

    @property
    def enabled(self) -> bool:
        """True when an observability context was active at construction."""
        return self._enabled

    def baseline_span(self) -> AbstractContextManager[Any]:
        """Span around the baseline scoring of the original agent."""
        return self._span("holodeck.optimize.baseline")

    def cycle_span(self, cycle: int) -> AbstractContextManager[Any]:
        """Span around one coordinate-descent cycle."""
        return self._span("holodeck.optimize.cycle", {"holodeck.optimize.cycle": cycle})

    def _span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> AbstractContextManager[Any]:
        """Start an active span, or a no-op context when disabled."""
        if not self._enabled or self._tracer is None:
            return nullcontext()
        # Bind to a typed local: start_as_current_span is typed as Any in the
        # OTel stubs, and a bare return trips mypy's warn_return_any.
        span: AbstractContextManager[Any] = self._tracer.start_as_current_span(
            name, attributes=attributes or {}
        )
        return span
