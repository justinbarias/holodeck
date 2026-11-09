"""Shared fixtures for observability tests."""

import logging
from collections.abc import Iterator

import opentelemetry._logs._internal as otel_logs
import opentelemetry.metrics._internal as otel_metrics
import opentelemetry.trace as otel_trace
import pytest
from opentelemetry.util._once import Once

from holodeck.lib.observability import providers


@pytest.fixture
def isolated_otel() -> Iterator[None]:
    """Give a test fresh OTel globals and undo what initialize_observability did.

    ``initialize_observability`` installs set-once global providers, adds an
    OTel handler to the root logger, and changes the root level. Without this
    fixture those survive the test: later tests cannot install providers, and
    the batch export thread keeps writing to pytest's closed capture stream.
    """
    root = logging.getLogger()
    holodeck_logger = logging.getLogger("holodeck")
    saved_root_handlers = root.handlers[:]
    saved_root_level = root.level
    saved_holodeck_handlers = holodeck_logger.handlers[:]
    saved_globals = [
        (module, name, getattr(module, name))
        for module, name in (
            (otel_trace, "_TRACER_PROVIDER"),
            (otel_trace, "_TRACER_PROVIDER_SET_ONCE"),
            (otel_logs, "_LOGGER_PROVIDER"),
            (otel_logs, "_LOGGER_PROVIDER_SET_ONCE"),
            (otel_metrics, "_METER_PROVIDER"),
            (otel_metrics, "_METER_PROVIDER_SET_ONCE"),
        )
    ]
    for module, name, _ in saved_globals:
        setattr(module, name, Once() if name.endswith("_SET_ONCE") else None)

    try:
        yield
    finally:
        context = providers._observability_context
        if context is not None:
            providers.shutdown_observability(context)
        root.handlers[:] = saved_root_handlers
        root.setLevel(saved_root_level)
        holodeck_logger.handlers[:] = saved_holodeck_handlers
        for module, name, value in saved_globals:
            setattr(module, name, value)
