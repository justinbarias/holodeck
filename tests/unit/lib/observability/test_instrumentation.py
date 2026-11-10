"""Tests for third-party GenAI telemetry instrumentation."""

from __future__ import annotations

import pytest

from holodeck.models.observability import ObservabilityConfig, TracingConfig


@pytest.mark.unit
class TestEnableLitellmTelemetry:
    """Tests for enable_litellm_telemetry function."""

    @pytest.fixture(autouse=True)
    def clean_litellm_callbacks(self) -> None:
        """Reset litellm.callbacks before and after each test."""
        import litellm

        saved = list(litellm.callbacks)
        litellm.callbacks.clear()
        yield
        litellm.callbacks.clear()
        litellm.callbacks.extend(saved)

    def test_registers_otel_callback(self) -> None:
        """An OpenTelemetry callback is appended to litellm.callbacks."""
        import litellm
        from litellm.integrations.opentelemetry import OpenTelemetry

        from holodeck.lib.observability.instrumentation import (
            enable_litellm_telemetry,
        )

        config = ObservabilityConfig(enabled=True)
        enable_litellm_telemetry(config)

        assert any(isinstance(cb, OpenTelemetry) for cb in litellm.callbacks)

    def test_idempotent(self) -> None:
        """A second call does not register a duplicate callback."""
        import litellm
        from litellm.integrations.opentelemetry import OpenTelemetry

        from holodeck.lib.observability.instrumentation import (
            enable_litellm_telemetry,
        )

        config = ObservabilityConfig(enabled=True)
        enable_litellm_telemetry(config)
        enable_litellm_telemetry(config)

        otel_callbacks = [
            cb for cb in litellm.callbacks if isinstance(cb, OpenTelemetry)
        ]
        assert len(otel_callbacks) == 1

    def test_capture_content_off_means_no_content(self) -> None:
        """capture_content=False configures NO_CONTENT capture mode."""
        import litellm
        from litellm.integrations.opentelemetry import OpenTelemetry

        from holodeck.lib.observability.instrumentation import (
            enable_litellm_telemetry,
        )

        config = ObservabilityConfig(
            enabled=True, traces=TracingConfig(capture_content=False)
        )
        enable_litellm_telemetry(config)

        callback = next(cb for cb in litellm.callbacks if isinstance(cb, OpenTelemetry))
        assert callback.config.capture_message_content == "NO_CONTENT"

    def test_capture_content_on_means_span_only(self) -> None:
        """capture_content=True uses SPAN_ONLY so content never rides on events.

        Span attributes are scrubbed by RedactingSpanProcessor; span events
        are not, so event capture must stay off.
        """
        import litellm
        from litellm.integrations.opentelemetry import OpenTelemetry

        from holodeck.lib.observability.instrumentation import (
            enable_litellm_telemetry,
        )

        config = ObservabilityConfig(
            enabled=True, traces=TracingConfig(capture_content=True)
        )
        enable_litellm_telemetry(config)

        callback = next(cb for cb in litellm.callbacks if isinstance(cb, OpenTelemetry))
        assert callback.config.capture_message_content == "SPAN_ONLY"
