"""Tests for Semantic Kernel telemetry instrumentation.

Task: Tests for T061 - enable_semantic_kernel_telemetry()
"""

from __future__ import annotations

import os

import pytest

from holodeck.lib.observability.instrumentation import (
    SK_OTEL_DIAGNOSTICS_ENV,
    SK_OTEL_SENSITIVE_ENV,
    enable_semantic_kernel_telemetry,
)
from holodeck.models.observability import ObservabilityConfig, TracingConfig


@pytest.mark.unit
class TestEnableSemanticKernelTelemetry:
    """Tests for enable_semantic_kernel_telemetry function."""

    @pytest.fixture(autouse=True)
    def clean_env(self) -> None:
        """Clean up environment variables before and after each test."""
        # Remove any existing SK telemetry env vars
        env_vars_to_clean = [SK_OTEL_DIAGNOSTICS_ENV, SK_OTEL_SENSITIVE_ENV]
        for var in env_vars_to_clean:
            os.environ.pop(var, None)
        yield
        # Clean up after test
        for var in env_vars_to_clean:
            os.environ.pop(var, None)

    def test_enables_basic_diagnostics(self) -> None:
        """Test that basic SK diagnostics env var is set."""
        config = ObservabilityConfig(enabled=True)

        enable_semantic_kernel_telemetry(config)

        assert os.environ.get(SK_OTEL_DIAGNOSTICS_ENV) == "true"

    def test_sensitive_diagnostics_disabled_by_default(self) -> None:
        """Test that sensitive diagnostics is NOT set when capture_content=False."""
        config = ObservabilityConfig(
            enabled=True,
            traces=TracingConfig(capture_content=False),
        )

        enable_semantic_kernel_telemetry(config)

        assert os.environ.get(SK_OTEL_SENSITIVE_ENV) is None

    def test_enables_sensitive_diagnostics_when_capture_content_true(self) -> None:
        """Test that sensitive diagnostics IS set when capture_content=True."""
        config = ObservabilityConfig(
            enabled=True,
            traces=TracingConfig(capture_content=True),
        )

        enable_semantic_kernel_telemetry(config)

        assert os.environ.get(SK_OTEL_SENSITIVE_ENV) == "true"

    def test_both_env_vars_set_when_capture_content_enabled(self) -> None:
        """Test that both basic and sensitive diagnostics are enabled together."""
        config = ObservabilityConfig(
            enabled=True,
            traces=TracingConfig(capture_content=True),
        )

        enable_semantic_kernel_telemetry(config)

        assert os.environ.get(SK_OTEL_DIAGNOSTICS_ENV) == "true"
        assert os.environ.get(SK_OTEL_SENSITIVE_ENV) == "true"

    def test_uses_correct_env_var_names(self) -> None:
        """Test that the correct Semantic Kernel env var names are used."""
        # Verify the constant values match SK's expected env vars
        assert SK_OTEL_DIAGNOSTICS_ENV == (
            "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS"
        )
        assert SK_OTEL_SENSITIVE_ENV == (
            "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE"
        )

    def test_can_be_called_multiple_times(self) -> None:
        """Test that calling multiple times doesn't cause issues."""
        config = ObservabilityConfig(enabled=True)

        # Call multiple times
        enable_semantic_kernel_telemetry(config)
        enable_semantic_kernel_telemetry(config)

        # Should still work correctly
        assert os.environ.get(SK_OTEL_DIAGNOSTICS_ENV) == "true"


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
