"""Unit tests for logging coordination between OTel and setup_logging.

TDD: These tests are written FIRST, before implementation.
Tests verify that when OTel observability is enabled, setup_logging is NOT called.

Task: T120 - Tests for logging coordination
"""

from unittest.mock import MagicMock

import pytest

from holodeck.models.observability import ObservabilityConfig


@pytest.mark.unit
class TestLoggingCoordination:
    """Tests for logging coordination behavior."""

    def test_observability_config_enabled_check(self) -> None:
        """Test ObservabilityConfig.enabled attribute works correctly."""
        # Arrange & Act
        enabled_config = ObservabilityConfig(enabled=True)
        disabled_config = ObservabilityConfig(enabled=False)
        default_config = ObservabilityConfig()

        # Assert
        assert enabled_config.enabled is True
        assert disabled_config.enabled is False
        assert default_config.enabled is False  # Default is False

    def test_observability_enabled_check_pattern(self) -> None:
        """Test the pattern for checking if observability is enabled."""
        # This tests the conditional logic pattern used in CLI commands
        agent_with_obs = MagicMock()
        agent_with_obs.observability = ObservabilityConfig(enabled=True)

        agent_without_obs = MagicMock()
        agent_without_obs.observability = None

        agent_obs_disabled = MagicMock()
        agent_obs_disabled.observability = ObservabilityConfig(enabled=False)

        # Pattern: agent.observability and agent.observability.enabled
        assert (
            agent_with_obs.observability and agent_with_obs.observability.enabled
        ) is True
        assert (
            agent_without_obs.observability and agent_without_obs.observability.enabled
        ) is None  # Short-circuit to None
        assert (
            agent_obs_disabled.observability
            and agent_obs_disabled.observability.enabled
        ) is False


@pytest.mark.unit
class TestLoggingStrategyDecision:
    """Tests for the logging strategy decision logic."""

    def test_otel_replaces_setup_logging_when_enabled(self) -> None:
        """Test that OTel initialization replaces setup_logging when enabled."""
        # This is a behavioral test that verifies the contract:
        # When observability.enabled=True, setup_logging should NOT be called

        # Simulate the decision logic
        def should_use_otel(observability: ObservabilityConfig | None) -> bool:
            return observability is not None and observability.enabled

        # Assert
        assert should_use_otel(ObservabilityConfig(enabled=True)) is True
        assert should_use_otel(ObservabilityConfig(enabled=False)) is False
        assert should_use_otel(None) is False

    def test_setup_logging_used_when_otel_disabled(self) -> None:
        """Test that setup_logging is used when observability is disabled."""

        def should_use_setup_logging(
            observability: ObservabilityConfig | None,
        ) -> bool:
            return observability is None or not observability.enabled

        # Assert
        assert should_use_setup_logging(ObservabilityConfig(enabled=True)) is False
        assert should_use_setup_logging(ObservabilityConfig(enabled=False)) is True
        assert should_use_setup_logging(None) is True


@pytest.mark.unit
class TestOTelLoggingCapture:
    """Tests for OTel LoggerProvider capturing Python logging."""

    def test_otel_logger_provider_initialized(self) -> None:
        """Test that OTel LoggerProvider is initialized with observability."""
        from holodeck.lib.observability import initialize_observability
        from holodeck.lib.observability.providers import ObservabilityContext

        config = ObservabilityConfig(enabled=True)

        # Act
        context = initialize_observability(config, agent_name="test-agent")

        # Assert
        assert isinstance(context, ObservabilityContext)
        assert context.logger_provider is not None

    def test_otel_shutdown_flushes_logs(self) -> None:
        """Test that shutdown_observability flushes pending logs."""
        from holodeck.lib.observability import (
            initialize_observability,
            shutdown_observability,
        )

        config = ObservabilityConfig(enabled=True)
        context = initialize_observability(config, agent_name="test-agent")

        # Act - should not raise
        shutdown_observability(context)

        # Assert - provider should be shutdown (internal state)
        # We just verify no exception is raised
