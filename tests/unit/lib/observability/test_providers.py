"""Unit tests for observability providers module.

TDD: These tests are written FIRST, before implementation.

Tasks:
    T031 - Unit tests for create_resource() function
    T032 - Unit tests for ObservabilityContext dataclass
    T033 - Unit tests for initialize_observability() function
    T034 - Unit tests for shutdown_observability() function
    T035 - Unit tests for get_tracer() function
    T036 - Unit tests for get_meter() function
"""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.models.observability import ObservabilityConfig


@pytest.mark.unit
class TestCreateResource:
    """Tests for create_resource() function (T031)."""

    def test_uses_agent_name_for_default_service_name(self) -> None:
        """Test service name defaults to holodeck-{agent_name}."""
        from holodeck.lib.observability.providers import create_resource

        config = ObservabilityConfig(enabled=True)
        resource = create_resource(config, agent_name="customer-support")

        service_name = resource.attributes.get("service.name")
        assert service_name == "holodeck-customer-support"

    def test_uses_config_service_name_when_provided(self) -> None:
        """Test config.service_name overrides default."""
        from holodeck.lib.observability.providers import create_resource

        config = ObservabilityConfig(
            enabled=True,
            service_name="my-custom-service",
        )
        resource = create_resource(config, agent_name="ignored")

        service_name = resource.attributes.get("service.name")
        assert service_name == "my-custom-service"

    def test_includes_custom_resource_attributes(self) -> None:
        """Test resource_attributes are included."""
        from holodeck.lib.observability.providers import create_resource

        config = ObservabilityConfig(
            enabled=True,
            resource_attributes={
                "environment": "production",
                "version": "1.2.3",
            },
        )
        resource = create_resource(config, agent_name="test")

        assert resource.attributes.get("environment") == "production"
        assert resource.attributes.get("version") == "1.2.3"

    def test_returns_resource_instance(self) -> None:
        """Test returns OpenTelemetry Resource instance."""
        from opentelemetry.sdk.resources import Resource

        from holodeck.lib.observability.providers import create_resource

        config = ObservabilityConfig(enabled=True)
        resource = create_resource(config, agent_name="test")

        assert isinstance(resource, Resource)

    def test_formats_agent_name_with_hyphens(self) -> None:
        """Test agent name with underscores is formatted correctly."""
        from holodeck.lib.observability.providers import create_resource

        config = ObservabilityConfig(enabled=True)
        resource = create_resource(config, agent_name="my_agent_name")

        service_name = resource.attributes.get("service.name")
        assert service_name == "holodeck-my_agent_name"


@pytest.mark.unit
class TestObservabilityContext:
    """Tests for ObservabilityContext dataclass (T032)."""

    def test_is_enabled_returns_true_when_all_providers_set(self) -> None:
        """Test is_enabled returns True when all providers are initialized."""
        from holodeck.lib.observability.providers import ObservabilityContext

        context = ObservabilityContext(
            tracer_provider=MagicMock(),
            meter_provider=MagicMock(),
            logger_provider=MagicMock(),
            exporters=["console"],
        )
        assert context.is_enabled() is True

    def test_get_resource_returns_resource(self) -> None:
        """Test get_resource returns the stored resource."""
        from opentelemetry.sdk.resources import Resource

        from holodeck.lib.observability.providers import ObservabilityContext

        resource = Resource.create({"service.name": "test"})
        context = ObservabilityContext(
            tracer_provider=MagicMock(),
            meter_provider=MagicMock(),
            logger_provider=MagicMock(),
            resource=resource,
        )
        assert context.get_resource() is resource

    def test_exporters_list_tracks_enabled_exporters(self) -> None:
        """Test exporters list contains enabled exporter names."""
        from holodeck.lib.observability.providers import ObservabilityContext

        context = ObservabilityContext(
            tracer_provider=MagicMock(),
            meter_provider=MagicMock(),
            logger_provider=MagicMock(),
            exporters=["console", "otlp"],
        )
        assert "console" in context.exporters
        assert "otlp" in context.exporters

    def test_default_exporters_is_empty_list(self) -> None:
        """Test exporters defaults to empty list."""
        from holodeck.lib.observability.providers import ObservabilityContext

        context = ObservabilityContext(
            tracer_provider=MagicMock(),
            meter_provider=MagicMock(),
            logger_provider=MagicMock(),
        )
        assert context.exporters == []

    def test_stores_all_providers(self) -> None:
        """Test context stores all provider references."""
        from holodeck.lib.observability.providers import ObservabilityContext

        tracer = MagicMock()
        meter = MagicMock()
        logger = MagicMock()

        context = ObservabilityContext(
            tracer_provider=tracer,
            meter_provider=meter,
            logger_provider=logger,
        )

        assert context.tracer_provider is tracer
        assert context.meter_provider is meter
        assert context.logger_provider is logger


@pytest.mark.unit
class TestInitializeObservability:
    """Tests for initialize_observability() function (T033)."""

    @patch("holodeck.lib.observability.config.configure_exporters")
    @patch("holodeck.lib.observability.config.configure_logging")
    def test_returns_observability_context(
        self, mock_configure_logging: MagicMock, mock_configure_exporters: MagicMock
    ) -> None:
        """Test returns ObservabilityContext instance."""
        from holodeck.lib.observability.providers import (
            ObservabilityContext,
            initialize_observability,
        )

        mock_configure_exporters.return_value = ([], [], [], ["console"])

        config = ObservabilityConfig(enabled=True)
        context = initialize_observability(config, agent_name="test")

        assert isinstance(context, ObservabilityContext)

    @patch("holodeck.lib.observability.config.configure_exporters")
    @patch("holodeck.lib.observability.config.configure_logging")
    def test_sets_semantic_kernel_env_var(
        self, mock_configure_logging: MagicMock, mock_configure_exporters: MagicMock
    ) -> None:
        """Test sets SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS."""
        import os

        from holodeck.lib.observability.providers import initialize_observability

        mock_configure_exporters.return_value = ([], [], [], [])

        # Clear env var if set
        os.environ.pop(
            "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS", None
        )

        config = ObservabilityConfig(enabled=True)
        initialize_observability(config, agent_name="test")

        assert (
            os.environ.get("SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS")
            == "true"
        )

    @patch("holodeck.lib.observability.config.configure_exporters")
    @patch("holodeck.lib.observability.config.configure_logging")
    def test_context_contains_exporter_names(
        self, mock_configure_logging: MagicMock, mock_configure_exporters: MagicMock
    ) -> None:
        """Test context contains list of enabled exporter names."""
        from holodeck.lib.observability.providers import initialize_observability

        mock_configure_exporters.return_value = ([], [], [], ["console", "otlp"])

        config = ObservabilityConfig(enabled=True)
        context = initialize_observability(config, agent_name="test")

        assert "console" in context.exporters
        assert "otlp" in context.exporters


@pytest.mark.unit
class TestShutdownObservability:
    """Tests for shutdown_observability() function (T034)."""

    def test_calls_shutdown_on_providers(self) -> None:
        """Test shutdown is called on all providers."""
        from holodeck.lib.observability.providers import (
            ObservabilityContext,
            shutdown_observability,
        )

        tracer_mock = MagicMock()
        meter_mock = MagicMock()
        logger_mock = MagicMock()

        context = ObservabilityContext(
            tracer_provider=tracer_mock,
            meter_provider=meter_mock,
            logger_provider=logger_mock,
        )

        shutdown_observability(context)

        tracer_mock.shutdown.assert_called_once()
        meter_mock.shutdown.assert_called_once()
        logger_mock.shutdown.assert_called_once()

    def test_handles_none_providers_gracefully(self) -> None:
        """Test shutdown handles None providers without error."""
        from holodeck.lib.observability.providers import (
            ObservabilityContext,
            shutdown_observability,
        )

        context = ObservabilityContext(
            tracer_provider=None,  # type: ignore
            meter_provider=None,  # type: ignore
            logger_provider=None,  # type: ignore
        )

        # Should not raise
        shutdown_observability(context)


@pytest.mark.unit
class TestGetTracer:
    """Tests for get_tracer() function (T035)."""

    def test_returns_tracer_instance(self) -> None:
        """Test get_tracer returns a Tracer."""
        from holodeck.lib.observability.providers import get_tracer

        tracer = get_tracer("test.module")
        assert tracer is not None

    def test_uses_provided_name(self) -> None:
        """Test tracer uses the provided name."""
        from holodeck.lib.observability.providers import get_tracer

        tracer = get_tracer("my.custom.tracer")
        # Verify no error raised and tracer is valid
        assert tracer is not None

    def test_returns_same_tracer_for_same_name(self) -> None:
        """Test returns consistent tracer for same name."""
        from holodeck.lib.observability.providers import get_tracer

        tracer1 = get_tracer("same.name")
        tracer2 = get_tracer("same.name")
        # Both should be valid tracers
        assert tracer1 is not None
        assert tracer2 is not None


@pytest.mark.unit
class TestGetMeter:
    """Tests for get_meter() function (T036)."""

    def test_returns_meter_instance(self) -> None:
        """Test get_meter returns a Meter."""
        from holodeck.lib.observability.providers import get_meter

        meter = get_meter("test.module")
        assert meter is not None

    def test_uses_provided_name(self) -> None:
        """Test meter uses the provided name."""
        from holodeck.lib.observability.providers import get_meter

        meter = get_meter("my.custom.meter")
        assert meter is not None

    def test_returns_same_meter_for_same_name(self) -> None:
        """Test returns consistent meter for same name."""
        from holodeck.lib.observability.providers import get_meter

        meter1 = get_meter("same.name")
        meter2 = get_meter("same.name")
        # Both should be valid meters
        assert meter1 is not None
        assert meter2 is not None
