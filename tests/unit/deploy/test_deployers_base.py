"""Unit tests for BaseDeployer interface."""

from __future__ import annotations

import inspect

import pytest

from holodeck.deploy.deployers.base import BaseDeployer


class TestBaseDeployerInterface:
    """Tests for BaseDeployer abstract interface."""

    def test_base_deployer_is_abstract(self) -> None:
        """Test BaseDeployer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDeployer()  # type: ignore[abstract]

    def test_base_deployer_abstract_methods(self) -> None:
        """Test BaseDeployer declares required abstract methods."""
        expected_methods = {"deploy", "get_status", "destroy", "stream_logs"}
        assert expected_methods.issubset(BaseDeployer.__abstractmethods__)

    def test_base_deployer_method_signatures(self) -> None:
        """Test BaseDeployer method signatures include expected parameters."""
        deploy_params = inspect.signature(BaseDeployer.deploy).parameters
        assert "service_name" in deploy_params
        assert "image_uri" in deploy_params
        assert "port" in deploy_params
        assert "env_vars" in deploy_params
        assert "health_check_path" in deploy_params

        for name in (
            "service_name",
            "image_uri",
            "port",
            "env_vars",
            "health_check_path",
        ):
            assert (
                deploy_params[name].kind == inspect.Parameter.KEYWORD_ONLY
            ), f"{name} must be keyword-only"

        assert deploy_params["health_check_path"].default == "/health"

        status_params = inspect.signature(BaseDeployer.get_status).parameters
        assert "service_id" in status_params

        destroy_params = inspect.signature(BaseDeployer.destroy).parameters
        assert "service_id" in destroy_params

        logs_params = inspect.signature(BaseDeployer.stream_logs).parameters
        assert "service_id" in logs_params

    def test_base_deployer_missing_method_validation(self) -> None:
        """Test abstract methods are required for subclasses."""

        class IncompleteDeployer(BaseDeployer):
            def deploy(  # type: ignore[override]
                self,
                *,
                service_name: str,
                image_uri: str,
                port: int,
                env_vars: dict[str, str],
                health_check_path: str = "/health",
                **kwargs: object,
            ) -> dict[str, str | None]:
                return {}

        with pytest.raises(TypeError):
            IncompleteDeployer()  # type: ignore[abstract]
