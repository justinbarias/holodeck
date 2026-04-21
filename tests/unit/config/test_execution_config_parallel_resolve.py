"""Tests for parallel_test_cases propagation through the resolve chain (T041)."""

import pytest

from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.config.loader import ConfigLoader
from holodeck.models.config import ExecutionConfig


@pytest.mark.unit
class TestParallelTestCasesResolveChain:
    def test_default_is_one_when_all_none(self) -> None:
        loader = ConfigLoader()
        resolved = loader.resolve_execution_config(
            cli_config=None,
            yaml_config=None,
            project_config=None,
            user_config=None,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )
        assert resolved.parallel_test_cases == 1

    def test_cli_overrides_yaml(self) -> None:
        loader = ConfigLoader()
        resolved = loader.resolve_execution_config(
            cli_config=ExecutionConfig(parallel_test_cases=8),
            yaml_config=ExecutionConfig(parallel_test_cases=2),
            project_config=None,
            user_config=None,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )
        assert resolved.parallel_test_cases == 8

    def test_yaml_used_when_cli_unset(self) -> None:
        loader = ConfigLoader()
        # cli_config=None → yaml_config wins, per resolver priority.
        resolved = loader.resolve_execution_config(
            cli_config=None,
            yaml_config=ExecutionConfig(parallel_test_cases=3),
            project_config=None,
            user_config=None,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )
        assert resolved.parallel_test_cases == 3
