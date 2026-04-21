"""Tests for ExecutionConfig.parallel_test_cases (T007)."""

import pytest
from pydantic import ValidationError

from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.models.config import ExecutionConfig


@pytest.mark.unit
class TestExecutionConfigParallel:
    def test_model_default_is_none_unset_sentinel(self) -> None:
        """Model-level default is None so the resolver can distinguish unset
        from explicitly set; the real default (1) lives in DEFAULT_EXECUTION_CONFIG."""
        cfg = ExecutionConfig()
        assert cfg.parallel_test_cases is None

    def test_rejects_less_than_one(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(parallel_test_cases=0)
        with pytest.raises(ValidationError):
            ExecutionConfig(parallel_test_cases=-3)

    def test_accepts_positive_values(self) -> None:
        cfg = ExecutionConfig(parallel_test_cases=4)
        assert cfg.parallel_test_cases == 4

    def test_default_dict_carries_one(self) -> None:
        assert DEFAULT_EXECUTION_CONFIG["parallel_test_cases"] == 1
