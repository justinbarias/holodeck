"""Unit tests for ``CodeMetric`` load-time grader resolution (US4 Phase 4).

Covers T021–T027 in ``specs/032-multi-turn-test-cases/tasks-us4.md``.

``CodeMetric`` resolves ``grader: module:callable`` at config-load time via
``importlib.import_module`` + ``getattr``; failures surface as ``ConfigError``
per ``contracts/code-grader-contract.md`` §2. The callable is cached on the
instance so each turn's grader invocation doesn't re-import.
"""

from __future__ import annotations

import importlib
from unittest import mock

import pytest
from pydantic import ValidationError

from holodeck.lib.errors import ConfigError
from holodeck.models.evaluation import CodeMetric, EvaluationConfig


@pytest.mark.unit
class TestGraderPathFormat:
    def test_valid_path_parses(self) -> None:
        m = CodeMetric(grader="my_benchmarks:numeric_equal")
        assert m.grader == "my_benchmarks:numeric_equal"

    def test_missing_colon_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodeMetric(grader="bad-path")

    def test_double_colon_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodeMetric(grader="mod::fn")


@pytest.mark.unit
class TestGraderLoadTimeResolution:
    def test_load_time_import_failure_is_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc:
            CodeMetric(grader="nonexistent_module_xyz:fn")
        # ConfigError message carries the grader path and underlying error.
        msg = str(exc.value)
        assert "nonexistent_module_xyz" in msg
        assert "ModuleNotFoundError" in msg or "No module named" in msg

    def test_load_time_attribute_error(self) -> None:
        with pytest.raises(ConfigError) as exc:
            CodeMetric(grader="os:nonexistent_callable_xyz")
        msg = str(exc.value)
        assert "nonexistent_callable_xyz" in msg

    def test_non_callable_rejected(self) -> None:
        # ``os.sep`` is a module-level str constant, not callable.
        with pytest.raises(ConfigError) as exc:
            CodeMetric(grader="os:sep")
        assert "not callable" in str(exc.value)

    def test_resolved_callable_cached_on_instance(self) -> None:
        """``import_module`` is called exactly once across 10 accesses.

        We patch ``importlib.import_module`` with a wrapper so we count
        invocations rather than relying on ``sys.modules`` — per T025.
        """
        with mock.patch(
            "holodeck.models.evaluation.importlib.import_module",
            wraps=importlib.import_module,
        ) as spy:
            m = CodeMetric(grader="os:getcwd")
            for _ in range(10):
                assert m.resolved_callable is not None
            assert spy.call_count == 1

    def test_defaults(self) -> None:
        m = CodeMetric(grader="os:getcwd")
        assert m.enabled is True
        assert m.fail_on_error is False
        assert m.threshold is None
        # name defaults to callable name via ``display_name`` property.
        assert m.display_name == "getcwd"


@pytest.mark.unit
class TestMetricTypeUnionAcceptsCodeVariant:
    def test_union_parses_code_variant(self) -> None:
        config = EvaluationConfig(
            metrics=[
                {"type": "code", "grader": "os:getcwd"},
            ]
        )
        assert len(config.metrics) == 1
        assert isinstance(config.metrics[0], CodeMetric)
