"""Unit tests for DeepEval error classes."""

import pytest

from holodeck.lib.errors import EvaluationError
from holodeck.lib.evaluators.deepeval.errors import (
    DeepEvalError,
    ProviderNotSupportedError,
)


class TestProviderNotSupportedError:
    """Tests for ProviderNotSupportedError exception."""

    def test_inherits_from_evaluation_error(self) -> None:
        """ProviderNotSupportedError should inherit from EvaluationError."""
        error = ProviderNotSupportedError(
            message="Test error",
            evaluator_type="TestEvaluator",
            configured_provider="openai",
            supported_providers=["azure_openai"],
        )
        assert isinstance(error, EvaluationError)

    def test_stores_evaluator_type(self) -> None:
        """Should store evaluator_type attribute."""
        error = ProviderNotSupportedError(
            message="Test error",
            evaluator_type="AzureAIEvaluator",
            configured_provider="openai",
            supported_providers=["azure_openai"],
        )
        assert error.evaluator_type == "AzureAIEvaluator"

    def test_stores_configured_provider(self) -> None:
        """Should store configured_provider attribute."""
        error = ProviderNotSupportedError(
            message="Test error",
            evaluator_type="TestEvaluator",
            configured_provider="anthropic",
            supported_providers=["azure_openai"],
        )
        assert error.configured_provider == "anthropic"

    def test_stores_supported_providers(self) -> None:
        """Should store supported_providers attribute."""
        error = ProviderNotSupportedError(
            message="Test error",
            evaluator_type="TestEvaluator",
            configured_provider="openai",
            supported_providers=["azure_openai", "ollama"],
        )
        assert error.supported_providers == ["azure_openai", "ollama"]

    def test_message_is_accessible(self) -> None:
        """Error message should be accessible via str()."""
        error = ProviderNotSupportedError(
            message="Azure AI Evaluator requires Azure OpenAI provider",
            evaluator_type="AzureAIEvaluator",
            configured_provider="openai",
            supported_providers=["azure_openai"],
        )
        assert "Azure AI Evaluator requires Azure OpenAI provider" in str(error)

    def test_can_be_raised_and_caught(self) -> None:
        """Should be raisable and catchable."""
        with pytest.raises(ProviderNotSupportedError) as exc_info:
            raise ProviderNotSupportedError(
                message="Provider not supported",
                evaluator_type="TestEvaluator",
                configured_provider="openai",
                supported_providers=["azure_openai"],
            )

        assert exc_info.value.evaluator_type == "TestEvaluator"
        assert exc_info.value.configured_provider == "openai"


class TestDeepEvalError:
    """Tests for DeepEvalError exception."""

    def test_inherits_from_evaluation_error(self) -> None:
        """DeepEvalError should inherit from EvaluationError."""
        error = DeepEvalError(
            message="Test error",
            metric_name="GEval",
        )
        assert isinstance(error, EvaluationError)

    def test_stores_metric_name(self) -> None:
        """Should store metric_name attribute."""
        error = DeepEvalError(
            message="Test error",
            metric_name="FaithfulnessMetric",
        )
        assert error.metric_name == "FaithfulnessMetric"

    def test_stores_original_error(self) -> None:
        """Should store original_error attribute."""
        original = ValueError("Original error")
        error = DeepEvalError(
            message="Wrapped error",
            metric_name="GEval",
            original_error=original,
        )
        assert error.original_error is original
        assert isinstance(error.original_error, ValueError)

    def test_original_error_defaults_to_none(self) -> None:
        """original_error should default to None."""
        error = DeepEvalError(
            message="Test error",
            metric_name="GEval",
        )
        assert error.original_error is None

    def test_stores_test_case_summary(self) -> None:
        """Should store test_case_summary attribute."""
        summary = {
            "input": "Test query",
            "actual_output": "Test response",
        }
        error = DeepEvalError(
            message="Test error",
            metric_name="GEval",
            test_case_summary=summary,
        )
        assert error.test_case_summary == summary

    def test_test_case_summary_defaults_to_empty_dict(self) -> None:
        """test_case_summary should default to empty dict."""
        error = DeepEvalError(
            message="Test error",
            metric_name="GEval",
        )
        assert error.test_case_summary == {}

    def test_message_is_accessible(self) -> None:
        """Error message should be accessible via str()."""
        error = DeepEvalError(
            message="DeepEval metric failed: Invalid JSON",
            metric_name="GEval",
        )
        assert "DeepEval metric failed: Invalid JSON" in str(error)

    def test_can_be_raised_and_caught(self) -> None:
        """Should be raisable and catchable."""
        original = RuntimeError("LLM returned invalid response")

        with pytest.raises(DeepEvalError) as exc_info:
            raise DeepEvalError(
                message="Metric evaluation failed",
                metric_name="AnswerRelevancy",
                original_error=original,
                test_case_summary={"input": "test"},
            )

        assert exc_info.value.metric_name == "AnswerRelevancy"
        assert exc_info.value.original_error is original
        assert "input" in exc_info.value.test_case_summary

    def test_can_catch_as_evaluation_error(self) -> None:
        """Should be catchable as EvaluationError."""
        with pytest.raises(EvaluationError):
            raise DeepEvalError(
                message="Test error",
                metric_name="GEval",
            )


class TestExceptionChaining:
    """Tests for exception chaining behavior."""

    def test_deepeval_error_chains_original(self) -> None:
        """DeepEvalError should support exception chaining."""
        original = ValueError("Original cause")

        try:
            try:
                raise original
            except ValueError as e:
                raise DeepEvalError(
                    message="Wrapped error",
                    metric_name="GEval",
                    original_error=e,
                ) from e
        except DeepEvalError as error:
            assert error.__cause__ is original
            assert error.original_error is original
