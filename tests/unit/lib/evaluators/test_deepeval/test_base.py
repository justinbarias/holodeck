"""Unit tests for DeepEval base evaluator."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.evaluators.base import RetryConfig
from holodeck.lib.evaluators.deepeval.base import DeepEvalBaseEvaluator
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.evaluators.deepeval.errors import DeepEvalError
from holodeck.models.llm import ProviderEnum


class ConcreteDeepEvalEvaluator(DeepEvalBaseEvaluator):
    """Concrete implementation for testing the abstract base class."""

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
        mock_metric: Any = None,
    ) -> None:
        # Store mock before calling super().__init__
        self._mock_metric = mock_metric
        super().__init__(
            model_config=model_config,
            threshold=threshold,
            timeout=timeout,
            retry_config=retry_config,
        )

    def _create_metric(self) -> Any:
        """Return mock metric for testing."""
        return self._mock_metric


class FailingDeepEvalEvaluator(DeepEvalBaseEvaluator):
    """Evaluator that always raises an error."""

    def __init__(
        self,
        error_message: str = "Metric failed",
        **kwargs: Any,
    ) -> None:
        self._error_message = error_message
        super().__init__(**kwargs)

    def _create_metric(self) -> Any:
        """Return a metric that raises an error."""
        mock = MagicMock()
        mock.measure.side_effect = ValueError(self._error_message)
        return mock


class TestDeepEvalBaseEvaluatorInit:
    """Tests for evaluator initialization."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """DeepEvalBaseEvaluator is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            DeepEvalBaseEvaluator()  # type: ignore[abstract]

    @patch("deepeval.models.OllamaModel")
    def test_default_model_config(self, mock_ollama: MagicMock) -> None:
        """Evaluator should use default model config if none provided."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)

        assert evaluator._model_config.provider == ProviderEnum.OLLAMA
        assert evaluator._model_config.model_name == "gpt-oss:20b"

    @patch("deepeval.models.GPTModel")
    def test_custom_model_config(self, mock_gpt: MagicMock) -> None:
        """Evaluator should use provided model config."""
        mock_gpt.return_value = MagicMock()
        mock_metric = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.OPENAI,
            model_name="gpt-4o",
        )
        evaluator = ConcreteDeepEvalEvaluator(
            model_config=config,
            mock_metric=mock_metric,
        )

        assert evaluator._model_config.provider == ProviderEnum.OPENAI
        assert evaluator._model_config.model_name == "gpt-4o"

    @patch("deepeval.models.OllamaModel")
    def test_default_threshold(self, mock_ollama: MagicMock) -> None:
        """Default threshold should be 0.5."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)

        assert evaluator._threshold == 0.5

    @patch("deepeval.models.OllamaModel")
    def test_custom_threshold(self, mock_ollama: MagicMock) -> None:
        """Custom threshold should be set correctly."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(
            threshold=0.8,
            mock_metric=mock_metric,
        )

        assert evaluator._threshold == 0.8

    @patch("deepeval.models.OllamaModel")
    def test_inherits_timeout_from_base(self, mock_ollama: MagicMock) -> None:
        """Evaluator should inherit timeout from BaseEvaluator."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(
            timeout=120.0,
            mock_metric=mock_metric,
        )

        assert evaluator.timeout == 120.0

    @patch("deepeval.models.OllamaModel")
    def test_inherits_retry_config_from_base(self, mock_ollama: MagicMock) -> None:
        """Evaluator should inherit retry config from BaseEvaluator."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        retry_config = RetryConfig(max_retries=5, base_delay=1.0)
        evaluator = ConcreteDeepEvalEvaluator(
            retry_config=retry_config,
            mock_metric=mock_metric,
        )

        assert evaluator.retry_config.max_retries == 5
        assert evaluator.retry_config.base_delay == 1.0


class TestBuildTestCase:
    """Tests for _build_test_case() method."""

    @patch("deepeval.models.OllamaModel")
    def test_standard_parameter_names(self, mock_ollama: MagicMock) -> None:
        """Should build test case with standard DeepEval parameter names."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)
        test_case = evaluator._build_test_case(
            input="What is Python?",
            actual_output="Python is a programming language.",
            expected_output="A programming language.",
            retrieval_context=["Python info"],
        )

        assert test_case.input == "What is Python?"
        assert test_case.actual_output == "Python is a programming language."
        assert test_case.expected_output == "A programming language."
        assert test_case.retrieval_context == ["Python info"]

    @patch("deepeval.models.OllamaModel")
    def test_alias_query_to_input(self, mock_ollama: MagicMock) -> None:
        """Should accept 'query' as alias for 'input'."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)
        test_case = evaluator._build_test_case(
            query="What is Python?",
            actual_output="Python is a language.",
        )

        assert test_case.input == "What is Python?"

    @patch("deepeval.models.OllamaModel")
    def test_alias_response_to_actual_output(self, mock_ollama: MagicMock) -> None:
        """Should accept 'response' as alias for 'actual_output'."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)
        test_case = evaluator._build_test_case(
            input="What is Python?",
            response="Python is a language.",
        )

        assert test_case.actual_output == "Python is a language."

    @patch("deepeval.models.OllamaModel")
    def test_alias_ground_truth_to_expected_output(
        self, mock_ollama: MagicMock
    ) -> None:
        """Should accept 'ground_truth' as alias for 'expected_output'."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)
        test_case = evaluator._build_test_case(
            input="What is Python?",
            actual_output="Python is a language.",
            ground_truth="A programming language.",
        )

        assert test_case.expected_output == "A programming language."

    @patch("deepeval.models.OllamaModel")
    def test_standard_names_take_precedence_over_aliases(
        self, mock_ollama: MagicMock
    ) -> None:
        """Standard parameter names should take precedence over aliases."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)
        test_case = evaluator._build_test_case(
            input="Standard input",
            query="Alias query",  # Should be ignored
            actual_output="Standard output",
            response="Alias response",  # Should be ignored
        )

        assert test_case.input == "Standard input"
        assert test_case.actual_output == "Standard output"


class TestSummarizeTestCase:
    """Tests for _summarize_test_case() method."""

    @patch("deepeval.models.OllamaModel")
    def test_summarize_truncates_long_values(self, mock_ollama: MagicMock) -> None:
        """Should truncate values longer than 100 characters."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)
        test_case = evaluator._build_test_case(
            input="A" * 150,
            actual_output="B" * 150,
        )

        summary = evaluator._summarize_test_case(test_case)

        assert len(summary["input"]) == 103  # 100 + "..."
        assert summary["input"].endswith("...")
        assert len(summary["actual_output"]) == 103

    @patch("deepeval.models.OllamaModel")
    def test_summarize_excludes_none_values(self, mock_ollama: MagicMock) -> None:
        """Should exclude None values from summary."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)
        test_case = evaluator._build_test_case(
            input="Test input",
            actual_output="Test output",
        )

        summary = evaluator._summarize_test_case(test_case)

        assert "input" in summary
        assert "actual_output" in summary
        assert "expected_output" not in summary
        assert "retrieval_context" not in summary


class TestEvaluateImpl:
    """Tests for _evaluate_impl() method."""

    @pytest.mark.asyncio
    @patch("deepeval.models.OllamaModel")
    async def test_successful_evaluation(self, mock_ollama: MagicMock) -> None:
        """Should return normalized result on successful evaluation."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = "Good response"

        evaluator = ConcreteDeepEvalEvaluator(
            threshold=0.7,
            mock_metric=mock_metric,
        )

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        assert result["score"] == 0.85
        assert result["passed"] is True
        assert result["reasoning"] == "Good response"
        assert result["threshold"] == 0.7
        mock_metric.measure.assert_called_once()

    @pytest.mark.asyncio
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_below_threshold(self, mock_ollama: MagicMock) -> None:
        """Should return passed=False when score is below threshold."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.4
        mock_metric.reason = "Poor response"

        evaluator = ConcreteDeepEvalEvaluator(
            threshold=0.5,
            mock_metric=mock_metric,
        )

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        assert result["score"] == 0.4
        assert result["passed"] is False

    @pytest.mark.asyncio
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_at_threshold(self, mock_ollama: MagicMock) -> None:
        """Should return passed=True when score equals threshold."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.5
        mock_metric.reason = "Acceptable response"

        evaluator = ConcreteDeepEvalEvaluator(
            threshold=0.5,
            mock_metric=mock_metric,
        )

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        assert result["score"] == 0.5
        assert result["passed"] is True

    @pytest.mark.asyncio
    @patch("deepeval.models.OllamaModel")
    async def test_empty_reasoning(self, mock_ollama: MagicMock) -> None:
        """Should handle None reasoning gracefully."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.7
        mock_metric.reason = None

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        assert result["reasoning"] == ""

    @pytest.mark.asyncio
    @patch("deepeval.models.OllamaModel")
    async def test_metric_failure_raises_deepeval_error(
        self, mock_ollama: MagicMock
    ) -> None:
        """Should raise DeepEvalError when metric fails."""
        mock_ollama.return_value = MagicMock()

        evaluator = FailingDeepEvalEvaluator(error_message="Evaluation failed")

        with pytest.raises(DeepEvalError) as exc_info:
            await evaluator._evaluate_impl(
                input="Test query",
                actual_output="Test response",
            )

        assert "Evaluation failed" in str(exc_info.value)
        assert exc_info.value.metric_name == "FailingDeepEvalEvaluator"
        assert exc_info.value.original_error is not None

    @pytest.mark.asyncio
    @patch("deepeval.models.OllamaModel")
    async def test_metric_failure_includes_test_case_summary(
        self, mock_ollama: MagicMock
    ) -> None:
        """DeepEvalError should include test case summary."""
        mock_ollama.return_value = MagicMock()

        evaluator = FailingDeepEvalEvaluator(error_message="Parse error")

        with pytest.raises(DeepEvalError) as exc_info:
            await evaluator._evaluate_impl(
                input="Test query",
                actual_output="Test response",
            )

        assert "input" in exc_info.value.test_case_summary
        assert "actual_output" in exc_info.value.test_case_summary


class TestEvaluatorName:
    """Tests for evaluator name property."""

    @patch("deepeval.models.OllamaModel")
    def test_name_returns_class_name(self, mock_ollama: MagicMock) -> None:
        """Name property should return the class name."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)

        assert evaluator.name == "ConcreteDeepEvalEvaluator"


class TestPublicEvaluateMethod:
    """Tests for the public evaluate() method (inherited from BaseEvaluator)."""

    @pytest.mark.asyncio
    @patch("deepeval.models.OllamaModel")
    async def test_evaluate_calls_evaluate_impl(self, mock_ollama: MagicMock) -> None:
        """Public evaluate() should call _evaluate_impl()."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = "Excellent"

        evaluator = ConcreteDeepEvalEvaluator(mock_metric=mock_metric)

        result = await evaluator.evaluate(
            input="Test query",
            actual_output="Test response",
        )

        assert result["score"] == 0.9
        assert result["passed"] is True
        mock_metric.measure.assert_called_once()
