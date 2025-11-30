"""Unit tests for ContextualPrecision evaluator."""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.evaluators.base import RetryConfig
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.evaluators.deepeval.contextual_precision import (
    ContextualPrecisionEvaluator,
)
from holodeck.models.llm import ProviderEnum

# =============================================================================
# Phase 5 Tests (T026) - ContextualPrecisionEvaluator
# =============================================================================


class TestContextualPrecisionEvaluatorInit:
    """Tests for ContextualPrecisionEvaluator initialization."""

    @patch("deepeval.models.OllamaModel")
    def test_default_parameters(self, mock_ollama: MagicMock) -> None:
        """Should use default Ollama provider with threshold 0.5."""
        mock_ollama.return_value = MagicMock()

        evaluator = ContextualPrecisionEvaluator()

        assert evaluator._threshold == 0.5
        assert evaluator._include_reason is True
        assert evaluator._model_config.provider == ProviderEnum.OLLAMA

    @patch("deepeval.models.OllamaModel")
    def test_custom_threshold(self, mock_ollama: MagicMock) -> None:
        """Custom threshold should be set correctly."""
        mock_ollama.return_value = MagicMock()

        evaluator = ContextualPrecisionEvaluator(threshold=0.8)

        assert evaluator._threshold == 0.8

    @patch("deepeval.models.OllamaModel")
    def test_custom_include_reason_false(self, mock_ollama: MagicMock) -> None:
        """include_reason=False should be honored."""
        mock_ollama.return_value = MagicMock()

        evaluator = ContextualPrecisionEvaluator(include_reason=False)

        assert evaluator._include_reason is False

    @patch("deepeval.models.OllamaModel")
    def test_inherits_timeout_from_base(self, mock_ollama: MagicMock) -> None:
        """Should inherit timeout from base class."""
        mock_ollama.return_value = MagicMock()

        evaluator = ContextualPrecisionEvaluator(timeout=120.0)

        assert evaluator.timeout == 120.0

    @patch("deepeval.models.OllamaModel")
    def test_inherits_retry_config_from_base(self, mock_ollama: MagicMock) -> None:
        """Should inherit retry_config from base class."""
        mock_ollama.return_value = MagicMock()

        retry_config = RetryConfig(max_retries=5, base_delay=1.0)
        evaluator = ContextualPrecisionEvaluator(retry_config=retry_config)

        assert evaluator.retry_config.max_retries == 5
        assert evaluator.retry_config.base_delay == 1.0


class TestContextualPrecisionCreateMetric:
    """Tests for _create_metric() method."""

    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    def test_creates_contextual_precision_metric(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should create ContextualPrecisionMetric with correct parameters."""
        mock_ollama.return_value = MagicMock()
        mock_contextual_precision_instance = MagicMock()
        mock_contextual_precision.return_value = mock_contextual_precision_instance

        evaluator = ContextualPrecisionEvaluator(threshold=0.7)

        evaluator._create_metric()

        mock_contextual_precision.assert_called_once()

    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    def test_model_passed_to_metric(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should pass model to ContextualPrecisionMetric."""
        mock_model = MagicMock()
        mock_ollama.return_value = mock_model
        mock_contextual_precision.return_value = MagicMock()

        evaluator = ContextualPrecisionEvaluator()

        evaluator._create_metric()

        call_kwargs = mock_contextual_precision.call_args[1]
        assert call_kwargs["model"] == mock_model

    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    def test_threshold_passed_to_metric(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should pass threshold to ContextualPrecisionMetric."""
        mock_ollama.return_value = MagicMock()
        mock_contextual_precision.return_value = MagicMock()

        evaluator = ContextualPrecisionEvaluator(threshold=0.9)

        evaluator._create_metric()

        call_kwargs = mock_contextual_precision.call_args[1]
        assert call_kwargs["threshold"] == 0.9

    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    def test_include_reason_passed_to_metric(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should pass include_reason to ContextualPrecisionMetric."""
        mock_ollama.return_value = MagicMock()
        mock_contextual_precision.return_value = MagicMock()

        evaluator = ContextualPrecisionEvaluator(include_reason=False)

        evaluator._create_metric()

        call_kwargs = mock_contextual_precision.call_args[1]
        assert call_kwargs["include_reason"] is False


class TestContextualPrecisionEvaluation:
    """Tests for evaluation functionality."""

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_successful_evaluation_good_ranking(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should return high score when relevant chunks are ranked first."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 1.0
        mock_metric.reason = "Relevant chunks are ranked first"
        mock_contextual_precision.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator(threshold=0.7)

        result = await evaluator._evaluate_impl(
            input="What is X?",
            actual_output="X is the definition.",
            expected_output="X is the correct definition.",
            retrieval_context=[
                "X is the definition",  # Relevant - ranked first
                "Irrelevant info",  # Irrelevant - ranked last
            ],
        )

        assert result["score"] == 1.0
        mock_metric.measure.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_poor_ranking(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should return low score when irrelevant chunks are ranked first."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.3
        mock_metric.reason = "Irrelevant chunks ranked before relevant ones"
        mock_contextual_precision.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator(threshold=0.7)

        result = await evaluator._evaluate_impl(
            input="What is X?",
            actual_output="X is the definition.",
            expected_output="X is the correct definition.",
            retrieval_context=[
                "Irrelevant info",  # Irrelevant - ranked first (bad)
                "X is the definition",  # Relevant - ranked last (bad)
            ],
        )

        assert result["score"] == 0.3

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_passes_when_above_threshold(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should return passed=True when score >= threshold."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.8
        mock_metric.reason = "Good ranking"
        mock_contextual_precision.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator(threshold=0.7)

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
            expected_output="Expected response",
            retrieval_context=["Test context"],
        )

        assert result["passed"] is True

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_fails_when_below_threshold(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should return passed=False when score < threshold."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.4
        mock_metric.reason = "Poor ranking"
        mock_contextual_precision.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator(threshold=0.7)

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
            expected_output="Expected response",
            retrieval_context=["Test context"],
        )

        assert result["passed"] is False

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_reasoning_included_in_result(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should include reasoning in result."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.75
        mock_metric.reason = "Ranking is mostly correct with minor issues"
        mock_contextual_precision.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator()

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
            expected_output="Expected response",
            retrieval_context=["Test context"],
        )

        assert result["reasoning"] == "Ranking is mostly correct with minor issues"

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_metric_name_in_result(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should include 'ContextualPrecision' in result['metric_name']."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.75
        mock_metric.reason = "Good"
        mock_contextual_precision.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator()

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
            expected_output="Expected response",
            retrieval_context=["Test context"],
        )

        assert result["metric_name"] == "ContextualPrecision"


class TestContextualPrecisionProviderSupport:
    """Tests for multi-provider support."""

    @patch("deepeval.models.GPTModel")
    def test_openai_provider(self, mock_gpt: MagicMock) -> None:
        """Should support OpenAI provider."""
        mock_gpt.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.OPENAI,
            model_name="gpt-4o",
        )
        evaluator = ContextualPrecisionEvaluator(model_config=config)

        assert evaluator._model_config.provider == ProviderEnum.OPENAI

    @patch("deepeval.models.AnthropicModel")
    def test_anthropic_provider(self, mock_anthropic: MagicMock) -> None:
        """Should support Anthropic provider."""
        mock_anthropic.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.ANTHROPIC,
            model_name="claude-3-5-sonnet-latest",
        )
        evaluator = ContextualPrecisionEvaluator(model_config=config)

        assert evaluator._model_config.provider == ProviderEnum.ANTHROPIC

    @patch("deepeval.models.OllamaModel")
    def test_ollama_provider_default(self, mock_ollama: MagicMock) -> None:
        """Should support Ollama provider (default)."""
        mock_ollama.return_value = MagicMock()

        evaluator = ContextualPrecisionEvaluator()

        assert evaluator._model_config.provider == ProviderEnum.OLLAMA

    @patch("deepeval.models.AzureOpenAIModel")
    def test_azure_openai_provider(self, mock_azure: MagicMock) -> None:
        """Should support Azure OpenAI provider."""
        mock_azure.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.AZURE_OPENAI,
            model_name="gpt-4o",
            api_key="test-key",
            endpoint="https://test.openai.azure.com/",
            deployment_name="test-deployment",
        )
        evaluator = ContextualPrecisionEvaluator(model_config=config)

        assert evaluator._model_config.provider == ProviderEnum.AZURE_OPENAI


class TestContextualPrecisionRequiredInputs:
    """Tests for required inputs validation."""

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_works_with_all_required_inputs(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Should work when all required inputs are provided."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = "Good"
        mock_contextual_precision.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator()

        result = await evaluator._evaluate_impl(
            input="What is the capital?",
            actual_output="Paris is the capital of France.",
            expected_output="Paris is the capital of France.",
            retrieval_context=["Paris is the capital of France."],
        )

        assert result["score"] == 0.9
        mock_metric.measure.assert_called_once()


class TestContextualPrecisionWithPublicEvaluate:
    """Tests for the public evaluate() method."""

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_evaluate_calls_evaluate_impl(
        self, mock_ollama: MagicMock, mock_contextual_precision: MagicMock
    ) -> None:
        """Public evaluate() should call _evaluate_impl()."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = "Excellent"
        mock_contextual_precision.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator()

        result = await evaluator.evaluate(
            input="Test query",
            actual_output="Test response",
            expected_output="Expected response",
            retrieval_context=["Test context"],
        )

        assert result["score"] == 0.9
        assert result["passed"] is True
        mock_metric.measure.assert_called_once()
