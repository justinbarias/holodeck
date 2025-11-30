"""Unit tests for Faithfulness evaluator."""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.evaluators.base import RetryConfig
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.evaluators.deepeval.faithfulness import FaithfulnessEvaluator
from holodeck.models.llm import ProviderEnum

# =============================================================================
# Phase 5 Tests (T024) - FaithfulnessEvaluator
# =============================================================================


class TestFaithfulnessEvaluatorInit:
    """Tests for FaithfulnessEvaluator initialization."""

    @patch("deepeval.models.OllamaModel")
    def test_default_parameters(self, mock_ollama: MagicMock) -> None:
        """Should use default Ollama provider with threshold 0.5."""
        mock_ollama.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator()

        assert evaluator._threshold == 0.5
        assert evaluator._include_reason is True
        assert evaluator._model_config.provider == ProviderEnum.OLLAMA

    @patch("deepeval.models.OllamaModel")
    def test_custom_threshold(self, mock_ollama: MagicMock) -> None:
        """Custom threshold should be set correctly."""
        mock_ollama.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator(threshold=0.8)

        assert evaluator._threshold == 0.8

    @patch("deepeval.models.OllamaModel")
    def test_custom_include_reason_false(self, mock_ollama: MagicMock) -> None:
        """include_reason=False should be honored."""
        mock_ollama.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator(include_reason=False)

        assert evaluator._include_reason is False

    @patch("deepeval.models.OllamaModel")
    def test_inherits_timeout_from_base(self, mock_ollama: MagicMock) -> None:
        """Should inherit timeout from base class."""
        mock_ollama.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator(timeout=120.0)

        assert evaluator.timeout == 120.0

    @patch("deepeval.models.OllamaModel")
    def test_inherits_retry_config_from_base(self, mock_ollama: MagicMock) -> None:
        """Should inherit retry_config from base class."""
        mock_ollama.return_value = MagicMock()

        retry_config = RetryConfig(max_retries=5, base_delay=1.0)
        evaluator = FaithfulnessEvaluator(retry_config=retry_config)

        assert evaluator.retry_config.max_retries == 5
        assert evaluator.retry_config.base_delay == 1.0


class TestFaithfulnessCreateMetric:
    """Tests for _create_metric() method."""

    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    def test_creates_faithfulness_metric(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should create FaithfulnessMetric with correct parameters."""
        mock_ollama.return_value = MagicMock()
        mock_faithfulness_instance = MagicMock()
        mock_faithfulness.return_value = mock_faithfulness_instance

        evaluator = FaithfulnessEvaluator(threshold=0.7)

        evaluator._create_metric()

        mock_faithfulness.assert_called_once()

    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    def test_model_passed_to_metric(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should pass model to FaithfulnessMetric."""
        mock_model = MagicMock()
        mock_ollama.return_value = mock_model
        mock_faithfulness.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator()

        evaluator._create_metric()

        call_kwargs = mock_faithfulness.call_args[1]
        assert call_kwargs["model"] == mock_model

    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    def test_threshold_passed_to_metric(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should pass threshold to FaithfulnessMetric."""
        mock_ollama.return_value = MagicMock()
        mock_faithfulness.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator(threshold=0.9)

        evaluator._create_metric()

        call_kwargs = mock_faithfulness.call_args[1]
        assert call_kwargs["threshold"] == 0.9

    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    def test_include_reason_passed_to_metric(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should pass include_reason to FaithfulnessMetric."""
        mock_ollama.return_value = MagicMock()
        mock_faithfulness.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator(include_reason=False)

        evaluator._create_metric()

        call_kwargs = mock_faithfulness.call_args[1]
        assert call_kwargs["include_reason"] is False


class TestFaithfulnessEvaluation:
    """Tests for evaluation functionality."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_successful_evaluation_high_score(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should return high score for faithful response."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.95
        mock_metric.reason = "Response is fully grounded in context"
        mock_faithfulness.return_value = mock_metric

        evaluator = FaithfulnessEvaluator(threshold=0.8)

        result = await evaluator._evaluate_impl(
            input="What are the store hours?",
            actual_output="Store is open Mon-Fri 9am-5pm.",
            retrieval_context=["Store hours: Mon-Fri 9am-5pm"],
        )

        assert result["score"] == 0.95
        mock_metric.measure.assert_called_once()

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_low_score_hallucination(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should return low score for hallucinated response."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.2
        mock_metric.reason = "Response contains information not in context"
        mock_faithfulness.return_value = mock_metric

        evaluator = FaithfulnessEvaluator(threshold=0.8)

        result = await evaluator._evaluate_impl(
            input="What are the store hours?",
            actual_output="Store is open 24/7.",
            retrieval_context=["Store hours: Mon-Fri 9am-5pm"],
        )

        assert result["score"] == 0.2

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_passes_when_above_threshold(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should return passed=True when score >= threshold."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = "Good"
        mock_faithfulness.return_value = mock_metric

        evaluator = FaithfulnessEvaluator(threshold=0.7)

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
            retrieval_context=["Test context"],
        )

        assert result["passed"] is True

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_fails_when_below_threshold(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should return passed=False when score < threshold."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.4
        mock_metric.reason = "Poor"
        mock_faithfulness.return_value = mock_metric

        evaluator = FaithfulnessEvaluator(threshold=0.7)

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
            retrieval_context=["Test context"],
        )

        assert result["passed"] is False

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_reasoning_included_in_result(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should include reasoning in result."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.75
        mock_metric.reason = "The response is mostly grounded but has minor issues"
        mock_faithfulness.return_value = mock_metric

        evaluator = FaithfulnessEvaluator()

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
            retrieval_context=["Test context"],
        )

        assert (
            result["reasoning"]
            == "The response is mostly grounded but has minor issues"
        )

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_metric_name_in_result(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should include 'Faithfulness' in result['metric_name']."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.75
        mock_metric.reason = "Good"
        mock_faithfulness.return_value = mock_metric

        evaluator = FaithfulnessEvaluator()

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
            retrieval_context=["Test context"],
        )

        assert result["metric_name"] == "Faithfulness"


class TestFaithfulnessProviderSupport:
    """Tests for multi-provider support."""

    @patch("deepeval.models.GPTModel")
    def test_openai_provider(self, mock_gpt: MagicMock) -> None:
        """Should support OpenAI provider."""
        mock_gpt.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.OPENAI,
            model_name="gpt-4o",
        )
        evaluator = FaithfulnessEvaluator(model_config=config)

        assert evaluator._model_config.provider == ProviderEnum.OPENAI

    @patch("deepeval.models.AnthropicModel")
    def test_anthropic_provider(self, mock_anthropic: MagicMock) -> None:
        """Should support Anthropic provider."""
        mock_anthropic.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.ANTHROPIC,
            model_name="claude-3-5-sonnet-latest",
        )
        evaluator = FaithfulnessEvaluator(model_config=config)

        assert evaluator._model_config.provider == ProviderEnum.ANTHROPIC

    @patch("deepeval.models.OllamaModel")
    def test_ollama_provider_default(self, mock_ollama: MagicMock) -> None:
        """Should support Ollama provider (default)."""
        mock_ollama.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator()

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
        evaluator = FaithfulnessEvaluator(model_config=config)

        assert evaluator._model_config.provider == ProviderEnum.AZURE_OPENAI


class TestFaithfulnessRequiredInputs:
    """Tests for required inputs validation."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_works_with_all_required_inputs(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Should work when all required inputs are provided."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = "Good"
        mock_faithfulness.return_value = mock_metric

        evaluator = FaithfulnessEvaluator()

        result = await evaluator._evaluate_impl(
            input="What is the capital?",
            actual_output="Paris is the capital of France.",
            retrieval_context=["Paris is the capital of France."],
        )

        assert result["score"] == 0.9
        mock_metric.measure.assert_called_once()


class TestFaithfulnessWithPublicEvaluate:
    """Tests for the public evaluate() method."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_evaluate_calls_evaluate_impl(
        self, mock_ollama: MagicMock, mock_faithfulness: MagicMock
    ) -> None:
        """Public evaluate() should call _evaluate_impl()."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = "Excellent"
        mock_faithfulness.return_value = mock_metric

        evaluator = FaithfulnessEvaluator()

        result = await evaluator.evaluate(
            input="Test query",
            actual_output="Test response",
            retrieval_context=["Test context"],
        )

        assert result["score"] == 0.9
        assert result["passed"] is True
        mock_metric.measure.assert_called_once()
