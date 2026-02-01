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

    @pytest.mark.parametrize(
        "param,value,attr,expected",
        [
            ("threshold", 0.8, "_threshold", 0.8),
            ("include_reason", False, "_include_reason", False),
            ("timeout", 120.0, "timeout", 120.0),
        ],
        ids=["threshold", "include_reason", "timeout"],
    )
    @patch("deepeval.models.OllamaModel")
    def test_custom_parameters(
        self,
        mock_ollama: MagicMock,
        param: str,
        value: float,
        attr: str,
        expected: float,
    ) -> None:
        """Custom parameters should be set correctly."""
        mock_ollama.return_value = MagicMock()

        evaluator = FaithfulnessEvaluator(**{param: value})

        assert getattr(evaluator, attr) == expected

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

    @pytest.mark.parametrize(
        "provider,mock_path,model_name,extra_config",
        [
            (ProviderEnum.OPENAI, "deepeval.models.GPTModel", "gpt-4o", {}),
            (
                ProviderEnum.ANTHROPIC,
                "deepeval.models.AnthropicModel",
                "claude-3-5-sonnet-latest",
                {},
            ),
            (ProviderEnum.OLLAMA, "deepeval.models.OllamaModel", "llama3", {}),
            (
                ProviderEnum.AZURE_OPENAI,
                "deepeval.models.AzureOpenAIModel",
                "gpt-4o",
                {
                    "api_key": "test-key",
                    "endpoint": "https://test.openai.azure.com/",
                    "deployment_name": "test-deployment",
                },
            ),
        ],
        ids=["openai", "anthropic", "ollama", "azure_openai"],
    )
    def test_provider_support(
        self,
        provider: ProviderEnum,
        mock_path: str,
        model_name: str,
        extra_config: dict,
    ) -> None:
        """Should support various LLM providers."""
        with patch(mock_path) as mock_model:
            mock_model.return_value = MagicMock()

            config = DeepEvalModelConfig(
                provider=provider,
                model_name=model_name,
                **extra_config,
            )
            evaluator = FaithfulnessEvaluator(model_config=config)

            assert evaluator._model_config.provider == provider


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
