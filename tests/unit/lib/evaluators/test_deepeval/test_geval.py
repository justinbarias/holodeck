"""Unit tests for GEval evaluator."""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.evaluators.base import RetryConfig
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.evaluators.deepeval.geval import GEvalEvaluator
from holodeck.models.llm import ProviderEnum

# =============================================================================
# Phase 3 Tests (T012) - Basic GEvalEvaluator
# =============================================================================


class TestGEvalEvaluatorInit:
    """Tests for GEvalEvaluator initialization."""

    @patch("deepeval.models.OllamaModel")
    def test_required_parameters_name_and_criteria(
        self, mock_ollama: MagicMock
    ) -> None:
        """GEvalEvaluator requires name and criteria parameters."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Evaluate the quality of the response",
        )

        assert evaluator._metric_name == "TestMetric"
        assert evaluator._criteria == "Evaluate the quality of the response"

    @patch("deepeval.models.OllamaModel")
    def test_default_evaluation_params(self, mock_ollama: MagicMock) -> None:
        """Default evaluation_params should be ['actual_output']."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
        )

        assert evaluator._evaluation_params == ["actual_output"]

    @patch("deepeval.models.OllamaModel")
    def test_custom_evaluation_params(self, mock_ollama: MagicMock) -> None:
        """Custom evaluation_params should be set correctly."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            evaluation_params=["input", "actual_output", "expected_output"],
        )

        assert evaluator._evaluation_params == [
            "input",
            "actual_output",
            "expected_output",
        ]

    @patch("deepeval.models.OllamaModel")
    def test_custom_evaluation_steps(self, mock_ollama: MagicMock) -> None:
        """Custom evaluation_steps should be set correctly."""
        mock_ollama.return_value = MagicMock()

        steps = ["Step 1: Check grammar", "Step 2: Check relevance"]
        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            evaluation_steps=steps,
        )

        assert evaluator._evaluation_steps == steps

    @patch("deepeval.models.OllamaModel")
    def test_evaluation_steps_none_by_default(self, mock_ollama: MagicMock) -> None:
        """evaluation_steps should be None by default (auto-generated)."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
        )

        assert evaluator._evaluation_steps is None

    @patch("deepeval.models.OllamaModel")
    def test_strict_mode_default_false(self, mock_ollama: MagicMock) -> None:
        """strict_mode should be False by default."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
        )

        assert evaluator._strict_mode is False

    @patch("deepeval.models.OllamaModel")
    def test_strict_mode_enabled(self, mock_ollama: MagicMock) -> None:
        """strict_mode can be enabled."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            strict_mode=True,
        )

        assert evaluator._strict_mode is True

    @patch("deepeval.models.OllamaModel")
    def test_inherits_threshold_from_base(self, mock_ollama: MagicMock) -> None:
        """Should inherit threshold from base class."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            threshold=0.8,
        )

        assert evaluator._threshold == 0.8

    @patch("deepeval.models.OllamaModel")
    def test_inherits_timeout_from_base(self, mock_ollama: MagicMock) -> None:
        """Should inherit timeout from base class."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            timeout=120.0,
        )

        assert evaluator.timeout == 120.0

    @patch("deepeval.models.OllamaModel")
    def test_inherits_retry_config_from_base(self, mock_ollama: MagicMock) -> None:
        """Should inherit retry_config from base class."""
        mock_ollama.return_value = MagicMock()

        retry_config = RetryConfig(max_retries=5, base_delay=1.0)
        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            retry_config=retry_config,
        )

        assert evaluator.retry_config.max_retries == 5
        assert evaluator.retry_config.base_delay == 1.0


class TestGEvalCreateMetric:
    """Tests for _create_metric() method."""

    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    def test_creates_geval_with_correct_params(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should create GEval metric with correct parameters."""
        mock_ollama.return_value = MagicMock()
        mock_geval_instance = MagicMock()
        mock_geval.return_value = mock_geval_instance

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            threshold=0.7,
        )

        evaluator._create_metric()

        mock_geval.assert_called_once()
        call_kwargs = mock_geval.call_args[1]
        assert call_kwargs["name"] == "TestMetric"
        assert call_kwargs["criteria"] == "Test criteria"
        assert call_kwargs["threshold"] == 0.7

    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    def test_evaluation_params_mapping(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should map string evaluation_params to LLMTestCaseParams."""
        mock_ollama.return_value = MagicMock()
        mock_geval.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            evaluation_params=["input", "actual_output"],
        )

        evaluator._create_metric()

        call_kwargs = mock_geval.call_args[1]
        # Verify evaluation_params are converted to enum values
        assert "evaluation_params" in call_kwargs
        assert len(call_kwargs["evaluation_params"]) == 2

    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    def test_threshold_passed_to_metric(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should pass threshold to GEval metric."""
        mock_ollama.return_value = MagicMock()
        mock_geval.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            threshold=0.9,
        )

        evaluator._create_metric()

        call_kwargs = mock_geval.call_args[1]
        assert call_kwargs["threshold"] == 0.9

    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    def test_model_passed_to_metric(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should pass model to GEval metric."""
        mock_model = MagicMock()
        mock_ollama.return_value = mock_model
        mock_geval.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
        )

        evaluator._create_metric()

        call_kwargs = mock_geval.call_args[1]
        assert call_kwargs["model"] == mock_model

    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    def test_strict_mode_passed_to_metric(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should pass strict_mode to GEval metric."""
        mock_ollama.return_value = MagicMock()
        mock_geval.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            strict_mode=True,
        )

        evaluator._create_metric()

        call_kwargs = mock_geval.call_args[1]
        assert call_kwargs["strict_mode"] is True


class TestGEvalEvaluation:
    """Tests for evaluation functionality."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    async def test_successful_evaluation_returns_normalized_score(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should return normalized score on successful evaluation."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.85
        mock_metric.reason = "Good response quality"
        mock_geval.return_value = mock_metric

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            threshold=0.7,
        )

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        assert result["score"] == 0.85
        mock_metric.measure.assert_called_once()

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_passes_when_above_threshold(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should return passed=True when score >= threshold."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.8
        mock_metric.reason = "Good"
        mock_geval.return_value = mock_metric

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            threshold=0.7,
        )

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        assert result["passed"] is True

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_fails_when_below_threshold(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should return passed=False when score < threshold."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.4
        mock_metric.reason = "Poor"
        mock_geval.return_value = mock_metric

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            threshold=0.7,
        )

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        assert result["passed"] is False

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    async def test_reasoning_included_in_result(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should include reasoning in result."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.75
        mock_metric.reason = "The response was clear and helpful"
        mock_geval.return_value = mock_metric

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
        )

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        assert result["reasoning"] == "The response was clear and helpful"

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    async def test_metric_name_in_result(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Should include metric name in result."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.75
        mock_metric.reason = "Good"
        mock_geval.return_value = mock_metric

        evaluator = GEvalEvaluator(
            name="Helpfulness",
            criteria="Test criteria",
        )

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        assert result["metric_name"] == "Helpfulness"


class TestGEvalProviderSupport:
    """Tests for multi-provider support."""

    @patch("deepeval.models.GPTModel")
    def test_openai_provider(self, mock_gpt: MagicMock) -> None:
        """Should support OpenAI provider."""
        mock_gpt.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.OPENAI,
            model_name="gpt-4o",
        )
        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            model_config=config,
        )

        assert evaluator._model_config.provider == ProviderEnum.OPENAI

    @patch("deepeval.models.AnthropicModel")
    def test_anthropic_provider(self, mock_anthropic: MagicMock) -> None:
        """Should support Anthropic provider."""
        mock_anthropic.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.ANTHROPIC,
            model_name="claude-3-5-sonnet-latest",
        )
        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            model_config=config,
        )

        assert evaluator._model_config.provider == ProviderEnum.ANTHROPIC

    @patch("deepeval.models.OllamaModel")
    def test_ollama_provider(self, mock_ollama: MagicMock) -> None:
        """Should support Ollama provider (default)."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
        )

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
        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            model_config=config,
        )

        assert evaluator._model_config.provider == ProviderEnum.AZURE_OPENAI


class TestGEvalNameProperty:
    """Tests for name property override."""

    @patch("deepeval.models.OllamaModel")
    def test_name_returns_custom_metric_name(self, mock_ollama: MagicMock) -> None:
        """Name property should return the custom metric name."""
        mock_ollama.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="Professionalism",
            criteria="Test criteria",
        )

        assert evaluator.name == "Professionalism"


# =============================================================================
# Phase 4 Tests (T019) - Custom Evaluation Criteria
# =============================================================================


class TestGEvalCustomCriteria:
    """Tests for custom criteria support (User Story 2)."""

    @patch("deepeval.models.OllamaModel")
    def test_evaluation_params_all_valid_options(self, mock_ollama: MagicMock) -> None:
        """Should accept all valid evaluation_params options."""
        mock_ollama.return_value = MagicMock()

        valid_params = [
            "input",
            "actual_output",
            "expected_output",
            "context",
            "retrieval_context",
        ]

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            evaluation_params=valid_params,
        )

        assert evaluator._evaluation_params == valid_params

    @patch("deepeval.models.OllamaModel")
    def test_evaluation_params_invalid_raises_error(
        self, mock_ollama: MagicMock
    ) -> None:
        """Should raise ValueError for invalid evaluation_params."""
        mock_ollama.return_value = MagicMock()

        with pytest.raises(ValueError, match="Invalid evaluation_param"):
            GEvalEvaluator(
                name="TestMetric",
                criteria="Test criteria",
                evaluation_params=["invalid_param"],
            )

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    async def test_strict_mode_returns_binary_score(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """strict_mode should result in binary scores (0.0 or 1.0)."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        # DeepEval returns binary score in strict mode
        mock_metric.score = 1.0
        mock_metric.reason = "Passed strict evaluation"
        mock_geval.return_value = mock_metric

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            strict_mode=True,
            threshold=0.5,
        )

        result = await evaluator._evaluate_impl(
            input="Test query",
            actual_output="Test response",
        )

        # Verify strict_mode was passed to GEval
        call_kwargs = mock_geval.call_args[1]
        assert call_kwargs["strict_mode"] is True
        # Score should be binary (1.0 or 0.0)
        assert result["score"] in [0.0, 1.0]

    @patch("deepeval.models.OllamaModel")
    def test_threshold_validation_bounds(self, mock_ollama: MagicMock) -> None:
        """Threshold should be between 0.0 and 1.0."""
        mock_ollama.return_value = MagicMock()

        # Valid thresholds
        evaluator_low = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            threshold=0.0,
        )
        assert evaluator_low._threshold == 0.0

        evaluator_high = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            threshold=1.0,
        )
        assert evaluator_high._threshold == 1.0

        evaluator_mid = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
            threshold=0.5,
        )
        assert evaluator_mid._threshold == 0.5

    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    def test_auto_generation_of_evaluation_steps(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """When evaluation_steps is None, GEval auto-generates them."""
        mock_ollama.return_value = MagicMock()
        mock_geval.return_value = MagicMock()

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Evaluate if the response is professional",
            # evaluation_steps not provided - should be None
        )

        evaluator._create_metric()

        # Verify evaluation_steps is not passed (or passed as None)
        call_kwargs = mock_geval.call_args[1]
        assert call_kwargs.get("evaluation_steps") is None

    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    def test_custom_steps_override_auto_generation(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Custom evaluation_steps should override auto-generation."""
        mock_ollama.return_value = MagicMock()
        mock_geval.return_value = MagicMock()

        custom_steps = [
            "Step 1: Check for professional tone",
            "Step 2: Verify grammar correctness",
            "Step 3: Assess clarity of communication",
        ]

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Evaluate professionalism",
            evaluation_steps=custom_steps,
        )

        evaluator._create_metric()

        call_kwargs = mock_geval.call_args[1]
        assert call_kwargs["evaluation_steps"] == custom_steps


class TestGEvalWithPublicEvaluate:
    """Tests for the public evaluate() method."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.geval.GEval")
    @patch("deepeval.models.OllamaModel")
    async def test_evaluate_calls_evaluate_impl(
        self, mock_ollama: MagicMock, mock_geval: MagicMock
    ) -> None:
        """Public evaluate() should call _evaluate_impl()."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.9
        mock_metric.reason = "Excellent"
        mock_geval.return_value = mock_metric

        evaluator = GEvalEvaluator(
            name="TestMetric",
            criteria="Test criteria",
        )

        result = await evaluator.evaluate(
            input="Test query",
            actual_output="Test response",
        )

        assert result["score"] == 0.9
        assert result["passed"] is True
        mock_metric.measure.assert_called_once()
