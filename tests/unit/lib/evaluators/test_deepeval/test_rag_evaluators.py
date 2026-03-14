"""Parameterized unit tests for RAG evaluators (Faithfulness, ContextualRelevancy,
ContextualPrecision, ContextualRecall).

These four evaluators share identical structure: init, _create_metric, evaluation,
provider support, required inputs, and public evaluate. This module consolidates
those into parameterized tests to eliminate duplication.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.evaluators.base import RetryConfig
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.evaluators.deepeval.contextual_precision import (
    ContextualPrecisionEvaluator,
)
from holodeck.lib.evaluators.deepeval.contextual_recall import (
    ContextualRecallEvaluator,
)
from holodeck.lib.evaluators.deepeval.contextual_relevancy import (
    ContextualRelevancyEvaluator,
)
from holodeck.lib.evaluators.deepeval.faithfulness import FaithfulnessEvaluator
from holodeck.models.llm import ProviderEnum

# ---------------------------------------------------------------------------
# Shared test data:
# (evaluator_cls, metric_mock_path, metric_name, needs_expected_output)
# ---------------------------------------------------------------------------
RAG_EVALUATORS = [
    pytest.param(
        FaithfulnessEvaluator,
        "holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric",
        "Faithfulness",
        False,
        id="faithfulness",
    ),
    pytest.param(
        ContextualRelevancyEvaluator,
        "holodeck.lib.evaluators.deepeval.contextual_relevancy.ContextualRelevancyMetric",
        "ContextualRelevancy",
        False,
        id="contextual_relevancy",
    ),
    pytest.param(
        ContextualPrecisionEvaluator,
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric",
        "ContextualPrecision",
        True,
        id="contextual_precision",
    ),
    pytest.param(
        ContextualRecallEvaluator,
        "holodeck.lib.evaluators.deepeval.contextual_recall.ContextualRecallMetric",
        "ContextualRecall",
        True,
        id="contextual_recall",
    ),
]


def _eval_kwargs(
    needs_expected: bool,
    *,
    input_text: str = "Test query",
    actual_output: str = "Test response",
    expected_output: str = "Expected response",
    retrieval_context: list[str] | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for _evaluate_impl / evaluate calls."""
    kwargs: dict[str, Any] = {
        "input": input_text,
        "actual_output": actual_output,
        "retrieval_context": retrieval_context or ["Test context"],
    }
    if needs_expected:
        kwargs["expected_output"] = expected_output
    return kwargs


# =============================================================================
# Initialization tests
# =============================================================================


class TestRAGEvaluatorInit:
    """Tests for RAG evaluator initialization (common across all four)."""

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_default_parameters(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should use default Ollama provider with threshold 0.5."""
        mock_ollama.return_value = MagicMock()

        evaluator = evaluator_cls()

        assert evaluator._threshold == 0.5
        assert evaluator._include_reason is True
        assert evaluator._model_config.provider == ProviderEnum.OLLAMA

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_custom_threshold(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Custom threshold should be set correctly."""
        mock_ollama.return_value = MagicMock()

        evaluator = evaluator_cls(threshold=0.8)

        assert evaluator._threshold == 0.8

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_custom_include_reason_false(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """include_reason=False should be honored."""
        mock_ollama.return_value = MagicMock()

        evaluator = evaluator_cls(include_reason=False)

        assert evaluator._include_reason is False

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_inherits_timeout_from_base(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should inherit timeout from base class."""
        mock_ollama.return_value = MagicMock()

        evaluator = evaluator_cls(timeout=120.0)

        assert evaluator.timeout == 120.0

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_inherits_retry_config_from_base(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should inherit retry_config from base class."""
        mock_ollama.return_value = MagicMock()

        retry_config = RetryConfig(max_retries=5, base_delay=1.0)
        evaluator = evaluator_cls(retry_config=retry_config)

        assert evaluator.retry_config.max_retries == 5
        assert evaluator.retry_config.base_delay == 1.0


# =============================================================================
# _create_metric tests
# =============================================================================


class TestRAGCreateMetric:
    """Tests for _create_metric() method (common across all four)."""

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_creates_metric(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should create the corresponding metric with correct parameters."""
        mock_ollama.return_value = MagicMock()

        with patch(metric_path) as mock_metric_cls:
            mock_metric_cls.return_value = MagicMock()
            evaluator = evaluator_cls(threshold=0.7)
            evaluator._create_metric()
            mock_metric_cls.assert_called_once()

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_model_passed_to_metric(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should pass model to the metric."""
        mock_model = MagicMock()
        mock_ollama.return_value = mock_model

        with patch(metric_path) as mock_metric_cls:
            mock_metric_cls.return_value = MagicMock()
            evaluator = evaluator_cls()
            evaluator._create_metric()
            call_kwargs = mock_metric_cls.call_args[1]
            assert call_kwargs["model"] == mock_model

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_threshold_passed_to_metric(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should pass threshold to the metric."""
        mock_ollama.return_value = MagicMock()

        with patch(metric_path) as mock_metric_cls:
            mock_metric_cls.return_value = MagicMock()
            evaluator = evaluator_cls(threshold=0.9)
            evaluator._create_metric()
            call_kwargs = mock_metric_cls.call_args[1]
            assert call_kwargs["threshold"] == 0.9

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_include_reason_passed_to_metric(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should pass include_reason to the metric."""
        mock_ollama.return_value = MagicMock()

        with patch(metric_path) as mock_metric_cls:
            mock_metric_cls.return_value = MagicMock()
            evaluator = evaluator_cls(include_reason=False)
            evaluator._create_metric()
            call_kwargs = mock_metric_cls.call_args[1]
            assert call_kwargs["include_reason"] is False


# =============================================================================
# Evaluation tests (common pass/fail/reasoning/metric_name)
# =============================================================================


class TestRAGEvaluation:
    """Tests for evaluation functionality (common across all four)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_passes_when_above_threshold(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should return passed=True when score >= threshold."""
        mock_ollama.return_value = MagicMock()

        with patch(metric_path) as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.score = 0.85
            mock_metric.reason = "Good"
            mock_metric_cls.return_value = mock_metric

            evaluator = evaluator_cls(threshold=0.7)
            result = await evaluator._evaluate_impl(**_eval_kwargs(needs_expected))

            assert result["passed"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    async def test_evaluation_fails_when_below_threshold(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should return passed=False when score < threshold."""
        mock_ollama.return_value = MagicMock()

        with patch(metric_path) as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.score = 0.4
            mock_metric.reason = "Poor"
            mock_metric_cls.return_value = mock_metric

            evaluator = evaluator_cls(threshold=0.7)
            result = await evaluator._evaluate_impl(**_eval_kwargs(needs_expected))

            assert result["passed"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    async def test_reasoning_included_in_result(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should include reasoning in result."""
        mock_ollama.return_value = MagicMock()

        with patch(metric_path) as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.score = 0.75
            mock_metric.reason = "Detailed reasoning text"
            mock_metric_cls.return_value = mock_metric

            evaluator = evaluator_cls()
            result = await evaluator._evaluate_impl(**_eval_kwargs(needs_expected))

            assert result["reasoning"] == "Detailed reasoning text"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    async def test_metric_name_in_result(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should include the correct metric name in result."""
        mock_ollama.return_value = MagicMock()

        with patch(metric_path) as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.score = 0.75
            mock_metric.reason = "Good"
            mock_metric_cls.return_value = mock_metric

            evaluator = evaluator_cls()
            result = await evaluator._evaluate_impl(**_eval_kwargs(needs_expected))

            assert result["metric_name"] == metric_name


# =============================================================================
# Provider support tests
# =============================================================================


class TestRAGProviderSupport:
    """Tests for multi-provider support (common across all four)."""

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.GPTModel")
    def test_openai_provider(
        self,
        mock_gpt: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should support OpenAI provider."""
        mock_gpt.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.OPENAI,
            model_name="gpt-4o",
        )
        evaluator = evaluator_cls(model_config=config)

        assert evaluator._model_config.provider == ProviderEnum.OPENAI

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.AnthropicModel")
    def test_anthropic_provider(
        self,
        mock_anthropic: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should support Anthropic provider."""
        mock_anthropic.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.ANTHROPIC,
            model_name="claude-3-5-sonnet-latest",
        )
        evaluator = evaluator_cls(model_config=config)

        assert evaluator._model_config.provider == ProviderEnum.ANTHROPIC

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    def test_ollama_provider_default(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should support Ollama provider (default)."""
        mock_ollama.return_value = MagicMock()

        evaluator = evaluator_cls()

        assert evaluator._model_config.provider == ProviderEnum.OLLAMA

    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.AzureOpenAIModel")
    def test_azure_openai_provider(
        self,
        mock_azure: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should support Azure OpenAI provider."""
        mock_azure.return_value = MagicMock()

        config = DeepEvalModelConfig(
            provider=ProviderEnum.AZURE_OPENAI,
            model_name="gpt-4o",
            api_key="test-key",
            endpoint="https://test.openai.azure.com/",
            deployment_name="test-deployment",
        )
        evaluator = evaluator_cls(model_config=config)

        assert evaluator._model_config.provider == ProviderEnum.AZURE_OPENAI


# =============================================================================
# Required inputs tests
# =============================================================================


class TestRAGRequiredInputs:
    """Tests for required inputs validation (common across all four)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    async def test_works_with_all_required_inputs(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Should work when all required inputs are provided."""
        mock_ollama.return_value = MagicMock()

        with patch(metric_path) as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.score = 0.9
            mock_metric.reason = "Good"
            mock_metric_cls.return_value = mock_metric

            evaluator = evaluator_cls()

            kwargs: dict[str, Any] = {
                "input": "What is the capital?",
                "actual_output": "Paris is the capital of France.",
                "retrieval_context": ["Paris is the capital of France."],
            }
            if needs_expected:
                kwargs["expected_output"] = "Paris is the capital of France."

            result = await evaluator._evaluate_impl(**kwargs)

            assert result["score"] == 0.9
            mock_metric.measure.assert_called_once()


# =============================================================================
# Public evaluate() tests
# =============================================================================


class TestRAGPublicEvaluate:
    """Tests for the public evaluate() method (common across all four)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "evaluator_cls,metric_path,metric_name,needs_expected",
        RAG_EVALUATORS,
    )
    @patch("deepeval.models.OllamaModel")
    async def test_evaluate_calls_evaluate_impl(
        self,
        mock_ollama: MagicMock,
        evaluator_cls: type,
        metric_path: str,
        metric_name: str,
        needs_expected: bool,
    ) -> None:
        """Public evaluate() should call _evaluate_impl()."""
        mock_ollama.return_value = MagicMock()

        with patch(metric_path) as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.score = 0.9
            mock_metric.reason = "Excellent"
            mock_metric_cls.return_value = mock_metric

            evaluator = evaluator_cls()

            result = await evaluator.evaluate(**_eval_kwargs(needs_expected))

            assert result["score"] == 0.9
            assert result["passed"] is True
            mock_metric.measure.assert_called_once()


# =============================================================================
# Evaluator-specific scenario tests
# =============================================================================


class TestFaithfulnessSpecificScenarios:
    """Faithfulness-specific evaluation scenarios."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.faithfulness.FaithfulnessMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_high_score_faithful_response(
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
    async def test_low_score_hallucination(
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


class TestContextualRelevancySpecificScenarios:
    """ContextualRelevancy-specific evaluation scenarios."""

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_relevancy.ContextualRelevancyMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_all_chunks_relevant(
        self, mock_ollama: MagicMock, mock_metric_cls: MagicMock
    ) -> None:
        """Should return 1.0 when all chunks are relevant."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 1.0
        mock_metric.reason = "All retrieved chunks are relevant to the query"
        mock_metric_cls.return_value = mock_metric

        evaluator = ContextualRelevancyEvaluator(threshold=0.6)

        result = await evaluator._evaluate_impl(
            input="What is the pricing?",
            actual_output="Basic plan is $10/month.",
            retrieval_context=["Pricing: Basic $10, Pro $25"],
        )

        assert result["score"] == 1.0
        mock_metric.measure.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_relevancy.ContextualRelevancyMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_partial_relevance(
        self, mock_ollama: MagicMock, mock_metric_cls: MagicMock
    ) -> None:
        """Should return proportion when some chunks are relevant."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.5
        mock_metric.reason = "Only 1 of 2 chunks is relevant"
        mock_metric_cls.return_value = mock_metric

        evaluator = ContextualRelevancyEvaluator(threshold=0.6)

        result = await evaluator._evaluate_impl(
            input="What is the pricing?",
            actual_output="Basic plan is $10/month.",
            retrieval_context=[
                "Pricing: Basic $10, Pro $25",
                "Company founded in 2020",
            ],
        )

        assert result["score"] == 0.5

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_relevancy.ContextualRelevancyMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_no_relevant_chunks(
        self, mock_ollama: MagicMock, mock_metric_cls: MagicMock
    ) -> None:
        """Should return low score when no chunks are relevant."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.0
        mock_metric.reason = "No chunks are relevant to the query"
        mock_metric_cls.return_value = mock_metric

        evaluator = ContextualRelevancyEvaluator(threshold=0.6)

        result = await evaluator._evaluate_impl(
            input="What is the pricing?",
            actual_output="Basic plan is $10/month.",
            retrieval_context=[
                "Company founded in 2020",
                "Office located in NYC",
            ],
        )

        assert result["score"] == 0.0


class TestContextualPrecisionSpecificScenarios:
    """ContextualPrecision-specific evaluation scenarios."""

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_good_ranking(
        self, mock_ollama: MagicMock, mock_metric_cls: MagicMock
    ) -> None:
        """Should return high score when relevant chunks are ranked first."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 1.0
        mock_metric.reason = "Relevant chunks are ranked first"
        mock_metric_cls.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator(threshold=0.7)

        result = await evaluator._evaluate_impl(
            input="What is X?",
            actual_output="X is the definition.",
            expected_output="X is the correct definition.",
            retrieval_context=[
                "X is the definition",
                "Irrelevant info",
            ],
        )

        assert result["score"] == 1.0
        mock_metric.measure.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "holodeck.lib.evaluators.deepeval.contextual_precision.ContextualPrecisionMetric"
    )
    @patch("deepeval.models.OllamaModel")
    async def test_poor_ranking(
        self, mock_ollama: MagicMock, mock_metric_cls: MagicMock
    ) -> None:
        """Should return low score when irrelevant chunks are ranked first."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.3
        mock_metric.reason = "Irrelevant chunks ranked before relevant ones"
        mock_metric_cls.return_value = mock_metric

        evaluator = ContextualPrecisionEvaluator(threshold=0.7)

        result = await evaluator._evaluate_impl(
            input="What is X?",
            actual_output="X is the definition.",
            expected_output="X is the correct definition.",
            retrieval_context=[
                "Irrelevant info",
                "X is the definition",
            ],
        )

        assert result["score"] == 0.3


class TestContextualRecallSpecificScenarios:
    """ContextualRecall-specific evaluation scenarios."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.contextual_recall.ContextualRecallMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_complete_retrieval(
        self, mock_ollama: MagicMock, mock_metric_cls: MagicMock
    ) -> None:
        """Should return high score when all facts are retrieved."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 1.0
        mock_metric.reason = "All facts in expected output are in retrieval context"
        mock_metric_cls.return_value = mock_metric

        evaluator = ContextualRecallEvaluator(threshold=0.8)

        result = await evaluator._evaluate_impl(
            input="List all features",
            actual_output="Features are A, B, and C",
            expected_output="Features are A, B, and C",
            retrieval_context=[
                "Feature A: description",
                "Feature B: description",
                "Feature C: description",
            ],
        )

        assert result["score"] == 1.0
        mock_metric.measure.assert_called_once()

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.contextual_recall.ContextualRecallMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_incomplete_retrieval(
        self, mock_ollama: MagicMock, mock_metric_cls: MagicMock
    ) -> None:
        """Should return lower score when some facts are missing."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.67
        mock_metric.reason = "Missing Feature C in retrieval context"
        mock_metric_cls.return_value = mock_metric

        evaluator = ContextualRecallEvaluator(threshold=0.8)

        result = await evaluator._evaluate_impl(
            input="List all features",
            actual_output="Features are A and B",
            expected_output="Features are A, B, and C",
            retrieval_context=[
                "Feature A: description",
                "Feature B: description",
            ],
        )

        assert result["score"] == 0.67

    @pytest.mark.asyncio
    @patch("holodeck.lib.evaluators.deepeval.contextual_recall.ContextualRecallMetric")
    @patch("deepeval.models.OllamaModel")
    async def test_no_facts_retrieved(
        self, mock_ollama: MagicMock, mock_metric_cls: MagicMock
    ) -> None:
        """Should return low score when no relevant facts are retrieved."""
        mock_ollama.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.score = 0.0
        mock_metric.reason = "No facts from expected output found in retrieval context"
        mock_metric_cls.return_value = mock_metric

        evaluator = ContextualRecallEvaluator(threshold=0.8)

        result = await evaluator._evaluate_impl(
            input="List all features",
            actual_output="Features are...",
            expected_output="Features are A, B, and C",
            retrieval_context=[
                "Unrelated info 1",
                "Unrelated info 2",
            ],
        )

        assert result["score"] == 0.0
