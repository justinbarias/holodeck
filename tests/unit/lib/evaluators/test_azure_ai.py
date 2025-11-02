"""Unit tests for Azure AI Evaluation SDK integration.

Tests cover:
- GroundednessEvaluator, RelevanceEvaluator, CoherenceEvaluator,
  FluencyEvaluator, SimilarityEvaluator
- Per-metric model configuration (global vs per-metric overrides)
- Retry logic with exponential backoff (3 attempts, 2s/4s/8s delays)
- LLM API error handling and timeouts
- Query as optional parameter for GroundednessEvaluator

References:
- specs/006-agent-test-execution/research/test-execution-integration-research.md
  Lines 809-1384 (Azure AI Evaluation SDK section)
- specs/006-agent-test-execution/research.md
  Lines 107-165 (Azure AI Evaluation decision and integration pattern)
"""

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from holodeck.lib.evaluators.azure_ai import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    ModelConfig,
    RelevanceEvaluator,
    SimilarityEvaluator,
)
from holodeck.lib.evaluators.base import EvaluationError, RetryConfig


class TestModelConfig:
    """Test ModelConfig model for Azure OpenAI configuration."""

    def test_default_values(self) -> None:
        """Test default model configuration values."""
        config = ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            azure_deployment="gpt-4o",
        )
        assert config.azure_endpoint == "https://test.openai.azure.com/"
        assert config.api_key == "test-key"
        assert config.azure_deployment == "gpt-4o"
        assert config.api_version == "2024-02-15-preview"  # Default

    def test_custom_api_version(self) -> None:
        """Test custom API version."""
        config = ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
        )
        assert config.api_version == "2024-08-01-preview"

    def test_validation_missing_fields(self) -> None:
        """Test validation fails when required fields are missing."""
        with pytest.raises(ValueError):
            ModelConfig(
                azure_endpoint="https://test.com/", api_key="key"
            )  # Missing deployment

        with pytest.raises(ValueError):
            ModelConfig(azure_deployment="gpt-4o", api_key="key")  # Missing endpoint

        with pytest.raises(ValueError):
            ModelConfig(
                azure_endpoint="https://test.com/", azure_deployment="gpt-4o"
            )  # Missing api_key


class TestAzureAIEvaluator:
    """Test AzureAIEvaluator base class."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create test model configuration."""
        return ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-api-key",
            azure_deployment="gpt-4o-mini",
        )

    def test_initialization_with_model_config(self, model_config: ModelConfig) -> None:
        """Test evaluator initialization with model config."""
        evaluator = GroundednessEvaluator(model_config=model_config)
        assert evaluator.model_config == model_config
        assert evaluator.timeout == 60.0  # Default timeout
        assert evaluator.retry_config.max_retries == 3  # Default retries

    def test_initialization_with_custom_timeout(
        self, model_config: ModelConfig
    ) -> None:
        """Test evaluator initialization with custom timeout."""
        evaluator = GroundednessEvaluator(model_config=model_config, timeout=90.0)
        assert evaluator.timeout == 90.0

    def test_initialization_with_custom_retry_config(
        self, model_config: ModelConfig
    ) -> None:
        """Test evaluator initialization with custom retry config."""
        retry_config = RetryConfig(max_retries=5, base_delay=1.0)
        evaluator = GroundednessEvaluator(
            model_config=model_config, retry_config=retry_config
        )
        assert evaluator.retry_config.max_retries == 5
        assert evaluator.retry_config.base_delay == 1.0


class TestGroundednessEvaluator:
    """Test GroundednessEvaluator implementation."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create test model configuration."""
        return ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-api-key",
            azure_deployment="gpt-4o",
        )

    @pytest.mark.asyncio
    async def test_groundedness_with_query(self, model_config: ModelConfig) -> None:
        """Test groundedness evaluation with query parameter."""
        evaluator = GroundednessEvaluator(model_config=model_config)

        with patch(
            "holodeck.lib.evaluators.azure_ai.GroundednessEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.return_value = {
                "groundedness": 4.0,
                "reasoning": "Test reasoning",
            }

            result = await evaluator.evaluate(
                query="What is the capital of France?",
                response="The capital of France is Paris.",
                context="France is a country in Europe. Paris is its capital.",
            )

            assert result["score"] == 0.8  # 4.0 / 5.0 normalized to 0-1
            assert result["groundedness"] == 4.0
            assert result["reasoning"] == "Test reasoning"
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_groundedness_without_query(self, model_config: ModelConfig) -> None:
        """Test groundedness evaluation without query (optional parameter)."""
        evaluator = GroundednessEvaluator(model_config=model_config)

        with patch(
            "holodeck.lib.evaluators.azure_ai.GroundednessEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.return_value = {
                "groundedness": 3.5,
                "reasoning": "Partial grounding",
            }

            result = await evaluator.evaluate(
                response="The capital of France is Paris.",
                context="France is a country in Europe. Paris is its capital.",
            )

            assert result["score"] == 0.7  # 3.5 / 5.0
            assert result["groundedness"] == 3.5
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_groundedness_name_property(self, model_config: ModelConfig) -> None:
        """Test evaluator name property."""
        evaluator = GroundednessEvaluator(model_config=model_config)
        assert evaluator.name == "GroundednessEvaluator"


class TestRelevanceEvaluator:
    """Test RelevanceEvaluator implementation."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create test model configuration."""
        return ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-api-key",
            azure_deployment="gpt-4o-mini",
        )

    @pytest.mark.asyncio
    async def test_relevance_evaluation(self, model_config: ModelConfig) -> None:
        """Test relevance evaluation."""
        evaluator = RelevanceEvaluator(model_config=model_config)

        with patch(
            "holodeck.lib.evaluators.azure_ai.RelevanceEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.return_value = {"relevance": 5.0, "reasoning": "Highly relevant"}

            result = await evaluator.evaluate(
                query="What is machine learning?",
                response=(
                    "Machine learning is a subset of AI that "
                    "enables computers to learn from data."
                ),
            )

            assert result["score"] == 1.0  # 5.0 / 5.0
            assert result["relevance"] == 5.0
            assert result["reasoning"] == "Highly relevant"

    @pytest.mark.asyncio
    async def test_relevance_name_property(self, model_config: ModelConfig) -> None:
        """Test evaluator name property."""
        evaluator = RelevanceEvaluator(model_config=model_config)
        assert evaluator.name == "RelevanceEvaluator"


class TestCoherenceEvaluator:
    """Test CoherenceEvaluator implementation."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create test model configuration."""
        return ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-api-key",
            azure_deployment="gpt-4o-mini",
        )

    @pytest.mark.asyncio
    async def test_coherence_evaluation(self, model_config: ModelConfig) -> None:
        """Test coherence evaluation."""
        evaluator = CoherenceEvaluator(model_config=model_config)

        with patch(
            "holodeck.lib.evaluators.azure_ai.CoherenceEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.return_value = {
                "coherence": 4.5,
                "reasoning": "Coherent response",
            }

            result = await evaluator.evaluate(
                query="Explain photosynthesis",
                response="Plants convert sunlight into energy through photosynthesis.",
            )

            assert result["score"] == 0.9  # 4.5 / 5.0
            assert result["coherence"] == 4.5

    @pytest.mark.asyncio
    async def test_coherence_name_property(self, model_config: ModelConfig) -> None:
        """Test evaluator name property."""
        evaluator = CoherenceEvaluator(model_config=model_config)
        assert evaluator.name == "CoherenceEvaluator"


class TestFluencyEvaluator:
    """Test FluencyEvaluator implementation."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create test model configuration."""
        return ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-api-key",
            azure_deployment="gpt-4o-mini",
        )

    @pytest.mark.asyncio
    async def test_fluency_evaluation(self, model_config: ModelConfig) -> None:
        """Test fluency evaluation."""
        evaluator = FluencyEvaluator(model_config=model_config)

        with patch(
            "holodeck.lib.evaluators.azure_ai.FluencyEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.return_value = {"fluency": 5.0, "reasoning": "Perfect fluency"}

            result = await evaluator.evaluate(
                query="Test", response="This is a well-written response."
            )

            assert result["score"] == 1.0
            assert result["fluency"] == 5.0

    @pytest.mark.asyncio
    async def test_fluency_name_property(self, model_config: ModelConfig) -> None:
        """Test evaluator name property."""
        evaluator = FluencyEvaluator(model_config=model_config)
        assert evaluator.name == "FluencyEvaluator"


class TestSimilarityEvaluator:
    """Test SimilarityEvaluator implementation."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create test model configuration."""
        return ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-api-key",
            azure_deployment="gpt-4o-mini",
        )

    @pytest.mark.asyncio
    async def test_similarity_evaluation(self, model_config: ModelConfig) -> None:
        """Test similarity evaluation with ground_truth."""
        evaluator = SimilarityEvaluator(model_config=model_config)

        with patch(
            "holodeck.lib.evaluators.azure_ai.SimilarityEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.return_value = {"similarity": 4.0, "reasoning": "Very similar"}

            result = await evaluator.evaluate(
                query="What is 2+2?",
                response="The answer is 4.",
                ground_truth="2+2 equals 4.",
            )

            assert result["score"] == 0.8
            assert result["similarity"] == 4.0

    @pytest.mark.asyncio
    async def test_similarity_name_property(self, model_config: ModelConfig) -> None:
        """Test evaluator name property."""
        evaluator = SimilarityEvaluator(model_config=model_config)
        assert evaluator.name == "SimilarityEvaluator"


class TestPerMetricModelConfiguration:
    """Test per-metric model configuration (critical vs general metrics)."""

    @pytest.mark.asyncio
    async def test_expensive_model_for_groundedness(self) -> None:
        """Test using GPT-4o (expensive) for critical groundedness metric."""
        expensive_config = ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            azure_deployment="gpt-4o",  # Expensive model
        )
        evaluator = GroundednessEvaluator(model_config=expensive_config)

        assert evaluator.model_config.azure_deployment == "gpt-4o"

    @pytest.mark.asyncio
    async def test_cheap_model_for_fluency(self) -> None:
        """Test using GPT-4o-mini (cheap) for less critical fluency metric."""
        cheap_config = ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            azure_deployment="gpt-4o-mini",  # Cheap model
        )
        evaluator = FluencyEvaluator(model_config=cheap_config)

        assert evaluator.model_config.azure_deployment == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_different_models_for_different_metrics(self) -> None:
        """Test that different evaluators can use different models."""
        critical_config = ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            azure_deployment="gpt-4o",
        )
        general_config = ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            azure_deployment="gpt-4o-mini",
        )

        groundedness = GroundednessEvaluator(model_config=critical_config)
        relevance = RelevanceEvaluator(model_config=critical_config)
        coherence = CoherenceEvaluator(model_config=general_config)
        fluency = FluencyEvaluator(model_config=general_config)

        assert groundedness.model_config.azure_deployment == "gpt-4o"
        assert relevance.model_config.azure_deployment == "gpt-4o"
        assert coherence.model_config.azure_deployment == "gpt-4o-mini"
        assert fluency.model_config.azure_deployment == "gpt-4o-mini"


class TestErrorHandling:
    """Test error handling for Azure AI evaluators."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create test model configuration."""
        return ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-api-key",
            azure_deployment="gpt-4o-mini",
        )

    @pytest.mark.asyncio
    async def test_api_error_raises_evaluation_error(
        self, model_config: ModelConfig
    ) -> None:
        """Test that API errors raise EvaluationError after retries."""
        evaluator = GroundednessEvaluator(
            model_config=model_config,
            retry_config=RetryConfig(max_retries=2, base_delay=0.1),
        )

        with patch(
            "holodeck.lib.evaluators.azure_ai.GroundednessEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.side_effect = ConnectionError("API connection failed")

            with pytest.raises(EvaluationError) as exc_info:
                await evaluator.evaluate(query="test", response="test", context="test")

            assert "API connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error_is_retried(self, model_config: ModelConfig) -> None:
        """Test that timeout errors trigger retry."""
        evaluator = RelevanceEvaluator(
            model_config=model_config,
            retry_config=RetryConfig(max_retries=3, base_delay=0.1),
        )

        call_count = 0

        async def mock_call_with_timeout(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Request timeout")
            return {"relevance": 4.0, "reasoning": "Success"}

        with patch(
            "holodeck.lib.evaluators.azure_ai.RelevanceEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.side_effect = mock_call_with_timeout

            result = await evaluator.evaluate(query="test", response="test")

            assert result["score"] == 0.8
            assert call_count == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_validation_error_not_retried(
        self, model_config: ModelConfig
    ) -> None:
        """Test that validation errors (non-retryable) fail immediately."""
        evaluator = CoherenceEvaluator(
            model_config=model_config,
            retry_config=RetryConfig(max_retries=3, base_delay=0.1),
        )

        call_count = 0

        async def mock_call_with_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input parameters")

        with patch(
            "holodeck.lib.evaluators.azure_ai.CoherenceEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.side_effect = mock_call_with_error

            with pytest.raises(EvaluationError):
                await evaluator.evaluate(query="test", response="test")

            # Should not retry on ValueError
            assert call_count == 1


class TestRetryLogic:
    """Test retry logic with exponential backoff for Azure AI evaluators."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create test model configuration."""
        return ModelConfig(
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-api-key",
            azure_deployment="gpt-4o-mini",
        )

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(
        self, model_config: ModelConfig
    ) -> None:
        """Test retry logic with 2s/4s/8s exponential backoff."""
        retry_config = RetryConfig(max_retries=3, base_delay=2.0, exponential_base=2.0)
        evaluator = GroundednessEvaluator(
            model_config=model_config, retry_config=retry_config
        )

        call_count = 0

        async def mock_call_with_failures(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise ConnectionError(f"Transient error (attempt {call_count})")
            return {"groundedness": 4.0, "reasoning": "Success"}

        with patch(
            "holodeck.lib.evaluators.azure_ai.GroundednessEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.side_effect = mock_call_with_failures

            start_time = asyncio.get_event_loop().time()
            result = await evaluator.evaluate(
                query="test", response="test", context="test"
            )
            elapsed_time = asyncio.get_event_loop().time() - start_time

            # Expected delays: 2s (after 1st failure), 4s (after 2nd failure) = 6s total
            assert elapsed_time >= 5.5, f"Elapsed time {elapsed_time}s is too short"
            assert (
                elapsed_time <= 7.5
            ), f"Elapsed time {elapsed_time}s is too long"  # Allow overhead

            assert result["score"] == 0.8
            assert call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, model_config: ModelConfig) -> None:
        """Test that retries are exhausted after max attempts."""
        retry_config = RetryConfig(max_retries=3, base_delay=0.1)
        evaluator = RelevanceEvaluator(
            model_config=model_config, retry_config=retry_config
        )

        call_count = 0

        async def mock_call_always_fails(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"Persistent error (attempt {call_count})")

        with patch(
            "holodeck.lib.evaluators.azure_ai.RelevanceEvaluator._call_azure_evaluator"
        ) as mock_call:
            mock_call.side_effect = mock_call_always_fails

            with pytest.raises(EvaluationError) as exc_info:
                await evaluator.evaluate(query="test", response="test")

            assert "Persistent error" in str(exc_info.value)
            assert call_count == 3  # max_retries attempts
