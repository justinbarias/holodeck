"""Azure AI Evaluation SDK integration for test case evaluation.

This module provides evaluators using Azure AI Evaluation SDK for AI-assisted
quality metrics (groundedness, relevance, coherence, fluency, similarity).

Implements per-metric model configuration for cost optimization:
- Critical metrics (groundedness, safety): GPT-4o (expensive, high quality)
- General metrics (relevance, coherence, fluency): GPT-4o-mini (cheaper)

References:
- Research: specs/006-agent-test-execution/research/
  test-execution-integration-research.md Lines 809-1384
- Research: specs/006-agent-test-execution/research.md Lines 107-165
- Azure AI Evaluation SDK:
  https://learn.microsoft.com/azure/ai-foundry/how-to/develop/evaluate-sdk
"""

from typing import Any, ClassVar

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation import CoherenceEvaluator as AzureCoherenceEvaluator
from azure.ai.evaluation import FluencyEvaluator as AzureFluencyEvaluator
from azure.ai.evaluation import GroundednessEvaluator as AzureGroundednessEvaluator
from azure.ai.evaluation import RelevanceEvaluator as AzureRelevanceEvaluator
from azure.ai.evaluation import SimilarityEvaluator as AzureSimilarityEvaluator
from pydantic import BaseModel, Field

from holodeck.lib.evaluators.base import BaseEvaluator, RetryConfig
from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec


class ModelConfig(BaseModel):
    """Azure OpenAI model configuration for evaluators.

    Attributes:
        azure_endpoint: Azure OpenAI endpoint URL
        api_key: Azure OpenAI API key
        azure_deployment: Azure deployment name (e.g., "gpt-4o", "gpt-4o-mini")
        api_version: Azure OpenAI API version (default: "2024-02-15-preview")

    Example:
        >>> config = ModelConfig(
        ...     azure_endpoint="https://my-resource.openai.azure.com/",
        ...     api_key="my-api-key",
        ...     azure_deployment="gpt-4o"
        ... )
    """

    azure_endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    azure_deployment: str = Field(..., description="Azure deployment name")
    api_version: str = Field(
        default="2024-02-15-preview", description="Azure OpenAI API version"
    )


class AzureAIEvaluator(BaseEvaluator):
    """Base class for Azure AI Evaluation SDK evaluators.

    Provides common functionality for all Azure AI evaluators:
    - Model configuration
    - Retry logic with exponential backoff
    - Timeout handling
    - Score normalization (5-point scale to 0-1)

    Attributes:
        model_config: Azure OpenAI model configuration
        timeout: Timeout in seconds (default: 60s)
        retry_config: Retry configuration with exponential backoff

    Example:
        >>> config = ModelConfig(
        ...     azure_endpoint="https://test.openai.azure.com/",
        ...     api_key="key",
        ...     azure_deployment="gpt-4o-mini"
        ... )
        >>> evaluator = RelevanceEvaluator(model_config=config)
        >>> result = await evaluator.evaluate(query="test", response="answer")
    """

    def __init__(
        self,
        model_config: ModelConfig,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize Azure AI evaluator.

        Args:
            model_config: Azure OpenAI model configuration
            timeout: Timeout in seconds (default: 60s, None for no timeout)
            retry_config: Retry configuration (uses defaults if not provided)
        """
        super().__init__(timeout=timeout, retry_config=retry_config)
        self.model_config = model_config

    def _create_azure_config(self) -> AzureOpenAIModelConfiguration:
        """Create Azure OpenAI model configuration for SDK.

        Returns:
            AzureOpenAIModelConfiguration instance
        """
        return AzureOpenAIModelConfiguration(
            azure_endpoint=self.model_config.azure_endpoint,
            api_key=self.model_config.api_key,
            azure_deployment=self.model_config.azure_deployment,
            api_version=self.model_config.api_version,
        )

    def _normalize_score(self, score: float, scale: float = 5.0) -> float:
        """Normalize score from Azure scale (1-5) to 0-1.

        Args:
            score: Score from Azure evaluator (typically 1-5 scale)
            scale: Maximum value of the scale (default: 5.0)

        Returns:
            Normalized score (0.0-1.0)

        Example:
            >>> evaluator._normalize_score(4.0)  # 4.0 / 5.0
            0.8
            >>> evaluator._normalize_score(5.0)  # 5.0 / 5.0
            1.0
        """
        return score / scale

    async def _call_azure_evaluator(
        self, evaluator: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Call Azure evaluator and return raw result.

        This method wraps the synchronous Azure SDK evaluator call.
        Override in subclasses if needed for specific evaluator types.

        Args:
            evaluator: Azure SDK evaluator instance
            **kwargs: Parameters for evaluator (query, response, context, etc.)

        Returns:
            Raw result from Azure evaluator

        Raises:
            Exception: Any errors from Azure SDK
        """
        # Azure SDK evaluators are synchronous, not async
        # Call them directly and return the result
        result: dict[str, Any] = evaluator(**kwargs)
        return result


class GroundednessEvaluator(AzureAIEvaluator):
    """Groundedness evaluator using Azure AI Evaluation SDK.

    Assesses correspondence between claims in AI-generated answers and source context.
    Measures factual accuracy by verifying that all claims in the response are
    supported by the provided context.

    Query parameter is optional but recommended for better accuracy.

    Scale: 1-5 (normalized to 0.0-1.0)

    Example:
        >>> config = ModelConfig(
        ...     azure_endpoint="https://test.openai.azure.com/",
        ...     api_key="key",
        ...     azure_deployment="gpt-4o"  # Use expensive model for critical metric
        ... )
        >>> evaluator = GroundednessEvaluator(model_config=config)
        >>> result = await evaluator.evaluate(
        ...     query="What is the capital?",
        ...     response="The capital is Paris.",
        ...     context="France's capital is Paris."
        ... )
        >>> print(result["score"])  # 0.0-1.0
        0.95
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE, EvalParam.CONTEXT}),
        optional=frozenset({EvalParam.QUERY}),
        uses_context=True,
    )

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement groundedness evaluation.

        Args:
            **kwargs: Evaluation parameters
                - query (str, optional): User query
                - response (str, required): Agent response to evaluate
                - context (str, required): Ground truth context

        Returns:
            Dictionary containing:
                - score: Normalized score (0.0-1.0)
                - groundedness: Raw score (1-5)
                - reasoning: Explanation of the score

        Raises:
            Exception: Any errors from Azure SDK
        """
        # Create Azure configuration
        azure_config = self._create_azure_config()

        # Create evaluator instance
        evaluator = AzureGroundednessEvaluator(model_config=azure_config)

        # Call Azure evaluator
        result = await self._call_azure_evaluator(evaluator, **kwargs)

        # Normalize score to 0-1 range
        normalized_score = self._normalize_score(result["groundedness"])

        return {
            "score": normalized_score,
            "groundedness": result["groundedness"],
            "reasoning": result.get("reasoning", ""),
        }


class RelevanceEvaluator(AzureAIEvaluator):
    """Relevance evaluator using Azure AI Evaluation SDK.

    Measures relevance of response to query. Assesses whether the response
    directly addresses the user's question or request.

    Scale: 1-5 (normalized to 0.0-1.0)

    Example:
        >>> config = ModelConfig(
        ...     azure_endpoint="https://test.openai.azure.com/",
        ...     api_key="key",
        ...     azure_deployment="gpt-4o"  # Critical metric
        ... )
        >>> evaluator = RelevanceEvaluator(model_config=config)
        >>> result = await evaluator.evaluate(
        ...     query="What is ML?",
        ...     response="ML is machine learning, a subset of AI."
        ... )
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE, EvalParam.QUERY}),
        optional=frozenset({EvalParam.CONTEXT}),
        uses_context=True,
    )

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement relevance evaluation.

        Args:
            **kwargs: Evaluation parameters
                - query (str, required): User query
                - response (str, required): Agent response to evaluate

        Returns:
            Dictionary containing:
                - score: Normalized score (0.0-1.0)
                - relevance: Raw score (1-5)
                - reasoning: Explanation of the score
        """
        azure_config = self._create_azure_config()
        evaluator = AzureRelevanceEvaluator(model_config=azure_config)

        result = await self._call_azure_evaluator(evaluator, **kwargs)

        normalized_score = self._normalize_score(result["relevance"])

        return {
            "score": normalized_score,
            "relevance": result["relevance"],
            "reasoning": result.get("reasoning", ""),
        }


class CoherenceEvaluator(AzureAIEvaluator):
    """Coherence evaluator using Azure AI Evaluation SDK.

    Evaluates logical flow and readability. Measures how well the response
    is organized and whether ideas connect logically.

    Scale: 1-5 (normalized to 0.0-1.0)

    Example:
        >>> config = ModelConfig(
        ...     azure_endpoint="https://test.openai.azure.com/",
        ...     api_key="key",
        ...     azure_deployment="gpt-4o-mini"  # Less critical metric
        ... )
        >>> evaluator = CoherenceEvaluator(model_config=config)
        >>> result = await evaluator.evaluate(
        ...     query="Explain X",
        ...     response="X is... Furthermore... In conclusion..."
        ... )
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE, EvalParam.QUERY}),
    )

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement coherence evaluation.

        Args:
            **kwargs: Evaluation parameters
                - query (str, required): User query
                - response (str, required): Agent response to evaluate

        Returns:
            Dictionary containing:
                - score: Normalized score (0.0-1.0)
                - coherence: Raw score (1-5)
                - reasoning: Explanation of the score
        """
        azure_config = self._create_azure_config()
        evaluator = AzureCoherenceEvaluator(model_config=azure_config)

        result = await self._call_azure_evaluator(evaluator, **kwargs)

        normalized_score = self._normalize_score(result["coherence"])

        return {
            "score": normalized_score,
            "coherence": result["coherence"],
            "reasoning": result.get("reasoning", ""),
        }


class FluencyEvaluator(AzureAIEvaluator):
    """Fluency evaluator using Azure AI Evaluation SDK.

    Assesses language quality. Measures grammar, spelling, punctuation,
    word choice, and sentence structure.

    Scale: 1-5 (normalized to 0.0-1.0)

    Example:
        >>> config = ModelConfig(
        ...     azure_endpoint="https://test.openai.azure.com/",
        ...     api_key="key",
        ...     azure_deployment="gpt-4o-mini"  # Less critical metric
        ... )
        >>> evaluator = FluencyEvaluator(model_config=config)
        >>> result = await evaluator.evaluate(
        ...     query="Test",
        ...     response="This is a well-written response."
        ... )
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE, EvalParam.QUERY}),
    )

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement fluency evaluation.

        Args:
            **kwargs: Evaluation parameters
                - query (str, required): User query
                - response (str, required): Agent response to evaluate

        Returns:
            Dictionary containing:
                - score: Normalized score (0.0-1.0)
                - fluency: Raw score (1-5)
                - reasoning: Explanation of the score
        """
        azure_config = self._create_azure_config()
        evaluator = AzureFluencyEvaluator(model_config=azure_config)

        result = await self._call_azure_evaluator(evaluator, **kwargs)

        normalized_score = self._normalize_score(result["fluency"])

        return {
            "score": normalized_score,
            "fluency": result["fluency"],
            "reasoning": result.get("reasoning", ""),
        }


class SimilarityEvaluator(AzureAIEvaluator):
    """Similarity evaluator using Azure AI Evaluation SDK.

    Compares semantic similarity between response and ground truth.
    Measures how closely the response matches the expected answer.

    Requires ground_truth parameter.

    Scale: 1-5 (normalized to 0.0-1.0)

    Example:
        >>> config = ModelConfig(
        ...     azure_endpoint="https://test.openai.azure.com/",
        ...     api_key="key",
        ...     azure_deployment="gpt-4o-mini"
        ... )
        >>> evaluator = SimilarityEvaluator(model_config=config)
        >>> result = await evaluator.evaluate(
        ...     query="What is 2+2?",
        ...     response="The answer is 4.",
        ...     ground_truth="2+2 equals 4."
        ... )
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset(
            {EvalParam.RESPONSE, EvalParam.QUERY, EvalParam.GROUND_TRUTH}
        ),
    )

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement similarity evaluation.

        Args:
            **kwargs: Evaluation parameters
                - query (str, required): User query
                - response (str, required): Agent response to evaluate
                - ground_truth (str, required): Expected answer

        Returns:
            Dictionary containing:
                - score: Normalized score (0.0-1.0)
                - similarity: Raw score (1-5)
                - reasoning: Explanation of the score
        """
        azure_config = self._create_azure_config()
        evaluator = AzureSimilarityEvaluator(model_config=azure_config)

        result = await self._call_azure_evaluator(evaluator, **kwargs)

        normalized_score = self._normalize_score(result["similarity"])

        return {
            "score": normalized_score,
            "similarity": result["similarity"],
            "reasoning": result.get("reasoning", ""),
        }


# Export public API
__all__ = [
    "ModelConfig",
    "AzureAIEvaluator",
    "GroundednessEvaluator",
    "RelevanceEvaluator",
    "CoherenceEvaluator",
    "FluencyEvaluator",
    "SimilarityEvaluator",
]
