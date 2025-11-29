# Contracts: Evaluator Interfaces

**Feature**: 012-deepeval-metrics
**Date**: 2025-01-30

## Overview

This document defines the Python interfaces (protocols/abstract classes) for DeepEval evaluators. These contracts ensure consistency across all evaluator implementations and enable proper integration with the existing HoloDeck evaluation framework.

---

## 1. DeepEvalBaseEvaluator Interface

Base class for all DeepEval-based evaluators. Extends `BaseEvaluator`.

```python
from abc import abstractmethod
from typing import Any

from holodeck.lib.evaluators.base import BaseEvaluator, RetryConfig
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig


class DeepEvalBaseEvaluator(BaseEvaluator):
    """Base class for DeepEval metric evaluators.

    All DeepEval evaluators inherit from this class to get:
    - Model configuration and initialization
    - Retry logic with exponential backoff
    - Timeout handling
    - Standard logging

    Subclasses must implement:
    - _create_metric(): Returns the DeepEval metric instance
    - _extract_result(): Extracts and normalizes the metric result
    """

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize DeepEval evaluator.

        Args:
            model_config: LLM judge configuration. Defaults to Ollama gpt-oss:20b.
            threshold: Pass/fail score threshold (0.0-1.0). Default: 0.5.
            timeout: Evaluation timeout in seconds. Default: 60.0.
            retry_config: Retry configuration for transient failures.
        """
        ...

    @abstractmethod
    def _create_metric(self) -> Any:
        """Create and return the DeepEval metric instance.

        Returns:
            DeepEval metric object (e.g., GEval, AnswerRelevancyMetric)
        """
        ...

    @abstractmethod
    def _extract_result(self, metric: Any) -> dict[str, Any]:
        """Extract evaluation result from DeepEval metric.

        Args:
            metric: DeepEval metric after measure() has been called

        Returns:
            Dictionary with at minimum:
                - score: float (0.0-1.0)
                - passed: bool
                - reasoning: str
        """
        ...

    def _build_test_case(self, **kwargs: Any) -> Any:
        """Build DeepEval LLMTestCase from evaluation inputs.

        Args:
            **kwargs: Evaluation parameters:
                - input/query: str (user query)
                - actual_output/response: str (agent response)
                - expected_output/ground_truth: str | None
                - context: str | None
                - retrieval_context: list[str] | None

        Returns:
            LLMTestCase instance

        Raises:
            ValueError: If required parameters are missing
        """
        ...

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement evaluation using DeepEval metric.

        This method is called by BaseEvaluator.evaluate() with retry/timeout.

        Args:
            **kwargs: Evaluation parameters (see _build_test_case)

        Returns:
            Evaluation result dictionary

        Raises:
            DeepEvalError: If DeepEval metric evaluation fails
            ValueError: If required parameters are missing
        """
        ...
```

---

## 2. GEvalEvaluator Interface

Custom criteria evaluator using G-Eval algorithm.

```python
class GEvalEvaluator(DeepEvalBaseEvaluator):
    """G-Eval custom criteria evaluator.

    Evaluates LLM outputs against user-defined criteria using
    chain-of-thought prompting and token probability scoring.

    Example:
        >>> evaluator = GEvalEvaluator(
        ...     name="Professionalism",
        ...     criteria="Evaluate if the response uses professional language",
        ...     evaluation_params=["actual_output"],
        ...     threshold=0.7
        ... )
        >>> result = await evaluator.evaluate(
        ...     input="Help me write an email",
        ...     actual_output="Yo, here's ur email bro..."
        ... )
        >>> print(result["score"])  # Low score due to informal language
    """

    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_params: list[str] | None = None,
        evaluation_steps: list[str] | None = None,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        strict_mode: bool = False,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize G-Eval evaluator.

        Args:
            name: Metric identifier (e.g., "Correctness", "Helpfulness")
            criteria: Natural language evaluation criteria
            evaluation_params: Test case fields to include.
                Valid: ["input", "actual_output", "expected_output",
                        "context", "retrieval_context"]
                Default: ["actual_output"]
            evaluation_steps: Explicit evaluation steps (auto-generated if None)
            model_config: LLM judge configuration
            threshold: Pass/fail score threshold
            strict_mode: If True, score is 1.0 or 0.0 only
            timeout: Evaluation timeout in seconds
            retry_config: Retry configuration
        """
        ...
```

---

## 3. RAG Metric Evaluator Interfaces

### 3.1 AnswerRelevancyEvaluator

```python
class AnswerRelevancyEvaluator(DeepEvalBaseEvaluator):
    """Evaluates relevance of response to user query.

    Required inputs:
        - input: User query
        - actual_output: Agent response

    Example:
        >>> evaluator = AnswerRelevancyEvaluator(threshold=0.7)
        >>> result = await evaluator.evaluate(
        ...     input="What is the return policy?",
        ...     actual_output="We offer 30-day returns with full refund."
        ... )
        >>> print(result["score"])  # High relevance score
    """

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        ...
```

### 3.2 FaithfulnessEvaluator

```python
class FaithfulnessEvaluator(DeepEvalBaseEvaluator):
    """Detects hallucinations by comparing response to retrieval context.

    Required inputs:
        - input: User query
        - actual_output: Agent response
        - retrieval_context: List of retrieved text chunks

    Example:
        >>> evaluator = FaithfulnessEvaluator(threshold=0.8)
        >>> result = await evaluator.evaluate(
        ...     input="What are the store hours?",
        ...     actual_output="Store is open 24/7.",
        ...     retrieval_context=["Store hours: Mon-Fri 9am-5pm"]
        ... )
        >>> print(result["score"])  # Low score (hallucination detected)
    """

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        ...
```

### 3.3 ContextualRelevancyEvaluator

```python
class ContextualRelevancyEvaluator(DeepEvalBaseEvaluator):
    """Measures relevance of retrieved context to user query.

    Required inputs:
        - input: User query
        - actual_output: Agent response
        - retrieval_context: List of retrieved text chunks

    Returns proportion of chunks that are relevant to the query.

    Example:
        >>> evaluator = ContextualRelevancyEvaluator(threshold=0.6)
        >>> result = await evaluator.evaluate(
        ...     input="What is the pricing?",
        ...     actual_output="Basic plan is $10/month.",
        ...     retrieval_context=[
        ...         "Pricing: Basic $10, Pro $25",  # Relevant
        ...         "Company founded in 2020",       # Irrelevant
        ...     ]
        ... )
        >>> print(result["score"])  # 0.5 (1 of 2 chunks relevant)
    """

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        ...
```

### 3.4 ContextualPrecisionEvaluator

```python
class ContextualPrecisionEvaluator(DeepEvalBaseEvaluator):
    """Evaluates ranking quality of retrieved chunks.

    Measures whether relevant chunks appear before irrelevant ones.

    Required inputs:
        - input: User query
        - actual_output: Agent response
        - expected_output: Ground truth answer
        - retrieval_context: List of retrieved text chunks (order matters)

    Example:
        >>> evaluator = ContextualPrecisionEvaluator(threshold=0.7)
        >>> result = await evaluator.evaluate(
        ...     input="What is X?",
        ...     actual_output="X is...",
        ...     expected_output="X is the correct definition.",
        ...     retrieval_context=[
        ...         "Irrelevant info",      # Bad: irrelevant first
        ...         "X is the definition",  # Good: relevant
        ...     ]
        ... )
        >>> print(result["score"])  # Lower due to poor ranking
    """

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        ...
```

### 3.5 ContextualRecallEvaluator

```python
class ContextualRecallEvaluator(DeepEvalBaseEvaluator):
    """Measures retrieval completeness against expected output.

    Evaluates whether retrieval context contains all facts needed
    to produce the expected output.

    Required inputs:
        - input: User query
        - actual_output: Agent response
        - expected_output: Ground truth answer
        - retrieval_context: List of retrieved text chunks

    Example:
        >>> evaluator = ContextualRecallEvaluator(threshold=0.8)
        >>> result = await evaluator.evaluate(
        ...     input="List all features",
        ...     actual_output="Features are A and B",
        ...     expected_output="Features are A, B, and C",
        ...     retrieval_context=["Feature A: ...", "Feature B: ..."]
        ... )
        >>> print(result["score"])  # ~0.67 (missing Feature C)
    """

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        ...
```

---

## 4. Model Configuration Interface

```python
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ProviderEnum(str, Enum):
    """Supported LLM providers for DeepEval."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class DeepEvalModelConfig(BaseModel):
    """Configuration for DeepEval LLM judge model.

    Supports multiple providers with appropriate validation for each.

    Example (OpenAI):
        >>> config = DeepEvalModelConfig(
        ...     provider=ProviderEnum.OPENAI,
        ...     model_name="gpt-4o",
        ...     api_key="sk-..."
        ... )

    Example (Ollama - default):
        >>> config = DeepEvalModelConfig(
        ...     provider=ProviderEnum.OLLAMA,
        ...     model_name="gpt-oss:20b"
        ... )
    """

    provider: ProviderEnum = Field(
        default=ProviderEnum.OLLAMA,
        description="LLM provider"
    )
    model_name: str = Field(
        default="gpt-oss:20b",
        description="Model name or identifier"
    )
    api_key: str | None = Field(
        default=None,
        description="API key (required for cloud providers)"
    )
    endpoint: str | None = Field(
        default=None,
        description="API endpoint (required for Azure, optional for Ollama)"
    )
    api_version: str | None = Field(
        default="2024-02-15-preview",
        description="API version (Azure OpenAI only)"
    )
    deployment_name: str | None = Field(
        default=None,
        description="Deployment name (Azure OpenAI only)"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Model temperature (0.0 for deterministic)"
    )

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "DeepEvalModelConfig":
        """Validate required fields based on provider."""
        ...

    def to_deepeval_model(self) -> Any:
        """Convert to DeepEval native model instance.

        Uses DeepEval's built-in model classes - no custom implementation required:
        - GPTModel for OpenAI
        - AzureOpenAIModel for Azure OpenAI
        - AnthropicModel for Anthropic
        - OllamaModel for Ollama

        Returns:
            Native DeepEval model instance (GPTModel, AzureOpenAIModel,
            AnthropicModel, or OllamaModel)

        Example:
            >>> config = DeepEvalModelConfig(provider=ProviderEnum.OLLAMA, model_name="gpt-oss:20b")
            >>> model = config.to_deepeval_model()
            >>> # Returns: OllamaModel(model="gpt-oss:20b", base_url="http://localhost:11434", temperature=0)
        """
        from deepeval.models import GPTModel, AzureOpenAIModel, AnthropicModel, OllamaModel

        if self.provider == ProviderEnum.OPENAI:
            return GPTModel(model=self.model_name, temperature=self.temperature)
        elif self.provider == ProviderEnum.AZURE_OPENAI:
            return AzureOpenAIModel(
                model_name=self.model_name,
                deployment_name=self.deployment_name,
                azure_endpoint=self.endpoint,
                azure_openai_api_key=self.api_key,
                openai_api_version=self.api_version,
                temperature=self.temperature
            )
        elif self.provider == ProviderEnum.ANTHROPIC:
            return AnthropicModel(model=self.model_name, temperature=self.temperature)
        elif self.provider == ProviderEnum.OLLAMA:
            return OllamaModel(
                model=self.model_name,
                base_url=self.endpoint or "http://localhost:11434",
                temperature=self.temperature
            )
        ...
```

---

## 5. Error Types

```python
from holodeck.lib.evaluators.base import EvaluationError


class ProviderNotSupportedError(EvaluationError):
    """Raised when evaluator is used with incompatible provider.

    Attributes:
        evaluator_type: Name of the evaluator class
        configured_provider: Provider that was configured
        supported_providers: List of supported providers
    """

    def __init__(
        self,
        message: str,
        evaluator_type: str,
        configured_provider: str,
        supported_providers: list[str],
    ) -> None:
        ...


class DeepEvalError(EvaluationError):
    """Wraps errors from DeepEval library.

    Attributes:
        metric_name: Name of the DeepEval metric that failed
        original_error: Original exception from DeepEval
        test_case_summary: Truncated input/output for debugging
    """

    def __init__(
        self,
        message: str,
        metric_name: str,
        original_error: Exception | None = None,
        test_case_summary: dict[str, str] | None = None,
    ) -> None:
        ...
```

---

## 6. Return Type Contract

All evaluators return a consistent structure:

```python
from typing import TypedDict, Any


class EvaluationResult(TypedDict):
    """Standard return type for all evaluators."""

    score: float           # 0.0 - 1.0 normalized score
    passed: bool           # score >= threshold
    reasoning: str         # LLM-generated explanation
    metric_name: str       # e.g., "GEval", "AnswerRelevancy"
    threshold: float       # Configured threshold
    raw_score: float | None  # Original score (if different from normalized)
    evaluation_steps: list[str] | None  # Steps used (G-Eval only)
    metadata: dict[str, Any]  # Metric-specific additional data
```

---

## 7. Integration Points

### 7.1 Test Runner Integration

DeepEval evaluators integrate with existing test runner via `BaseEvaluator.evaluate()`:

```python
# In test_runner/executor.py
async def run_evaluation(
    evaluator: BaseEvaluator,
    test_case: TestCaseConfig,
) -> EvaluationResult:
    """Run evaluation using any BaseEvaluator subclass."""
    return await evaluator.evaluate(
        input=test_case.input,
        actual_output=agent_response,
        expected_output=test_case.ground_truth,
        retrieval_context=test_case.retrieval_context,
    )
```

### 7.2 YAML Configuration

Evaluators are configured via agent YAML:

```yaml
evaluations:
  model:
    provider: ollama
    name: gpt-oss:20b
  metrics:
    - type: geval
      name: Helpfulness
      criteria: "Evaluate if the response is helpful and actionable"
      threshold: 0.7
    - type: answer_relevancy
      threshold: 0.6
    - type: faithfulness
      threshold: 0.8
```
