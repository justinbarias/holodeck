"""DeepEval metrics integration for HoloDeck.

This module provides LLM-as-a-judge evaluation capabilities with multi-provider
support (OpenAI, Azure OpenAI, Anthropic, Ollama).

The DeepEval integration allows users to evaluate agent responses using
industry-standard metrics without being locked into a specific LLM provider.

Key components:
- DeepEvalModelConfig: Configuration adapter for LLM providers
- DeepEvalBaseEvaluator: Abstract base class for DeepEval metrics
- GEvalEvaluator: Custom criteria evaluator using G-Eval algorithm
- RAG Evaluators: Faithfulness, ContextualRelevancy, ContextualPrecision, Recall
- Error classes for handling evaluation failures

Example:
    >>> from holodeck.lib.evaluators.deepeval import (
    ...     GEvalEvaluator,
    ...     FaithfulnessEvaluator,
    ...     DeepEvalModelConfig,
    ... )
    >>> config = DeepEvalModelConfig()  # Default: Ollama with gpt-oss:20b
    >>> evaluator = GEvalEvaluator(
    ...     name="Helpfulness",
    ...     criteria="Evaluate if the response is helpful",
    ...     model_config=config
    ... )
    >>> rag_evaluator = FaithfulnessEvaluator(threshold=0.8)
"""

from holodeck.lib.evaluators.deepeval.answer_relevancy import AnswerRelevancyEvaluator
from holodeck.lib.evaluators.deepeval.base import DeepEvalBaseEvaluator
from holodeck.lib.evaluators.deepeval.config import (
    DEFAULT_MODEL_CONFIG,
    DeepEvalModelConfig,
)
from holodeck.lib.evaluators.deepeval.contextual_precision import (
    ContextualPrecisionEvaluator,
)
from holodeck.lib.evaluators.deepeval.contextual_recall import ContextualRecallEvaluator
from holodeck.lib.evaluators.deepeval.contextual_relevancy import (
    ContextualRelevancyEvaluator,
)
from holodeck.lib.evaluators.deepeval.errors import (
    DeepEvalError,
    ProviderNotSupportedError,
)
from holodeck.lib.evaluators.deepeval.faithfulness import FaithfulnessEvaluator
from holodeck.lib.evaluators.deepeval.geval import GEvalEvaluator

__all__ = [
    # Base classes
    "DeepEvalBaseEvaluator",
    "DeepEvalModelConfig",
    "DEFAULT_MODEL_CONFIG",
    # Errors
    "DeepEvalError",
    "ProviderNotSupportedError",
    # Evaluators
    "GEvalEvaluator",
    # RAG Evaluators
    "AnswerRelevancyEvaluator",
    "ContextualPrecisionEvaluator",
    "ContextualRecallEvaluator",
    "ContextualRelevancyEvaluator",
    "FaithfulnessEvaluator",
]
