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

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from holodeck.lib.evaluators.deepeval.config import (
    DEFAULT_MODEL_CONFIG,
    DeepEvalModelConfig,
)
from holodeck.lib.evaluators.deepeval.errors import (
    DeepEvalError,
    ProviderNotSupportedError,
)

if TYPE_CHECKING:
    from holodeck.lib.evaluators.deepeval.answer_relevancy import (
        AnswerRelevancyEvaluator,
    )
    from holodeck.lib.evaluators.deepeval.base import DeepEvalBaseEvaluator
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

# Evaluator classes import the DeepEval SDK (the ``eval`` extra), so they load
# on first use. The configuration and error types above do not need it.
_LAZY_EXPORTS = {
    "DeepEvalBaseEvaluator": "base",
    "GEvalEvaluator": "geval",
    "AnswerRelevancyEvaluator": "answer_relevancy",
    "ContextualPrecisionEvaluator": "contextual_precision",
    "ContextualRecallEvaluator": "contextual_recall",
    "ContextualRelevancyEvaluator": "contextual_relevancy",
    "FaithfulnessEvaluator": "faithfulness",
}

DEEPEVAL_FEATURE = "DeepEval metrics"


def require_deepeval() -> None:
    """Raise ``MissingExtraError`` unless the DeepEval SDK is installed."""
    from holodeck.lib.extras import require_extra

    require_extra("deepeval", "eval", DEEPEVAL_FEATURE)


def __getattr__(name: str) -> object:
    """Import an evaluator class on first use, after checking the extra."""
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    require_deepeval()
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
