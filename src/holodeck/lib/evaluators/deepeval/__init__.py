"""DeepEval metrics integration for HoloDeck.

This module provides LLM-as-a-judge evaluation capabilities with multi-provider
support (OpenAI, Azure OpenAI, Anthropic, Ollama).

The DeepEval integration allows users to evaluate agent responses using
industry-standard metrics without being locked into a specific LLM provider.

Key components:
- DeepEvalModelConfig: Configuration adapter for LLM providers
- DeepEvalBaseEvaluator: Abstract base class for DeepEval metrics
- GEvalEvaluator: Custom criteria evaluator using G-Eval algorithm
- Error classes for handling evaluation failures

Example:
    >>> from holodeck.lib.evaluators.deepeval import (
    ...     GEvalEvaluator,
    ...     DeepEvalModelConfig,
    ... )
    >>> config = DeepEvalModelConfig()  # Default: Ollama with gpt-oss:20b
    >>> evaluator = GEvalEvaluator(
    ...     name="Helpfulness",
    ...     criteria="Evaluate if the response is helpful",
    ...     model_config=config
    ... )
"""

from holodeck.lib.evaluators.deepeval.base import DeepEvalBaseEvaluator
from holodeck.lib.evaluators.deepeval.config import (
    DEFAULT_MODEL_CONFIG,
    DeepEvalModelConfig,
)
from holodeck.lib.evaluators.deepeval.errors import (
    DeepEvalError,
    ProviderNotSupportedError,
)
from holodeck.lib.evaluators.deepeval.geval import GEvalEvaluator

__all__ = [
    "DeepEvalBaseEvaluator",
    "DeepEvalModelConfig",
    "DEFAULT_MODEL_CONFIG",
    "DeepEvalError",
    "GEvalEvaluator",
    "ProviderNotSupportedError",
]
