"""DeepEval metrics integration for HoloDeck.

This module provides LLM-as-a-judge evaluation capabilities with multi-provider
support (OpenAI, Azure OpenAI, Anthropic, Ollama).

The DeepEval integration allows users to evaluate agent responses using
industry-standard metrics without being locked into a specific LLM provider.

Key components:
- DeepEvalModelConfig: Configuration adapter for LLM providers
- DeepEvalBaseEvaluator: Abstract base class for DeepEval metrics
- Error classes for handling evaluation failures

Example:
    >>> from holodeck.lib.evaluators.deepeval import (
    ...     DeepEvalModelConfig,
    ...     DeepEvalBaseEvaluator,
    ... )
    >>> config = DeepEvalModelConfig()  # Default: Ollama with gpt-oss:20b
    >>> # Use with specific evaluators (GEvalEvaluator, etc.)
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

__all__ = [
    "DeepEvalBaseEvaluator",
    "DeepEvalModelConfig",
    "DEFAULT_MODEL_CONFIG",
    "DeepEvalError",
    "ProviderNotSupportedError",
]
