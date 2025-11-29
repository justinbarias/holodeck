"""Custom exceptions for DeepEval evaluators.

This module defines exception classes specific to DeepEval-based evaluation,
including provider compatibility errors and DeepEval library errors.
"""

from typing import Any

from holodeck.lib.errors import EvaluationError


class ProviderNotSupportedError(EvaluationError):
    """Raised when an evaluator is used with an incompatible LLM provider.

    This error is raised early during evaluator initialization to prevent
    confusing runtime errors when users misconfigure provider settings.

    Attributes:
        evaluator_type: The type of evaluator that requires specific providers
        configured_provider: The provider that was incorrectly configured
        supported_providers: List of providers that are supported
    """

    def __init__(
        self,
        message: str,
        evaluator_type: str,
        configured_provider: str,
        supported_providers: list[str],
    ) -> None:
        """Initialize ProviderNotSupportedError with context.

        Args:
            message: Human-readable error message
            evaluator_type: The evaluator class that raised the error
            configured_provider: The provider that was configured
            supported_providers: List of valid provider names
        """
        super().__init__(message)
        self.evaluator_type = evaluator_type
        self.configured_provider = configured_provider
        self.supported_providers = supported_providers


class DeepEvalError(EvaluationError):
    """Wraps errors from the DeepEval library with additional context.

    This exception provides debugging information when DeepEval metrics
    fail, including the metric name and a summary of the test case
    that triggered the error.

    Attributes:
        metric_name: Name of the DeepEval metric that failed
        original_error: The underlying exception from DeepEval
        test_case_summary: Truncated input/output data for debugging
    """

    def __init__(
        self,
        message: str,
        metric_name: str,
        original_error: Exception | None = None,
        test_case_summary: dict[str, Any] | None = None,
    ) -> None:
        """Initialize DeepEvalError with context.

        Args:
            message: Human-readable error message
            metric_name: Name of the metric that failed
            original_error: The underlying exception from DeepEval
            test_case_summary: Dictionary with truncated test case fields
        """
        super().__init__(message)
        self.metric_name = metric_name
        self.original_error = original_error
        self.test_case_summary = test_case_summary or {}
