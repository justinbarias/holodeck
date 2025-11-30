"""Parameter specification for evaluator inputs.

This module defines standard parameter names and specifications for evaluator
inputs across all evaluator types (Azure AI, NLP, DeepEval).

The key difference between evaluator types:
- Azure AI & NLP: use response, query, ground_truth
- DeepEval: use actual_output, input, expected_output
"""

from enum import Enum
from typing import NamedTuple


class EvalParam(str, Enum):
    """Standard evaluation parameter names.

    Two naming conventions are supported:
    - Azure AI / NLP: RESPONSE, QUERY, GROUND_TRUTH
    - DeepEval: ACTUAL_OUTPUT, INPUT, EXPECTED_OUTPUT

    Both conventions share CONTEXT and RETRIEVAL_CONTEXT.
    """

    # Azure AI / NLP parameter names
    RESPONSE = "response"
    QUERY = "query"
    GROUND_TRUTH = "ground_truth"

    # DeepEval parameter names
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"

    # Shared parameter names
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"


# DeepEval parameters for type checking
DEEPEVAL_PARAMS = frozenset(
    {EvalParam.INPUT, EvalParam.ACTUAL_OUTPUT, EvalParam.EXPECTED_OUTPUT}
)


class ParamSpec(NamedTuple):
    """Parameter specification for an evaluator.

    Declares which parameters an evaluator requires and optionally accepts,
    plus flags for special context handling.

    Attributes:
        required: Parameters that must be provided for evaluation.
        optional: Parameters that may be provided but aren't required.
        uses_context: Whether file content should be passed as context.
        uses_retrieval_context: Whether retrieval context from tools is needed.

    Example:
        >>> spec = ParamSpec(
        ...     required=frozenset({EvalParam.RESPONSE, EvalParam.QUERY}),
        ...     optional=frozenset({EvalParam.CONTEXT}),
        ...     uses_context=True,
        ... )
    """

    required: frozenset[EvalParam]
    optional: frozenset[EvalParam] = frozenset()
    uses_context: bool = False
    uses_retrieval_context: bool = False

    def uses_deepeval_params(self) -> bool:
        """Check if this spec uses DeepEval parameter naming convention.

        Returns:
            True if any required or optional param is a DeepEval param.
        """
        all_params = self.required | self.optional
        return bool(all_params & DEEPEVAL_PARAMS)
