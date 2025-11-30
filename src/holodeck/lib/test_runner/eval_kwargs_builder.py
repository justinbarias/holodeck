"""Builder for evaluation kwargs based on evaluator parameter specifications.

This module provides type-safe construction of evaluation kwargs based on
what each evaluator actually needs, replacing hardcoded metric name checks.

The key insight is that different evaluator types use different parameter names:
- Azure AI / NLP: response, query, ground_truth, context
- DeepEval: actual_output, input, expected_output, context, retrieval_context

The EvalKwargsBuilder handles this by:
1. Reading the evaluator's PARAM_SPEC to know what params it expects
2. Mapping test case data to the correct parameter names
3. Adding context/retrieval_context based on uses_* flags
"""

from typing import Any

from holodeck.lib.evaluators.base import BaseEvaluator
from holodeck.lib.evaluators.param_spec import EvalParam
from holodeck.lib.logging_config import get_logger

logger = get_logger(__name__)


class EvalKwargsBuilder:
    """Builder for evaluation kwargs based on evaluator specifications.

    Constructs eval_kwargs dictionaries based on:
    1. Evaluator's ParamSpec (required/optional parameters)
    2. Available data (test case inputs, file content, tool results)
    3. Evaluator type (DeepEval vs Azure AI/NLP param names)

    Example:
        >>> builder = EvalKwargsBuilder(
        ...     input_query="What is X?",
        ...     agent_response="X is...",
        ...     ground_truth="X is the answer",
        ...     file_content="Context from files...",
        ...     retrieval_context=["chunk1", "chunk2"],
        ... )
        >>> kwargs = builder.build_for(evaluator)
        >>> result = await evaluator.evaluate(**kwargs)
    """

    def __init__(
        self,
        agent_response: str,
        input_query: str | None = None,
        ground_truth: str | None = None,
        file_content: str | None = None,
        retrieval_context: list[str] | None = None,
    ) -> None:
        """Initialize the kwargs builder.

        Args:
            agent_response: Agent's response text (always required).
            input_query: User's input query.
            ground_truth: Expected ground truth answer.
            file_content: Combined content from processed files.
            retrieval_context: List of retrieved text chunks for RAG metrics.
        """
        self._agent_response = agent_response
        self._input_query = input_query
        self._ground_truth = ground_truth
        self._file_content = file_content
        self._retrieval_context = retrieval_context

    def build_for(self, evaluator: BaseEvaluator) -> dict[str, Any]:
        """Build eval_kwargs for a specific evaluator.

        The method:
        1. Gets the evaluator's PARAM_SPEC
        2. Determines if it uses DeepEval param names (input/actual_output)
           or standard names (query/response)
        3. Builds kwargs with the appropriate keys

        Args:
            evaluator: The evaluator instance to build kwargs for.

        Returns:
            Dictionary of kwargs ready for evaluator.evaluate().
        """
        spec = evaluator.get_param_spec()
        uses_deepeval = spec.uses_deepeval_params()

        kwargs: dict[str, Any] = {}

        # Add response/actual_output (always included)
        if uses_deepeval:
            kwargs["actual_output"] = self._agent_response
        else:
            kwargs["response"] = self._agent_response

        # Add query/input if needed and available
        needs_query = self._should_include(
            EvalParam.QUERY, spec
        ) or self._should_include(EvalParam.INPUT, spec)
        if needs_query and self._input_query:
            if uses_deepeval:
                kwargs["input"] = self._input_query
            else:
                kwargs["query"] = self._input_query

        # Add ground_truth/expected_output if needed and available
        needs_ground_truth = self._should_include(
            EvalParam.GROUND_TRUTH, spec
        ) or self._should_include(EvalParam.EXPECTED_OUTPUT, spec)
        if needs_ground_truth and self._ground_truth:
            if uses_deepeval:
                kwargs["expected_output"] = self._ground_truth
            else:
                kwargs["ground_truth"] = self._ground_truth

        # Add context if evaluator uses it and available
        needs_context = spec.uses_context or self._should_include(
            EvalParam.CONTEXT, spec
        )
        if needs_context and self._file_content:
            kwargs["context"] = self._file_content

        # Add retrieval_context if evaluator uses it and available
        needs_retrieval = spec.uses_retrieval_context or self._should_include(
            EvalParam.RETRIEVAL_CONTEXT, spec
        )
        if needs_retrieval and self._retrieval_context:
            kwargs["retrieval_context"] = self._retrieval_context

        logger.debug(
            f"Built kwargs for {evaluator.name}: "
            f"keys={list(kwargs.keys())}, uses_deepeval={uses_deepeval}"
        )

        return kwargs

    def _should_include(self, param: EvalParam, spec: Any) -> bool:
        """Check if a parameter should be included based on ParamSpec."""
        return param in spec.required or param in spec.optional


def build_retrieval_context_from_tools(
    tool_results: list[dict[str, Any]],
    retrieval_tool_names: set[str],
) -> list[str] | None:
    """Extract retrieval context from tool results.

    Only includes results from tools marked as retrieval tools.

    Args:
        tool_results: List of tool result dicts with 'name' and 'result' keys.
            The 'result' value can be a string, list of strings, or other types.
        retrieval_tool_names: Set of tool names that provide retrieval context.

    Returns:
        List of retrieval context strings, or None if none found.
    """
    context: list[str] = []
    for result in tool_results:
        tool_name = result.get("name", "")
        result_content: Any = result.get("result", "")
        if tool_name in retrieval_tool_names and result_content:
            if isinstance(result_content, str):
                context.append(result_content)
            elif isinstance(result_content, list):
                # Safely convert list items to strings, filtering out empty values
                item: Any
                for item in result_content:
                    if item is not None:
                        str_item = str(item)
                        if str_item:
                            context.append(str_item)

    return context if context else None
