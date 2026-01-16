"""Tool filter manager for automatic tool filtering.

This module provides the ToolFilterManager class that orchestrates
tool filtering for agent invocations. It integrates the ToolIndex
with Semantic Kernel's FunctionChoiceBehavior to dynamically filter
tools based on query relevance.

Key responsibilities:
- Initialize tool index from kernel plugins
- Filter tools per request based on semantic similarity
- Create FunctionChoiceBehavior with filtered tool list
- Track tool usage for adaptive optimization
"""

from typing import Any

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.prompt_execution_settings import (
    PromptExecutionSettings,
)

from holodeck.lib.logging_config import get_logger
from holodeck.lib.tool_filter.index import ToolIndex
from holodeck.lib.tool_filter.models import ToolFilterConfig

logger = get_logger(__name__)


class ToolFilterManager:
    """Manages tool filtering for agent invocations.

    Coordinates between the ToolIndex (for semantic search) and
    Semantic Kernel's FunctionChoiceBehavior (for tool filtering)
    to reduce token usage by only including relevant tools.

    Attributes:
        config: ToolFilterConfig with filtering parameters.
        kernel: Semantic Kernel with registered plugins.
        embedding_service: TextEmbedding service for semantic search.
        index: ToolIndex for fast tool searching.
        _initialized: Whether the manager has been initialized.
    """

    def __init__(
        self,
        config: ToolFilterConfig,
        kernel: Kernel,
        embedding_service: Any | None = None,
    ) -> None:
        """Initialize the tool filter manager.

        Args:
            config: Tool filtering configuration.
            kernel: Semantic Kernel with registered plugins.
            embedding_service: Optional TextEmbedding service for semantic search.
        """
        self.config = config
        self.kernel = kernel
        self.embedding_service = embedding_service
        self.index = ToolIndex()
        self._initialized = False

        logger.debug(
            f"ToolFilterManager created: enabled={config.enabled}, "
            f"top_k={config.top_k}, method={config.search_method}"
        )

    async def initialize(
        self,
        defer_loading_map: dict[str, bool] | None = None,
    ) -> None:
        """Initialize the tool index from kernel plugins.

        Must be called after all tools are registered on the kernel
        and before any filtering operations.

        Args:
            defer_loading_map: Optional mapping of tool names to defer_loading flags.
        """
        if self._initialized:
            logger.debug("ToolFilterManager already initialized")
            return

        logger.debug("Initializing ToolFilterManager index")

        await self.index.build_from_kernel(
            kernel=self.kernel,
            embedding_service=self.embedding_service,
            defer_loading_map=defer_loading_map,
        )

        self._initialized = True
        logger.info(f"ToolFilterManager initialized with {len(self.index.tools)} tools")

    async def filter_tools(self, query: str) -> list[str]:
        """Filter tools based on query relevance.

        Returns a list of tool names that should be included in
        the LLM call based on semantic similarity to the query.

        Args:
            query: User query for filtering.

        Returns:
            List of tool full_names to include in the request.
        """
        if not self._initialized:
            logger.warning("ToolFilterManager not initialized, returning all tools")
            return self.index.get_all_tool_names()

        # Start with always_include tools
        included_tools: set[str] = set()

        # Add always_include tools
        for tool_name in self.config.always_include:
            # Match against full_name or just function name
            for full_name in self.index.get_all_tool_names():
                if tool_name == full_name or full_name.endswith(f"-{tool_name}"):
                    included_tools.add(full_name)
                    break

        # Add top-N most used tools
        if self.config.always_include_top_n_used > 0:
            top_used = self.index.get_top_n_used(self.config.always_include_top_n_used)
            for tool in top_used:
                included_tools.add(tool.full_name)

        # Search for relevant tools
        remaining_slots = max(0, self.config.top_k - len(included_tools))

        if remaining_slots > 0:
            search_results = await self.index.search(
                query=query,
                top_k=remaining_slots + len(included_tools),  # Over-fetch to filter
                method=self.config.search_method,
                threshold=self.config.similarity_threshold,
                embedding_service=self.embedding_service,
            )

            for tool, score in search_results:
                if len(included_tools) >= self.config.top_k:
                    break
                # Skip if already included
                if tool.full_name in included_tools:
                    continue
                # Skip deferred tools if below threshold
                if tool.defer_loading and score < self.config.similarity_threshold:
                    continue
                included_tools.add(tool.full_name)
                logger.debug(f"Included tool {tool.full_name} (score={score:.3f})")

        logger.debug(
            f"Filtered tools: {len(included_tools)}/{len(self.index.tools)} "
            f"for query: {query[:50]}..."
        )

        return list(included_tools)

    def create_function_choice_behavior(
        self, filtered_tools: list[str]
    ) -> FunctionChoiceBehavior:
        """Create FunctionChoiceBehavior with filtered tool list.

        Uses Semantic Kernel's native filtering mechanism to restrict
        which functions are available to the LLM.

        Args:
            filtered_tools: List of tool full_names to include.

        Returns:
            FunctionChoiceBehavior configured with the filtered tool list.
        """
        return FunctionChoiceBehavior.Auto(
            filters={"included_functions": filtered_tools}
        )

    async def prepare_execution_settings(
        self,
        query: str,
        base_settings: PromptExecutionSettings | dict[str, PromptExecutionSettings],
    ) -> PromptExecutionSettings | dict[str, PromptExecutionSettings]:
        """Prepare execution settings with filtered tools.

        Filters tools based on the query and creates new execution
        settings with the appropriate FunctionChoiceBehavior.

        Args:
            query: User query for filtering.
            base_settings: Base execution settings to modify.

        Returns:
            Modified execution settings with filtered function choice behavior.
        """
        if not self.config.enabled:
            return base_settings

        # Filter tools
        filtered_tools = await self.filter_tools(query)

        # Create function choice behavior
        function_choice = self.create_function_choice_behavior(filtered_tools)

        # Handle both single settings and dict of settings
        if isinstance(base_settings, dict):
            # Clone and modify each settings object
            modified_settings: dict[str, PromptExecutionSettings] = {}
            for key, settings in base_settings.items():
                cloned = self._clone_settings(settings)
                if hasattr(cloned, "function_choice_behavior"):
                    cloned.function_choice_behavior = function_choice
                modified_settings[key] = cloned
            return modified_settings
        else:
            # Single settings object
            cloned = self._clone_settings(base_settings)
            if hasattr(cloned, "function_choice_behavior"):
                cloned.function_choice_behavior = function_choice
            return cloned

    def _clone_settings(
        self, settings: PromptExecutionSettings
    ) -> PromptExecutionSettings:
        """Create a shallow clone of execution settings.

        Args:
            settings: Settings to clone.

        Returns:
            Cloned settings object.
        """
        # Use model_copy if available (Pydantic v2), otherwise fallback
        if hasattr(settings, "model_copy"):
            return settings.model_copy()
        elif hasattr(settings, "copy"):
            return settings.copy()
        else:
            # Last resort: create new instance with same values
            return type(settings)(**vars(settings))

    def record_tool_usage(self, tool_calls: list[dict[str, Any]]) -> None:
        """Record tool usage for adaptive optimization.

        Updates usage counts in the index based on which tools
        were actually called during agent execution.

        Args:
            tool_calls: List of tool call dicts with 'name' key.
        """
        for call in tool_calls:
            tool_name = call.get("name", "")
            if tool_name:
                self.index.update_usage(tool_name)

    def get_filter_stats(self) -> dict[str, Any]:
        """Get statistics about tool filtering.

        Returns:
            Dictionary with filtering statistics.
        """
        return {
            "enabled": self.config.enabled,
            "total_tools": len(self.index.tools),
            "top_k": self.config.top_k,
            "similarity_threshold": self.config.similarity_threshold,
            "search_method": self.config.search_method,
            "always_include": self.config.always_include,
            "always_include_top_n_used": self.config.always_include_top_n_used,
        }
