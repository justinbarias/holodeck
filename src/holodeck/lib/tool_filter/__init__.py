"""Tool filtering for automatic tool selection.

This package implements Anthropic's Tool Search pattern for reducing
token usage by dynamically filtering tools per request based on
semantic similarity to the user's query.

Key components:
- ToolMetadata: Metadata about tools for indexing and search
- ToolFilterConfig: Configuration for filtering behavior
- ToolIndex: In-memory index for fast tool searching
- ToolFilterManager: Orchestrates filtering for agent invocations

Example usage:
    from holodeck.lib.tool_filter import ToolFilterConfig, ToolFilterManager

    config = ToolFilterConfig(
        enabled=True,
        top_k=5,
        similarity_threshold=0.3,
    )

    manager = ToolFilterManager(config, kernel, embedding_service)
    await manager.initialize()

    filtered_tools = await manager.filter_tools("What's the weather?")
"""

from holodeck.lib.tool_filter.index import ToolIndex
from holodeck.lib.tool_filter.manager import ToolFilterManager
from holodeck.lib.tool_filter.models import ToolFilterConfig, ToolMetadata

__all__ = [
    "ToolMetadata",
    "ToolFilterConfig",
    "ToolIndex",
    "ToolFilterManager",
]
