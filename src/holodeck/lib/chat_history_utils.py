"""Utilities for extracting tool information from agent execution results.

Provides shared functions for extracting tool names from tool call records,
used by the test_runner module.
"""

from typing import Any


def extract_tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Extract tool names from tool calls list.

    Tool calls are represented as list of dicts with 'name' and 'arguments' keys.

    Args:
        tool_calls: List of tool call dicts from agent.

    Returns:
        List of tool names that were called.
    """
    return [call.get("name", "") for call in tool_calls if "name" in call]
