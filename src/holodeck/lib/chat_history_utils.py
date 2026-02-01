"""Utilities for extracting content from Semantic Kernel ChatHistory.

Provides shared functions for extracting assistant messages and tool calls
from ChatHistory objects, used by both chat and test_runner modules.
"""

from typing import Any

from semantic_kernel.contents import ChatHistory

from holodeck.lib.logging_config import get_logger

logger = get_logger(__name__)


def extract_last_assistant_content(history: ChatHistory) -> str:
    """Extract the last assistant message content from chat history.

    Searches the chat history for the most recent assistant message and
    returns its text content.

    Args:
        history: Semantic Kernel ChatHistory object.

    Returns:
        Content of the last assistant message, or empty string if not found.
    """
    try:
        if not history or not hasattr(history, "messages") or not history.messages:
            return ""

        # Search from end to find last assistant message
        for message in reversed(history.messages):
            if (
                hasattr(message, "role")
                and message.role == "assistant"
                and hasattr(message, "content")
            ):
                content = message.content
                return str(content) if content else ""
        return ""
    except Exception as e:
        logger.warning(f"Failed to extract content from history: {e}")
        return ""


def extract_tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Extract tool names from tool calls list.

    Tool calls are represented as list of dicts with 'name' and 'arguments' keys.

    Args:
        tool_calls: List of tool call dicts from agent.

    Returns:
        List of tool names that were called.
    """
    return [call.get("name", "") for call in tool_calls if "name" in call]
