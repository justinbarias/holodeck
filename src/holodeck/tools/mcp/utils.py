"""Utility functions for MCP tool module.

Provides helper functions for MCP tool operations.
"""

import re


def normalize_tool_name(name: str) -> str:
    """Normalize tool name by replacing invalid characters with '-'.

    Per Semantic Kernel pattern, tool names must be valid identifiers.
    This method replaces any character that is not alphanumeric or
    underscore with a hyphen.

    Args:
        name: Original tool name from MCP server

    Returns:
        Normalized name safe for use as identifier

    Example:
        >>> normalize_tool_name("read.file")
        'read-file'
        >>> normalize_tool_name("read/write")
        'read-write'
        >>> normalize_tool_name("read_file_v2")
        'read_file_v2'
    """
    return re.sub(r"[^a-zA-Z0-9_]", "-", name)
