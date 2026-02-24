"""MCP Bridge for Claude Agent SDK.

Translates HoloDeck MCPTool configurations into Claude Agent SDK
McpStdioServerConfig format for subprocess-based MCP servers.
"""

import json
import logging

from claude_agent_sdk.types import McpStdioServerConfig

from holodeck.config.env_loader import load_env_file, substitute_env_vars
from holodeck.models.tool import MCPTool, TransportType

logger = logging.getLogger(__name__)


def _resolve_mcp_env(tool: MCPTool) -> dict[str, str]:
    """Resolve environment variables for an MCP tool.

    Precedence (highest to lowest):
    1. Explicit env vars from tool.env (with ${VAR} substitution)
    2. Variables loaded from tool.env_file
    3. Process environment (for ${VAR} substitution only)

    Args:
        tool: MCP tool configuration.

    Returns:
        Dictionary of resolved environment variables.
    """
    resolved: dict[str, str] = {}

    # Load env_file first (lower precedence)
    if tool.env_file:
        resolved.update(load_env_file(tool.env_file))

    # Apply explicit env vars (higher precedence, with substitution)
    if tool.env:
        for key, value in tool.env.items():
            resolved[key] = substitute_env_vars(value)

    # Serialize config to MCP_CONFIG env var if provided
    if tool.config:
        resolved["MCP_CONFIG"] = json.dumps(tool.config)

    return resolved


def build_claude_mcp_configs(
    mcp_tools: list[MCPTool],
) -> dict[str, McpStdioServerConfig]:
    """Translate HoloDeck MCPTool configs to Claude SDK MCP server configs.

    Only stdio transport tools are supported by the Claude subprocess.
    Non-stdio tools are skipped with a warning.

    Args:
        mcp_tools: List of MCPTool configurations from agent YAML.

    Returns:
        Dictionary mapping tool names to McpStdioServerConfig TypedDicts.
    """
    configs: dict[str, McpStdioServerConfig] = {}

    for tool in mcp_tools:
        if tool.transport != TransportType.STDIO:
            logger.warning(
                "Skipping MCP tool '%s': %s transport is not supported by "
                "Claude subprocess (only stdio is supported)",
                tool.name,
                tool.transport.value,
            )
            continue

        command = tool.command.value if tool.command else "npx"
        args = tool.args or []
        env = _resolve_mcp_env(tool)

        entry: McpStdioServerConfig = {
            "command": command,
            "args": args,
        }

        if env:
            entry["env"] = env

        configs[tool.name] = entry

    return configs
