"""needs_nodejs is derived from stdio MCP server command fields, not provider.

Model note: MCPTool.command is a CommandType enum (str-enum with values
"node", "npx", "uvx", "docker") — not a list[str] as the task spec assumed.
_agent_needs_nodejs inspects the string value of the enum, so it also handles
any absolute-path variants via a final path-component split.
"""

from __future__ import annotations

import pytest

from holodeck.cli.commands.deploy import _agent_needs_nodejs
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import CommandType, MCPTool, TransportType


def _make_mcp_tool(name: str, command: CommandType) -> MCPTool:
    """Construct a valid stdio MCPTool with the given command."""
    return MCPTool(
        name=name,
        description=f"{name} mcp server",
        type="mcp",
        transport=TransportType.STDIO,
        command=command,
    )


def _claude_agent_with_tools(tools: list) -> Agent:
    return Agent(
        name="t",
        description="test agent",
        model=LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-6"),
        instructions=Instructions(inline="You are a helpful assistant."),
        tools=tools,
    )


@pytest.mark.unit
def test_no_mcp_tools_means_no_nodejs() -> None:
    """Anthropic agent with no tools does not need Node."""
    agent = _claude_agent_with_tools([])
    assert _agent_needs_nodejs(agent) is False


@pytest.mark.unit
def test_uvx_mcp_server_means_no_nodejs() -> None:
    """A Python/uvx-launched MCP server does not require Node."""
    agent = _claude_agent_with_tools(
        [
            _make_mcp_tool("qdrant", CommandType.UVX),
        ]
    )
    assert _agent_needs_nodejs(agent) is False


@pytest.mark.unit
def test_docker_mcp_server_means_no_nodejs() -> None:
    """A Docker-launched MCP server does not require Node."""
    agent = _claude_agent_with_tools(
        [
            _make_mcp_tool("docker_mcp", CommandType.DOCKER),
        ]
    )
    assert _agent_needs_nodejs(agent) is False


@pytest.mark.unit
def test_node_mcp_server_means_nodejs_required() -> None:
    """An MCP server launched via 'node' requires Node."""
    agent = _claude_agent_with_tools(
        [
            _make_mcp_tool("filesystem", CommandType.NODE),
        ]
    )
    assert _agent_needs_nodejs(agent) is True


@pytest.mark.unit
def test_npx_mcp_server_means_nodejs_required() -> None:
    """An MCP server launched via 'npx' requires Node."""
    agent = _claude_agent_with_tools(
        [
            _make_mcp_tool("brave", CommandType.NPX),
        ]
    )
    assert _agent_needs_nodejs(agent) is True


@pytest.mark.unit
def test_mixed_tools_node_and_uvx_means_nodejs_required() -> None:
    """If any tool needs Node, result is True even if others do not."""
    agent = _claude_agent_with_tools(
        [
            _make_mcp_tool("qdrant", CommandType.UVX),
            _make_mcp_tool("filesystem", CommandType.NODE),
        ]
    )
    assert _agent_needs_nodejs(agent) is True


@pytest.mark.unit
def test_none_tools_means_no_nodejs() -> None:
    """Agent with tools=None does not need Node."""
    agent = _claude_agent_with_tools([])
    agent_with_none = agent.model_copy(update={"tools": None})
    assert _agent_needs_nodejs(agent_with_none) is False
