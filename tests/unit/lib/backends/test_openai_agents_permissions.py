"""Unit tests for the openai_agents `disallowed_tools` config-time filter (F2).

Covers FR-034: named tools are removed from the resolved `Agent.tools` /
`mcp_servers` at build time, and a name appearing in both `allowed_tools` and
`disallowed_tools` fails load.

Matching is on the HoloDeck *config* name (the name as written in YAML), not the
SDK tool name — so a disallowed vectorstore/hier-doc tool is filtered before it
can produce a `{name}_search` SDK tool. SDK interactions are mocked; the
`openai-agents` package is installed (dev extra) so the lazy imports resolve.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.backends.openai_agents_backend import (
    OpenAIAgentsBackend,
    _disallowed_tool_names,
)
from holodeck.lib.backends.openai_agents_mcp import build_mcp_servers
from holodeck.lib.backends.openai_agents_tool_adapters import build_sdk_tools
from holodeck.lib.errors import ConfigError
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.openai_config import OpenAIConfig, OpenAIPermissionsConfig
from holodeck.models.tool import (
    FunctionTool,
    MCPTool,
    TransportType,
    VectorstoreTool,
)


def _agent(
    *,
    tools: list | None = None,
    openai: OpenAIConfig | None = None,
) -> Agent:
    """Build a minimal openai-provider Agent for permission tests."""
    return Agent(
        name="test-agent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="Be helpful."),
        tools=tools,
        openai=openai,
    )


def _func_tool(name: str) -> FunctionTool:
    """A function tool config (the callable is patched via load_function_tool)."""
    return FunctionTool(
        name=name,
        description=f"{name} desc",
        file="tool.py",
        function="run",
    )


def _vectorstore_tool(name: str) -> VectorstoreTool:
    return VectorstoreTool(
        name=name,
        description=f"{name} desc",
        source="docs/",
    )


def _mcp_http_tool(name: str) -> MCPTool:
    return MCPTool(
        name=name,
        description=f"{name} desc",
        transport=TransportType.HTTP,
        url="https://example.test/mcp",
    )


# ---------------------------------------------------------------------------
# _disallowed_tool_names — union of the two disallow sources
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDisallowedToolNames:
    """`openai.disallowed_tools` ∪ `openai.permissions.disallowed_tools`."""

    def test_none_when_no_openai_config(self) -> None:
        assert _disallowed_tool_names(_agent()) == set()

    def test_top_level_disallowed(self) -> None:
        agent = _agent(openai=OpenAIConfig(disallowed_tools=["a", "b"]))
        assert _disallowed_tool_names(agent) == {"a", "b"}

    def test_permissions_disallowed(self) -> None:
        agent = _agent(
            openai=OpenAIConfig(
                permissions=OpenAIPermissionsConfig(disallowed_tools=["c"])
            )
        )
        assert _disallowed_tool_names(agent) == {"c"}

    def test_union_of_both_sources(self) -> None:
        agent = _agent(
            openai=OpenAIConfig(
                disallowed_tools=["a"],
                permissions=OpenAIPermissionsConfig(disallowed_tools=["b"]),
            )
        )
        assert _disallowed_tool_names(agent) == {"a", "b"}


# ---------------------------------------------------------------------------
# build_sdk_tools — config-time filter by HoloDeck config name
# ---------------------------------------------------------------------------


def _stub_callable() -> str:
    return "ok"


@pytest.mark.unit
class TestBuildSdkToolsDisallowed:
    """Disallowed function / vectorstore tools never reach the SDK tool list."""

    def test_disallowed_function_tool_absent(self) -> None:
        configs = [_func_tool("keep"), _func_tool("drop")]
        with patch(
            "holodeck.lib.backends.openai_agents_tool_adapters.load_function_tool",
            return_value=_stub_callable,
        ):
            tools = build_sdk_tools(configs, base_dir=None, disallowed={"drop"})
        names = {t.name for t in tools}
        assert names == {"keep"}

    def test_disallowed_vectorstore_skips_search_tool(self) -> None:
        configs = [_vectorstore_tool("docs")]
        instances = {"docs": MagicMock()}
        tools = build_sdk_tools(
            configs,
            base_dir=None,
            tool_instances=instances,
            disallowed={"docs"},
        )
        # Matching is on the config name ("docs"), so the "docs_search" SDK
        # tool is never constructed.
        assert tools == []

    def test_allowed_vectorstore_still_builds_search_tool(self) -> None:
        configs = [_vectorstore_tool("docs")]
        instances = {"docs": MagicMock()}
        tools = build_sdk_tools(
            configs, base_dir=None, tool_instances=instances, disallowed=set()
        )
        assert {t.name for t in tools} == {"docs_search"}

    def test_no_disallowed_set_builds_all(self) -> None:
        configs = [_func_tool("a"), _func_tool("b")]
        with patch(
            "holodeck.lib.backends.openai_agents_tool_adapters.load_function_tool",
            return_value=_stub_callable,
        ):
            tools = build_sdk_tools(configs, base_dir=None)
        assert {t.name for t in tools} == {"a", "b"}


# ---------------------------------------------------------------------------
# build_mcp_servers — drop the whole server when its name is disallowed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildMcpServersDisallowed:
    """A disallowed MCP tool name drops the whole server."""

    def test_disallowed_mcp_server_dropped(self) -> None:
        keep = _mcp_http_tool("keep")
        drop = _mcp_http_tool("drop")
        built = MagicMock(name="keep_server")
        with patch(
            "holodeck.lib.backends.openai_agents_mcp._build_http_server",
            return_value=built,
        ) as build_http:
            servers = build_mcp_servers(
                [keep, drop], base_dir=None, disallowed={"drop"}
            )
        # Only the kept tool was built into a server.
        assert servers == [built]
        build_http.assert_called_once_with(keep)

    def test_no_disallowed_builds_all_mcp_servers(self) -> None:
        a = _mcp_http_tool("a")
        b = _mcp_http_tool("b")
        with patch(
            "holodeck.lib.backends.openai_agents_mcp._build_http_server",
            side_effect=[MagicMock(), MagicMock()],
        ) as build_http:
            servers = build_mcp_servers([a, b], base_dir=None)
        assert len(servers) == 2
        assert build_http.call_count == 2


# ---------------------------------------------------------------------------
# initialize() — wires the filter and enforces allow ∩ disallow load-fail
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitializeEnforcesPermissions:
    """The backend filters the surface and fails load on allow ∩ disallow."""

    @pytest.mark.asyncio
    async def test_disallowed_function_tool_absent_from_built_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _agent(
            tools=[_func_tool("keep"), _func_tool("drop")],
            openai=OpenAIConfig(disallowed_tools=["drop"]),
        )
        backend = OpenAIAgentsBackend(agent, base_dir=Path("."))
        sdk_agent = MagicMock(name="sdk_agent")
        with (
            patch("agents.Agent", return_value=sdk_agent) as agent_ctor,
            patch("agents.ModelSettings"),
            patch("agents.set_default_openai_key"),
            patch(
                "holodeck.lib.backends.openai_agents_tool_adapters.load_function_tool",
                return_value=_stub_callable,
            ),
        ):
            await backend.initialize()
        _, kwargs = agent_ctor.call_args
        tool_names = {t.name for t in kwargs["tools"]}
        assert tool_names == {"keep"}

    @pytest.mark.asyncio
    async def test_allow_intersect_disallow_fails_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _agent(
            tools=[_func_tool("x")],
            openai=OpenAIConfig(
                disallowed_tools=["x"],
                permissions=OpenAIPermissionsConfig(allowed_tools=["x"]),
            ),
        )
        backend = OpenAIAgentsBackend(agent, base_dir=Path("."))
        with (
            patch("agents.Agent"),
            patch("agents.ModelSettings"),
            patch("agents.set_default_openai_key"),
            pytest.raises(ConfigError, match="x"),
        ):
            await backend.initialize()
