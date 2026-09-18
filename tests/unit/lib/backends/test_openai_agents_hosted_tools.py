"""Unit tests for hosted tools on the openai_agents backend (T4 / D17).

Covers spec 035 FR-083 (code-interpreter safety gate), FR-084 (unsupported
approval gates fail closed), FR-034 (permission filtering includes hosted
entries), FR-110 (one validation pass reports every problem), D06 (five
hosted classes; ComputerTool rejected), and the hosted call → event / result
mapping. Factories are asserted against the exact SDK constructor arguments
for openai-agents 0.17.x; the SDK is installed (dev extra) so the lazy
imports resolve, and no network is touched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from holodeck.lib.backends.openai_agents_backend import (
    OpenAIAgentsBackend,
    OpenAIAgentsSession,
    _hosted_capability_hint,
    _to_execution_result,
)
from holodeck.lib.backends.openai_agents_events import (
    hosted_call_for,
    tool_events_for_run_items,
)
from holodeck.lib.backends.openai_agents_tool_adapters import (
    CODE_INTERPRETER_OPT_IN_MESSAGE,
    build_hosted_tool,
    build_sdk_tools,
    sdk_tool_name_for,
)
from holodeck.lib.backends.validators import (
    HOSTED_TOOLS_CLAUDE_MESSAGE,
    validate_no_hosted_tools,
    validate_openai_agents,
)
from holodeck.lib.errors import ConfigError
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.openai_config import OpenAIConfig, OpenAIPermissionsConfig


def _agent(
    tools: list[dict],
    *,
    openai: OpenAIConfig | None = None,
    provider: ProviderEnum = ProviderEnum.OPENAI,
) -> Agent:
    return Agent(
        name="hosted-agent",
        model=LLMProvider(
            provider=provider,
            name="gpt-4o-mini",
            **(
                {"endpoint": "https://r.openai.azure.com"}
                if provider is ProviderEnum.AZURE_OPENAI
                else {}
            ),
        ),
        instructions=Instructions(inline="Be helpful."),
        tools=tools,
        openai=openai,
    )


def _hosted(name: str, tool: str, **params: object) -> dict:
    entry: dict = {"name": name, "type": "hosted", "tool": tool}
    if params:
        entry["params"] = params
    return entry


WEB = _hosted("web", "WebSearchTool")
FILES = _hosted("files", "FileSearchTool", vector_store_ids=["vs_1"])
CODE = _hosted("code", "CodeInterpreterTool", container={"type": "auto"})
IMAGE = _hosted("image", "ImageGenerationTool")
MCP = _hosted("docs", "HostedMCPTool", server_label="docs", server_url="https://mcp")


# ---------------------------------------------------------------------------
# Factories — exact SDK arguments
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHostedFactories:
    def test_web_search_defaults(self) -> None:
        from agents import WebSearchTool

        tool = build_hosted_tool(_agent([WEB]).tools[0])
        assert isinstance(tool, WebSearchTool)
        assert tool.name == "web_search"
        assert tool.user_location is None
        assert tool.filters is None
        assert tool.search_context_size == "medium"
        assert tool.external_web_access is None

    def test_web_search_forwards_location_domains_and_access(self) -> None:
        cfg = _agent(
            [
                _hosted(
                    "web",
                    "WebSearchTool",
                    user_location={"city": "Austin", "country": "US"},
                    allowed_domains=["example.com"],
                    search_context_size="low",
                    external_web_access=False,
                )
            ]
        ).tools[0]
        tool = build_hosted_tool(cfg)
        assert tool.user_location == {
            "type": "approximate",
            "city": "Austin",
            "country": "US",
        }
        from openai.types.responses.web_search_tool import Filters

        assert isinstance(tool.filters, Filters)
        assert tool.filters.allowed_domains == ["example.com"]
        assert tool.search_context_size == "low"
        assert tool.external_web_access is False

    def test_file_search_forwards_ids_limits_and_ranking(self) -> None:
        from agents import FileSearchTool

        cfg = _agent(
            [
                _hosted(
                    "files",
                    "FileSearchTool",
                    vector_store_ids=["vs_1", "vs_2"],
                    max_num_results=5,
                    include_search_results=True,
                    ranking_options={"ranker": "auto", "score_threshold": 0.4},
                    filters={"type": "eq", "key": "lang", "value": "en"},
                )
            ]
        ).tools[0]
        tool = build_hosted_tool(cfg)
        assert isinstance(tool, FileSearchTool)
        assert tool.name == "file_search"
        assert tool.vector_store_ids == ["vs_1", "vs_2"]
        assert tool.max_num_results == 5
        assert tool.include_search_results is True
        assert tool.ranking_options == {"ranker": "auto", "score_threshold": 0.4}
        assert tool.filters == {"type": "eq", "key": "lang", "value": "en"}

    def test_web_search_serialises_through_sdk_converter(self) -> None:
        from agents.models.openai_responses import Converter

        cfg = _agent(
            [
                _hosted(
                    "web",
                    "WebSearchTool",
                    allowed_domains=["example.com"],
                    user_location={"country": "US"},
                )
            ]
        ).tools[0]
        converted, _include = Converter._convert_tool(build_hosted_tool(cfg))
        assert converted["type"] == "web_search"
        assert converted["filters"] == {"allowed_domains": ["example.com"]}
        assert converted["user_location"] == {"type": "approximate", "country": "US"}

    def test_all_five_serialise_through_sdk_converter(self) -> None:
        from agents.models.openai_responses import Converter

        agent = _agent([WEB, FILES, CODE, IMAGE, MCP])
        tools = build_sdk_tools(agent.tools, None, allow_unsafe_hosted=True)
        params = [Converter._convert_tool(t)[0] for t in tools]
        assert [p["type"] for p in params] == [
            "web_search",
            "file_search",
            "code_interpreter",
            "image_generation",
            "mcp",
        ]
        assert params[1]["vector_store_ids"] == ["vs_1"]
        assert params[2]["container"] == {"type": "auto"}
        assert params[4]["require_approval"] == "never"

    def test_file_search_minimal(self) -> None:
        tool = build_hosted_tool(_agent([FILES]).tools[0])
        assert tool.vector_store_ids == ["vs_1"]
        assert tool.max_num_results is None
        assert tool.include_search_results is False
        assert tool.ranking_options is None
        assert tool.filters is None

    def test_code_interpreter_builds_nested_auto_container(self) -> None:
        from agents import CodeInterpreterTool

        cfg = _agent(
            [
                _hosted(
                    "code",
                    "CodeInterpreterTool",
                    container={
                        "type": "auto",
                        "file_ids": ["file_1"],
                        "memory_limit": "4g",
                        "network_policy": {"type": "disabled"},
                    },
                )
            ]
        ).tools[0]
        tool = build_hosted_tool(cfg, allow_unsafe=True)
        assert isinstance(tool, CodeInterpreterTool)
        assert tool.name == "code_interpreter"
        assert tool.tool_config == {
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": ["file_1"],
                "memory_limit": "4g",
                "network_policy": {"type": "disabled"},
            },
        }

    def test_code_interpreter_container_id_passthrough(self) -> None:
        cfg = _agent(
            [_hosted("code", "CodeInterpreterTool", container="cntr_abc")]
        ).tools[0]
        tool = build_hosted_tool(cfg, allow_unsafe=True)
        assert tool.tool_config == {"type": "code_interpreter", "container": "cntr_abc"}

    def test_code_interpreter_without_opt_in_is_canonical_error(self) -> None:
        cfg = _agent([CODE]).tools[0]
        with pytest.raises(ConfigError, match="i_understand_this_is_unsafe") as info:
            build_hosted_tool(cfg)
        assert CODE_INTERPRETER_OPT_IN_MESSAGE in str(info.value)
        assert "tools.code" in str(info.value)

    def test_image_generation_forwards_only_set_fields(self) -> None:
        from agents import ImageGenerationTool

        default = build_hosted_tool(_agent([IMAGE]).tools[0])
        assert isinstance(default, ImageGenerationTool)
        assert default.name == "image_generation"
        assert default.tool_config == {"type": "image_generation"}

        cfg = _agent(
            [
                _hosted(
                    "image",
                    "ImageGenerationTool",
                    model="gpt-image-1",
                    quality="high",
                    size="1024x1536",
                    output_format="webp",
                    output_compression=80,
                    background="transparent",
                )
            ]
        ).tools[0]
        tool = build_hosted_tool(cfg)
        assert tool.tool_config == {
            "type": "image_generation",
            "model": "gpt-image-1",
            "quality": "high",
            "size": "1024x1536",
            "output_format": "webp",
            "output_compression": 80,
            "background": "transparent",
        }

    def test_hosted_mcp_builds_mcp_config_with_env_substitution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agents import HostedMCPTool

        monkeypatch.setenv("MCP_TOKEN", "tok-123")
        cfg = _agent(
            [
                _hosted(
                    "docs",
                    "HostedMCPTool",
                    server_label="docs",
                    server_url="https://mcp.example.com",
                    server_description="Company docs",
                    authorization="${MCP_TOKEN}",
                    headers={"X-Team": "${MCP_TOKEN}"},
                    allowed_tools=["search", "read"],
                )
            ]
        ).tools[0]
        tool = build_hosted_tool(cfg)
        assert isinstance(tool, HostedMCPTool)
        assert tool.name == "hosted_mcp"
        assert tool.on_approval_request is None
        assert tool.tool_config == {
            "type": "mcp",
            "server_label": "docs",
            "require_approval": "never",
            "server_url": "https://mcp.example.com",
            "server_description": "Company docs",
            "authorization": "tok-123",
            "headers": {"X-Team": "tok-123"},
            "allowed_tools": ["search", "read"],
        }

    def test_hosted_mcp_connector_form(self) -> None:
        cfg = _agent(
            [
                _hosted(
                    "drive",
                    "HostedMCPTool",
                    server_label="drive",
                    connector_id="connector_googledrive",
                    authorization="oauth-token",
                )
            ]
        ).tools[0]
        tool = build_hosted_tool(cfg)
        assert tool.tool_config == {
            "type": "mcp",
            "server_label": "drive",
            "require_approval": "never",
            "connector_id": "connector_googledrive",
            "authorization": "oauth-token",
        }

    def test_non_hosted_entry_rejected(self) -> None:
        cfg = _agent(
            [
                {
                    "name": "f",
                    "type": "function",
                    "description": "d",
                    "file": "t.py",
                    "function": "run",
                }
            ]
        ).tools[0]
        with pytest.raises(ConfigError, match="not a hosted tool"):
            build_hosted_tool(cfg)

    def test_sdk_agent_accepts_all_five(self) -> None:
        from agents import Agent as SDKAgent

        agent = _agent([WEB, FILES, CODE, IMAGE, MCP])
        tools = build_sdk_tools(agent.tools, None, allow_unsafe_hosted=True)
        sdk_agent = SDKAgent(name="x", tools=tools)
        assert [t.name for t in sdk_agent.tools] == [
            "web_search",
            "file_search",
            "code_interpreter",
            "image_generation",
            "hosted_mcp",
        ]


# ---------------------------------------------------------------------------
# build_sdk_tools — filtering, gate, collisions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildSdkToolsHosted:
    def test_hosted_entries_built_alongside_function_tools(self) -> None:
        agent = _agent([WEB, IMAGE])
        names = [t.name for t in build_sdk_tools(agent.tools, None)]
        assert names == ["web_search", "image_generation"]

    def test_disallowed_hosted_entry_filtered_by_config_name(self) -> None:
        agent = _agent([WEB, IMAGE])
        names = [t.name for t in build_sdk_tools(agent.tools, None, disallowed={"web"})]
        assert names == ["image_generation"]

    def test_disallowed_code_interpreter_needs_no_opt_in(self) -> None:
        agent = _agent([WEB, CODE])
        names = [
            t.name for t in build_sdk_tools(agent.tools, None, disallowed={"code"})
        ]
        assert names == ["web_search"]

    def test_code_interpreter_gate_enforced_in_build(self) -> None:
        agent = _agent([CODE])
        with pytest.raises(ConfigError, match="i_understand_this_is_unsafe"):
            build_sdk_tools(agent.tools, None)
        assert [
            t.name for t in build_sdk_tools(agent.tools, None, allow_unsafe_hosted=True)
        ] == ["code_interpreter"]

    def test_two_entries_of_one_class_collide(self) -> None:
        agent = _agent([WEB, _hosted("web2", "WebSearchTool")])
        with pytest.raises(ConfigError, match="both surface as SDK tool 'web_search'"):
            build_sdk_tools(agent.tools, None)

    def test_function_tool_named_like_hosted_collides(self) -> None:
        agent = _agent(
            [
                WEB,
                {
                    "name": "web_search",
                    "type": "function",
                    "description": "d",
                    "file": "t.py",
                    "function": "run",
                },
            ]
        )
        with pytest.raises(ConfigError, match="web_search"):
            build_sdk_tools(agent.tools, None)

    def test_sdk_tool_name_for_hosted(self) -> None:
        agent = _agent([WEB, FILES, CODE, IMAGE, MCP])
        assert [sdk_tool_name_for(t) for t in agent.tools] == [
            "web_search",
            "file_search",
            "code_interpreter",
            "image_generation",
            "hosted_mcp",
        ]


# ---------------------------------------------------------------------------
# Validators — FR-083 / FR-110 / Claude rejection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHostedValidation:
    def test_code_interpreter_without_opt_in_fails_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with pytest.raises(ConfigError, match="tools.code") as info:
            validate_openai_agents(_agent([CODE]))
        assert CODE_INTERPRETER_OPT_IN_MESSAGE in str(info.value)

    def test_code_interpreter_with_opt_in_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        validate_openai_agents(
            _agent([CODE], openai=OpenAIConfig(i_understand_this_is_unsafe=True))
        )

    def test_disallowed_code_interpreter_passes_without_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        validate_openai_agents(
            _agent([CODE], openai=OpenAIConfig(disallowed_tools=["code"]))
        )
        validate_openai_agents(
            _agent(
                [CODE],
                openai=OpenAIConfig(
                    permissions=OpenAIPermissionsConfig(disallowed_tools=["code"])
                ),
            )
        )

    def test_all_problems_reported_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = _agent(
            [CODE, WEB],
            openai=OpenAIConfig(
                disallowed_tools=["web"],
                permissions=OpenAIPermissionsConfig(allowed_tools=["web"]),
            ),
        )
        with pytest.raises(ConfigError) as info:
            validate_openai_agents(agent)
        text = str(info.value)
        assert "OPENAI_API_KEY" in text
        assert "both allowed_tools and disallowed_tools: web" in text
        assert "tools.code" in text

    def test_claude_backend_rejects_hosted_tools(self) -> None:
        agent = _agent([WEB, MCP], provider=ProviderEnum.OPENAI)
        with pytest.raises(ConfigError, match="web, docs") as info:
            validate_no_hosted_tools(agent)
        assert HOSTED_TOOLS_CLAUDE_MESSAGE in str(info.value)

    def test_claude_validator_passes_without_hosted_tools(self) -> None:
        validate_no_hosted_tools(_agent([]))
        validate_no_hosted_tools(
            _agent(
                [
                    {
                        "name": "f",
                        "type": "function",
                        "description": "d",
                        "file": "t.py",
                        "function": "run",
                    }
                ]
            )
        )


# ---------------------------------------------------------------------------
# initialize() wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitializeHosted:
    @pytest.mark.asyncio
    async def test_initialize_builds_hosted_tools_with_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agents import Agent as SDKAgent
        from agents import CodeInterpreterTool, WebSearchTool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _agent(
            [WEB, CODE], openai=OpenAIConfig(i_understand_this_is_unsafe=True)
        )
        backend = OpenAIAgentsBackend(agent, base_dir=Path("."))
        await backend.initialize()
        sdk_agent = backend._sdk_agent
        assert isinstance(sdk_agent, SDKAgent)
        assert [type(t) for t in sdk_agent.tools] == [
            WebSearchTool,
            CodeInterpreterTool,
        ]

    @pytest.mark.asyncio
    async def test_initialize_fails_closed_without_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        backend = OpenAIAgentsBackend(_agent([WEB, CODE]), base_dir=Path("."))
        with pytest.raises(ConfigError, match="i_understand_this_is_unsafe"):
            await backend.initialize()
        assert backend._sdk_agent is None

    @pytest.mark.asyncio
    async def test_initialize_on_azure_loads_hosted_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agents import WebSearchTool

        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com")
        backend = OpenAIAgentsBackend(
            _agent([WEB], provider=ProviderEnum.AZURE_OPENAI), base_dir=Path(".")
        )
        await backend.initialize()
        assert backend._sdk_agent is not None
        assert isinstance(backend._sdk_agent.tools[0], WebSearchTool)

    @pytest.mark.asyncio
    async def test_hosted_tools_grantable_to_subagents_by_config_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agents import WebSearchTool

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _agent(
            [WEB, IMAGE],
            openai=OpenAIConfig(
                agents={
                    "searcher": {
                        "description": "Searches",
                        "prompt": "S",
                        "tools": ["web"],
                    }
                }
            ),
        )
        backend = OpenAIAgentsBackend(agent, base_dir=Path("."))
        await backend.initialize()
        handoff = backend._sdk_agent.handoffs[0]
        assert [type(t) for t in handoff.tools] == [WebSearchTool]


# ---------------------------------------------------------------------------
# Hosted call items → events and ExecutionResult
# ---------------------------------------------------------------------------


def _hosted_items() -> list:
    from agents import Agent as SDKAgent
    from agents.items import ToolCallItem
    from openai.types.responses import (
        ResponseCodeInterpreterToolCall,
        ResponseFileSearchToolCall,
        ResponseFunctionWebSearch,
    )
    from openai.types.responses.response_function_web_search import ActionSearch
    from openai.types.responses.response_output_item import (
        ImageGenerationCall,
        McpCall,
    )

    sdk_agent = SDKAgent(name="x")
    raws = [
        ResponseFunctionWebSearch(
            id="ws_1",
            action=ActionSearch(type="search", query="qdrant"),
            status="completed",
            type="web_search_call",
        ),
        ResponseFileSearchToolCall(
            id="fs_1",
            queries=["policy"],
            status="completed",
            type="file_search_call",
            results=None,
        ),
        ResponseCodeInterpreterToolCall(
            id="ci_1",
            code="print(1)",
            container_id="cntr",
            outputs=[{"type": "logs", "logs": "1\n"}],
            status="completed",
            type="code_interpreter_call",
        ),
        McpCall(
            id="m_1",
            arguments='{"path": "a"}',
            name="read",
            server_label="docs",
            type="mcp_call",
            output="ok",
        ),
        ImageGenerationCall(
            id="ig_1", result="AAAA", status="completed", type="image_generation_call"
        ),
    ]
    return [ToolCallItem(agent=sdk_agent, raw_item=raw) for raw in raws]


@pytest.mark.unit
class TestHostedCallMapping:
    def test_function_call_is_not_hosted(self) -> None:
        from openai.types.responses import ResponseFunctionToolCall

        raw = ResponseFunctionToolCall(
            call_id="c1", name="add", arguments="{}", type="function_call"
        )
        assert hosted_call_for(raw) is None
        assert hosted_call_for({"type": "function_call", "name": "add"}) is None

    def test_hosted_call_views(self) -> None:
        views = [hosted_call_for(item.raw_item) for item in _hosted_items()]
        assert [(v.name, v.call_id) for v in views] == [
            ("web_search", "ws_1"),
            ("file_search", "fs_1"),
            ("code_interpreter", "ci_1"),
            ("read", "m_1"),
            ("image_generation", "ig_1"),
        ]
        assert views[0].arguments == {"action": {"type": "search", "query": "qdrant"}}
        assert views[1].arguments == {"queries": ["policy"]}
        assert views[2].arguments == {"code": "print(1)"}
        assert '"logs": "1\\n"' in views[2].response
        assert views[3].arguments == {"server_label": "docs", "path": "a"}
        assert views[3].response == "ok"
        assert views[4].response == "completed (image data omitted)"
        assert "AAAA" not in views[4].response

    def test_mcp_call_server_label_cannot_be_spoofed_by_arguments(self) -> None:
        from openai.types.responses.response_output_item import McpCall

        raw = McpCall(
            id="m_2",
            arguments='{"server_label": "spoof", "path": "a"}',
            name="read",
            server_label="actual",
            type="mcp_call",
        )
        view = hosted_call_for(raw)
        assert view is not None
        assert view.arguments == {"path": "a", "server_label": "actual"}

    def test_mcp_call_error_surfaces_as_response(self) -> None:
        view = hosted_call_for(
            {
                "type": "mcp_call",
                "id": "m",
                "name": "read",
                "error": "boom",
                "output": None,
            }
        )
        assert view is not None
        assert view.response == "boom"

    def test_run_items_emit_start_and_end_per_hosted_call(self) -> None:
        events = tool_events_for_run_items(_hosted_items())
        assert [(e.kind, e.tool_name, e.tool_use_id) for e in events] == [
            ("start", "web_search", "ws_1"),
            ("end", "web_search", "ws_1"),
            ("start", "file_search", "fs_1"),
            ("end", "file_search", "fs_1"),
            ("start", "code_interpreter", "ci_1"),
            ("end", "code_interpreter", "ci_1"),
            ("start", "read", "m_1"),
            ("end", "read", "m_1"),
            ("start", "image_generation", "ig_1"),
            ("end", "image_generation", "ig_1"),
        ]
        assert events[0].tool_input == {"action": {"type": "search", "query": "qdrant"}}
        assert events[7].tool_response == "ok"

    def test_execution_result_pairs_hosted_calls_and_results(self) -> None:
        result = MagicMock()
        result.final_output = "done"
        result.new_items = _hosted_items()
        result.raw_responses = [MagicMock()]
        result.context_wrapper.usage = None
        mapped = _to_execution_result(result)
        assert [c["name"] for c in mapped.tool_calls] == [
            "web_search",
            "file_search",
            "code_interpreter",
            "read",
            "image_generation",
        ]
        assert [r["name"] for r in mapped.tool_results] == [
            c["name"] for c in mapped.tool_calls
        ]
        assert mapped.tool_calls[0]["call_id"] == "ws_1"
        assert mapped.tool_results[0]["call_id"] == "ws_1"
        assert mapped.tool_calls[2]["arguments"] == {"code": "print(1)"}
        assert mapped.tool_results[3]["result"] == "ok"


# ---------------------------------------------------------------------------
# Azure runtime capability hint
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAzureCapabilityHint:
    def test_no_hint_on_openai_provider(self) -> None:
        agent = _agent([WEB])
        assert _hosted_capability_hint(agent, RuntimeError("tool not supported")) == ""

    def test_no_hint_without_hosted_tools(self) -> None:
        agent = _agent([], provider=ProviderEnum.AZURE_OPENAI)
        assert _hosted_capability_hint(agent, RuntimeError("tool not supported")) == ""

    def test_no_hint_for_unrelated_error(self) -> None:
        agent = _agent([WEB], provider=ProviderEnum.AZURE_OPENAI)
        assert _hosted_capability_hint(agent, RuntimeError("rate limited")) == ""

    def test_hint_names_declared_hosted_classes(self) -> None:
        agent = _agent([WEB, IMAGE], provider=ProviderEnum.AZURE_OPENAI)
        hint = _hosted_capability_hint(
            agent, RuntimeError("Invalid value: 'web_search' tool is not supported")
        )
        assert "WebSearchTool, ImageGenerationTool" in hint
        assert "Azure" in hint

    @pytest.mark.asyncio
    async def test_send_preserves_sdk_error_and_appends_hint(self) -> None:
        agent = _agent([WEB], provider=ProviderEnum.AZURE_OPENAI)
        session = OpenAIAgentsSession(MagicMock(), MagicMock(), agent_config=agent)
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(
                side_effect=RuntimeError("400 tool type web_search is not supported")
            )
            result = await session.send("hi")
        assert result.is_error
        assert result.error_reason.startswith(
            "RuntimeError: 400 tool type web_search is not supported"
        )
        assert "WebSearchTool" in result.error_reason
        assert "Azure" in result.error_reason
