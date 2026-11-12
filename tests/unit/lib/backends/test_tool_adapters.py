"""Tests for Phase 5 tool adapters (T042-T045).

Tests for VectorStoreToolAdapter, HierarchicalDocToolAdapter,
build_holodeck_sdk_server(), and create_tool_adapters().
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from holodeck.lib.backends.base import BackendInitError, BackendSessionError
from holodeck.lib.backends.tool_adapters import (
    HierarchicalDocToolAdapter,
    TemporalMemoryToolAdapter,
    VectorStoreToolAdapter,
    build_holodeck_sdk_server,
    create_tool_adapters,
)
from holodeck.lib.hybrid_search import SearchResult
from holodeck.lib.memory.backends.fake import FakeBackend
from holodeck.lib.memory.tool_api import (
    MEMORY_FUNCTION_DESCRIPTIONS,
    MEMORY_FUNCTION_NAMES,
    MEMORY_TOOL_MODELS,
)
from holodeck.models.tool import (
    HierarchicalDocumentToolConfig,
    MCPTool,
    PromptTool,
    VectorstoreTool,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_vectorstore_config(
    name: str = "kb_search",
    description: str = "Search the knowledge base",
) -> VectorstoreTool:
    """Build a minimal VectorstoreTool config for testing."""
    return VectorstoreTool(
        name=name,
        description=description,
        type="vectorstore",
        source="/data/fake",  # noqa: S108
    )


def _make_hierdoc_config(
    name: str = "policy_search",
    description: str = "Search policy documents",
) -> HierarchicalDocumentToolConfig:
    """Build a minimal HierarchicalDocumentToolConfig for testing."""
    return HierarchicalDocumentToolConfig(
        name=name,
        description=description,
        type="hierarchical_document",
        source="/data/fake",  # noqa: S108
    )


def _make_search_result(
    content: str = "Test content",
    score: float = 0.85,
    source_path: str = "/docs/test.md",
    parent_chain: list[str] | None = None,
    section_id: str = "1.1",
) -> SearchResult:
    """Create a real SearchResult instance for testing."""
    return SearchResult(
        chunk_id="test_chunk_0",
        content=content,
        fused_score=score,
        source_path=source_path,
        parent_chain=parent_chain or ["Chapter 1"],
        section_id=section_id,
    )


def _make_mock_vectorstore_tool(
    config: VectorstoreTool,
    search_return: str = "Found 1 result(s):\n\n[1] Score: 0.89\nContent",
    is_initialized: bool = True,
) -> MagicMock:
    """Create a MagicMock mimicking VectorStoreTool."""
    mock = MagicMock()
    mock.is_initialized = is_initialized
    mock.search = AsyncMock(return_value=search_return)
    mock.config = config
    return mock


def _make_mock_hierdoc_tool(
    config: HierarchicalDocumentToolConfig,
    search_return: list[SearchResult] | None = None,
    initialized: bool = True,
) -> MagicMock:
    """Create a MagicMock mimicking HierarchicalDocumentTool."""
    mock = MagicMock()
    mock._initialized = initialized
    mock.search = AsyncMock(
        return_value=search_return if search_return is not None else []
    )
    mock.config = config
    return mock


# ---------------------------------------------------------------------------
# T042: TestVectorStoreToolAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorStoreToolAdapter:
    """Tests for VectorStoreToolAdapter (T042)."""

    def test_init_stores_config_and_instance(self) -> None:
        """Adapter stores config and instance as public attributes."""
        config = _make_vectorstore_config()
        instance = _make_mock_vectorstore_tool(config)

        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        assert adapter.config is config
        assert adapter.instance is instance

    def test_to_sdk_tool_returns_correct_name(self) -> None:
        """to_sdk_tool() returns SdkMcpTool with name '<config.name>_search'."""
        from claude_agent_sdk import SdkMcpTool

        config = _make_vectorstore_config(name="my_kb")
        instance = _make_mock_vectorstore_tool(config)
        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()

        assert isinstance(sdk_tool, SdkMcpTool)
        assert sdk_tool.name == "my_kb_search"

    @pytest.mark.asyncio
    async def test_handler_calls_search_returns_content(self) -> None:
        """Handler invokes search(query) and returns content dict."""
        config = _make_vectorstore_config()
        search_text = "Found 2 result(s):\n\n[1] Score: 0.92 | doc.md\nHello"
        instance = _make_mock_vectorstore_tool(config, search_return=search_text)
        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()
        result = await sdk_tool.handler({"query": "test query"})

        instance.search.assert_awaited_once_with("test query")
        assert result == {"content": [{"type": "text", "text": search_text}]}

    @pytest.mark.asyncio
    async def test_handler_empty_string_returns_no_results(self) -> None:
        """Empty string from search() produces 'No results found.' message."""
        config = _make_vectorstore_config()
        instance = _make_mock_vectorstore_tool(config, search_return="")
        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()
        result = await sdk_tool.handler({"query": "nothing"})

        assert result == {"content": [{"type": "text", "text": "No results found."}]}

    @pytest.mark.asyncio
    async def test_handler_propagates_exceptions(self) -> None:
        """RuntimeError from search() bubbles up to caller."""
        config = _make_vectorstore_config()
        instance = _make_mock_vectorstore_tool(config)
        instance.search = AsyncMock(side_effect=RuntimeError("connection lost"))
        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()

        with pytest.raises(RuntimeError, match="connection lost"):
            await sdk_tool.handler({"query": "boom"})

    @pytest.mark.asyncio
    async def test_factory_function_binds_correct_instance(self) -> None:
        """Two adapters each call their own mock — no closure capture bug."""
        config_a = _make_vectorstore_config(name="tool_a", description="Tool A")
        config_b = _make_vectorstore_config(name="tool_b", description="Tool B")
        instance_a = _make_mock_vectorstore_tool(config_a, search_return="result A")
        instance_b = _make_mock_vectorstore_tool(config_b, search_return="result B")

        adapters = create_tool_adapters(
            tool_configs=[config_a, config_b],
            tool_instances={"tool_a": instance_a, "tool_b": instance_b},
        )

        result_a = await adapters[0].to_sdk_tool().handler({"query": "q"})
        result_b = await adapters[1].to_sdk_tool().handler({"query": "q"})

        instance_a.search.assert_awaited_once_with("q")
        instance_b.search.assert_awaited_once_with("q")
        assert result_a["content"][0]["text"] == "result A"
        assert result_b["content"][0]["text"] == "result B"

    @pytest.mark.asyncio
    async def test_handler_raises_when_not_initialized(self) -> None:
        """BackendSessionError raised when is_initialized is False."""
        config = _make_vectorstore_config(name="lazy_tool")
        instance = _make_mock_vectorstore_tool(config, is_initialized=False)
        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()

        with pytest.raises(BackendSessionError, match="lazy_tool"):
            await sdk_tool.handler({"query": "test"})
        instance.search.assert_not_awaited()


# ---------------------------------------------------------------------------
# T043: TestHierarchicalDocToolAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHierarchicalDocToolAdapter:
    """Tests for HierarchicalDocToolAdapter (T043)."""

    def test_init_stores_config_and_instance(self) -> None:
        """Adapter stores config and instance as public attributes."""
        config = _make_hierdoc_config()
        instance = _make_mock_hierdoc_tool(config)

        adapter = HierarchicalDocToolAdapter(config=config, instance=instance)

        assert adapter.config is config
        assert adapter.instance is instance

    def test_to_sdk_tool_returns_correct_name(self) -> None:
        """to_sdk_tool() returns SdkMcpTool with name '<config.name>_search'."""
        from claude_agent_sdk import SdkMcpTool

        config = _make_hierdoc_config(name="doc_search")
        instance = _make_mock_hierdoc_tool(config)
        adapter = HierarchicalDocToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()

        assert isinstance(sdk_tool, SdkMcpTool)
        assert sdk_tool.name == "doc_search_search"

    @pytest.mark.asyncio
    async def test_handler_serializes_via_search_result_format(self) -> None:
        """Handler serializes via SearchResult.format() with separator."""
        config = _make_hierdoc_config()
        results = [
            _make_search_result(content="First result", score=0.9),
            _make_search_result(content="Second result", score=0.8),
        ]
        instance = _make_mock_hierdoc_tool(config, search_return=results)
        adapter = HierarchicalDocToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()
        result = await sdk_tool.handler({"query": "test"})

        expected_text = "\n---\n".join(r.format() for r in results)
        assert result == {"content": [{"type": "text", "text": expected_text}]}

    @pytest.mark.asyncio
    async def test_serialization_includes_score_location_content(self) -> None:
        """Serialized output includes score, source path, location, and content."""
        config = _make_hierdoc_config()
        sr = _make_search_result(
            content="Force Majeure clause",
            score=0.92,
            source_path="/docs/policy.md",
            parent_chain=["Chapter 1", "Definitions"],
            section_id="1.2",
        )
        instance = _make_mock_hierdoc_tool(config, search_return=[sr])
        adapter = HierarchicalDocToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()
        result = await sdk_tool.handler({"query": "force majeure"})
        text = result["content"][0]["text"]

        assert "0.920" in text
        assert "/docs/policy.md" in text
        assert "Chapter 1 > Definitions" in text
        assert "Force Majeure clause" in text
        assert "1.2" in text

    @pytest.mark.asyncio
    async def test_handler_empty_list_returns_no_results(self) -> None:
        """Empty list from search() produces 'No results found.' message."""
        config = _make_hierdoc_config()
        instance = _make_mock_hierdoc_tool(config, search_return=[])
        adapter = HierarchicalDocToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()
        result = await sdk_tool.handler({"query": "nothing"})

        assert result == {"content": [{"type": "text", "text": "No results found."}]}

    @pytest.mark.asyncio
    async def test_handler_propagates_exceptions(self) -> None:
        """RuntimeError from search() bubbles up to caller."""
        config = _make_hierdoc_config()
        instance = _make_mock_hierdoc_tool(config)
        instance.search = AsyncMock(side_effect=RuntimeError("search failed"))
        adapter = HierarchicalDocToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()

        with pytest.raises(RuntimeError, match="search failed"):
            await sdk_tool.handler({"query": "boom"})

    @pytest.mark.asyncio
    async def test_handler_raises_when_not_initialized(self) -> None:
        """BackendSessionError raised when _initialized is False."""
        config = _make_hierdoc_config(name="uninit_doc")
        instance = _make_mock_hierdoc_tool(config, initialized=False)
        adapter = HierarchicalDocToolAdapter(config=config, instance=instance)

        sdk_tool = adapter.to_sdk_tool()

        with pytest.raises(BackendSessionError, match="uninit_doc"):
            await sdk_tool.handler({"query": "test"})
        instance.search.assert_not_awaited()


# ---------------------------------------------------------------------------
# T044: TestBuildHolodeckSdkServer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildHolodeckSdkServer:
    """Tests for build_holodeck_sdk_server() (T044)."""

    def test_returns_tuple_of_config_and_allowed_tools(self) -> None:
        """Returns (server_config_dict, list[str]) with correct keys."""
        config = _make_vectorstore_config()
        instance = _make_mock_vectorstore_tool(config)
        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        server_config, allowed_tools = build_holodeck_sdk_server([adapter])

        assert server_config["type"] == "sdk"
        assert "name" in server_config
        assert "instance" in server_config
        assert isinstance(allowed_tools, list)

    def test_empty_adapters(self) -> None:
        """Empty adapter list produces server with no tools."""
        server_config, allowed_tools = build_holodeck_sdk_server([])

        assert server_config["type"] == "sdk"
        assert server_config["name"] == "holodeck_tools"
        assert allowed_tools == []

    def test_one_vectorstore_adapter(self) -> None:
        """Single vectorstore adapter produces correct server and allowed_tools."""
        config = _make_vectorstore_config(name="my_kb")
        instance = _make_mock_vectorstore_tool(config)
        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        server_config, allowed_tools = build_holodeck_sdk_server([adapter])

        assert server_config["name"] == "holodeck_tools"
        assert allowed_tools == ["mcp__holodeck_tools__my_kb_search"]

    def test_one_hierarchical_adapter(self) -> None:
        """Single hierarchical adapter produces correct server and allowed_tools."""
        config = _make_hierdoc_config(name="docs")
        instance = _make_mock_hierdoc_tool(config)
        adapter = HierarchicalDocToolAdapter(config=config, instance=instance)

        server_config, allowed_tools = build_holodeck_sdk_server([adapter])

        assert server_config["name"] == "holodeck_tools"
        assert allowed_tools == ["mcp__holodeck_tools__docs_search"]

    def test_mixed_adapters(self) -> None:
        """Mixed adapters (2 vectorstore + 1 hierarchical) produce 3 entries."""
        vs_config_a = _make_vectorstore_config(name="kb_a")
        vs_config_b = _make_vectorstore_config(name="kb_b")
        hd_config = _make_hierdoc_config(name="policy")
        adapters = [
            VectorStoreToolAdapter(
                config=vs_config_a,
                instance=_make_mock_vectorstore_tool(vs_config_a),
            ),
            VectorStoreToolAdapter(
                config=vs_config_b,
                instance=_make_mock_vectorstore_tool(vs_config_b),
            ),
            HierarchicalDocToolAdapter(
                config=hd_config,
                instance=_make_mock_hierdoc_tool(hd_config),
            ),
        ]

        server_config, allowed_tools = build_holodeck_sdk_server(adapters)

        assert len(allowed_tools) == 3
        assert "mcp__holodeck_tools__kb_a_search" in allowed_tools
        assert "mcp__holodeck_tools__kb_b_search" in allowed_tools
        assert "mcp__holodeck_tools__policy_search" in allowed_tools

    def test_allowed_tool_name_format(self) -> None:
        """Allowed tool names follow mcp__holodeck_tools__<name>_search convention."""
        config = _make_vectorstore_config(name="special_tool")
        instance = _make_mock_vectorstore_tool(config)
        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        _, allowed_tools = build_holodeck_sdk_server([adapter])

        assert allowed_tools[0] == "mcp__holodeck_tools__special_tool_search"

    def test_calls_create_sdk_mcp_server_with_correct_args(self) -> None:
        """Verifies create_sdk_mcp_server is called with name and tools."""
        from unittest.mock import patch

        config = _make_vectorstore_config(name="kb")
        instance = _make_mock_vectorstore_tool(config)
        adapter = VectorStoreToolAdapter(config=config, instance=instance)

        with patch(
            "holodeck.lib.backends.tool_adapters.create_sdk_mcp_server"
        ) as mock_create:
            mock_create.return_value = {
                "type": "sdk",
                "name": "holodeck_tools",
                "instance": MagicMock(),
            }
            build_holodeck_sdk_server([adapter])

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs.get("name") == "holodeck_tools" or (
                call_kwargs.args[0] == "holodeck_tools" if call_kwargs.args else False
            )


# ---------------------------------------------------------------------------
# T045: TestCreateToolAdapters
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateToolAdapters:
    """Tests for create_tool_adapters() factory (T045)."""

    def test_returns_correct_adapter_types(self) -> None:
        """Returns VectorStoreToolAdapter and HierarchicalDocToolAdapter."""
        vs_config = _make_vectorstore_config(name="kb")
        hd_config = _make_hierdoc_config(name="docs")
        instances = {
            "kb": _make_mock_vectorstore_tool(vs_config),
            "docs": _make_mock_hierdoc_tool(hd_config),
        }

        adapters = create_tool_adapters(
            tool_configs=[vs_config, hd_config],
            tool_instances=instances,
        )

        assert len(adapters) == 2
        assert isinstance(adapters[0], VectorStoreToolAdapter)
        assert isinstance(adapters[1], HierarchicalDocToolAdapter)

    def test_filters_only_vectorstore_and_hierarchical(self) -> None:
        """Ignores MCP and Prompt tool types (FunctionTool now produces an adapter)."""
        vs_config = _make_vectorstore_config(name="kb")
        mcp_config = MCPTool(
            name="api",
            description="An API tool",
            type="mcp",
            command="npx",
            args=["server"],
        )
        prompt_config = PromptTool(
            name="summarize",
            description="Summarize text",
            type="prompt",
            template="Summarize: {{input}}",
            parameters={"input": {"type": "string"}},
        )
        instances = {"kb": _make_mock_vectorstore_tool(vs_config)}

        adapters = create_tool_adapters(
            tool_configs=[vs_config, mcp_config, prompt_config],
            tool_instances=instances,
        )

        assert len(adapters) == 1
        assert isinstance(adapters[0], VectorStoreToolAdapter)

    def test_matches_by_config_name(self) -> None:
        """Adapters are matched to instances by config.name."""
        config_a = _make_vectorstore_config(name="alpha")
        config_b = _make_vectorstore_config(name="beta")
        instance_a = _make_mock_vectorstore_tool(config_a, search_return="A")
        instance_b = _make_mock_vectorstore_tool(config_b, search_return="B")

        adapters = create_tool_adapters(
            tool_configs=[config_a, config_b],
            tool_instances={"alpha": instance_a, "beta": instance_b},
        )

        assert adapters[0].config.name == "alpha"
        assert adapters[0].instance is instance_a
        assert adapters[1].config.name == "beta"
        assert adapters[1].instance is instance_b

    def test_raises_backend_init_error_when_instance_missing(self) -> None:
        """BackendInitError raised when no matching instance for a config."""
        config = _make_vectorstore_config(name="missing_tool")

        with pytest.raises(BackendInitError, match="missing_tool"):
            create_tool_adapters(
                tool_configs=[config],
                tool_instances={},
            )

    def test_returns_empty_for_no_matching_tools(self) -> None:
        """Empty list returned when no vectorstore/hierarchical tools configured."""
        mcp_config = MCPTool(
            name="api",
            description="An API tool",
            type="mcp",
            command="npx",
            args=["server"],
        )

        adapters = create_tool_adapters(
            tool_configs=[mcp_config],
            tool_instances={},
        )

        assert adapters == []


class TestBuildHolodeckSdkServerCompat:
    """The in-process MCP server must build on the installed mcp major."""

    def test_server_registers_tool_list_and_call_handlers(self) -> None:
        from claude_agent_sdk import tool

        from holodeck.lib.backends.tool_adapters import build_holodeck_sdk_server

        @tool("echo", "Echo input", {"text": str})
        async def echo(args: dict[str, str]) -> dict[str, object]:
            return {"content": [{"type": "text", "text": args["text"]}]}

        adapter = MagicMock()
        adapter.to_sdk_tool.return_value = echo
        adapter.config.name = "echo"

        config, allowed = build_holodeck_sdk_server([adapter])

        assert config["type"] == "sdk"
        assert allowed == ["mcp__holodeck_tools__echo_search"]
        server = config["instance"]
        # mcp 1.x keeps handlers in `request_handlers`, 2.x in `_request_handlers`.
        handlers = vars(server).get("_request_handlers") or vars(server).get(
            "request_handlers"
        )
        assert handlers is not None
        keys = {str(k) for k in handlers}
        assert "tools/list" in keys or any("ListTools" in k for k in keys)


# ---------------------------------------------------------------------------
# T13 (spec 043): TemporalMemoryToolAdapter
# ---------------------------------------------------------------------------

_MEMORY_DOC = (
    "---\neffective_date: 2020-01-01\n---\n# Rates\n\n"
    "## Pension\n\nThe pension rate is 100 dollars per fortnight.\n\n"
    "## Carer\n\nThe carer payment is paid weekly to a carer.\n\n"
    "## Youth\n\nYouth support is paid to students under age limits.\n"
)


def _memory_agent(source: str) -> Any:
    from holodeck.models.agent import Agent

    return Agent(
        name="a",
        model={"provider": "openai", "name": "gpt-4o"},
        instructions={"inline": "hi"},
        tools=[
            {
                "name": "acts",
                "type": "hierarchical_document",
                "description": "acts",
                "source": source,
                "search_mode": "keyword",
            },
            {
                "name": "law_memory",
                "type": "temporal_memory",
                "description": "Family law memory.",
                "source_tools": ["acts"],
                "backend": {"driver": "kuzu"},
            },
        ],
    )


@pytest_asyncio.fixture
async def memory_rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Any, Any, FakeBackend]]:
    """A started ``TemporalMemoryTool`` over a keyword tool and FakeBackend."""
    from holodeck.tools import temporal_memory as composition
    from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
    from holodeck.tools.temporal_memory_tool import TemporalMemoryTool

    (tmp_path / "doc.md").write_text(_MEMORY_DOC)
    backend = FakeBackend()
    monkeypatch.setattr(composition, "create_backend", lambda *args: backend)
    monkeypatch.setattr(composition, "create_embedding_service", lambda agent: None)
    agent = _memory_agent(str(tmp_path))
    source_config, memory_config = agent.tools
    source = HierarchicalDocumentTool(source_config)
    await source.initialize()
    memory = TemporalMemoryTool(
        agent, memory_config, {"acts": source}, embedding_service=None
    )
    await memory.initialize()
    yield memory_config, memory, backend
    await memory.close()
    await source.close()


@pytest.mark.unit
class TestTemporalMemoryToolAdapter:
    """The Claude binding of the memory tool: three fixed names, thin calls."""

    @pytest.mark.asyncio
    async def test_hst_01_exposes_exactly_the_three_names_with_arg_schemas(
        self, memory_rig: tuple[Any, Any, FakeBackend]
    ) -> None:
        config, memory, _ = memory_rig

        sdk_tools = TemporalMemoryToolAdapter(config, memory).to_sdk_tools()

        assert [t.name for t in sdk_tools] == list(MEMORY_FUNCTION_NAMES)
        for sdk_tool in sdk_tools:
            args_model, _ = MEMORY_TOOL_MODELS[sdk_tool.name]
            assert sdk_tool.input_schema == args_model.model_json_schema()

    @pytest.mark.asyncio
    async def test_hst_01_allowed_tools_are_the_fixed_names_only(
        self, memory_rig: tuple[Any, Any, FakeBackend]
    ) -> None:
        config, memory, _ = memory_rig
        hd_config = _make_hierdoc_config(name="acts")
        adapters = [
            HierarchicalDocToolAdapter(
                config=hd_config, instance=_make_mock_hierdoc_tool(hd_config)
            ),
            TemporalMemoryToolAdapter(config, memory),
        ]

        _, allowed = build_holodeck_sdk_server(adapters)

        assert allowed == [
            "mcp__holodeck_tools__acts_search",
            "mcp__holodeck_tools__memory_search",
            "mcp__holodeck_tools__memory_timeline",
            "mcp__holodeck_tools__memory_provenance",
        ]
        assert "mcp__holodeck_tools__law_memory_search" not in allowed

    @pytest.mark.asyncio
    async def test_descriptions_put_the_yaml_text_before_the_built_in_text(
        self, memory_rig: tuple[Any, Any, FakeBackend]
    ) -> None:
        config, memory, _ = memory_rig

        sdk_tools = TemporalMemoryToolAdapter(config, memory).to_sdk_tools()

        for sdk_tool in sdk_tools:
            assert sdk_tool.description == (
                f"Family law memory.\n\n{MEMORY_FUNCTION_DESCRIPTIONS[sdk_tool.name]}"
            )
        search = next(t for t in sdk_tools if t.name == "memory_search")
        assert "prefer facts when they disagree" in search.description

    @pytest.mark.asyncio
    async def test_call_passes_through_as_one_text_block(
        self, memory_rig: tuple[Any, Any, FakeBackend]
    ) -> None:
        config, memory, backend = memory_rig
        tools = {
            t.name: t for t in TemporalMemoryToolAdapter(config, memory).to_sdk_tools()
        }
        # Both instants fixed, so two calls return the same text.
        args = {
            "query": "pension rate",
            "valid_at": "2024-01-01",
            "known_at": "2030-01-01",
        }

        result = await tools["memory_search"].handler(args)

        assert result["content"][0]["type"] == "text"
        assert len(result["content"]) == 1
        assert "is_error" not in result
        text = result["content"][0]["text"]
        assert text == await memory.call("memory_search", args)
        body = json.loads(text)
        assert body["hits"], body
        assert body["valid_at"] == "2024-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_bad_valid_at_returns_the_error_result_with_no_backend_call(
        self, memory_rig: tuple[Any, Any, FakeBackend]
    ) -> None:
        config, memory, backend = memory_rig
        tools = {
            t.name: t for t in TemporalMemoryToolAdapter(config, memory).to_sdk_tools()
        }
        before = dict(backend.calls)

        result = await tools["memory_search"].handler(
            {"query": "rate", "valid_at": "last July"}
        )

        body = json.loads(result["content"][0]["text"])
        assert body["error"]["code"] == "invalid_argument"
        assert dict(backend.calls) == before

    @pytest.mark.asyncio
    async def test_uninitialized_memory_tool_raises_backend_session_error(
        self, memory_rig: tuple[Any, Any, FakeBackend]
    ) -> None:
        config, memory, _ = memory_rig
        tools = {
            t.name: t for t in TemporalMemoryToolAdapter(config, memory).to_sdk_tools()
        }
        await memory.close()

        with pytest.raises(BackendSessionError, match="law_memory"):
            await tools["memory_timeline"].handler({"entity": "x"})

    @pytest.mark.asyncio
    async def test_create_tool_adapters_builds_the_memory_adapter(
        self, memory_rig: tuple[Any, Any, FakeBackend]
    ) -> None:
        config, memory, _ = memory_rig

        adapters = create_tool_adapters(
            tool_configs=[config], tool_instances={"law_memory": memory}
        )

        assert len(adapters) == 1
        assert isinstance(adapters[0], TemporalMemoryToolAdapter)
        assert adapters[0].instance is memory

    @pytest.mark.asyncio
    async def test_create_tool_adapters_requires_an_initialized_memory_tool(
        self, memory_rig: tuple[Any, Any, FakeBackend]
    ) -> None:
        config, _, _ = memory_rig

        with pytest.raises(BackendInitError, match="law_memory"):
            create_tool_adapters(tool_configs=[config], tool_instances={})
