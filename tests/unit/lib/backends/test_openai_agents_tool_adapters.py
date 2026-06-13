"""Unit tests for holodeck.lib.backends.openai_agents_tool_adapters.

Covers the function-tool happy path (callable loaded, schema derived, the SDK
``on_invoke_tool`` dispatches to it) and the fail-fast errors for unsupported
tool types. The `openai-agents` SDK is installed (dev extra); ``FunctionTool``
is the real low-level class.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from holodeck.lib.backends.openai_agents_tool_adapters import (
    _derive_params_schema,
    build_sdk_tools,
)
from holodeck.lib.errors import ConfigError
from holodeck.models.tool import FunctionTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOOL_SRC = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def greet(name: str) -> str:
    return f"Hello, {name}!"
'''


@pytest.fixture
def tool_dir(tmp_path: Path) -> Path:
    """Write a module with sync + async tool callables and return its dir."""
    (tmp_path / "tools.py").write_text(_TOOL_SRC)
    return tmp_path


def _fn_tool(name: str, function: str, **extra: object) -> FunctionTool:
    return FunctionTool(
        name=name,
        description=f"{function} tool",
        file="tools.py",
        function=function,
        **extra,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Schema derivation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeriveSchema:
    def test_derived_from_signature(self) -> None:
        def fn(a: int, b: str = "x") -> str:
            return b * a

        schema = _derive_params_schema(fn, None)
        assert schema["type"] == "object"
        assert schema["properties"]["a"] == {"type": "integer"}
        assert schema["properties"]["b"] == {"type": "string"}
        assert schema["required"] == ["a"]  # b has a default

    def test_declared_parameters_take_precedence(self) -> None:
        def fn(a: int) -> int:
            return a

        declared = {"a": {"type": "number", "description": "the input"}}
        schema = _derive_params_schema(fn, declared)
        assert schema["properties"] == declared
        assert schema["required"] == ["a"]


# ---------------------------------------------------------------------------
# build_sdk_tools — function tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildFunctionTools:
    def test_sync_function_tool_invokes_callable(self, tool_dir: Path) -> None:
        import asyncio

        tools = build_sdk_tools([_fn_tool("add", "add")], base_dir=tool_dir)
        assert len(tools) == 1
        sdk_tool = tools[0]
        assert sdk_tool.name == "add"
        # The model-facing schema reflects the signature.
        assert set(sdk_tool.params_json_schema["properties"]) == {"a", "b"}
        # Drive the on_invoke handler the way the SDK loop would.
        out = asyncio.run(sdk_tool.on_invoke_tool(None, '{"a": 2, "b": 3}'))
        assert out == "5"

    def test_async_function_tool_invokes_callable(self, tool_dir: Path) -> None:
        import asyncio

        tools = build_sdk_tools([_fn_tool("greet", "greet")], base_dir=tool_dir)
        out = asyncio.run(tools[0].on_invoke_tool(None, '{"name": "Ada"}'))
        assert out == "Hello, Ada!"

    def test_empty_config_returns_empty_list(self) -> None:
        assert build_sdk_tools(None, base_dir=None) == []
        assert build_sdk_tools([], base_dir=None) == []

    def test_malformed_json_arguments_raise_config_error(self, tool_dir: Path) -> None:
        import asyncio

        tools = build_sdk_tools([_fn_tool("add", "add")], base_dir=tool_dir)
        with pytest.raises(ConfigError, match="malformed JSON"):
            asyncio.run(tools[0].on_invoke_tool(None, "{not json"))

    def test_non_dict_arguments_call_with_no_kwargs(self, tool_dir: Path) -> None:
        import asyncio

        # ``greet`` takes no required-without-default kwargs beyond name; use a
        # zero-arg callable to confirm a JSON non-object degrades to no kwargs.
        (tool_dir / "noargs.py").write_text('def ping() -> str:\n    return "pong"\n')
        cfg = FunctionTool(
            name="ping", description="ping tool", file="noargs.py", function="ping"
        )
        tools = build_sdk_tools([cfg], base_dir=tool_dir)
        out = asyncio.run(tools[0].on_invoke_tool(None, "[1, 2, 3]"))
        assert out == "pong"

    def test_empty_input_calls_with_no_kwargs(self, tool_dir: Path) -> None:
        import asyncio

        (tool_dir / "noargs2.py").write_text('def ping() -> str:\n    return "pong"\n')
        cfg = FunctionTool(
            name="ping", description="ping tool", file="noargs2.py", function="ping"
        )
        tools = build_sdk_tools([cfg], base_dir=tool_dir)
        out = asyncio.run(tools[0].on_invoke_tool(None, ""))
        assert out == "pong"


# ---------------------------------------------------------------------------
# build_sdk_tools — unsupported types fail fast
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnsupportedTools:
    @pytest.mark.parametrize(
        "tool_type",
        ["skill"],
    )
    def test_unsupported_type_raises_naming_type(self, tool_type: str) -> None:
        cfg = SimpleNamespace(type=tool_type, name="x")
        with pytest.raises(ConfigError, match=tool_type):
            build_sdk_tools([cfg], base_dir=None)  # type: ignore[list-item]

    def test_error_mentions_backend(self) -> None:
        cfg = SimpleNamespace(type="skill", name="srv")
        with pytest.raises(ConfigError, match="openai_agents"):
            build_sdk_tools([cfg], base_dir=None)  # type: ignore[list-item]


@pytest.mark.unit
class TestMCPToolsSkipped:
    """MCP tools are not wrapped as FunctionTools (they become mcp_servers)."""

    def test_mcp_tool_skipped_no_function_tool(self) -> None:
        from holodeck.models.tool import MCPTool

        cfg = MCPTool(
            name="files",
            description="filesystem server",
            transport="stdio",
            command="npx",
            args=["-y", "server"],
        )
        assert build_sdk_tools([cfg], base_dir=None) == []


# ---------------------------------------------------------------------------
# build_sdk_tools — vectorstore / hierarchical_document / prompt (B1)
# ---------------------------------------------------------------------------


class _FakeVectorStore:
    """Stand-in for an initialized VectorStoreTool."""

    is_initialized = True

    def __init__(self, text: str) -> None:
        self._text = text

    async def search(self, query: str) -> str:
        return self._text


class _FakeResult:
    def __init__(self, text: str) -> None:
        self._text = text

    def format(self) -> str:
        return self._text


class _FakeHierDoc:
    """Stand-in for an initialized HierarchicalDocumentTool."""

    _initialized = True

    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results

    async def search(self, query: str) -> list[_FakeResult]:
        return self._results


@pytest.mark.unit
class TestVectorstoreAdapter:
    def _cfg(self) -> object:
        from holodeck.models.tool import VectorstoreTool

        return VectorstoreTool(name="kb", description="knowledge base", source=".")

    def test_builds_search_tool_and_invokes(self) -> None:
        import asyncio

        cfg = self._cfg()
        inst = _FakeVectorStore("found: 42")
        tools = build_sdk_tools([cfg], base_dir=None, tool_instances={"kb": inst})  # type: ignore[list-item]
        assert len(tools) == 1
        assert tools[0].name == "kb_search"
        assert set(tools[0].params_json_schema["properties"]) == {"query"}
        out = asyncio.run(tools[0].on_invoke_tool(None, '{"query": "life"}'))
        assert out == "found: 42"

    def test_empty_result_returns_no_results_sentinel(self) -> None:
        import asyncio

        tools = build_sdk_tools(
            [self._cfg()],  # type: ignore[list-item]
            base_dir=None,
            tool_instances={"kb": _FakeVectorStore("")},
        )
        out = asyncio.run(tools[0].on_invoke_tool(None, '{"query": "x"}'))
        assert out == "No results found."

    def test_missing_instance_raises_backend_init_error(self) -> None:
        from holodeck.lib.backends.base import BackendInitError

        with pytest.raises(BackendInitError, match="kb"):
            build_sdk_tools([self._cfg()], base_dir=None, tool_instances={})  # type: ignore[list-item]


@pytest.mark.unit
class TestHierarchicalDocAdapter:
    def _cfg(self) -> object:
        from holodeck.models.tool import HierarchicalDocumentToolConfig

        return HierarchicalDocumentToolConfig(
            name="docs", description="docs", source="."
        )

    def test_joins_results_with_separator(self) -> None:
        import asyncio

        inst = _FakeHierDoc([_FakeResult("A"), _FakeResult("B")])
        tools = build_sdk_tools(
            [self._cfg()],  # type: ignore[list-item]
            base_dir=None,
            tool_instances={"docs": inst},
        )
        assert tools[0].name == "docs_search"
        out = asyncio.run(tools[0].on_invoke_tool(None, '{"query": "x"}'))
        assert out == "A\n---\nB"

    def test_empty_results_returns_sentinel(self) -> None:
        import asyncio

        tools = build_sdk_tools(
            [self._cfg()],  # type: ignore[list-item]
            base_dir=None,
            tool_instances={"docs": _FakeHierDoc([])},
        )
        out = asyncio.run(tools[0].on_invoke_tool(None, '{"query": "x"}'))
        assert out == "No results found."

    def test_missing_instance_raises_backend_init_error(self) -> None:
        from holodeck.lib.backends.base import BackendInitError

        with pytest.raises(BackendInitError, match="docs"):
            build_sdk_tools([self._cfg()], base_dir=None, tool_instances={})  # type: ignore[list-item]


@pytest.mark.unit
class TestPromptToolSkipped:
    def test_prompt_tool_skipped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from holodeck.models.tool import PromptTool

        cfg = PromptTool(
            name="p",
            description="a prompt",
            template="Do {{topic}}",
            parameters={"topic": {"type": "string"}},
        )
        with caplog.at_level(logging.WARNING):
            tools = build_sdk_tools([cfg], base_dir=None)
        assert tools == []
        assert any("prompt" in r.message and "p" in r.message for r in caplog.records)
