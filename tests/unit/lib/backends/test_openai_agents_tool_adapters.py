"""Unit tests for holodeck.lib.backends.openai_agents_tool_adapters.

Covers the function-tool happy path (callable loaded, schema derived, the SDK
``on_invoke_tool`` dispatches to it) and the fail-fast errors for unsupported
tool types. The `openai-agents` SDK is installed (dev extra); ``FunctionTool``
is the real low-level class.
"""

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

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


@pytest.mark.unit
class TestSdkNameCollision:
    """Two configs surfacing as one SDK tool name fail load (T3 review)."""

    def test_vectorstore_and_function_sharing_search_name_rejected(self) -> None:
        from holodeck.lib.errors import ConfigError
        from holodeck.models.tool import FunctionTool, VectorstoreTool

        configs = [
            VectorstoreTool(name="kb", description="d", source="data/"),
            FunctionTool(
                name="kb_search", description="d", file="tools.py", function="f"
            ),
        ]
        with pytest.raises(ConfigError, match="both surface as SDK tool 'kb_search'"):
            build_sdk_tools(configs, base_dir=None, tool_instances={"kb": object()})


# ---------------------------------------------------------------------------
# temporal_memory tool (spec 043, HST-01)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def memory_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Any]:
    """A started ``TemporalMemoryTool`` over a keyword tool and FakeBackend."""
    from holodeck.lib.memory.backends.fake import FakeBackend
    from holodeck.models.agent import Agent
    from holodeck.models.tool import HierarchicalDocumentToolConfig, SearchMode
    from holodeck.tools import temporal_memory as composition
    from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
    from holodeck.tools.temporal_memory_tool import TemporalMemoryTool

    (tmp_path / "doc.md").write_text(
        "# Rates\n\n## Pension\n\nThe pension rate is 100 dollars.\n\n"
        "## Carer\n\nThe carer payment is paid weekly.\n\n"
        "## Youth\n\nYouth support is paid to students.\n"
    )
    backend = FakeBackend()
    monkeypatch.setattr(composition, "create_backend", lambda *args: backend)
    monkeypatch.setattr(composition, "create_embedding_service", lambda agent: None)
    source_config = HierarchicalDocumentToolConfig(
        name="acts",
        description="acts",
        source=str(tmp_path),
        search_mode=SearchMode.KEYWORD,
    )
    source = HierarchicalDocumentTool(source_config)
    await source.initialize()
    config = _memory_config()
    agent = Agent(
        name="a",
        model={"provider": "openai", "name": "gpt-4o"},  # type: ignore[arg-type]
        instructions={"inline": "hi"},  # type: ignore[arg-type]
        tools=[source_config, config],
    )
    memory = TemporalMemoryTool(agent, config, {"acts": source}, embedding_service=None)
    await memory.initialize()
    yield memory
    await memory.close()
    await source.close()


def _memory_config(name: str = "law_memory") -> Any:
    from holodeck.models.tool import TemporalMemoryToolConfig

    return TemporalMemoryToolConfig(
        name=name,
        description="Family law memory.",
        source_tools=["acts"],
        backend={"driver": "kuzu"},  # type: ignore[arg-type]
    )


@pytest.mark.unit
class TestTemporalMemoryTools:
    """The memory tool is three fixed-name function tools, not an error."""

    @pytest.mark.asyncio
    async def test_hst_01_builds_the_three_fixed_names_without_instance_suffix(
        self, memory_instance: Any
    ) -> None:
        from holodeck.lib.memory.tool_api import (
            MEMORY_FUNCTION_NAMES,
            MEMORY_TOOL_MODELS,
            function_description,
        )

        cfg = _memory_config()

        tools = build_sdk_tools(
            [cfg], base_dir=None, tool_instances={"law_memory": memory_instance}
        )

        assert [t.name for t in tools] == list(MEMORY_FUNCTION_NAMES)
        assert not any(t.name.startswith("law_memory") for t in tools)
        for tool in tools:
            args_model, _ = MEMORY_TOOL_MODELS[tool.name]
            assert tool.params_json_schema == args_model.model_json_schema()
            assert tool.description == function_description(tool.name, cfg.description)

    def test_missing_memory_instance_raises_backend_init_error(self) -> None:
        from holodeck.lib.backends.base import BackendInitError

        with pytest.raises(BackendInitError, match="law_memory"):
            build_sdk_tools([_memory_config()], base_dir=None, tool_instances={})

    def test_an_instance_of_another_type_is_not_a_memory_instance(self) -> None:
        from holodeck.lib.backends.base import BackendInitError

        with pytest.raises(BackendInitError, match="law_memory"):
            build_sdk_tools(
                [_memory_config()],
                base_dir=None,
                tool_instances={"law_memory": object()},
            )

    @pytest.mark.asyncio
    async def test_uninitialized_instance_raises_backend_session_error_on_call(
        self, memory_instance: Any
    ) -> None:
        from holodeck.lib.backends.base import BackendSessionError

        tools = build_sdk_tools(
            [_memory_config()],
            base_dir=None,
            tool_instances={"law_memory": memory_instance},
        )
        await memory_instance.close()

        with pytest.raises(BackendSessionError, match="law_memory"):
            await tools[1].on_invoke_tool(None, '{"entity": "x"}')

    @pytest.mark.parametrize("raw", ["not json", "[1, 2]", '"text"', "null"])
    @pytest.mark.asyncio
    async def test_arguments_that_are_not_a_json_object_are_an_error_result(
        self, memory_instance: Any, raw: str
    ) -> None:
        import json

        tools = build_sdk_tools(
            [_memory_config()],
            base_dir=None,
            tool_instances={"law_memory": memory_instance},
        )

        text = await tools[0].on_invoke_tool(None, raw)

        assert json.loads(text)["error"]["code"] == "invalid_argument"

    @pytest.mark.parametrize(
        "other",
        [
            lambda: FunctionTool(
                name="memory_search", description="d", file="t.py", function="f"
            ),
            lambda: __import__("holodeck.models.tool", fromlist=["x"]).VectorstoreTool(
                name="memory", description="d", source="data/"
            ),
        ],
        ids=["function-tool", "vectorstore-memory_search"],
    )
    def test_hst_01_a_tool_producing_a_fixed_name_is_rejected(self, other: Any) -> None:
        with pytest.raises(
            ConfigError, match="both surface as SDK tool 'memory_search'"
        ):
            build_sdk_tools(
                [_memory_config(), other()],
                base_dir=None,
                tool_instances={"memory": object()},
            )

    def test_sdk_tool_names_for_lists_the_fixed_names_only(self) -> None:
        from holodeck.lib.backends.openai_agents_tool_adapters import (
            sdk_tool_names_for,
        )

        assert sdk_tool_names_for(_memory_config()) == (
            "memory_search",
            "memory_timeline",
            "memory_provenance",
        )

    def test_agent_without_memory_tool_does_not_import_the_memory_stack(
        self, tmp_path: Path
    ) -> None:
        import json
        import subprocess
        import sys

        # A fresh interpreter: other tests in this worker import the memory stack.
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "doc.md").write_text("# Rates\n\nThe rate is 100 dollars.\n")
        watched = (
            "graphiti_core",
            "holodeck.lib.memory.backends.graphiti",
            "holodeck.lib.memory.service",
            "holodeck.lib.memory.tool_api",
            "holodeck.tools.temporal_memory",
            "holodeck.tools.temporal_memory_tool",
        )
        code = f"""
import asyncio, json, sys
from holodeck.lib import tool_initializer
from holodeck.lib.backends.openai_agents_tool_adapters import build_sdk_tools
from holodeck.models.agent import Agent

tool_initializer.create_embedding_service = lambda agent, **kw: None
agent = Agent(
    name="a",
    model={{"provider": "openai", "name": "gpt-4o"}},
    instructions={{"inline": "hi"}},
    tools=[{{
        "name": "docs", "type": "hierarchical_document", "description": "d",
        "source": {str(folder)!r}, "search_mode": "keyword",
    }}],
)

async def main():
    instances = await tool_initializer.initialize_tools(agent)
    tools = build_sdk_tools(agent.tools, None, instances)
    await tool_initializer.close_tools(instances.values())
    return [t.name for t in tools]

names = asyncio.run(main())
print(json.dumps({{
    "names": names,
    "loaded": [m for m in {watched!r} if m in sys.modules],
}}))
"""
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )

        report = json.loads(out.stdout.strip().splitlines()[-1])
        assert report["names"] == ["docs_search"]
        assert report["loaded"] == []
