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


# ---------------------------------------------------------------------------
# build_sdk_tools — unsupported types fail fast
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnsupportedTools:
    @pytest.mark.parametrize(
        "tool_type",
        ["vectorstore", "mcp", "hierarchical_document", "skill", "prompt"],
    )
    def test_unsupported_type_raises_naming_type(self, tool_type: str) -> None:
        cfg = SimpleNamespace(type=tool_type, name="x")
        with pytest.raises(ConfigError, match=tool_type):
            build_sdk_tools([cfg], base_dir=None)  # type: ignore[list-item]

    def test_error_mentions_backend(self) -> None:
        cfg = SimpleNamespace(type="vectorstore", name="kb")
        with pytest.raises(ConfigError, match="openai_agents"):
            build_sdk_tools([cfg], base_dir=None)  # type: ignore[list-item]
