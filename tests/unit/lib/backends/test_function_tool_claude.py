"""Tests for FunctionTool support in the Claude Agent SDK tool adapters (Phase 0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from holodeck.lib.backends.tool_adapters import (
    FunctionToolAdapter,
    build_holodeck_sdk_server,
    create_tool_adapters,
)
from holodeck.models.tool import FunctionTool

ECHO_FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "tools"


def _echo_config(name: str = "echo") -> FunctionTool:
    return FunctionTool(
        name=name,
        description="Echo back the input message.",
        type="function",
        file="echo.py",
        function="echo",
    )


@pytest.mark.unit
class TestFunctionToolAdapterCreation:
    """create_tool_adapters() produces a FunctionToolAdapter for FunctionTool."""

    def test_returns_function_tool_adapter(self) -> None:
        cfg = _echo_config()

        adapters = create_tool_adapters(
            tool_configs=[cfg],
            tool_instances={},
            base_dir=ECHO_FIXTURE_DIR,
        )

        assert len(adapters) == 1
        assert isinstance(adapters[0], FunctionToolAdapter)
        assert adapters[0].config.name == "echo"
        assert callable(adapters[0].callable)

    def test_mixed_with_other_tool_types(self) -> None:
        from tests.unit.lib.backends.test_tool_adapters import (
            _make_hierdoc_config,
            _make_mock_hierdoc_tool,
        )

        hd_cfg = _make_hierdoc_config(name="docs")
        fn_cfg = _echo_config()
        adapters = create_tool_adapters(
            tool_configs=[hd_cfg, fn_cfg],
            tool_instances={"docs": _make_mock_hierdoc_tool(hd_cfg)},
            base_dir=ECHO_FIXTURE_DIR,
        )

        assert {type(a).__name__ for a in adapters} == {
            "HierarchicalDocToolAdapter",
            "FunctionToolAdapter",
        }


@pytest.mark.unit
class TestFunctionToolAdapterSdkTool:
    """The adapter produces an @tool-decorated SdkMcpTool wrapping the callable."""

    def test_to_sdk_tool_uses_tool_name_directly(self) -> None:
        cfg = _echo_config(name="shout")
        adapter = FunctionToolAdapter(
            config=cfg,
            callable=lambda message: message.upper(),
        )

        sdk_tool = adapter.to_sdk_tool()

        assert sdk_tool.name == "shout"
        assert "Echo" in sdk_tool.description

    @pytest.mark.asyncio
    async def test_invokes_sync_callable_and_returns_mcp_payload(self) -> None:
        cfg = _echo_config(name="echo")
        adapter = FunctionToolAdapter(
            config=cfg,
            callable=lambda message: f"echoed: {message}",
        )
        sdk_tool = adapter.to_sdk_tool()

        result = await sdk_tool.handler({"message": "hi"})

        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "echoed: hi"

    @pytest.mark.asyncio
    async def test_invokes_async_callable(self) -> None:
        async def async_callable(message: str) -> str:
            return f"async: {message}"

        cfg = _echo_config(name="echo_async")
        adapter = FunctionToolAdapter(config=cfg, callable=async_callable)
        sdk_tool = adapter.to_sdk_tool()

        result = await sdk_tool.handler({"message": "yo"})

        assert result["content"][0]["text"] == "async: yo"

    @pytest.mark.asyncio
    async def test_stringifies_non_string_results(self) -> None:
        cfg = FunctionTool(
            name="add",
            description="Sum two ints.",
            type="function",
            file="echo.py",  # unused — callable is injected directly
            function="echo",
        )
        adapter = FunctionToolAdapter(config=cfg, callable=lambda a, b: a + b)
        sdk_tool = adapter.to_sdk_tool()

        result = await sdk_tool.handler({"a": 2, "b": 3})

        assert result["content"][0]["text"] == "5"


@pytest.mark.unit
class TestFunctionToolAllowedTool:
    """build_holodeck_sdk_server emits MCP-prefixed allowed tool names."""

    def test_allowed_tool_name(self) -> None:
        cfg = _echo_config(name="my_echo")

        adapters = create_tool_adapters(
            tool_configs=[cfg],
            tool_instances={},
            base_dir=ECHO_FIXTURE_DIR,
        )

        _, allowed = build_holodeck_sdk_server(adapters)

        assert "mcp__holodeck_tools__my_echo" in allowed


@pytest.mark.unit
class TestDeriveInputSchema:
    """JSON schema derivation from callable type hints."""

    def test_primitive_types(self) -> None:
        from holodeck.lib.backends.tool_adapters import _derive_input_schema

        def f(a: int, b: str, c: float, d: bool) -> str:
            return f"{a}{b}{c}{d}"

        schema = _derive_input_schema(f)

        assert schema == {"a": int, "b": str, "c": float, "d": bool}

    def test_skips_self(self) -> None:
        from holodeck.lib.backends.tool_adapters import _derive_input_schema

        class C:
            def method(self, x: int) -> int:
                return x

        schema = _derive_input_schema(C().method)
        assert "self" not in schema
        assert schema == {"x": int}

    def test_missing_annotation_falls_back_to_any(self) -> None:
        from holodeck.lib.backends.tool_adapters import _derive_input_schema

        def f(a, b: int) -> int:  # type: ignore[no-untyped-def]
            return b

        schema = _derive_input_schema(f)
        assert schema == {"a": Any, "b": int}
