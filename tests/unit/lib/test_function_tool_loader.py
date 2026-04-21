"""Tests for the FunctionTool dynamic loader (Phase 0 / US6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from holodeck.lib.errors import ConfigError
from holodeck.lib.function_tool_loader import load_function_tool
from holodeck.models.tool import FunctionTool

FIXTURES_ROOT = Path(__file__).parent.parent.parent / "fixtures"
ECHO_FIXTURE_DIR = FIXTURES_ROOT / "tools"


@pytest.mark.unit
class TestLoadFunctionTool:
    """Covers the happy path and the four documented failure shapes."""

    def test_imports_file_and_returns_callable(self) -> None:
        tool = FunctionTool(
            name="echo",
            description="Echo back the input message.",
            type="function",
            file="echo.py",
            function="echo",
        )

        func = load_function_tool(tool, base_dir=ECHO_FIXTURE_DIR)

        assert callable(func)
        assert func("hello") == "hello"

    def test_absolute_path_is_honored(self) -> None:
        tool = FunctionTool(
            name="echo",
            description="Echo back the input message.",
            type="function",
            file=str(ECHO_FIXTURE_DIR / "echo.py"),
            function="echo",
        )

        func = load_function_tool(tool, base_dir=None)

        assert func("hi") == "hi"

    def test_loads_async_callables(self) -> None:
        tool = FunctionTool(
            name="echo_async",
            description="Async echo.",
            type="function",
            file="echo.py",
            function="async_echo",
        )

        func = load_function_tool(tool, base_dir=ECHO_FIXTURE_DIR)

        import asyncio

        assert callable(func)
        assert asyncio.run(func("async-hi")) == "async-hi"

    def test_missing_file_raises_config_error(self, tmp_path: Path) -> None:
        tool = FunctionTool(
            name="nope",
            description="Tool pointing at a file that does not exist.",
            type="function",
            file="does_not_exist.py",
            function="irrelevant",
        )

        with pytest.raises(ConfigError) as exc_info:
            load_function_tool(tool, base_dir=tmp_path)

        assert "nope" in str(exc_info.value)
        assert "does_not_exist.py" in str(exc_info.value)

    def test_missing_function_raises_config_error(self) -> None:
        tool = FunctionTool(
            name="ghost",
            description="Tool pointing at a function that is not defined.",
            type="function",
            file="echo.py",
            function="does_not_exist",
        )

        with pytest.raises(ConfigError) as exc_info:
            load_function_tool(tool, base_dir=ECHO_FIXTURE_DIR)

        msg = str(exc_info.value)
        assert "ghost" in msg
        assert "does_not_exist" in msg

    def test_non_callable_attribute_raises_config_error(self) -> None:
        tool = FunctionTool(
            name="string_thing",
            description="Tool pointing at a module attribute that is not callable.",
            type="function",
            file="echo.py",
            function="NOT_CALLABLE",
        )

        with pytest.raises(ConfigError) as exc_info:
            load_function_tool(tool, base_dir=ECHO_FIXTURE_DIR)

        msg = str(exc_info.value)
        assert "string_thing" in msg
        assert "NOT_CALLABLE" in msg

    def test_import_error_in_target_module_is_wrapped(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.py"
        bad.write_text("import __nonexistent_module_for_tests__\n")

        tool = FunctionTool(
            name="broken",
            description="Module that fails to import.",
            type="function",
            file="bad.py",
            function="anything",
        )

        with pytest.raises(ConfigError) as exc_info:
            load_function_tool(tool, base_dir=tmp_path)

        assert "broken" in str(exc_info.value)
