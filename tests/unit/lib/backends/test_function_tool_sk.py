"""Tests for FunctionTool support in the Semantic Kernel backend (Phase 0).

Focuses on the AgentFactory registration hook — `_register_function_tools`
should load the callable via `function_tool_loader` and add it to the SK
kernel under the tool's configured name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holodeck.lib.test_runner.agent_factory import AgentFactory
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import FunctionTool

ECHO_FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "tools"


def _agent_with_echo_tool() -> Agent:
    return Agent(
        name="echo-agent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="Echo things."),
        tools=[
            FunctionTool(
                name="echo",
                description="Echo back the input message.",
                type="function",
                file="echo.py",
                function="echo",
            )
        ],
    )


@pytest.mark.unit
class TestSKFunctionToolRegistration:
    """SK kernel registers FunctionTool callables under their configured name."""

    @pytest.mark.asyncio
    async def test_has_function_tools_detects_function_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-test")

        factory = AgentFactory(agent_config=_agent_with_echo_tool())

        assert factory._has_function_tools() is True

    @pytest.mark.asyncio
    async def test_register_function_tools_adds_kernel_function(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-test")
        from holodeck.config.context import agent_base_dir

        token = agent_base_dir.set(str(ECHO_FIXTURE_DIR))
        try:
            factory = AgentFactory(agent_config=_agent_with_echo_tool())
            await factory._register_function_tools()

            plugins = factory.kernel.plugins
            assert "function_tools" in plugins
            assert "echo" in plugins["function_tools"].functions
        finally:
            agent_base_dir.reset(token)

    @pytest.mark.asyncio
    async def test_registered_function_is_invokable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-test")
        from holodeck.config.context import agent_base_dir

        token = agent_base_dir.set(str(ECHO_FIXTURE_DIR))
        try:
            factory = AgentFactory(agent_config=_agent_with_echo_tool())
            await factory._register_function_tools()

            kf = factory.kernel.plugins["function_tools"].functions["echo"]
            result = await kf.invoke(factory.kernel, message="pinged")
            assert str(result) == "pinged"
        finally:
            agent_base_dir.reset(token)

    @pytest.mark.asyncio
    async def test_no_tools_no_plugin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-test")
        agent = Agent(
            name="no-tools",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
            instructions=Instructions(inline="Hi."),
        )

        factory = AgentFactory(agent_config=agent)
        await factory._register_function_tools()

        assert "function_tools" not in factory.kernel.plugins
