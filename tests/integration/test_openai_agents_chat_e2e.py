"""Azure OpenAI end-to-end smoke test for the OpenAI Agents backend.

Creds-gated: skipped unless ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_ENDPOINT``
are set (loaded from ``tests/integration/.env`` if present). With creds, this runs
a real tool-calling turn against a live Azure deployment through the
``OpenAIAgentsBackend`` (initialize → session → send / send_streaming) and
verifies a grounded response plus token streaming.

    AZURE_OPENAI_API_KEY=...  AZURE_OPENAI_ENDPOINT=...  make test-integration

The deployment name defaults to ``gpt-4o-mini``; override with
``AZURE_OPENAI_DEPLOYMENT``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from holodeck.lib.backends.openai_agents_backend import OpenAIAgentsBackend
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import FunctionTool

# ---------------------------------------------------------------------------
# Environment & skip logic
# ---------------------------------------------------------------------------

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Values shipped in the committed .env template — treat them as "unset" so the
# test skips rather than dialling a fake endpoint.
_PLACEHOLDERS = {
    "your-azure-openai-api-key-here",
    "https://your-instance.openai.azure.com",
    "",
}


def _real(value: str | None) -> str | None:
    """Return the value only if it is set and not a known template placeholder."""
    if value is None or value.strip() in _PLACEHOLDERS:
        return None
    return value


SKIP_LLM_TESTS = os.getenv("SKIP_LLM_INTEGRATION_TESTS", "false").lower() == "true"
AZURE_OPENAI_API_KEY = _real(os.getenv("AZURE_OPENAI_API_KEY"))
AZURE_OPENAI_ENDPOINT = _real(os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    or "gpt-4o-mini"
)

skip_if_no_azure = pytest.mark.skipif(
    SKIP_LLM_TESTS or not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT),
    reason="AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT not configured "
    "(or SKIP_LLM_INTEGRATION_TESTS=true)",
)

# A tool with a value the model cannot guess, so a correct answer proves the
# tool was actually invoked rather than hallucinated.
_TOOL_SRC = '''
def get_secret_code(label: str) -> str:
    """Return the secret code for a given label."""
    return f"The secret code for {label} is 8472."
'''


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_with_tool(tmp_path: Path) -> tuple[Agent, Path]:
    """Write a one-function tool module and build an Azure agent that uses it."""
    (tmp_path / "tools.py").write_text(_TOOL_SRC)
    agent = Agent(
        name="azure-openai-agents-e2e",
        model=LLMProvider(
            provider=ProviderEnum.AZURE_OPENAI,
            name=AZURE_OPENAI_DEPLOYMENT,
            endpoint=AZURE_OPENAI_ENDPOINT,
            temperature=0.0,
            max_tokens=200,
        ),
        instructions=Instructions(
            inline=(
                "You must call the get_secret_code tool to answer questions about "
                "secret codes. Report the exact code the tool returns."
            )
        ),
        tools=[
            FunctionTool(
                name="get_secret_code",
                description="Look up the secret code for a label.",
                file="tools.py",
                function="get_secret_code",
            )
        ],
    )
    return agent, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestAzureOpenAIAgentsE2E:
    """Live Azure smoke for the OpenAI Agents backend (skipped without creds)."""

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_tool_calling_turn_returns_grounded_response(
        self, agent_with_tool: tuple[Agent, Path]
    ) -> None:
        """A tool-calling turn invokes the Python tool and grounds the answer."""
        agent, base_dir = agent_with_tool
        backend = OpenAIAgentsBackend(agent, base_dir=base_dir)
        await backend.initialize()
        try:
            session = await backend.create_session()
            result = await session.send("What is the secret code for orbit?")

            assert result.is_error is False, result.error_reason
            assert "8472" in result.response
            assert any(c["name"] == "get_secret_code" for c in result.tool_calls)
            assert result.token_usage is not None
            assert result.token_usage.total_tokens > 0
            await session.close()
        finally:
            await backend.teardown()

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_streaming_yields_chunks(
        self, agent_with_tool: tuple[Agent, Path]
    ) -> None:
        """The streaming path yields non-empty text chunks from a live turn."""
        agent, base_dir = agent_with_tool
        backend = OpenAIAgentsBackend(agent, base_dir=base_dir)
        await backend.initialize()
        try:
            session = await backend.create_session()
            chunks = [c async for c in session.send_streaming("Say hello in one word.")]

            assert len(chunks) >= 1, "Expected at least one streamed chunk"
            assert all(isinstance(c, str) for c in chunks)
            assert "".join(chunks).strip(), "Streamed content must not be empty"
            await session.close()
        finally:
            await backend.teardown()
