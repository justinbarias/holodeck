"""Backend selector for routing agent configurations to the appropriate backend.

Routes agent configurations to the correct backend implementation based on
the configured LLM provider.
"""

from typing import Any

from holodeck.lib.backends.base import AgentBackend, BackendInitError
from holodeck.lib.backends.claude_backend import ClaudeBackend
from holodeck.models.agent import Agent
from holodeck.models.llm import ProviderEnum


class BackendSelector:
    """Selects and initializes the appropriate backend for an agent configuration."""

    @staticmethod
    async def select(
        agent: Agent,
        tool_instances: dict[str, Any] | None = None,
        mode: str = "test",
    ) -> AgentBackend:
        """Select and initialize the appropriate backend for the given agent.

        Args:
            agent: Agent configuration with model provider information.
            tool_instances: Initialized tool instances for Claude backend.
            mode: Execution mode (``"test"`` or ``"chat"``).

        Returns:
            An initialized AgentBackend instance ready for use.

        Raises:
            BackendInitError: If the provider is not supported or initialization fails.
        """
        provider = agent.model.provider

        if provider in (ProviderEnum.OPENAI, ProviderEnum.AZURE_OPENAI):
            # Lazy import keeps the optional openai-agents SDK off the import
            # path for non-OpenAI providers (SC-005).
            from holodeck.lib.backends.openai_agents_backend import OpenAIAgentsBackend

            openai_backend = OpenAIAgentsBackend(agent=agent)
            await openai_backend.initialize()
            return openai_backend

        if provider in (ProviderEnum.ANTHROPIC, ProviderEnum.OLLAMA):
            claude_backend = ClaudeBackend(
                agent=agent,
                tool_instances=tool_instances,
                mode=mode,
            )
            await claude_backend.initialize()
            return claude_backend

        raise BackendInitError(f"Unsupported provider: {provider}")
