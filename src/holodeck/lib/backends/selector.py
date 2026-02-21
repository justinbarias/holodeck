"""Backend selector for routing agent configurations to the appropriate backend.

Routes agent configurations to the correct backend implementation based on
the configured LLM provider.
"""

from holodeck.lib.backends.base import AgentBackend, BackendInitError
from holodeck.lib.backends.sk_backend import SKBackend
from holodeck.models.agent import Agent
from holodeck.models.llm import ProviderEnum


class BackendSelector:
    """Selects and initializes the appropriate backend for an agent configuration."""

    @staticmethod
    async def select(agent: Agent) -> AgentBackend:
        """Select and initialize the appropriate backend for the given agent.

        Args:
            agent: Agent configuration with model provider information.

        Returns:
            An initialized AgentBackend instance ready for use.

        Raises:
            BackendInitError: If the provider is not supported or initialization fails.
        """
        provider = agent.model.provider

        if provider in (
            ProviderEnum.OPENAI,
            ProviderEnum.AZURE_OPENAI,
            ProviderEnum.OLLAMA,
        ):
            backend = SKBackend(agent_config=agent)
            await backend.initialize()
            return backend

        if provider == ProviderEnum.ANTHROPIC:
            raise BackendInitError(
                "Anthropic provider is not yet supported via Semantic Kernel backend. "
                "Claude Agent SDK backend will be added in a future phase."
            )

        raise BackendInitError(f"Unsupported provider: {provider}")
