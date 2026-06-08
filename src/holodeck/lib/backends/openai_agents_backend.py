"""OpenAI Agents SDK backend for HoloDeck agent execution.

Implements the provider-agnostic ``AgentBackend`` / ``AgentSession`` protocols
on top of the `openai-agents` SDK (``Agent`` + ``Runner`` + ``SQLiteSession``).
Routes ``provider: openai`` and ``provider: azure_openai`` agents through the
SDK agent loop with custom Python function tools.

The `openai-agents` package is an optional extra. Every ``import agents`` (and
``openai``) is performed *inside* functions/methods here so that importing this
module — or, more importantly, ``holodeck.lib.backends.selector`` — never pulls
the SDK in. Other backends therefore incur no import cost or failure when the
extra is not installed (SC-005).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from holodeck.lib.backends.base import BackendInitError
from holodeck.models.agent import Agent
from holodeck.models.llm import ProviderEnum

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    pass

# A current Azure OpenAI GA API version. Used when the agent config does not
# pin one and ``AZURE_OPENAI_API_VERSION`` is unset. ``AsyncAzureOpenAI``
# requires an explicit ``api_version``.
_DEFAULT_AZURE_API_VERSION = "2024-10-21"


def _resolve_secret(value: Any) -> str | None:
    """Return the plain string for a config value that may be a ``SecretStr``."""
    if value is None:
        return None
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        secret = getter()
        return str(secret) if secret else None
    text = str(value)
    return text or None


def _build_model(agent: Agent) -> Any:
    """Build the SDK ``model=`` argument for *agent* and validate credentials.

    For ``provider: openai`` the default Responses client is used, so the model
    name string is returned; ``OPENAI_API_KEY`` (or an explicit
    ``model.api_key``) must be available.

    For ``provider: azure_openai`` an ``AsyncAzureOpenAI`` client is wrapped as
    an ``OpenAIChatCompletionsModel`` (Azure speaks Chat Completions, not the
    Responses API). SDK tracing is disabled because no OpenAI-platform key is
    present to upload traces with.

    Args:
        agent: The agent configuration whose ``model`` selects the provider.

    Returns:
        Either the model-name string (OpenAI) or an
        ``OpenAIChatCompletionsModel`` instance (Azure), ready to pass as the
        SDK ``Agent(model=...)`` argument.

    Raises:
        BackendInitError: If a required credential is missing, or the provider
            is not supported by this backend.
    """
    model_cfg = agent.model
    provider = model_cfg.provider

    if provider == ProviderEnum.OPENAI:
        api_key = _resolve_secret(model_cfg.api_key) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise BackendInitError(
                "OPENAI_API_KEY is required for provider 'openai'. "
                "Set it in the environment or as model.api_key in the agent config."
            )
        # The default Responses client reads OPENAI_API_KEY from the env; make
        # an explicit config-supplied key authoritative for this process.
        from agents import set_default_openai_key

        set_default_openai_key(api_key)
        return model_cfg.name

    if provider == ProviderEnum.AZURE_OPENAI:
        api_key = _resolve_secret(model_cfg.api_key) or os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )
        if not api_key:
            raise BackendInitError(
                "AZURE_OPENAI_API_KEY is required for provider 'azure_openai'. "
                "Set it in the environment or as model.api_key in the agent config."
            )
        endpoint = model_cfg.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise BackendInitError(
                "AZURE_OPENAI_ENDPOINT is required for provider 'azure_openai'. "
                "Set it in the environment or as model.endpoint in the agent config."
            )
        api_version = (
            model_cfg.api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or _DEFAULT_AZURE_API_VERSION
        )

        from agents import OpenAIChatCompletionsModel, set_tracing_disabled
        from openai import AsyncAzureOpenAI

        # No OpenAI-platform key is available in the Azure dev path, so disable
        # the SDK's trace upload (it would otherwise fail / leak).
        set_tracing_disabled(True)

        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        # For Azure, ``model`` is the deployment name (model_cfg.name).
        return OpenAIChatCompletionsModel(model=model_cfg.name, openai_client=client)

    raise BackendInitError(
        f"The openai_agents backend does not support provider '{provider.value}'."
    )
