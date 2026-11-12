"""Graphiti's model clients and graph driver, built from HoloDeck config.

Design seam 4 (spec 043, GPH-01 to GPH-05, D24). Every model call Graphiti
makes goes to an endpoint the agent YAML names:

* The LLM client is chosen by ``extraction_model.provider`` and, for
  Anthropic, ``auth_provider``. No environment variable selects it.
* Embeddings go through HoloDeck's ``LiteLLMEmbeddingService``, the same
  service the source tools use.
* The cross-encoder refuses every call, so Graphiti never builds its OpenAI
  reranker.

This module imports ``graphiti_core`` at import time. Import it only after
``require_graphiti()`` in ``graphiti.py`` has passed.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from anthropic import AsyncAnthropic
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from graphiti_core.tracer import Tracer
from openai import AsyncOpenAI
from pydantic import BaseModel, SecretStr, create_model

from holodeck.lib.backends.validators import validate_credentials
from holodeck.lib.errors import ConfigError
from holodeck.lib.extras import install_command
from holodeck.lib.litellm_support import _azure_v1_base_url
from holodeck.lib.memory.backends.graphiti import EmbeddingService
from holodeck.lib.memory.backends.graphiti_agent_sdk import AgentSdkLLMClient
from holodeck.lib.memory.backends.graphiti_prompts import holodeck_prompt_library
from holodeck.lib.memory.backends.graphiti_secrets import (
    credential_values,
    install_log_redaction,
)
from holodeck.lib.memory.errors import MemoryConfigError, TemporalMemoryError
from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import GraphBackendConfig, memory_extraction_model_error

logger = logging.getLogger(__name__)

TELEMETRY_ENV_VAR = "GRAPHITI_TELEMETRY_ENABLED"
GRAPHITI_REQUIREMENT = "graphiti-core[{extra}]>=0.31,<0.32"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
OLLAMA_DEFAULT_ENDPOINT = "http://localhost:11434"
OLLAMA_PLACEHOLDER_KEY = "ollama"
FALKORDB_DEFAULT_PORT = 6379
NEPTUNE_DEFAULT_PORT = 8182
KUZU_IN_MEMORY = ":memory:"
DEFAULT_TOOL_NAME = "temporal_memory"


def disable_telemetry() -> None:
    """Turn Graphiti's PostHog telemetry off (GPH-02).

    Graphiti reads the variable when it sends, so this must run before any
    Graphiti object is constructed. It overrides any value already set.
    """
    os.environ[TELEMETRY_ENV_VAR] = "false"


def driver_install_command(driver: str) -> str:
    """Return the pip command that installs a Graphiti driver's packages.

    The neo4j driver comes with ``graphiti-core``, so it is the ``graphiti``
    extra; every other driver has its own ``graphiti-core`` extra.
    """
    if driver == "neo4j":
        return install_command("graphiti")
    return f"pip install '{GRAPHITI_REQUIREMENT.format(extra=driver)}'"


# --- LLM client -----------------------------------------------------------


class CountingLLMClient(LLMClient):
    """Delegates to one Graphiti LLM client and counts the requests to it.

    Graphiti's token tracker counts calls only for the clients that record
    tokens; ``OpenAIGenericClient`` records nothing. This wrapper counts every
    ``generate_response`` request Graphiti makes, whatever the client. Retries
    inside the wrapped client are not counted separately.
    """

    def __init__(self, inner: LLMClient) -> None:
        super().__init__(inner.config, cache=False)
        self.inner = inner
        self.model = inner.model
        self.small_model = inner.small_model
        self.temperature = inner.temperature
        self.max_tokens = inner.max_tokens
        self.token_tracker = inner.token_tracker
        self.calls = 0

    def set_tracer(self, tracer: Tracer) -> None:
        super().set_tracer(tracer)
        self.inner.set_tracer(tracer)

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
        model: str | None = None,
    ) -> dict[str, object]:
        self.calls += 1
        response: dict[str, object] = await self.inner.generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
            model=model,
        )
        return response

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 0,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, object]:
        raise TemporalMemoryError(
            "CountingLLMClient delegates generate_response; this is never called"
        )


def _secret(value: SecretStr | None) -> str | None:
    """Unwrap a secret at the call that needs it. Never log the result."""
    return value.get_secret_value() if value is not None else None


def _llm_config(
    model: LLMProvider, *, api_key: str | None, base_url: str | None
) -> LLMConfig:
    """Map the settings table of design seam 4 onto Graphiti's ``LLMConfig``."""
    config = LLMConfig(
        api_key=api_key,
        model=model.name,
        base_url=base_url,
        small_model=model.name,
    )
    if model.temperature is not None:
        config.temperature = model.temperature
    max_tokens = _explicit_max_tokens(model)
    if max_tokens is not None:
        config.max_tokens = max_tokens
    return config


def _explicit_max_tokens(model: LLMProvider) -> int | None:
    """``max_tokens`` only when the YAML sets it.

    HoloDeck's default of 1000 would cut extraction output short, so an unset
    value leaves Graphiti's own default in place.
    """
    if "max_tokens" in model.model_fields_set and model.max_tokens is not None:
        return model.max_tokens
    return None


def _key_or_env(model: LLMProvider, env_var: str) -> str:
    key = _secret(model.api_key) or os.environ.get(env_var)
    if not key:
        raise MemoryConfigError(
            f"{env_var} is required for extraction_model provider "
            f"'{model.provider.value}'. Set it in the environment or as "
            "extraction_model.api_key."
        )
    return key


def _openai_client(model: LLMProvider) -> LLMClient:
    # The endpoint is always given. With none, the OpenAI SDK reads
    # OPENAI_BASE_URL, and extraction would go to a host the YAML does not name.
    config = _llm_config(
        model,
        api_key=_key_or_env(model, "OPENAI_API_KEY"),
        base_url=model.endpoint or OPENAI_DEFAULT_BASE_URL,
    )
    return OpenAIClient(config=config, cache=False, max_tokens=config.max_tokens)


def _azure_openai_client(model: LLMProvider) -> LLMClient:
    api_key = _key_or_env(model, "AZURE_OPENAI_API_KEY")
    base_url = _azure_v1_base_url(model.endpoint or "")
    config = _llm_config(model, api_key=api_key, base_url=base_url)
    default_query = {"api-version": model.api_version} if model.api_version else None
    azure_client = AsyncOpenAI(
        api_key=api_key, base_url=base_url, default_query=default_query
    )
    return AzureOpenAILLMClient(
        azure_client=azure_client, config=config, max_tokens=config.max_tokens
    )


def _ollama_base_url(endpoint: str | None) -> str:
    base = (endpoint or OLLAMA_DEFAULT_ENDPOINT).rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _ollama_client(model: LLMProvider) -> LLMClient:
    api_key = _secret(model.api_key) or OLLAMA_PLACEHOLDER_KEY
    config = _llm_config(
        model, api_key=api_key, base_url=_ollama_base_url(model.endpoint)
    )
    return OpenAIGenericClient(config=config, cache=False, max_tokens=config.max_tokens)


def _anthropic_api_key_client(model: LLMProvider) -> LLMClient:
    api_key = _secret(model.api_key)
    if api_key is None:
        try:
            api_key = validate_credentials(model)["ANTHROPIC_API_KEY"]
        except ConfigError as error:
            raise MemoryConfigError(str(error)) from error
    base_url = model.endpoint or ANTHROPIC_DEFAULT_BASE_URL
    # Built here with an explicit endpoint: the Anthropic SDK otherwise reads
    # ANTHROPIC_BASE_URL. max_retries matches Graphiti's own client.
    client = AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=1)
    return AnthropicClient(
        config=_llm_config(model, api_key=api_key, base_url=base_url),
        cache=False,
        client=client,
    )


def agent_sdk_llm_client(model: LLMProvider) -> LLMClient:
    """The Claude Agent SDK client for ``auth_provider: oauth_token`` (D24).

    The CLI reads the OAuth token from the environment; it is not read here.
    The CLI also chooses its provider and credential from the environment, so
    a variable that would send extraction elsewhere fails here.
    """
    try:
        validate_credentials(model)
    except ConfigError as error:
        raise MemoryConfigError(str(error)) from error
    conflicting = sorted(
        name for name in AGENT_SDK_CONFLICTING_ENV if os.environ.get(name)
    )
    if conflicting:
        raise MemoryConfigError(
            "extraction_model uses auth_provider 'oauth_token', but "
            f"{', '.join(conflicting)} is set and would change where the "
            "Claude Code CLI sends extraction calls or how they are billed. "
            "Unset it for this run."
        )
    ignored = sorted({"temperature", "max_tokens"} & model.model_fields_set)
    if ignored:
        logger.warning(
            "extraction_model %s is not applied with auth_provider "
            "'oauth_token': the Claude Agent SDK has no such option.",
            " and ".join(ignored),
        )
    return AgentSdkLLMClient(model.name)


def build_llm_client(
    model: LLMProvider, *, tool_name: str = DEFAULT_TOOL_NAME
) -> LLMClient:
    """Return Graphiti's LLM client for ``model`` (design seam 4 selection table)."""
    if model.provider == ProviderEnum.OPENAI:
        return _openai_client(model)
    if model.provider == ProviderEnum.AZURE_OPENAI:
        return _azure_openai_client(model)
    if model.provider == ProviderEnum.OLLAMA:
        return _ollama_client(model)
    auth = model.auth_provider
    if auth is None or auth == AuthProvider.api_key:
        return _anthropic_api_key_client(model)
    if auth == AuthProvider.oauth_token:
        return agent_sdk_llm_client(model)
    message = memory_extraction_model_error(tool_name, model) or (
        f"extraction_model auth_provider '{auth.value}' is not supported"
    )
    raise MemoryConfigError(message)


# --- Embedder and cross-encoder ------------------------------------------


class LiteLLMEmbedder(EmbedderClient):
    """Graphiti embedder over HoloDeck's embedding service.

    Graph vectors and chunk vectors come from the same model and dimension.
    """

    def __init__(self, service: EmbeddingService) -> None:
        self._service = service

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Return one vector for a string or a list holding one string."""
        if isinstance(input_data, str):
            text = input_data
        elif (
            isinstance(input_data, list)
            and len(input_data) == 1
            and isinstance(input_data[0], str)
        ):
            text = input_data[0]
        else:
            raise TypeError(
                "LiteLLMEmbedder.create takes one string or a list holding one "
                "string; token-id input is not supported"
            )
        vectors = await self._service.generate_embeddings([text])
        if len(vectors) != 1:
            raise TemporalMemoryError(
                f"the embedding service returned {len(vectors)} vectors for 1 text"
            )
        return vectors[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""
        vectors = await self._service.generate_embeddings(list(input_data_list))
        if len(vectors) != len(input_data_list):
            raise TemporalMemoryError(
                f"the embedding service returned {len(vectors)} vectors for "
                f"{len(input_data_list)} texts"
            )
        return vectors


class RefusingCrossEncoder(CrossEncoderClient):
    """A cross-encoder that refuses, so no passage is sent to a reranker."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        raise TemporalMemoryError(
            "the graphiti memory backend has no cross-encoder; use a search "
            "configuration without a cross-encoder reranker"
        )


@dataclass(frozen=True, slots=True)
class GraphitiClientSet:
    """The three clients Graphiti needs, none of them None."""

    llm: CountingLLMClient
    embedder: LiteLLMEmbedder
    cross_encoder: RefusingCrossEncoder


def build_clients(
    extraction_model: LLMProvider,
    embedding_service: EmbeddingService,
    *,
    tool_name: str = DEFAULT_TOOL_NAME,
) -> GraphitiClientSet:
    """Build Graphiti's LLM client, embedder and cross-encoder (design seam 4)."""
    disable_telemetry()
    llm = build_llm_client(extraction_model, tool_name=tool_name)
    return GraphitiClientSet(
        llm=CountingLLMClient(llm),
        embedder=LiteLLMEmbedder(embedding_service),
        cross_encoder=RefusingCrossEncoder(),
    )


# --- Graph driver ---------------------------------------------------------

_DRIVER_MODULES = {
    "neo4j": "graphiti_core.driver.neo4j_driver",
    "falkordb": "graphiti_core.driver.falkordb_driver",
    "neptune": "graphiti_core.driver.neptune_driver",
    "kuzu": "graphiti_core.driver.kuzu_driver",
}


_DRIVER_PACKAGES = {
    "neo4j": "neo4j",
    "falkordb": "falkordb",
    "neptune": "boto3, langchain-aws and opensearch-py",
    "kuzu": "kuzu",
}


def _driver_module(driver: str) -> None:
    """Import a Graphiti driver module, or name the missing package (GPH-05).

    Graphiti's FalkorDB module turns the missing package into a plain
    ``ImportError`` without a module name, so both forms are handled.
    """
    try:
        importlib.import_module(_DRIVER_MODULES[driver])
    except ImportError as error:
        if error.name and error.name.split(".")[0] == "graphiti_core":
            raise
        missing = error.name or _DRIVER_PACKAGES[driver]
        raise MemoryConfigError(
            f"Graph driver '{driver}' needs the package '{missing}', which is "
            f"not installed. Install it with: {driver_install_command(driver)}"
        ) from error


def _neo4j_driver(config: GraphBackendConfig) -> GraphDriver:
    _driver_module("neo4j")
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    return Neo4jDriver(
        uri=config.uri or "",
        user=config.user,
        password=_secret(config.password),
    )


def _falkordb_driver(config: GraphBackendConfig) -> GraphDriver:
    _driver_module("falkordb")
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    parts = urlsplit(config.uri or "")
    return FalkorDriver(
        host=parts.hostname or "localhost",
        port=parts.port or FALKORDB_DEFAULT_PORT,
        username=config.user or parts.username,
        password=_secret(config.password) or parts.password,
    )


def _neptune_driver(config: GraphBackendConfig) -> GraphDriver:
    _driver_module("neptune")
    from graphiti_core.driver.neptune_driver import NeptuneDriver

    uri = config.uri or ""
    parts = urlsplit(uri)
    host = f"{parts.scheme}://{parts.hostname}" if parts.hostname else uri
    return NeptuneDriver(
        host=host,
        aoss_host=config.aoss_host or "",
        port=parts.port or NEPTUNE_DEFAULT_PORT,
    )


def _kuzu_driver(config: GraphBackendConfig) -> GraphDriver:
    _driver_module("kuzu")
    from graphiti_core.driver.kuzu_driver import KuzuDriver

    logger.warning(
        "The kuzu graph driver is deprecated upstream and will be removed from "
        "Graphiti. Use neo4j or falkordb."
    )
    return KuzuDriver(db=config.uri or KUZU_IN_MEMORY)


def build_driver(config: GraphBackendConfig) -> GraphDriver:
    """Build the Graphiti driver for ``config.driver`` (GPH-03). No I/O for neo4j."""
    if config.driver == "neo4j":
        return _neo4j_driver(config)
    if config.driver == "falkordb":
        return _falkordb_driver(config)
    if config.driver == "neptune":
        return _neptune_driver(config)
    return _kuzu_driver(config)


# --- Graphiti -------------------------------------------------------------


def entity_type_models(names: Sequence[str]) -> dict[str, type[BaseModel]]:
    """One field-less Pydantic model per entity type name."""
    return {name: create_model(name, __base__=BaseModel) for name in names}


@dataclass(frozen=True, slots=True)
class GraphitiRuntime:
    """A built Graphiti instance and the counting client inside it."""

    graphiti: Graphiti
    llm: CountingLLMClient


AGENT_SDK_MAX_COROUTINES = 4

AGENT_SDK_CONFLICTING_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_VERTEX",
)
"""Variables that make the Claude Code CLI use another provider, endpoint or
credential than the OAuth token the YAML selects (D24)."""


def _max_coroutines(llm: LLMClient) -> int | None:
    """4 for the Agent SDK client, whose every call starts a CLI process.

    ``None`` keeps Graphiti's default for the API clients.
    """
    return AGENT_SDK_MAX_COROUTINES if isinstance(llm, AgentSdkLLMClient) else None


def build_graphiti(
    config: GraphBackendConfig,
    extraction_model: LLMProvider,
    embedding_service: EmbeddingService,
) -> GraphitiRuntime:
    """Build Graphiti with HoloDeck's clients and driver. Never with a None client."""
    disable_telemetry()
    clients = build_clients(extraction_model, embedding_service)
    driver = build_driver(config)
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=clients.llm,
        embedder=clients.embedder,
        cross_encoder=clients.cross_encoder,
        max_coroutines=_max_coroutines(clients.llm.inner),
        prompt_library=holodeck_prompt_library(),
    )
    # Graphiti logs provider exceptions itself, before the backend sees them.
    install_log_redaction(credential_values(extraction_model, config))
    return GraphitiRuntime(graphiti=graphiti, llm=clients.llm)
