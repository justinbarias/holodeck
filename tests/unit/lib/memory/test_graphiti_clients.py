"""Graphiti client factory and driver construction (spec 043, design seam 4, T7).

Real Graphiti client objects are constructed: none of them makes a network
call at construction. No model is called and no credential is read from the
user's environment.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from typing import Any
from urllib.parse import urlsplit

import pytest
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from pydantic import SecretStr

from holodeck.lib.memory import MemoryConfigError, TemporalMemoryError
from holodeck.lib.memory.backends import graphiti_clients
from holodeck.lib.memory.backends.graphiti_agent_sdk import AgentSdkLLMClient
from holodeck.lib.memory.backends.graphiti_clients import (
    CountingLLMClient,
    LiteLLMEmbedder,
    RefusingCrossEncoder,
    build_clients,
    build_driver,
    build_graphiti,
    build_llm_client,
)
from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import GraphBackendConfig

pytestmark = [pytest.mark.unit]

API_KEY = "sk-unit-not-real-9931"
CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
)


@pytest.fixture(autouse=True)
def no_user_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)


class FixedEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), float(i)] for i, text in enumerate(texts)]


def model(provider: ProviderEnum, **kwargs: Any) -> LLMProvider:
    if provider == ProviderEnum.AZURE_OPENAI:
        kwargs.setdefault("endpoint", "https://res.openai.azure.com")
    kwargs.setdefault("api_key", SecretStr(API_KEY))
    return LLMProvider(provider=provider, name="extract-model", **kwargs)


def api_key_of(client: LLMClient) -> str | None:
    if isinstance(client, AnthropicClient):
        return client.client.api_key
    assert isinstance(client, OpenAIClient | AzureOpenAILLMClient | OpenAIGenericClient)
    return client.client.api_key


# --- selection table (design seam 4) ----------------------------------------


@pytest.mark.parametrize(
    ("provider", "extra", "client_type", "base_url"),
    [
        (ProviderEnum.OPENAI, {}, OpenAIClient, "https://api.openai.com/v1/"),
        (
            ProviderEnum.AZURE_OPENAI,
            {},
            AzureOpenAILLMClient,
            "https://res.openai.azure.com/openai/v1/",
        ),
        (ProviderEnum.OLLAMA, {}, OpenAIGenericClient, "http://localhost:11434/v1/"),
        (
            ProviderEnum.OLLAMA,
            {"endpoint": "http://gpu-box:11434/"},
            OpenAIGenericClient,
            "http://gpu-box:11434/v1/",
        ),
    ],
)
def test_gph_01_openai_family_clients_use_the_yaml_endpoint(
    provider: ProviderEnum,
    extra: dict[str, Any],
    client_type: type[LLMClient],
    base_url: str,
) -> None:
    client = build_llm_client(model(provider, **extra))

    assert type(client) is client_type
    assert isinstance(client, OpenAIClient | AzureOpenAILLMClient | OpenAIGenericClient)
    assert str(client.client.base_url) == base_url
    assert client.model == "extract-model"


@pytest.mark.parametrize("auth", [None, AuthProvider.api_key])
def test_gph_04_anthropic_api_key_or_unset_uses_the_api_key_client(
    auth: AuthProvider | None,
) -> None:
    client = build_llm_client(model(ProviderEnum.ANTHROPIC, auth_provider=auth))

    assert type(client) is AnthropicClient
    assert api_key_of(client) == API_KEY


def test_gph_04_anthropic_key_comes_from_validate_credentials_when_yaml_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-not-real")

    client = build_llm_client(model(ProviderEnum.ANTHROPIC, api_key=None))

    assert api_key_of(client) == "sk-ant-env-not-real"


def test_gph_04_anthropic_without_any_key_names_the_variable() -> None:
    with pytest.raises(MemoryConfigError, match="ANTHROPIC_API_KEY"):
        build_llm_client(model(ProviderEnum.ANTHROPIC, api_key=None))


def test_gph_04_oauth_token_uses_the_agent_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-unit-not-real")
    client = build_llm_client(
        model(ProviderEnum.ANTHROPIC, auth_provider=AuthProvider.oauth_token)
    )

    assert type(client) is AgentSdkLLMClient
    assert client.model == "extract-model"


@pytest.mark.parametrize(
    "auth",
    [
        AuthProvider.bedrock,
        AuthProvider.vertex,
        AuthProvider.foundry,
        AuthProvider.custom,
    ],
)
def test_gph_04_other_auth_providers_fail_with_the_model_error(
    auth: AuthProvider,
) -> None:
    with pytest.raises(MemoryConfigError) as raised:
        build_llm_client(
            model(ProviderEnum.ANTHROPIC, auth_provider=auth), tool_name="family_memory"
        )

    message = str(raised.value)
    assert message == (
        "temporal_memory tool 'family_memory': extraction_model auth_provider "
        f"'{auth.value}' is not supported; use api_key or oauth_token"
    )
    assert API_KEY not in message


@pytest.mark.parametrize(
    ("provider", "variable"),
    [
        (ProviderEnum.OPENAI, "OPENAI_API_KEY"),
        (ProviderEnum.AZURE_OPENAI, "AZURE_OPENAI_API_KEY"),
    ],
)
def test_gph_01_missing_openai_key_names_the_variable(
    provider: ProviderEnum, variable: str
) -> None:
    with pytest.raises(MemoryConfigError, match=variable):
        build_llm_client(model(provider, api_key=None))


# --- settings table (design seam 4) -----------------------------------------

ALL_PROVIDERS = [
    ProviderEnum.OPENAI,
    ProviderEnum.AZURE_OPENAI,
    ProviderEnum.OLLAMA,
    ProviderEnum.ANTHROPIC,
]


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_d24_small_model_is_the_extraction_model(provider: ProviderEnum) -> None:
    client = build_llm_client(model(provider))

    assert client.model == "extract-model"
    assert client.small_model == "extract-model"


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_d24_holodeck_max_tokens_default_is_not_passed(provider: ProviderEnum) -> None:
    configured = model(provider)
    assert configured.max_tokens == 1000  # HoloDeck's default, not set in YAML

    client = build_llm_client(configured)

    assert client.max_tokens == 16384


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_d24_max_tokens_set_in_yaml_is_passed(provider: ProviderEnum) -> None:
    client = build_llm_client(model(provider, max_tokens=4000))

    assert client.max_tokens == 4000


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_d24_temperature_comes_from_the_config_and_cache_is_off(
    provider: ProviderEnum,
) -> None:
    client = build_llm_client(model(provider, temperature=0.1))

    assert client.temperature == 0.1
    assert client.cache_enabled is False


# --- embedder and cross-encoder ---------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["Agency B", ["Agency B"]])
async def test_gph_01_embedder_create_returns_one_vector(
    value: str | list[str],
) -> None:
    service = FixedEmbeddings()

    vector = await LiteLLMEmbedder(service).create(value)

    assert vector == [8.0, 0.0]
    assert service.calls == [["Agency B"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [[1, 2, 3], [[1, 2], [3]], ["a", "b"], []])
async def test_gph_01_embedder_create_rejects_other_input(value: Any) -> None:
    service = FixedEmbeddings()

    with pytest.raises(TypeError):
        await LiteLLMEmbedder(service).create(value)
    assert service.calls == []


@pytest.mark.asyncio
async def test_gph_01_embedder_create_batch_returns_one_vector_per_text() -> None:
    vectors = await LiteLLMEmbedder(FixedEmbeddings()).create_batch(["a", "bbb"])

    assert vectors == [[1.0, 0.0], [3.0, 1.0]]


@pytest.mark.asyncio
async def test_gph_01_cross_encoder_refuses_to_rank() -> None:
    with pytest.raises(TemporalMemoryError):
        await RefusingCrossEncoder().rank("query", ["passage"])


# --- Graphiti construction ---------------------------------------------------


class RecordingGraphiti:
    """Records what ``Graphiti(...)`` was given and the telemetry setting then."""

    built: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        RecordingGraphiti.built.append(
            {**kwargs, "telemetry": os.environ.get("GRAPHITI_TELEMETRY_ENABLED")}
        )


def neo4j_config() -> GraphBackendConfig:
    return GraphBackendConfig(
        driver="neo4j",
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password=SecretStr("db-not-real"),
    )


def test_gph_02_telemetry_is_disabled_before_graphiti_is_constructed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHITI_TELEMETRY_ENABLED", "true")
    monkeypatch.setattr(graphiti_clients, "Graphiti", RecordingGraphiti)
    RecordingGraphiti.built = []

    build_graphiti(neo4j_config(), model(ProviderEnum.OPENAI), FixedEmbeddings())

    assert [b["telemetry"] for b in RecordingGraphiti.built] == ["false"]


def test_gph_01_graphiti_is_never_built_with_a_none_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graphiti_clients, "Graphiti", RecordingGraphiti)
    RecordingGraphiti.built = []

    runtime = build_graphiti(
        neo4j_config(), model(ProviderEnum.OPENAI), FixedEmbeddings()
    )

    (kwargs,) = RecordingGraphiti.built
    assert isinstance(kwargs["llm_client"], CountingLLMClient)
    assert isinstance(kwargs["llm_client"].inner, OpenAIClient)
    assert isinstance(kwargs["embedder"], LiteLLMEmbedder)
    assert isinstance(kwargs["cross_encoder"], RefusingCrossEncoder)
    assert isinstance(kwargs["graph_driver"], Neo4jDriver)
    assert runtime.llm is kwargs["llm_client"]


@pytest.mark.asyncio
async def test_gph_01_real_graphiti_keeps_holodeck_clients() -> None:
    runtime = build_graphiti(
        neo4j_config(), model(ProviderEnum.OPENAI), FixedEmbeddings()
    )
    try:
        graphiti = runtime.graphiti
        assert graphiti.llm_client is runtime.llm
        assert isinstance(graphiti.embedder, LiteLLMEmbedder)
        assert isinstance(graphiti.cross_encoder, RefusingCrossEncoder)
        assert graphiti.token_tracker is runtime.llm.inner.token_tracker
    finally:
        await runtime.graphiti.close()


def test_gph_01_build_clients_returns_three_clients() -> None:
    clients = build_clients(model(ProviderEnum.OLLAMA), FixedEmbeddings())

    assert isinstance(clients.llm.inner, OpenAIGenericClient)
    assert isinstance(clients.embedder, LiteLLMEmbedder)
    assert isinstance(clients.cross_encoder, RefusingCrossEncoder)


@pytest.mark.asyncio
async def test_ing_06_counting_client_counts_and_delegates() -> None:
    replies: list[str | None] = []

    class Inner(OpenAIGenericClient):
        async def generate_response(  # type: ignore[override]
            self, messages: Any, response_model: Any = None, **kwargs: Any
        ) -> dict[str, Any]:
            replies.append(kwargs.get("prompt_name"))
            return {"ok": True}

    inner = Inner(config=LLMConfig(api_key="sk-not-real"))
    counting = CountingLLMClient(inner)

    first = await counting.generate_response(
        [], prompt_name="extract_nodes.extract_text"
    )
    await counting.generate_response([], prompt_name="extract_edges.edge")

    assert first == {"ok": True}
    assert counting.calls == 2
    assert replies == ["extract_nodes.extract_text", "extract_edges.edge"]
    assert counting.token_tracker is inner.token_tracker


# --- drivers (GPH-03, GPH-05) ------------------------------------------------


@pytest.mark.asyncio
async def test_gph_03_neo4j_driver_is_built_without_io() -> None:
    driver = build_driver(neo4j_config())
    try:
        assert isinstance(driver, Neo4jDriver)
    finally:
        await driver.close()


def fake_driver_module(
    monkeypatch: pytest.MonkeyPatch, module_name: str, class_name: str
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class FakeDriver:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    module = types.ModuleType(module_name)
    setattr(module, class_name, FakeDriver)
    monkeypatch.setitem(sys.modules, module_name, module)
    return calls


def test_gph_03_falkordb_driver_gets_host_and_port_from_the_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_driver_module(
        monkeypatch, "graphiti_core.driver.falkordb_driver", "FalkorDriver"
    )
    config = GraphBackendConfig(
        driver="falkordb",
        uri="redis://graph-host:6390",
        user="falkor",
        password=SecretStr("db-not-real"),
    )

    build_driver(config)

    assert calls == [
        {
            "host": "graph-host",
            "port": 6390,
            "username": "falkor",
            "password": "db-not-real",
        }
    ]


def test_gph_03_neptune_driver_gets_host_and_aoss_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_driver_module(
        monkeypatch, "graphiti_core.driver.neptune_driver", "NeptuneDriver"
    )
    config = GraphBackendConfig(
        driver="neptune",
        uri="neptune-db://cluster.example:8182",
        aoss_host="search.example",
    )

    build_driver(config)

    assert calls == [
        {
            "host": "neptune-db://cluster.example",
            "aoss_host": "search.example",
            "port": 8182,
        }
    ]


def test_gph_03_kuzu_driver_warns_it_is_deprecated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = fake_driver_module(
        monkeypatch, "graphiti_core.driver.kuzu_driver", "KuzuDriver"
    )

    with caplog.at_level(logging.WARNING, logger=graphiti_clients.__name__):
        build_driver(GraphBackendConfig(driver="kuzu"))

    assert calls == [{"db": ":memory:"}]
    assert "deprecated" in caplog.text


@pytest.mark.parametrize(
    ("driver", "package", "extra"),
    [
        ("falkordb", "falkordb", "falkordb"),
        ("kuzu", "kuzu", "kuzu"),
        ("neptune", "boto3", "neptune"),
    ],
)
def test_gph_05_missing_driver_package_names_its_install_command(
    monkeypatch: pytest.MonkeyPatch, driver: str, package: str, extra: str
) -> None:
    monkeypatch.delitem(
        sys.modules, f"graphiti_core.driver.{driver}_driver", raising=False
    )
    monkeypatch.setitem(sys.modules, package, None)
    config = GraphBackendConfig(
        driver=driver,  # type: ignore[arg-type]
        uri="neptune-db://h" if driver == "neptune" else "redis://h:1",
        aoss_host="search.example",
    )

    with pytest.raises(MemoryConfigError) as raised:
        build_driver(config)

    message = str(raised.value)
    assert f"'{package}'" in message
    assert f"pip install 'graphiti-core[{extra}]>=0.31,<0.32'" in message


def test_gph_01_openai_endpoint_is_explicit_whatever_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.example/v1")

    client = build_llm_client(model(ProviderEnum.OPENAI))

    base_url = str(client.client.base_url)  # type: ignore[attr-defined]
    assert urlsplit(base_url).hostname == "api.openai.com"


def test_gph_01_anthropic_endpoint_is_explicit_whatever_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://other.example")

    client = build_llm_client(model(ProviderEnum.ANTHROPIC))

    base_url = str(client.client.base_url)  # type: ignore[attr-defined]
    assert urlsplit(base_url).hostname == "api.anthropic.com"


@pytest.mark.parametrize(
    "variable",
    [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_VERTEX",
    ],
)
def test_gph_04_oauth_refuses_a_variable_that_reroutes_the_cli(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-unit-not-real")
    monkeypatch.setenv(variable, "1")

    with pytest.raises(MemoryConfigError, match=variable) as raised:
        build_llm_client(
            model(ProviderEnum.ANTHROPIC, auth_provider=AuthProvider.oauth_token)
        )
    assert "oauth-unit-not-real" not in str(raised.value)


def test_gph_04_oauth_without_a_token_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    with pytest.raises(MemoryConfigError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        build_llm_client(
            model(ProviderEnum.ANTHROPIC, auth_provider=AuthProvider.oauth_token)
        )


def test_d24_oauth_warns_that_temperature_and_max_tokens_are_not_applied(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-unit-not-real")
    configured = LLMProvider(
        provider=ProviderEnum.ANTHROPIC,
        name="extract-model",
        auth_provider=AuthProvider.oauth_token,
        temperature=0.1,
        max_tokens=100,
    )

    with caplog.at_level("WARNING"):
        build_llm_client(configured)

    assert "max_tokens and temperature is not applied" in caplog.text


def test_gph_01_credential_values_come_from_every_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from holodeck.lib.memory.backends.graphiti_secrets import credential_values

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-canary-7781")
    config = GraphBackendConfig(
        driver="falkordb",
        uri="redis://user:uri-canary-55@host:6379",
        password=SecretStr("field-canary-66"),
    )

    values = credential_values(
        LLMProvider(provider=ProviderEnum.OPENAI, name="m"), config
    )

    assert values == {"sk-env-canary-7781", "uri-canary-55", "field-canary-66"}


def test_gph_01_graphiti_log_records_are_redacted() -> None:
    import logging

    from holodeck.lib.memory.backends.graphiti_secrets import install_log_redaction

    child = logging.getLogger("graphiti_core.llm_client.unit_probe")
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture()
    child.addHandler(handler)
    try:
        install_log_redaction(frozenset({"sk-log-canary-9914"}))
        try:
            raise RuntimeError("Incorrect API key provided: sk-log-canary-9914")
        except RuntimeError as error:
            child.error("request failed: %s", error, exc_info=True)
    finally:
        child.removeHandler(handler)

    assert len(records) == 1
    assert records[0].getMessage() == "request failed: Incorrect API key provided: ***"
    assert "sk-log-canary-9914" not in (records[0].exc_text or "")
    assert "RuntimeError" in (records[0].exc_text or "")
