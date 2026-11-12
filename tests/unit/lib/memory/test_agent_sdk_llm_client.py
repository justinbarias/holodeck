"""Unit tests for the Graphiti LLM client over the Claude Agent SDK (T23).

``claude_agent_sdk.query`` is replaced by a scripted fake: no CLI process,
no network and no credentials.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import Message as SdkMessage
from graphiti_core.prompts.models import Message
from pydantic import BaseModel, SecretStr

from holodeck.lib.memory import TemporalMemoryError
from holodeck.lib.memory.backends import graphiti_agent_sdk, graphiti_clients
from holodeck.lib.memory.backends.graphiti_agent_sdk import AgentSdkLLMClient
from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import GraphBackendConfig

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

MODEL = "claude-extract-model"
CREDENTIAL_CANARY = "oauth-token-not-real-5512"


class Entities(BaseModel):
    names: list[str]


def result(
    *,
    structured_output: object = None,
    is_error: bool = False,
    subtype: str = "success",
    usage: dict[str, Any] | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=10,
        duration_api_ms=8,
        is_error=is_error,
        num_turns=1,
        session_id="session-1",
        usage=usage,
        structured_output=structured_output,
    )


def text(value: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=value)], model=MODEL)


@dataclass
class FakeQuery:
    """Plays one scripted list of SDK messages per ``query`` call."""

    replies: list[list[SdkMessage]]
    prompts: list[str] = field(default_factory=list)
    options: list[ClaudeAgentOptions] = field(default_factory=list)

    async def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[SdkMessage]:
        self.prompts.append(prompt)
        self.options.append(options)
        for message in self.replies[len(self.prompts) - 1]:
            yield message


@pytest.fixture
def fake_query(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(*replies: list[SdkMessage]) -> FakeQuery:
        fake = FakeQuery(list(replies))
        monkeypatch.setattr(graphiti_agent_sdk, "query", fake)
        return fake

    return install


def messages() -> list[Message]:
    return [
        Message(role="system", content="You extract entities."),
        Message(role="user", content="Ada met Grace."),
    ]


async def generate(
    client: AgentSdkLLMClient,
    response_model: type[BaseModel] | None = Entities,
    prompt_name: str | None = "extract_nodes.extract_message",
) -> dict[str, Any]:
    return await client.generate_response(
        messages(), response_model=response_model, prompt_name=prompt_name
    )


async def test_gph_04_structured_reply_returns_the_validated_dict(
    fake_query: Any,
) -> None:
    fake = fake_query([text("ignored"), result(structured_output={"names": ["Ada"]})])

    reply = await generate(AgentSdkLLMClient(MODEL))

    assert reply == {"names": ["Ada"]}
    assert len(fake.prompts) == 1


async def test_gph_04_fenced_json_text_reply_is_parsed(fake_query: Any) -> None:
    fake_query([text('```json\n{"names": ["Ada", "Grace"]}\n```'), result()])

    reply = await generate(AgentSdkLLMClient(MODEL))

    assert reply == {"names": ["Ada", "Grace"]}


async def test_gph_04_reply_without_a_response_model_is_returned_as_a_dict(
    fake_query: Any,
) -> None:
    fake_query([text('{"a": 1}'), result()], [text("[1, 2]"), result()])
    client = AgentSdkLLMClient(MODEL)

    assert await generate(client, response_model=None) == {"a": 1}
    assert await generate(client, response_model=None) == {"content": [1, 2]}


async def test_gph_04_invalid_reply_is_retried_with_the_validation_error(
    fake_query: Any,
) -> None:
    fake = fake_query(
        [result(structured_output={"names": "Ada"})],
        [result(structured_output={"names": ["Ada"]})],
    )

    reply = await generate(AgentSdkLLMClient(MODEL))

    assert reply == {"names": ["Ada"]}
    assert len(fake.prompts) == 2
    assert "Ada met Grace." in fake.prompts[1]
    assert "previous reply was not valid" in fake.prompts[1]
    assert "names" in fake.prompts[1] and "list" in fake.prompts[1]
    assert "previous reply was not valid" not in fake.prompts[0]


async def test_gph_04_three_invalid_replies_raise_after_three_queries(
    fake_query: Any,
) -> None:
    fake = fake_query(*[[text("not json"), result()] for _ in range(4)])

    with pytest.raises(TemporalMemoryError) as raised:
        await generate(AgentSdkLLMClient(MODEL))

    assert len(fake.prompts) == 3
    assert "Entities" in str(raised.value)
    assert "3 attempts" in str(raised.value)
    assert "Expecting value" in str(raised.value)


async def test_gph_04_error_result_raises_without_retry(
    fake_query: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", CREDENTIAL_CANARY)
    fake = fake_query(
        [result(is_error=True, subtype="error_max_turns")],
        [result(structured_output={"names": ["Ada"]})],
    )

    with pytest.raises(TemporalMemoryError, match="error_max_turns") as raised:
        await generate(AgentSdkLLMClient(MODEL))

    assert len(fake.prompts) == 1
    assert CREDENTIAL_CANARY not in str(raised.value)


async def test_gph_04_options_have_no_tools_no_settings_and_the_schema(
    fake_query: Any,
) -> None:
    fake = fake_query([result(structured_output={"names": []})])

    await generate(AgentSdkLLMClient(MODEL))

    (options,) = fake.options
    assert options.tools == []
    assert options.setting_sources == []
    assert options.permission_mode is None
    assert options.model == MODEL
    assert isinstance(options.system_prompt, str)
    assert options.system_prompt.startswith("You extract entities.")
    assert "You extract entities." not in fake.prompts[0]
    assert fake.prompts[0].startswith("Ada met Grace.")
    assert options.output_format == {
        "type": "json_schema",
        "schema": Entities.model_json_schema(),
    }
    assert options.max_turns == graphiti_agent_sdk.MAX_TURNS


async def test_gph_04_options_use_the_per_call_model_override(
    fake_query: Any,
) -> None:
    fake = fake_query([result(structured_output={"names": []})])

    await AgentSdkLLMClient(MODEL).generate_response(
        messages(), response_model=Entities, model="claude-other"
    )

    assert fake.options[0].model == "claude-other"


async def test_gph_04_tokens_from_every_attempt_are_recorded(
    fake_query: Any,
) -> None:
    usage = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 1000,
        "output_tokens": 7,
    }
    fake_query(
        [result(structured_output={"names": "x"}, usage=usage)],
        [result(structured_output={"names": ["x"]}, usage=usage)],
    )
    client = AgentSdkLLMClient(MODEL)

    await generate(client)

    usage_by_prompt = client.token_tracker.get_usage()
    assert list(usage_by_prompt) == ["extract_nodes.extract_message"]
    recorded = usage_by_prompt["extract_nodes.extract_message"]
    assert recorded.call_count == 1
    assert recorded.total_input_tokens == 2220
    assert recorded.total_output_tokens == 14


async def test_d24_small_model_is_the_model_and_cache_is_off() -> None:
    client = AgentSdkLLMClient(MODEL)

    assert client.model == MODEL
    assert client.small_model == MODEL
    assert client.cache_enabled is False


def oauth_model() -> LLMProvider:
    return LLMProvider(
        provider=ProviderEnum.ANTHROPIC,
        name=MODEL,
        auth_provider=AuthProvider.oauth_token,
    )


class RecordingGraphiti:
    built: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        RecordingGraphiti.built.append(kwargs)


@pytest.mark.parametrize(
    ("extraction_model", "expected"),
    [
        (oauth_model(), 4),
        (
            LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="extract-model",
                api_key=SecretStr("sk-unit-not-real"),
            ),
            None,
        ),
    ],
)
async def test_d24_agent_sdk_graphiti_runs_four_coroutines(
    monkeypatch: pytest.MonkeyPatch,
    extraction_model: LLMProvider,
    expected: int | None,
) -> None:
    class NoEmbeddings:
        async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
            return []

    monkeypatch.setattr(graphiti_clients, "Graphiti", RecordingGraphiti)
    monkeypatch.setattr(graphiti_clients, "build_driver", lambda config: None)
    RecordingGraphiti.built = []
    config = GraphBackendConfig(
        driver="neo4j",
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password=SecretStr("db-not-real"),
    )

    graphiti_clients.build_graphiti(config, extraction_model, NoEmbeddings())

    (kwargs,) = RecordingGraphiti.built
    assert kwargs["max_coroutines"] == expected


async def test_gph_04_tokens_of_an_error_result_are_still_recorded(
    fake_query: Any,
) -> None:
    fake_query(
        [text("not json"), result(usage={"input_tokens": 10, "output_tokens": 2})],
        [
            result(
                is_error=True,
                subtype="error_max_turns",
                usage={"input_tokens": 100, "output_tokens": 20},
            )
        ],
    )
    client = AgentSdkLLMClient(MODEL)

    with pytest.raises(TemporalMemoryError, match="error_max_turns"):
        await generate(client)

    total = client.token_tracker.get_total_usage()
    assert (total.input_tokens, total.output_tokens) == (110, 22)
