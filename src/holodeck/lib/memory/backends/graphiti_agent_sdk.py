"""Graphiti LLM client that calls Claude through the Claude Agent SDK (T23).

Design seam 4, "Agent SDK client" (spec 043, GPH-04, D24). This client is
chosen for ``extraction_model`` with ``provider: anthropic`` and
``auth_provider: oauth_token``. Each call starts the Claude Code CLI with no
tools and no filesystem settings, asks for structured output with the reply
model's JSON schema, and validates the reply with that model.

The CLI reads the OAuth token from the environment. This module never reads,
logs or reports it.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import dataclass

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel, ValidationError

from holodeck.lib.memory.errors import TemporalMemoryError

DEFAULT_MAX_ATTEMPTS = 3

# The CLI implements ``--json-schema`` as a ``StructuredOutput`` tool whose
# call ends the turn, so a valid reply takes one turn. A second turn lets the
# CLI send its own schema-validation error back to the model once before the
# query stops with ``error_max_turns``.
MAX_TURNS = 2

ERROR_DETAIL_LIMIT = 500
RETRY_DETAIL_LIMIT = 2000

_FENCE = re.compile(r"^```(?:json)?[ \t]*\n(.*?)\n?```$", re.DOTALL)

_prompt_name: ContextVar[str | None] = ContextVar(
    "holodeck_agent_sdk_prompt_name", default=None
)


@dataclass(frozen=True, slots=True)
class _Reply:
    """What one ``query`` call returned, converted from the SDK messages."""

    text: str
    structured_output: object
    input_tokens: int
    output_tokens: int


class _ResultError(TemporalMemoryError):
    """The SDK ended a query with an error result. Carries the tokens it used."""

    def __init__(self, message: str, input_tokens: int, output_tokens: int) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _join(messages: list[Message], *, system: bool) -> str:
    return "\n\n".join(
        message.content for message in messages if (message.role == "system") is system
    )


def _strip_fence(text: str) -> str:
    """Remove one surrounding Markdown code fence, if there is one."""
    stripped = text.strip()
    match = _FENCE.match(stripped)
    return match.group(1).strip() if match else stripped


def _usage_int(usage: dict[str, object], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _tokens(result: ResultMessage) -> tuple[int, int]:
    """Input and output tokens from the result message's ``usage``.

    Input counts uncached, cache-write and cache-read input tokens, because
    the CLI caches the prompt prefix and ``input_tokens`` alone leaves it out.
    """
    usage: dict[str, object] = dict(result.usage or {})
    input_tokens = (
        _usage_int(usage, "input_tokens")
        + _usage_int(usage, "cache_creation_input_tokens")
        + _usage_int(usage, "cache_read_input_tokens")
    )
    return input_tokens, _usage_int(usage, "output_tokens")


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


class AgentSdkLLMClient(LLMClient):
    """Graphiti ``LLMClient`` over ``claude_agent_sdk.query`` (design seam 4)."""

    def __init__(self, model: str, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        super().__init__(LLMConfig(model=model, small_model=model), cache=False)
        self.max_attempts = max_attempts

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
        """Keep ``prompt_name`` for the token tracker, then run Graphiti's flow."""
        token = _prompt_name.set(prompt_name)
        try:
            response: dict[str, object] = await super().generate_response(
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
        finally:
            _prompt_name.reset(token)

    def _options(
        self, system: str, response_model: type[BaseModel] | None, model: str
    ) -> ClaudeAgentOptions:
        output_format = (
            {"type": "json_schema", "schema": response_model.model_json_schema()}
            if response_model is not None
            else None
        )
        return ClaudeAgentOptions(
            model=model,
            system_prompt=system,
            tools=[],
            setting_sources=[],
            strict_mcp_config=True,
            max_turns=MAX_TURNS,
            output_format=output_format,
        )

    async def _query(self, prompt: str, options: ClaudeAgentOptions) -> _Reply:
        text_parts: list[str] = []
        result: ResultMessage | None = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                text_parts.extend(
                    block.text
                    for block in message.content
                    if isinstance(block, TextBlock)
                )
            elif isinstance(message, ResultMessage):
                result = message
        if result is None:
            raise TemporalMemoryError(
                "the Claude Agent SDK query ended without a result message"
            )
        input_tokens, output_tokens = _tokens(result)
        if result.is_error:
            status = (
                f", API status {result.api_error_status}"
                if result.api_error_status is not None
                else ""
            )
            raise _ResultError(
                f"the Claude Agent SDK returned an error result "
                f"(subtype '{result.subtype}'{status})",
                input_tokens,
                output_tokens,
            )
        return _Reply(
            text="".join(text_parts),
            structured_output=result.structured_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _parse(
        reply: _Reply, response_model: type[BaseModel] | None
    ) -> dict[str, object]:
        """Validate one reply. Raises ``ValueError`` (JSON) or ``ValidationError``."""
        value: object
        if response_model is not None and isinstance(reply.structured_output, dict):
            value = reply.structured_output
        else:
            value = json.loads(_strip_fence(reply.text))
        if response_model is not None:
            return response_model.model_validate(value).model_dump(mode="json")
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        return {"content": value}

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, object]:
        options = self._options(
            _join(messages, system=True), response_model, model or self.model or ""
        )
        user = _join(messages, system=False)
        prompt = user
        input_tokens = 0
        output_tokens = 0
        last_error = ""
        try:
            for _ in range(self.max_attempts):
                try:
                    reply = await self._query(prompt, options)
                except _ResultError as error:
                    # The failed call still used tokens.
                    input_tokens += error.input_tokens
                    output_tokens += error.output_tokens
                    raise
                input_tokens += reply.input_tokens
                output_tokens += reply.output_tokens
                try:
                    return self._parse(reply, response_model)
                except (ValueError, ValidationError) as error:
                    last_error = str(error)
                prompt = (
                    f"{user}\n\nYour previous reply was not valid: "
                    f"{_truncate(last_error, RETRY_DETAIL_LIMIT)}\n"
                    "Reply again with one JSON object that matches the schema."
                )
        finally:
            if input_tokens or output_tokens:
                self.token_tracker.record(
                    _prompt_name.get(), input_tokens, output_tokens
                )
        target = response_model.__name__ if response_model is not None else "JSON"
        raise TemporalMemoryError(
            f"the Claude Agent SDK reply did not validate against {target} after "
            f"{self.max_attempts} attempts: "
            f"{_truncate(last_error, ERROR_DETAIL_LIMIT)}"
        )
