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

import json
import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.lib.backends.base import (
    AgentSession,
    BackendInitError,
    BackendSessionError,
    ExecutionResult,
)
from holodeck.models.agent import Agent
from holodeck.models.llm import ProviderEnum
from holodeck.models.token_usage import TokenUsage

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


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Coerce a raw tool-call ``arguments`` value into a dict.

    The SDK surfaces function-call arguments as a JSON string; fall back to a
    ``{"raw": ...}`` wrapper if it is not valid JSON.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        return parsed if isinstance(parsed, dict) else {"raw": raw}
    return {}


def _to_execution_result(result: Any) -> ExecutionResult:
    """Map an SDK run result onto a provider-agnostic ``ExecutionResult``.

    Extracts the final text, tool calls / results (parallel lists for positional
    pairing, matching the SK contract), token usage, and the turn count.

    Args:
        result: A ``RunResult`` from ``Runner.run``.

    Returns:
        A populated ``ExecutionResult``.
    """
    from agents.items import ToolCallItem, ToolCallOutputItem

    final = result.final_output
    response = str(final) if final is not None else ""

    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    call_name_by_id: dict[str, str] = {}

    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            name = str(getattr(raw, "name", "") or "")
            call_id = str(getattr(raw, "call_id", "") or "")
            arguments = _parse_tool_arguments(getattr(raw, "arguments", None))
            if call_id:
                call_name_by_id[call_id] = name
            record: dict[str, Any] = {"name": name, "arguments": arguments}
            if call_id:
                record["call_id"] = call_id
            tool_calls.append(record)
        elif isinstance(item, ToolCallOutputItem):
            raw_out = item.raw_item
            call_id = ""
            if isinstance(raw_out, dict):
                call_id = str(raw_out.get("call_id", "") or "")
            name = call_name_by_id.get(call_id, "")
            tool_results.append(
                {
                    "name": name,
                    "result": str(item.output),
                    "call_id": call_id,
                }
            )

    token_usage = TokenUsage.zero()
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is not None:
        prompt = int(getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "output_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
        token_usage = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    raw_responses = getattr(result, "raw_responses", None) or []
    num_turns = max(1, len(raw_responses))

    return ExecutionResult(
        response=response,
        tool_calls=tool_calls,
        tool_results=tool_results,
        token_usage=token_usage,
        num_turns=num_turns,
        thinking="",
    )


class OpenAIAgentsSession:
    """Stateful multi-turn session backed by an SDK ``SQLiteSession``.

    Each ``send`` runs the SDK agent loop with the shared ``SQLiteSession`` so
    the SDK persists turn history. Idle sessions are SQLite rows, not held
    processes.
    """

    def __init__(self, sdk_agent: Any, sqlite_session: Any) -> None:
        """Bind the session to an SDK agent and its SQLite-backed history."""
        self._sdk_agent = sdk_agent
        self._session = sqlite_session

    async def prepare(self) -> None:
        """No-op. The SQLite session is ready at construction time."""
        return None

    async def send(self, message: str) -> ExecutionResult:
        """Run one turn against the persistent session.

        Args:
            message: The user message to send to the agent.

        Returns:
            ExecutionResult for this turn. Runtime failures are returned as an
            error result (``is_error=True``) rather than raised, so the multi-turn
            executor can record per-turn failures.
        """
        from agents import Runner

        try:
            result = await Runner.run(self._sdk_agent, message, session=self._session)
        except Exception as exc:  # noqa: BLE001 - surfaced via ExecutionResult
            return ExecutionResult(
                response="",
                is_error=True,
                error_reason=f"{type(exc).__name__}: {exc}",
            )
        return _to_execution_result(result)

    async def send_streaming(self, message: str) -> AsyncGenerator[str, None]:
        """Stream the agent response token by token.

        Runs the SDK agent loop via ``Runner.run_streamed`` and forwards each
        model text delta as it arrives. Text deltas surface as raw-response
        events carrying a ``ResponseTextDeltaEvent``; tool-call and lifecycle
        events are ignored for the streamed text channel.

        Args:
            message: The user message to send to the agent.

        Yields:
            String chunks of the agent response as the model produces them.
        """
        from agents import Runner
        from openai.types.responses import ResponseTextDeltaEvent

        result = Runner.run_streamed(self._sdk_agent, message, session=self._session)
        async for event in result.stream_events():
            if event.type != "raw_response_event":
                continue
            data = event.data
            if isinstance(data, ResponseTextDeltaEvent) and data.delta:
                yield data.delta

    async def close(self) -> None:
        """Release the SQLite session connection, if any."""
        close = getattr(self._session, "close", None)
        if callable(close):
            maybe = close()
            if hasattr(maybe, "__await__"):
                await maybe


class OpenAIAgentsBackend:
    """OpenAI Agents SDK backend implementing the ``AgentBackend`` protocol.

    Wraps an SDK ``Agent`` (built from the HoloDeck agent config) and drives it
    through ``Runner.run`` for single-turn invocations and ``SQLiteSession``-
    backed multi-turn sessions.
    """

    def __init__(self, agent: Agent, base_dir: Path | None = None) -> None:
        """Initialize the backend with agent configuration.

        Args:
            agent: The HoloDeck agent configuration.
            base_dir: Directory for resolving relative tool/instruction paths.
                Falls back to the ``agent_base_dir`` context variable.
        """
        self._agent_config = agent
        self._base_dir = base_dir
        self._sdk_agent: Any | None = None

    def _resolve_base_dir(self) -> Path | None:
        """Return the explicit base_dir or the ``agent_base_dir`` context value."""
        if self._base_dir is not None:
            return self._base_dir
        from holodeck.config.context import agent_base_dir

        base = agent_base_dir.get()
        return Path(base) if base else None

    async def initialize(self) -> None:
        """Build the SDK ``Agent`` — validating credentials and tools.

        Raises:
            BackendInitError: If credentials are missing or the provider is
                unsupported.
            ConfigError: If a tool config is unsupported or fails to load.
        """
        from agents import Agent as SDKAgent
        from agents import ModelSettings

        from holodeck.lib.backends.openai_agents_tool_adapters import build_sdk_tools
        from holodeck.lib.instruction_resolver import resolve_instructions

        base_dir = self._resolve_base_dir()
        model = _build_model(self._agent_config)
        instructions = resolve_instructions(
            self._agent_config.instructions, base_dir=base_dir
        )
        tools = build_sdk_tools(self._agent_config.tools, base_dir)

        model_cfg = self._agent_config.model
        model_settings = ModelSettings(
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
            max_tokens=model_cfg.max_tokens,
        )

        self._sdk_agent = SDKAgent(
            name=self._agent_config.name,
            instructions=instructions,
            model=model,
            tools=tools,
            model_settings=model_settings,
        )

    def _require_agent(self) -> Any:
        """Return the built SDK agent or raise if ``initialize`` was skipped."""
        if self._sdk_agent is None:
            raise BackendInitError(
                "OpenAIAgentsBackend.initialize() must be called before use."
            )
        return self._sdk_agent

    async def invoke_once(
        self,
        message: str,
        context: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        """Execute a single stateless agent turn.

        Args:
            message: The user message to send to the agent.
            context: Optional prior turns (unused in the MVP).

        Returns:
            ExecutionResult for the turn.

        Raises:
            BackendSessionError: If the SDK run fails at runtime.
        """
        del context  # not threaded into the SDK loop for the MVP
        sdk_agent = self._require_agent()
        from agents import Runner

        try:
            result = await Runner.run(sdk_agent, message)
        except Exception as exc:  # noqa: BLE001 - re-raised as backend error
            raise BackendSessionError(
                f"OpenAI Agents run failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _to_execution_result(result)

    async def create_session(self, *, eager_connect: bool = True) -> AgentSession:
        """Create a stateful multi-turn session backed by a fresh SQLiteSession.

        Args:
            eager_connect: Accepted for protocol compatibility; the SQLite
                session is created synchronously regardless.

        Returns:
            An ``OpenAIAgentsSession`` bound to this backend's SDK agent.
        """
        del eager_connect  # no lazy-connect transport for this backend
        sdk_agent = self._require_agent()
        from agents import SQLiteSession

        session = SQLiteSession(f"holodeck-{uuid.uuid4().hex}")
        return OpenAIAgentsSession(sdk_agent=sdk_agent, sqlite_session=session)

    async def teardown(self) -> None:
        """Release backend resources. No-op for this backend."""
        return None
