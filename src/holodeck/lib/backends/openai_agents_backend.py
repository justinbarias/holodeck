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
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.lib.backends.base import (
    AgentSession,
    BackendBudgetExceededError,
    BackendInitError,
    BackendSessionError,
    ExecutionResult,
)
from holodeck.models.agent import Agent
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.openai_config import OpenAIConfig
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool import HierarchicalDocumentToolConfig, VectorstoreTool

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from agents import ModelSettings, OpenAIResponsesModel, RunConfig, RunResult
    from agents.models.interface import Model
    from openai import AsyncOpenAI
    from openai.types.shared import Reasoning
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from pydantic import SecretStr

logger = logging.getLogger(__name__)


def _resolve_secret(value: SecretStr | None) -> str | None:
    """Return the plain string for a ``SecretStr`` credential, or ``None``."""
    if value is None:
        return None
    return value.get_secret_value() or None


def _azure_v1_base_url(endpoint: str) -> str:
    """Normalize an Azure endpoint to the OpenAI-compatible ``/openai/v1`` base.

    The Responses API is served from the Azure ``v1`` surface (both
    ``*.openai.azure.com`` and Foundry ``*.services.ai.azure.com`` resources).
    A bare resource endpoint has ``/openai/v1`` appended; an endpoint that
    already targets the v1 surface is used as-is.
    """
    base = endpoint.rstrip("/")
    if base.endswith("/openai/v1"):
        return base
    return f"{base}/openai/v1"


def _is_reasoning_model(name: str) -> bool:
    """Heuristic for OpenAI reasoning models (o-series, ``gpt-5``+).

    Reasoning models reject the sampling params ``temperature`` / ``top_p`` and
    use ``max_output_tokens`` rather than ``max_tokens``. For Azure the *name* is
    the deployment name, so this matches the common convention of embedding the
    base model in the deployment name (opaque deployment names won't be
    detected, in which case set sane sampling params in the config).
    """
    n = name.strip().lower()
    return n.startswith(("o1", "o3", "o4", "gpt-5"))


def _preflight_credentials(agent: Agent) -> tuple[str, str | None]:
    """Validate provider credentials for *agent* without any global side effects.

    Resolves and checks the required credentials for the configured provider and
    returns ``(api_key, endpoint)`` (``endpoint`` is ``None`` for ``openai``).
    Unlike :func:`_build_model`, this performs **no** SDK global mutation
    (``set_default_openai_key`` / ``set_tracing_disabled``), so it is safe to
    call from config-time validation (``validate_openai_agents``).

    Args:
        agent: The agent configuration whose ``model`` selects the provider.

    Returns:
        ``(api_key, endpoint)`` — ``endpoint`` is ``None`` for ``openai``.

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
        return api_key, None

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
        return api_key, endpoint

    raise BackendInitError(
        f"The openai_agents backend does not support provider '{provider.value}'."
    )


def _fallback_model_name(agent: Agent) -> str | None:
    """Return the configured ``openai.fallback_model`` for *agent*, or ``None``."""
    if agent.openai is not None:
        return agent.openai.fallback_model
    return None


def _build_azure_client(
    api_key: str, endpoint: str, api_version: str | None
) -> AsyncOpenAI:
    """Build the Azure ``AsyncOpenAI`` client for the ``/openai/v1`` surface.

    Args:
        api_key: The resolved Azure API key.
        endpoint: The resolved Azure resource endpoint.
        api_version: An optional pinned ``api-version`` query value.

    Returns:
        A configured ``AsyncOpenAI`` client targeting the Azure v1 surface.
    """
    from openai import AsyncOpenAI

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": _azure_v1_base_url(endpoint),
    }
    # The v1 surface does not require an api-version; honor one only if the
    # config pins it (e.g. to opt into a preview surface).
    if api_version:
        client_kwargs["default_query"] = {"api-version": api_version}
    return AsyncOpenAI(**client_kwargs)


def _build_model(agent: Agent) -> str | OpenAIResponsesModel | Model:
    """Build the SDK ``model=`` argument for *agent* and validate credentials.

    Both providers run on the Responses API. For ``provider: openai`` the
    default Responses client is used, so the model-name string is returned;
    ``OPENAI_API_KEY`` (or an explicit ``model.api_key``) must be available.

    For ``provider: azure_openai`` a plain ``AsyncOpenAI`` client pointed at the
    Azure ``/openai/v1`` surface is wrapped as an ``OpenAIResponsesModel``. SDK
    tracing is disabled because no OpenAI-platform key is present to upload
    traces with.

    When ``openai.fallback_model`` is set, the primary model is wrapped in a
    fallback ``Model`` (FR-033): on a retryable upstream error (429 / 5xx) the
    request is re-issued once against the fallback model, built with the *same*
    credentials/client as the primary (for Azure the fallback name is a
    deployment on the same endpoint). Without ``fallback_model`` the primary is
    returned directly, with no wrapper.

    Credential resolution and validation are delegated to the side-effect-free
    :func:`_preflight_credentials`; this function additionally performs the SDK
    global mutations (``set_default_openai_key`` / ``set_tracing_disabled``)
    that must NOT run during config-time validation.

    Args:
        agent: The agent configuration whose ``model`` selects the provider.

    Returns:
        The model-name string (OpenAI, no fallback), an ``OpenAIResponsesModel``
        instance (Azure, no fallback), or a wrapping fallback ``Model`` when
        ``openai.fallback_model`` is set — ready to pass as ``Agent(model=...)``.

    Raises:
        BackendInitError: If a required credential is missing, or the provider
            is not supported by this backend.
    """
    model_cfg = agent.model
    api_key, endpoint = _preflight_credentials(agent)
    fallback_name = _fallback_model_name(agent)

    if model_cfg.provider == ProviderEnum.OPENAI:
        # The default Responses client reads OPENAI_API_KEY from the env; make
        # an explicit config-supplied key authoritative for this process.
        from agents import set_default_openai_key

        set_default_openai_key(api_key)
        if fallback_name is None:
            return model_cfg.name
        # Resolve both names to concrete Models via the default OpenAI provider
        # so they share one client, then wrap them.
        from agents.models.openai_provider import OpenAIProvider

        from holodeck.lib.backends.openai_agents_fallback import build_fallback_model

        provider = OpenAIProvider()
        return build_fallback_model(
            provider.get_model(model_cfg.name),
            provider.get_model(fallback_name),
        )

    # AZURE_OPENAI — _preflight_credentials guarantees the provider is one of
    # the two supported values (otherwise it already raised) and a non-None
    # endpoint here.
    if endpoint is None:  # pragma: no cover - preflight guarantees non-None
        raise BackendInitError(
            "AZURE_OPENAI_ENDPOINT is required for provider 'azure_openai'."
        )

    from agents import OpenAIResponsesModel, set_tracing_disabled

    # No OpenAI-platform key is available in the Azure dev path, so disable
    # the SDK's trace upload (it would otherwise fail / leak).
    set_tracing_disabled(True)

    client = _build_azure_client(api_key, endpoint, model_cfg.api_version)
    # For Azure, ``model`` is the deployment name (model_cfg.name).
    primary = OpenAIResponsesModel(model=model_cfg.name, openai_client=client)
    if fallback_name is None:
        return primary
    # The fallback is a deployment name on the SAME Azure endpoint/client.
    from holodeck.lib.backends.openai_agents_fallback import build_fallback_model

    fallback = OpenAIResponsesModel(model=fallback_name, openai_client=client)
    return build_fallback_model(primary, fallback)


def _max_turns(agent: Agent) -> int:
    """Return the configured ``max_turns`` for *agent* (default 20 when unset)."""
    if agent.openai is not None:
        return agent.openai.max_turns
    return 20


def _max_budget_usd(agent: Agent) -> float | None:
    """Return the configured ``openai.max_budget_usd`` for *agent*, or ``None``.

    ``None`` means no budget is configured, so no cost-accountant hooks are
    attached and the run incurs zero accounting overhead (FR-032).
    """
    if agent.openai is not None:
        return agent.openai.max_budget_usd
    return None


def _disallowed_tool_names(agent: Agent) -> set[str]:
    """Return the HoloDeck config names to drop from the resolved tool surface.

    The disallow set is the union of the spec-026 top-level
    ``openai.disallowed_tools`` and ``openai.permissions.disallowed_tools``
    (FR-034). Both are config-time filters keyed on the HoloDeck config name,
    so callers compare against the tool's YAML ``name`` — not the SDK tool name.

    Args:
        agent: The agent configuration.

    Returns:
        The set of disallowed config names (empty when no ``openai`` block or no
        disallow lists are configured).
    """
    openai_cfg = agent.openai
    if openai_cfg is None:
        return set()
    blocked: set[str] = set(openai_cfg.disallowed_tools or [])
    if openai_cfg.permissions is not None:
        blocked |= set(openai_cfg.permissions.disallowed_tools or [])
    return blocked


def _build_run_config(agent: Agent, *, group_id: str | None = None) -> RunConfig:
    """Build an SDK ``RunConfig`` carrying trace identity and sensitivity.

    Every ``Runner.run`` call carries ``workflow_name`` (the agent name); session
    runs additionally carry ``group_id`` to correlate the session's turns in the
    trace. ``trace_include_sensitive_data`` is bound to
    ``observability.traces.capture_content`` — the SDK default is **True** (via
    env), which would upload raw tool inputs/outputs to platform.openai.com, so
    HoloDeck sets it explicitly to keep traces clean unless content capture is
    opted in.

    Args:
        agent: The agent configuration.
        group_id: The session id for session runs; ``None`` for ``invoke_once``.

    Returns:
        A populated ``RunConfig``.
    """
    from agents import RunConfig

    capture_content = False
    if agent.observability is not None:
        capture_content = agent.observability.traces.capture_content

    return RunConfig(
        workflow_name=agent.name,
        group_id=group_id,
        trace_metadata={"holodeck.agent": agent.name},
        trace_include_sensitive_data=capture_content,
    )


_EFFORT_TO_REASONING: dict[str, ReasoningEffort] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "xhigh",
}


def _build_reasoning(openai_cfg: OpenAIConfig | None) -> Reasoning | None:
    """Build an SDK ``Reasoning`` from ``openai.effort``, or ``None``.

    Maps the HoloDeck ``effort`` levels onto the OpenAI ``ReasoningEffort``
    literal. ``max`` maps to ``"xhigh"`` (the strongest level the installed
    client supports). When ``effort`` is unset, no reasoning settings are
    produced.

    Args:
        openai_cfg: The agent's ``openai`` config block, or ``None``.

    Returns:
        A ``Reasoning`` carrying the mapped effort, or ``None`` when no effort
        is configured.
    """
    if openai_cfg is None or openai_cfg.effort is None:
        return None
    from openai.types.shared import Reasoning

    return Reasoning(effort=_EFFORT_TO_REASONING[openai_cfg.effort])


def _build_model_settings(
    model_cfg: LLMProvider, openai_cfg: OpenAIConfig | None = None
) -> ModelSettings:
    """Build SDK ``ModelSettings`` for *model_cfg*, honoring reasoning models.

    Reasoning models (o-series, ``gpt-5``+) reject ``temperature`` / ``top_p``,
    so those are omitted for them; ``max_tokens`` is always forwarded (the SDK
    maps it to the Responses ``max_output_tokens``, which reasoning models
    accept). When ``openai.effort`` is set it is mapped onto the Responses
    ``reasoning.effort`` for both reasoning and non-reasoning models.

    Args:
        model_cfg: The agent's model provider config.
        openai_cfg: The agent's ``openai`` config block carrying ``effort``,
            or ``None``.

    Returns:
        A populated ``ModelSettings``.
    """
    from agents import ModelSettings

    reasoning = _build_reasoning(openai_cfg)

    if _is_reasoning_model(model_cfg.name):
        return ModelSettings(max_tokens=model_cfg.max_tokens, reasoning=reasoning)
    return ModelSettings(
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        max_tokens=model_cfg.max_tokens,
        reasoning=reasoning,
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


def _to_execution_result(result: RunResult) -> ExecutionResult:
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
    usage = getattr(result.context_wrapper, "usage", None)
    if usage is not None:
        prompt = int(getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "output_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
        token_usage = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    num_turns = max(1, len(result.raw_responses))

    return ExecutionResult(
        response=response,
        tool_calls=tool_calls,
        tool_results=tool_results,
        token_usage=token_usage,
        num_turns=num_turns,
        thinking="",
    )


def _budget_error_result(exc: BackendBudgetExceededError) -> ExecutionResult:
    """Map a budget-exceeded error onto an error ``ExecutionResult``.

    Preserves the partial response the model produced before the cap tripped and
    records the accumulated cost in ``error_reason`` (FR-032).

    Args:
        exc: The raised :class:`BackendBudgetExceededError`.

    Returns:
        An ``ExecutionResult`` with ``is_error=True`` and the partial response.
    """
    return ExecutionResult(
        response=exc.partial_response,
        is_error=True,
        error_reason=(
            f"max_budget_usd exceeded: accumulated cost "
            f"${exc.accumulated_cost_usd:.6f} >= budget ${exc.budget_usd:.6f}"
        ),
    )


class OpenAIAgentsSession:
    """Stateful multi-turn session backed by an SDK ``SQLiteSession``.

    Each ``send`` runs the SDK agent loop with the shared ``SQLiteSession`` so
    the SDK persists turn history. Idle sessions are SQLite rows, not held
    processes.
    """

    def __init__(
        self,
        sdk_agent: Any,
        sqlite_session: Any,
        *,
        agent_config: Agent | None = None,
        group_id: str | None = None,
        max_turns: int = 20,
        budget_usd: float | None = None,
    ) -> None:
        """Bind the session to an SDK agent and its SQLite-backed history.

        Args:
            sdk_agent: The built SDK ``Agent``.
            sqlite_session: The SDK ``SQLiteSession`` persisting turn history.
            agent_config: The HoloDeck agent config used to build the per-run
                ``RunConfig``. When ``None`` no ``RunConfig`` is attached.
            group_id: The session id, carried as ``RunConfig.group_id`` so the
                session's turns correlate in the trace.
            max_turns: The agent-loop cap passed to ``Runner.run``.
            budget_usd: The configured ``max_budget_usd``. When set, a single
                cost accountant is shared across every turn so the budget covers
                the whole session; when ``None`` no cost hooks are attached.
        """
        self._sdk_agent = sdk_agent
        self._session = sqlite_session
        self._agent_config = agent_config
        self._group_id = group_id
        self._max_turns = max_turns
        self._budget_usd = budget_usd
        # One accountant shared across the session's turns (FR-032).
        self._accountant: Any | None = None

    def _run_config(self) -> RunConfig | None:
        """Build the session ``RunConfig`` (carrying ``group_id``), or ``None``."""
        if self._agent_config is None:
            return None
        return _build_run_config(self._agent_config, group_id=self._group_id)

    def _hooks(self) -> Any | None:
        """Build budget hooks bound to this session's shared accountant, or None.

        Returns ``None`` (no hooks, zero overhead) when no budget is configured.
        The accountant is created lazily on first use and reused across turns so
        the budget covers the whole session rather than resetting each turn.
        """
        if self._budget_usd is None:
            return None
        from holodeck.lib.backends.openai_agents_cost import (
            CostAccountant,
            build_cost_hooks,
        )

        if self._accountant is None:
            self._accountant = CostAccountant(budget_usd=self._budget_usd)
        return build_cost_hooks(self._accountant)

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
            result = await Runner.run(
                self._sdk_agent,
                message,
                session=self._session,
                max_turns=self._max_turns,
                run_config=self._run_config(),
                hooks=self._hooks(),
            )
        except BackendBudgetExceededError as exc:
            return _budget_error_result(exc)
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

        result = Runner.run_streamed(
            self._sdk_agent,
            message,
            session=self._session,
            max_turns=self._max_turns,
            run_config=self._run_config(),
            hooks=self._hooks(),
        )
        try:
            async for event in result.stream_events():
                if event.type != "raw_response_event":
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent) and data.delta:
                    yield data.delta
        except BackendBudgetExceededError:
            # The budget tripped mid-stream; the deltas produced so far have
            # already been yielded, so end the stream gracefully (FR-032).
            return

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
        self._tool_instances: dict[str, Any] = {}
        self._owned_tools: list[Any] = []
        self._mcp_servers: list[Any] = []

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

        from holodeck.lib.backends.openai_agents_tool_adapters import build_sdk_tools
        from holodeck.lib.backends.validators import validate_openai_agents
        from holodeck.lib.instruction_resolver import resolve_instructions

        # FR-034 / FR-110: fail load on credential gaps and allow ∩ disallow
        # conflicts (all problems surfaced together) before any SDK side effects.
        validate_openai_agents(self._agent_config)

        base_dir = self._resolve_base_dir()
        disallowed = _disallowed_tool_names(self._agent_config)
        model = _build_model(self._agent_config)
        instructions = resolve_instructions(
            self._agent_config.instructions, base_dir=base_dir
        )
        await self._initialize_tool_instances()
        tools = build_sdk_tools(
            self._agent_config.tools,
            base_dir,
            tool_instances=self._tool_instances,
            disallowed=disallowed,
        )
        mcp_servers = await self._initialize_mcp_servers(base_dir, disallowed)

        model_settings = _build_model_settings(
            self._agent_config.model, self._agent_config.openai
        )

        self._sdk_agent = SDKAgent(
            name=self._agent_config.name,
            instructions=instructions,
            model=model,
            tools=tools,
            mcp_servers=mcp_servers,
            model_settings=model_settings,
        )

    async def _initialize_mcp_servers(
        self, base_dir: Path | None, disallowed: set[str] | None = None
    ) -> list[Any]:
        """Build and connect SDK MCP servers from the agent's MCP tools.

        Translates ``type: mcp`` tool configs into SDK MCP server objects and
        opens each connection (the SDK requires ``connect()`` before a server is
        passed to ``Agent(mcp_servers=...)``). Connected servers are recorded in
        ``self._mcp_servers`` so ``teardown`` can ``cleanup()`` them. If a
        connection fails, already-connected servers are cleaned up before the
        error is re-raised, so no connection is leaked.

        Args:
            base_dir: Directory for resolving relative stdio ``args`` paths.
            disallowed: MCP tool ``name`` values to drop entirely (FR-034).

        Returns:
            The list of connected SDK MCP server objects (empty when the agent
            declares no MCP tools).

        Raises:
            BackendInitError: If an MCP server fails to connect.
        """
        from holodeck.lib.backends.openai_agents_mcp import build_mcp_servers
        from holodeck.models.tool import MCPTool

        mcp_tools = [
            t for t in (self._agent_config.tools or []) if isinstance(t, MCPTool)
        ]
        servers = build_mcp_servers(mcp_tools, base_dir, disallowed)

        connected: list[Any] = []
        try:
            for server in servers:
                await server.connect()
                connected.append(server)
        except Exception as exc:  # noqa: BLE001 - normalized to BackendInitError
            for server in connected:
                try:
                    await server.cleanup()
                except Exception as cleanup_exc:  # noqa: BLE001 - best-effort
                    logger.warning("Error cleaning up MCP server: %s", cleanup_exc)
            raise BackendInitError(f"Failed to connect MCP server: {exc}") from exc

        self._mcp_servers = connected
        return connected

    async def _initialize_tool_instances(self) -> None:
        """Initialize vectorstore / hierarchical-document tool instances.

        Populates ``self._tool_instances`` (keyed by config name) so
        ``build_sdk_tools`` can wrap each ``.search()`` callable, and records the
        instances in ``self._owned_tools`` for cleanup at teardown. Skips when the
        agent declares no RAG tools. Embedding-provider validation runs first so a
        misconfiguration fails before any ingestion work.

        Raises:
            ConfigError: If the embedding provider is invalid for these tools.
            BackendInitError: If tool initialization fails.
        """
        agent = self._agent_config
        if not agent.tools:
            return
        has_rag = any(
            isinstance(t, VectorstoreTool | HierarchicalDocumentToolConfig)
            for t in agent.tools
        )
        if not has_rag:
            return

        from holodeck.lib.backends.validators import validate_embedding_provider
        from holodeck.lib.tool_initializer import ToolInitializerError, initialize_tools

        validate_embedding_provider(agent)

        base = self._resolve_base_dir()
        try:
            instances = await initialize_tools(
                agent=agent,
                execution_config=agent.execution,
                base_dir=str(base) if base is not None else None,
            )
        except ToolInitializerError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalized to BackendInitError
            raise BackendInitError(f"Failed to initialize tools: {exc}") from exc

        self._tool_instances = instances
        self._owned_tools = list(instances.values())

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
            result = await Runner.run(
                sdk_agent,
                message,
                max_turns=_max_turns(self._agent_config),
                run_config=_build_run_config(self._agent_config),
                hooks=self._invoke_hooks(),
            )
        except BackendBudgetExceededError as exc:
            # Surface the budget abort as an error result so the partial response
            # and accumulated cost are preserved (FR-032), not lost to a raise.
            return _budget_error_result(exc)
        except Exception as exc:  # noqa: BLE001 - re-raised as backend error
            raise BackendSessionError(
                f"OpenAI Agents run failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _to_execution_result(result)

    def _invoke_hooks(self) -> Any | None:
        """Build single-call budget hooks, or ``None`` when no budget is set.

        ``invoke_once`` is stateless, so each call gets a fresh accountant; the
        whole turn's spend is what the budget covers. Returns ``None`` (zero
        overhead) when ``openai.max_budget_usd`` is unset.
        """
        budget = _max_budget_usd(self._agent_config)
        if budget is None:
            return None
        from holodeck.lib.backends.openai_agents_cost import (
            CostAccountant,
            build_cost_hooks,
        )

        return build_cost_hooks(CostAccountant(budget_usd=budget))

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

        session_id = f"holodeck-{uuid.uuid4().hex}"
        session = SQLiteSession(session_id)
        return OpenAIAgentsSession(
            sdk_agent=sdk_agent,
            sqlite_session=session,
            agent_config=self._agent_config,
            group_id=session_id,
            max_turns=_max_turns(self._agent_config),
            budget_usd=_max_budget_usd(self._agent_config),
        )

    async def teardown(self) -> None:
        """Release backend resources, cleaning up RAG tools and MCP servers."""
        for tool_inst in self._owned_tools:
            cleanup = getattr(tool_inst, "cleanup", None)
            if callable(cleanup):
                try:
                    await cleanup()
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    logger.warning("Error cleaning up tool: %s", exc)
        self._owned_tools = []
        self._tool_instances = {}

        for server in self._mcp_servers:
            try:
                await server.cleanup()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Error cleaning up MCP server: %s", exc)
        self._mcp_servers = []
