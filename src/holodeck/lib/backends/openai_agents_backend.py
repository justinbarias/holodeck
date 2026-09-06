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

import asyncio
import contextlib
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
    ToolEvent,
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

    from holodeck.lib.backends.openai_agents_tracing import TracingPolicy

logger = logging.getLogger(__name__)

# Upper bound on buffered tool events per session (see OpenAIAgentsSession).
_TOOL_EVENT_QUEUE_MAXSIZE = 1000


def _tracing_enabled(agent: Agent) -> bool:
    """Return whether OTel tracing is enabled for *agent*.

    Mirrors how the CLI gates observability (``serve`` / ``chat``): tracing is
    on only when ``observability.enabled`` and ``observability.traces.enabled``
    are both true. When tracing is off no OTel mirror is built for the agent, so
    a run incurs no mirroring overhead and emits no OTel spans.

    Args:
        agent: The agent configuration.

    Returns:
        ``True`` when an OTel mirror should receive this agent's SDK spans.
    """
    obs = agent.observability
    return obs is not None and obs.enabled and obs.traces.enabled


def _provider_upload_permitted(agent: Agent) -> bool:
    """Return whether *agent*'s SDK traces may upload to platform.openai.com.

    Upload is permitted only for ``provider: openai`` without
    ``observability.disable_provider_tracing: true`` (FR-100 / FR-102). Azure
    never uploads (FR-101), independent of whether observability is enabled.

    Args:
        agent: The agent configuration.

    Returns:
        ``True`` when the provider exporter should receive the agent's traces.
    """
    if agent.model.provider != ProviderEnum.OPENAI:
        return False
    obs = agent.observability
    return obs is None or not obs.disable_provider_tracing


def _tracing_policy_for(agent: Agent) -> TracingPolicy:
    """Build the per-backend :class:`TracingPolicy` for *agent* (D13).

    The policy is registered with the process-global HoloDeck trace router at
    backend ``initialize()``, before any run emits spans, and is evaluated per
    trace — so several backends with different providers or overrides coexist
    in one process without one silently inheriting another's upload behaviour.

    * ``upload``: :func:`_provider_upload_permitted`.
    * ``mirror``: an OTel-mirroring ``TracingProcessor`` when
      :func:`_tracing_enabled`, else ``None``.

    Args:
        agent: The agent configuration (selects provider, override, and gate).

    Returns:
        The routing policy for this backend's SDK traces.
    """
    from holodeck.lib.backends.openai_agents_tracing import (
        TracingPolicy,
        build_tracing_mirror,
    )

    mirror = build_tracing_mirror(agent.name) if _tracing_enabled(agent) else None
    return TracingPolicy(upload=_provider_upload_permitted(agent), mirror=mirror)


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
    Azure ``/openai/v1`` surface is wrapped as an ``OpenAIResponsesModel``. The
    SDK trace upload is suppressed not here but by the tracing policy the
    backend registers at ``initialize()`` (FR-101 / D13): the HoloDeck trace
    router withholds Azure traces from the platform exporter while keeping
    spans flowing to OTel.

    When ``openai.fallback_model`` is set, the primary model is wrapped in a
    fallback ``Model`` (FR-033): on a retryable upstream error (429 / 5xx) the
    request is re-issued once against the fallback model, built with the *same*
    credentials/client as the primary (for Azure the fallback name is a
    deployment on the same endpoint). Without ``fallback_model`` the primary is
    returned directly, with no wrapper.

    Credential resolution and validation are delegated to the side-effect-free
    :func:`_preflight_credentials`; for ``provider: openai`` this function
    additionally performs the SDK global mutation ``set_default_openai_key``,
    which must NOT run during config-time validation.

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

    from agents import OpenAIResponsesModel

    # The SDK's default trace upload is suppressed without disabling the trace
    # provider: the backend registers an ``upload=False`` tracing policy at
    # initialize() (FR-101 / D13), so the HoloDeck router withholds this
    # backend's traces from the platform.openai.com exporter while spans still
    # reach the OTel mirror. Calling ``set_tracing_disabled(True)`` here would
    # make the provider return NoOp traces and starve that mirror, so it is
    # intentionally NOT called.
    client = _build_azure_client(api_key, endpoint, model_cfg.api_version)
    # For Azure, ``model`` is the deployment name (model_cfg.name).
    primary = OpenAIResponsesModel(model=model_cfg.name, openai_client=client)
    if fallback_name is None:
        return primary
    # The fallback is a deployment name on the SAME Azure endpoint/client.
    from holodeck.lib.backends.openai_agents_fallback import build_fallback_model

    fallback = OpenAIResponsesModel(model=fallback_name, openai_client=client)
    return build_fallback_model(primary, fallback)


def _resolve_subagent_model(agent: Agent, name: str) -> str | Model:
    """Return the SDK ``model=`` value for a subagent's explicit model *name*.

    For ``provider: openai`` the identifier string is returned and the SDK's
    default Responses client (already keyed by ``_build_model``) serves it.
    For ``provider: azure_openai`` *name* is a deployment on the parent's
    endpoint, so it is wrapped in an ``OpenAIResponsesModel`` bound to a
    client built from the same credentials. ``openai.fallback_model`` applies
    to the entry agent only; subagents with an explicit model get no fallback
    wrapper (``model: inherit`` reuses the parent's wrapped model as-is).

    Args:
        agent: The parent agent configuration (selects provider/credentials).
        name: The subagent's ``model`` value (not ``inherit``).

    Returns:
        A model-name string (OpenAI) or an ``OpenAIResponsesModel`` (Azure).
    """
    if agent.model.provider == ProviderEnum.OPENAI:
        return name
    api_key, endpoint = _preflight_credentials(agent)
    if endpoint is None:  # pragma: no cover - preflight guarantees non-None
        raise BackendInitError(
            "AZURE_OPENAI_ENDPOINT is required for provider 'azure_openai'."
        )
    from agents import OpenAIResponsesModel

    client = _build_azure_client(api_key, endpoint, agent.model.api_version)
    return OpenAIResponsesModel(model=name, openai_client=client)


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


def _build_run_config(
    agent: Agent,
    *,
    group_id: str | None = None,
    tracing_policy_id: str | None = None,
) -> RunConfig:
    """Build an SDK ``RunConfig`` carrying trace identity and sensitivity.

    Every ``Runner.run`` call carries ``workflow_name`` (the agent name); session
    runs additionally carry ``group_id`` to correlate the session's turns in the
    trace. ``trace_include_sensitive_data`` is bound to
    ``observability.traces.capture_content`` — the SDK default is **True** (via
    the ``OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA`` env default), which
    would upload raw tool inputs/outputs to platform.openai.com, so HoloDeck
    sets it explicitly on every run (FR-102); the environment never overrides
    a capture-disabled agent.

    Args:
        agent: The agent configuration.
        group_id: The session id for session runs; ``None`` for ``invoke_once``.
        tracing_policy_id: The backend's registered tracing-policy id, tagged
            onto the trace metadata so the HoloDeck trace router applies that
            backend's upload/mirror policy to this run (D13). ``None`` leaves
            the trace untagged.

    Returns:
        A populated ``RunConfig``.
    """
    from agents import RunConfig

    from holodeck.lib.backends.openai_agents_tracing import TRACE_POLICY_METADATA_KEY

    capture_content = False
    if agent.observability is not None:
        capture_content = agent.observability.traces.capture_content

    metadata: dict[str, Any] = {"holodeck.agent": agent.name}
    if tracing_policy_id is not None:
        metadata[TRACE_POLICY_METADATA_KEY] = tracing_policy_id

    return RunConfig(
        workflow_name=agent.name,
        group_id=group_id,
        trace_metadata=metadata,
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

    ``summary="auto"`` is requested alongside the effort so reasoning models
    emit reasoning summaries — the source of ``ExecutionResult.thinking`` (FR-004).
    Without it the run produces ``ReasoningItem``s with empty summaries and
    ``thinking`` stays empty.

    Args:
        openai_cfg: The agent's ``openai`` config block, or ``None``.

    Returns:
        A ``Reasoning`` carrying the mapped effort and ``summary="auto"``, or
        ``None`` when no effort is configured.
    """
    if openai_cfg is None or openai_cfg.effort is None:
        return None
    from openai.types.shared import Reasoning

    return Reasoning(effort=_EFFORT_TO_REASONING[openai_cfg.effort], summary="auto")


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


def _extract_thinking(result: RunResult) -> str:
    """Join reasoning-summary texts from a run's ``ReasoningItem``s (FR-004).

    Each ``ReasoningItem`` carries a ``summary`` list whose entries expose
    ``.text``. Summaries are only present when the request set
    ``Reasoning(summary="auto")`` (see :func:`_build_reasoning`); non-reasoning
    models or runs without summaries yield an empty string.

    Args:
        result: A ``RunResult`` from ``Runner.run``.

    Returns:
        The reasoning summaries joined with a blank-line separator, or ``""``.
    """
    from agents.items import ReasoningItem

    parts: list[str] = []
    for item in result.new_items:
        if not isinstance(item, ReasoningItem):
            continue
        for entry in getattr(item.raw_item, "summary", None) or []:
            text = getattr(entry, "text", "")
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _coerce_structured_output(final: object) -> dict[str, Any] | None:
    """Coerce a structured ``final_output`` into a dict, or ``None``.

    When an output schema is active the SDK parses the model's JSON into the
    validated object; ``JSONSchemaOutputSchema.validate_json`` returns a dict, so
    ``final_output`` is already a dict in that path. A pydantic model is dumped
    via ``model_dump``; a JSON string is parsed defensively. Anything that cannot
    be represented as a dict yields ``None``.

    Args:
        final: The run's ``final_output``.

    Returns:
        The structured output as a dict, or ``None``.
    """
    if isinstance(final, dict):
        return final
    model_dump = getattr(final, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else None
    if isinstance(final, str):
        try:
            parsed = json.loads(final)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _to_execution_result(
    result: RunResult, *, structured: bool = False
) -> ExecutionResult:
    """Map an SDK run result onto a provider-agnostic ``ExecutionResult``.

    Extracts the final text, tool calls / results (parallel lists for positional
    pairing, matching the SK contract), token usage, the turn count, reasoning
    ``thinking``, and — when *structured* — the parsed ``structured_output``.

    Args:
        result: A ``RunResult`` from ``Runner.run``.
        structured: Whether an output schema was set on the agent. When ``True``
            ``final_output`` is coerced into ``structured_output``; when ``False``
            ``structured_output`` stays ``None`` (the default behavior).

    Returns:
        A populated ``ExecutionResult``.
    """
    from agents.items import (
        HandoffCallItem,
        HandoffOutputItem,
        ToolCallItem,
        ToolCallOutputItem,
    )

    final = result.final_output
    structured_output = _coerce_structured_output(final) if structured else None
    if final is None:
        response = ""
    elif isinstance(final, str):
        response = final
    elif structured_output is not None:
        # The SDK parsed the model's JSON into an object; re-serialize it as
        # canonical JSON — str() would yield a Python repr (single quotes)
        # that downstream graders cannot json.loads().
        response = json.dumps(structured_output)
    else:
        response = str(final)

    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    call_name_by_id: dict[str, str] = {}

    from holodeck.lib.backends.openai_agents_events import hosted_call_for

    for item in result.new_items:
        if isinstance(item, ToolCallItem | HandoffCallItem):
            # Handoff calls (``transfer_to_<agent>``) are recorded alongside
            # ordinary tool calls so graders can assert a handoff happened.
            raw = item.raw_item
            hosted = hosted_call_for(raw)
            if hosted is not None:
                # Hosted tools run server-side and carry their outcome on the
                # call item itself; pair the call and result here.
                hosted_record: dict[str, Any] = {
                    "name": hosted.name,
                    "arguments": hosted.arguments,
                }
                hosted_result: dict[str, Any] = {
                    "name": hosted.name,
                    "result": hosted.response,
                }
                if hosted.call_id:
                    hosted_record["call_id"] = hosted.call_id
                    hosted_result["call_id"] = hosted.call_id
                tool_calls.append(hosted_record)
                tool_results.append(hosted_result)
                continue
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
        elif isinstance(item, HandoffOutputItem):
            raw_handoff = item.raw_item
            call_id = ""
            if isinstance(raw_handoff, dict):
                call_id = str(raw_handoff.get("call_id", "") or "")
            target = str(getattr(item.target_agent, "name", "") or "")
            tool_results.append(
                {
                    "name": call_name_by_id.get(call_id, ""),
                    "result": f"handoff:{target}",
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
        structured_output=structured_output,
        num_turns=num_turns,
        thinking=_extract_thinking(result),
    )


def _hosted_capability_hint(agent: Agent | None, exc: BaseException) -> str:
    """Return an Azure hosted-tool hint for a run failure, or ``""``.

    Hosted tools load on ``azure_openai`` (no blanket ban, D06/D17); a resource
    that lacks the capability rejects the request at run time. The SDK error
    is preserved verbatim and this hint is appended so the operator knows
    where to look.

    Args:
        agent: The HoloDeck agent config (``None`` when unavailable).
        exc: The exception raised by the SDK run.

    Returns:
        The hint text, or an empty string when it does not apply.
    """
    from holodeck.models.tool import HOSTED_TOOL_CLASSES

    if agent is None or agent.model.provider is not ProviderEnum.AZURE_OPENAI:
        return ""
    hosted = [t.tool for t in agent.tools or [] if isinstance(t, HOSTED_TOOL_CLASSES)]
    if not hosted:
        return ""
    text = str(exc).lower()
    if "tool" not in text and "not supported" not in text and "invalid" not in text:
        return ""
    return (
        f" (hosted tools declared: {', '.join(hosted)}; this Azure resource or "
        "API version may not support them — see the OpenAI backend guide, "
        '"Hosted tools on Azure")'
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
        structured_output: bool = False,
        tracing_policy_id: str | None = None,
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
            structured_output: Whether the agent has an output schema, so each
                turn's ``final_output`` is parsed into ``structured_output``.
            tracing_policy_id: The owning backend's registered tracing-policy
                id, tagged onto every turn's trace (D13); ``None`` when the
                backend registered no policy.
        """
        self._sdk_agent = sdk_agent
        self._session = sqlite_session
        self._agent_config = agent_config
        self._group_id = group_id
        self._tracing_policy_id = tracing_policy_id
        self._max_turns = max_turns
        self._budget_usd = budget_usd
        self._structured_output = structured_output
        # One accountant shared across the session's turns (FR-032).
        self._accountant: Any | None = None
        # Real-time tool / handoff / thinking events (FR-006), drained by the
        # chat tools panel and the AG-UI bridge via ``tool_events``. Bounded so
        # a consumer-less session (``holodeck test``) cannot grow it without
        # limit; on overflow the newest event is dropped (best-effort UI feed).
        self._tool_event_queue: asyncio.Queue[ToolEvent] = asyncio.Queue(
            maxsize=_TOOL_EVENT_QUEUE_MAXSIZE
        )

    @property
    def tool_events(self) -> asyncio.Queue[ToolEvent]:
        """Queue of ``ToolEvent`` records emitted during this session's turns."""
        return self._tool_event_queue

    def _publish(self, events: list[ToolEvent]) -> None:
        """Push *events* onto the queue; never block on a full queue."""
        for event in events:
            with contextlib.suppress(asyncio.QueueFull):
                self._tool_event_queue.put_nowait(event)

    def _run_config(self) -> RunConfig | None:
        """Build the session ``RunConfig`` (carrying ``group_id``), or ``None``."""
        if self._agent_config is None:
            return None
        return _build_run_config(
            self._agent_config,
            group_id=self._group_id,
            tracing_policy_id=self._tracing_policy_id,
        )

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

        from holodeck.lib.backends.openai_agents_tracing import active_tracing_policy

        try:
            with active_tracing_policy(self._tracing_policy_id):
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
            hint = _hosted_capability_hint(self._agent_config, exc)
            logger.warning(
                "OpenAI Agents run failed: %s: %s%s",
                type(exc).__name__,
                exc,
                hint,
                exc_info=True,
            )
            return ExecutionResult(
                response="",
                is_error=True,
                error_reason=f"{type(exc).__name__}: {exc}{hint}",
            )
        # Non-streaming runs have no live stream; reconstruct the ordered
        # tool / handoff events from the completed run's items (FR-006).
        from holodeck.lib.backends.openai_agents_events import (
            tool_events_for_run_items,
        )

        self._publish(tool_events_for_run_items(list(result.new_items)))
        return _to_execution_result(result, structured=self._structured_output)

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

        from holodeck.lib.backends.openai_agents_events import (
            HandoffTracker,
            tool_events_for_stream_event,
        )
        from holodeck.lib.backends.openai_agents_tracing import active_tracing_policy

        tracker = HandoffTracker()
        # The streamed run executes on a task created inside this scope, so it
        # inherits the policy id for spans opened outside HoloDeck's RunConfig.
        with active_tracing_policy(self._tracing_policy_id):
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
                    # Tool, handoff, and reasoning items feed the event queue
                    # in SDK order (FR-006); only text deltas are yielded.
                    self._publish(tool_events_for_stream_event(event, tracker))
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent) and data.delta:
                    yield data.delta
        except BackendBudgetExceededError as exc:
            # The budget tripped mid-stream; the deltas produced so far have
            # already been yielded, so end the stream gracefully (FR-032).
            # Open tool / handoff entries are closed as errors so the panel
            # does not show them running forever.
            self._publish(tracker.close(error=f"{type(exc).__name__}: {exc}"))
            return
        except BaseException as exc:
            hint = _hosted_capability_hint(self._agent_config, exc)
            if hint:
                logger.warning("OpenAI Agents streamed run failed: %s%s", exc, hint)
            self._publish(tracker.close(error=f"{type(exc).__name__}: {exc}{hint}"))
            raise
        # Handoffs stay "active" until the run ends (the target agent keeps
        # the conversation), so their ``end`` events are emitted here.
        self._publish(tracker.close())

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
        # Unique per backend instance: keys this backend's tracing policy in the
        # process-global router and tags every run's trace metadata (D13).
        self._tracing_policy_id = f"holodeck-backend-{uuid.uuid4().hex}"
        self._has_structured_output = False
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

        from holodeck.lib.backends.openai_agents_output import (
            build_output_schema,
            load_response_format_schema,
        )
        from holodeck.lib.backends.openai_agents_tool_adapters import build_sdk_tools
        from holodeck.lib.backends.validators import validate_openai_agents
        from holodeck.lib.instruction_resolver import resolve_instructions

        # FR-034 / FR-110: fail load on credential gaps and allow ∩ disallow
        # conflicts (all problems surfaced together) before any SDK side effects.
        validate_openai_agents(self._agent_config)

        # Register this backend's tracing policy (provider upload + OTel mirror)
        # with the process-global router before any run emits spans
        # (FR-100–FR-102, D13). Always registered — Azure upload suppression
        # must hold even with observability disabled.
        from holodeck.lib.backends.openai_agents_tracing import register_tracing_policy

        register_tracing_policy(
            self._tracing_policy_id, _tracing_policy_for(self._agent_config)
        )

        base_dir = self._resolve_base_dir()
        disallowed = _disallowed_tool_names(self._agent_config)
        model = _build_model(self._agent_config)
        instructions = resolve_instructions(
            self._agent_config.instructions, base_dir=base_dir
        )
        await self._initialize_tool_instances()
        openai_cfg = self._agent_config.openai
        tools = build_sdk_tools(
            self._agent_config.tools,
            base_dir,
            tool_instances=self._tool_instances,
            disallowed=disallowed,
            allow_unsafe_hosted=bool(
                openai_cfg is not None and openai_cfg.i_understand_this_is_unsafe
            ),
        )
        mcp_servers = await self._initialize_mcp_servers(base_dir, disallowed)

        model_settings = _build_model_settings(
            self._agent_config.model, self._agent_config.openai
        )

        # FR-004: wire response_format (dict | str path | None) to output_type.
        schema = load_response_format_schema(
            self._agent_config.response_format, base_dir
        )
        output_type = build_output_schema(schema) if schema is not None else None
        self._has_structured_output = output_type is not None

        # FR-060 / FR-070: subagents and skills become handoff targets sharing
        # the parent's built tool surface (inherit-all or explicit subsets),
        # output type, and model settings.
        from holodeck.lib.backends.openai_agents_subagents import (
            build_handoff_agents,
            index_parent_tools,
        )

        agent_cfg = self._agent_config
        handoffs: list[Any] = build_handoff_agents(
            agent_cfg,
            parent_model=model,
            parent_model_settings=model_settings,
            surface=index_parent_tools(agent_cfg.tools, tools, mcp_servers),
            base_dir=base_dir,
            resolve_model=lambda name: _resolve_subagent_model(agent_cfg, name),
            resolve_model_settings=lambda name: _build_model_settings(
                agent_cfg.model.model_copy(update={"name": name}), agent_cfg.openai
            ),
            output_type=output_type,
            disallowed=disallowed,
        )

        self._sdk_agent = SDKAgent(
            name=self._agent_config.name,
            instructions=instructions,
            model=model,
            tools=tools,
            mcp_servers=mcp_servers,
            handoffs=handoffs,
            model_settings=model_settings,
            output_type=output_type,
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

        from holodeck.lib.backends.openai_agents_tracing import active_tracing_policy

        try:
            with active_tracing_policy(self._tracing_policy_id):
                result = await Runner.run(
                    sdk_agent,
                    message,
                    max_turns=_max_turns(self._agent_config),
                    run_config=_build_run_config(
                        self._agent_config,
                        tracing_policy_id=self._tracing_policy_id,
                    ),
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
        return _to_execution_result(result, structured=self._has_structured_output)

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
            structured_output=self._has_structured_output,
            tracing_policy_id=self._tracing_policy_id,
        )

    async def teardown(self) -> None:
        """Release backend resources, cleaning up RAG tools and MCP servers.

        Also withdraws this backend's tracing policy from the process-global
        router, so a late trace tagged with its id is dropped rather than
        routed under a stale policy (D13).
        """
        from holodeck.lib.backends.openai_agents_tracing import (
            unregister_tracing_policy,
        )

        unregister_tracing_policy(self._tracing_policy_id)
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
