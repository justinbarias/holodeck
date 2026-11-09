"""Tool adapters bridging HoloDeck tools to the OpenAI Agents SDK.

Translates HoloDeck ``FunctionTool`` configs into low-level
``agents.FunctionTool`` instances. The user callable is resolved via the shared
:func:`holodeck.lib.function_tool_loader.load_function_tool` (the same loader the
SK and Claude backends use) and wrapped with an ``on_invoke_tool`` coroutine
that parses the model-supplied JSON arguments, invokes the callable, and returns
its stringified result.

Supported types: function, vectorstore, and hierarchical_document tools (the
last two wrap the same initialized ``.search()`` instances the Claude adapter
uses, supplied via ``tool_instances``). ``type: mcp`` tools are skipped here —
they become SDK ``mcp_servers`` built separately by
:mod:`holodeck.lib.backends.openai_agents_mcp`. ``type: skill`` tools are
likewise skipped — they become handoff-target agents built by
:mod:`holodeck.lib.backends.openai_agents_subagents`. ``type: prompt`` tools are
skipped with a warning — no backend has a runtime adapter for them. Any other
tool type raises :class:`ConfigError` naming the unsupported type, so
misconfigured agents fail fast rather than silently dropping tools.

All ``import agents`` happen inside functions to keep the optional SDK import
lazy (SC-005).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import typing
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.config.env_loader import substitute_env_vars
from holodeck.lib.backends.base import BackendInitError
from holodeck.lib.errors import ConfigError
from holodeck.lib.function_tool_loader import load_function_tool
from holodeck.models.tool import (
    HOSTED_TOOL_CLASSES,
    CodeInterpreterHostedTool,
    FileSearchHostedTool,
    FunctionTool,
    HierarchicalDocumentToolConfig,
    HostedMCPHostedTool,
    ImageGenerationHostedTool,
    MCPTool,
    PromptTool,
    SkillTool,
    ToolUnion,
    VectorstoreTool,
    WebSearchHostedTool,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from agents import Tool as SDKTool

logger = logging.getLogger(__name__)

# Sentinel returned to the model when a search yields no hits — mirrors the
# Claude adapter (``lib/backends/tool_adapters.py``).
_NO_RESULTS = "No results found."

# JSON Schema for the single-``query`` search tools, matching the Claude
# adapter's ``{"query": str}`` shape.
_SEARCH_PARAMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

# Python annotation -> JSON Schema type. Anything unrecognised falls back to a
# permissive ``string`` (the SDK is non-strict here, so the model still sees a
# usable parameter).
_JSON_TYPE_BY_PY: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type_for(annotation: Any) -> str:
    """Best-effort map a Python annotation to a JSON Schema ``type`` string."""
    if isinstance(annotation, type) and annotation in _JSON_TYPE_BY_PY:
        return _JSON_TYPE_BY_PY[annotation]
    # Handle typing constructs (e.g. ``list[str]``) via their origin.
    origin = typing.get_origin(annotation)
    if origin is not None and isinstance(origin, type) and origin in _JSON_TYPE_BY_PY:
        return _JSON_TYPE_BY_PY[origin]
    return "string"


def _derive_params_schema(
    func: Callable[..., Any],
    declared: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build a JSON Schema object for the SDK ``params_json_schema``.

    When the YAML config declares ``parameters`` it is used verbatim as the
    ``properties`` map; otherwise the schema is derived from *func*'s signature.
    Parameters without a default are marked required.

    Args:
        func: The resolved tool callable.
        declared: The ``FunctionTool.parameters`` mapping from YAML, or ``None``.

    Returns:
        A JSON Schema ``object`` with ``properties`` and ``required`` keys.
    """
    if declared:
        return {
            "type": "object",
            "properties": dict(declared),
            "required": list(declared.keys()),
            "additionalProperties": False,
        }

    properties: dict[str, Any] = {}
    required: list[str] = []
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except (NameError, TypeError):
        # Annotations referencing names not in scope (forward refs) or otherwise
        # unresolvable — fall back to the raw signature annotations below.
        hints = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, param.annotation)
        properties[name] = {"type": _json_type_for(annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _make_on_invoke(
    func: Callable[..., Any],
    tool_name: str,
) -> Callable[[Any, str], Any]:
    """Build the SDK ``on_invoke_tool`` coroutine that dispatches to *func*.

    The SDK passes a ``ToolContext`` and the raw JSON argument string. We parse
    the arguments, call the (sync or async) callable, and return its result as a
    string for the model.
    """
    is_coro = asyncio.iscoroutinefunction(func)

    async def _on_invoke(_ctx: Any, input_json: str) -> str:
        try:
            args = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"tools.{tool_name}",
                f"Tool '{tool_name}' received malformed JSON arguments: {exc}",
            ) from exc
        if not isinstance(args, dict):
            args = {}
        if is_coro:
            result = await func(**args)
        else:
            result = func(**args)
        return str(result)

    return _on_invoke


def _make_vectorstore_on_invoke(
    instance: Any,
    tool_name: str,
) -> Callable[[Any, str], Any]:
    """Build an ``on_invoke_tool`` coroutine for a vectorstore ``.search()``.

    The instance's ``search`` returns a single formatted string; an empty result
    is reported to the model as the ``_NO_RESULTS`` sentinel.
    """

    async def _on_invoke(_ctx: Any, input_json: str) -> str:
        query = _extract_query(input_json)
        result: str = await instance.search(query)
        return result if result else _NO_RESULTS

    del tool_name  # carried for symmetry / future failure-wrapping (E1)
    return _on_invoke


def _make_hierarchical_on_invoke(
    instance: Any,
    tool_name: str,
) -> Callable[[Any, str], Any]:
    """Build an ``on_invoke_tool`` coroutine for a hierarchical-doc ``.search()``.

    The instance's ``search`` returns a list of results; each is ``.format()``-ed
    and joined with ``\\n---\\n`` (matching the Claude adapter), or the
    ``_NO_RESULTS`` sentinel when empty.
    """

    async def _on_invoke(_ctx: Any, input_json: str) -> str:
        query = _extract_query(input_json)
        results = await instance.search(query)
        if not results:
            return _NO_RESULTS
        return "\n---\n".join(r.format() for r in results)

    del tool_name
    return _on_invoke


def _extract_query(input_json: str) -> str:
    """Parse the model-supplied JSON arguments and return the ``query`` string."""
    try:
        args = json.loads(input_json) if input_json else {}
    except json.JSONDecodeError:
        return ""
    if isinstance(args, dict):
        return str(args.get("query", ""))
    return ""


CODE_INTERPRETER_OPT_IN_MESSAGE = (
    "CodeInterpreterTool runs model-written code in an OpenAI-hosted container. "
    "Set `openai.i_understand_this_is_unsafe: true` to acknowledge this and "
    "load the tool, or remove the entry."
)


def code_interpreter_opt_in_error(tool_name: str) -> ConfigError:
    """Build the canonical FR-083 safety-gate error for *tool_name*."""
    return ConfigError(f"tools.{tool_name}", CODE_INTERPRETER_OPT_IN_MESSAGE)


def build_hosted_tool(cfg: ToolUnion, *, allow_unsafe: bool = False) -> SDKTool:
    """Construct the SDK hosted tool selected by a ``type: hosted`` entry.

    The nested ``tool_config`` objects the SDK expects (``CodeInterpreter``,
    ``ImageGeneration``, ``Mcp`` TypedDicts from ``openai.types.responses``)
    are built from the validated YAML params. Only fields set in YAML are
    forwarded so the API defaults apply to the rest.

    Args:
        cfg: A hosted tool config (one of ``HOSTED_TOOL_CLASSES``).
        allow_unsafe: The ``openai.i_understand_this_is_unsafe`` opt-in;
            required to construct ``CodeInterpreterTool`` (FR-083).

    Returns:
        The SDK tool instance.

    Raises:
        ConfigError: If the entry is not a hosted tool, or a
            ``CodeInterpreterTool`` is declared without the opt-in.
    """
    from agents import (
        CodeInterpreterTool,
        FileSearchTool,
        HostedMCPTool,
        ImageGenerationTool,
        WebSearchTool,
    )

    if isinstance(cfg, WebSearchHostedTool):
        params = cfg.params
        user_location: Any = None
        if params.user_location is not None:
            user_location = {
                "type": "approximate",
                **params.user_location.model_dump(exclude_none=True),
            }
        filters: Any = None
        if params.allowed_domains is not None:
            # The SDK converter calls ``filters.model_dump()``, so this must be
            # the pydantic model, not a mapping.
            from openai.types.responses.web_search_tool import (
                Filters as WebSearchToolFilters,
            )

            filters = WebSearchToolFilters(allowed_domains=list(params.allowed_domains))
        return WebSearchTool(
            user_location=user_location,
            filters=filters,
            search_context_size=params.search_context_size,
            external_web_access=params.external_web_access,
        )
    if isinstance(cfg, FileSearchHostedTool):
        fs = cfg.params
        ranking: Any = None
        if fs.ranking_options is not None:
            ranking = fs.ranking_options.model_dump(exclude_none=True) or None
        return FileSearchTool(
            vector_store_ids=list(fs.vector_store_ids),
            max_num_results=fs.max_num_results,
            include_search_results=fs.include_search_results,
            ranking_options=ranking,
            filters=fs.filters,  # type: ignore[arg-type]
        )
    if isinstance(cfg, CodeInterpreterHostedTool):
        if not allow_unsafe:
            raise code_interpreter_opt_in_error(cfg.name)
        container = cfg.params.container
        container_cfg: Any = (
            container
            if isinstance(container, str)
            else container.model_dump(exclude_none=True)
        )
        return CodeInterpreterTool(
            tool_config={"type": "code_interpreter", "container": container_cfg}
        )
    if isinstance(cfg, ImageGenerationHostedTool):
        image_cfg: dict[str, Any] = {"type": "image_generation"}
        image_cfg.update(cfg.params.model_dump(exclude_none=True))
        return ImageGenerationTool(tool_config=image_cfg)  # type: ignore[arg-type]
    if isinstance(cfg, HostedMCPHostedTool):
        mcp = cfg.params
        mcp_cfg: dict[str, Any] = {
            "type": "mcp",
            "server_label": mcp.server_label,
            "require_approval": mcp.require_approval,
        }
        if mcp.server_url is not None:
            mcp_cfg["server_url"] = mcp.server_url
        if mcp.connector_id is not None:
            mcp_cfg["connector_id"] = mcp.connector_id
        if mcp.server_description is not None:
            mcp_cfg["server_description"] = mcp.server_description
        if mcp.authorization is not None:
            mcp_cfg["authorization"] = substitute_env_vars(mcp.authorization)
        if mcp.headers is not None:
            mcp_cfg["headers"] = {
                key: substitute_env_vars(value) for key, value in mcp.headers.items()
            }
        if mcp.allowed_tools is not None:
            mcp_cfg["allowed_tools"] = list(mcp.allowed_tools)
        return HostedMCPTool(tool_config=mcp_cfg)  # type: ignore[arg-type]
    raise ConfigError(f"tools.{cfg.name}", f"'{cfg.type}' is not a hosted tool entry.")


def build_sdk_tools(
    tool_configs: list[ToolUnion] | None,
    base_dir: Path | None,
    tool_instances: dict[str, Any] | None = None,
    disallowed: set[str] | None = None,
    *,
    allow_unsafe_hosted: bool = False,
) -> list[SDKTool]:
    """Translate HoloDeck tool configs into SDK tool instances.

    Tools whose HoloDeck *config* name appears in *disallowed* are filtered out
    before any SDK tool is constructed (FR-034). Matching is on the config name
    as written in YAML — not the SDK tool name — so a disallowed vectorstore /
    hierarchical-document tool is dropped before it can produce its
    ``{name}_search`` SDK tool. This keeps the disallow list portable: an
    operator names the tool the way they declared it, regardless of how the
    backend renames it for the SDK.

    Args:
        tool_configs: All tool configurations from the agent YAML (may be None).
        base_dir: Directory used to resolve relative ``FunctionTool.file`` paths
            (typically the agent project root).
        tool_instances: Initialized vectorstore / hierarchical-document tool
            instances keyed by config name (built by ``initialize_tools``).
            Required for those two tool types.
        disallowed: HoloDeck config names to omit from the built tool surface.
            ``None`` (the default) applies no filtering.
        allow_unsafe_hosted: The ``openai.i_understand_this_is_unsafe`` opt-in
            gating ``CodeInterpreterTool`` (FR-083). Permission filtering runs
            first: a disallowed code interpreter is dropped, never built.

    Returns:
        A list of SDK tools (``FunctionTool`` plus any hosted tools) ready to
        pass to ``Agent``.

    Raises:
        BackendInitError: If a vectorstore / hierarchical-document tool has no
            matching initialized instance.
        ConfigError: If a tool type is unsupported on this backend, a function
            tool fails to load, or two configs would surface under the same
            SDK tool name (for example vectorstore ``kb`` and function
            ``kb_search``), which would make tool grants and disallow lists
            ambiguous.
    """
    from agents import FunctionTool as SDKFunctionTool

    blocked = disallowed or set()
    instances = tool_instances or {}
    tools: list[SDKTool] = []
    _reject_sdk_name_collisions(tool_configs)
    for cfg in tool_configs or []:
        if cfg.name in blocked:
            # Filter on the config name, before building, so a disallowed
            # vectorstore/hier-doc tool never produces a ``{name}_search`` tool.
            logger.debug("Skipping disallowed tool '%s'.", cfg.name)
            continue
        if isinstance(cfg, FunctionTool):
            func = load_function_tool(cfg, base_dir=base_dir)
            schema = _derive_params_schema(func, cfg.parameters)
            tools.append(
                SDKFunctionTool(
                    name=cfg.name,
                    description=cfg.description,
                    params_json_schema=schema,
                    on_invoke_tool=_make_on_invoke(func, cfg.name),
                    strict_json_schema=False,
                )
            )
        elif isinstance(cfg, VectorstoreTool):
            instance = _require_instance(instances, cfg.name, cfg.type)
            tools.append(
                SDKFunctionTool(
                    name=f"{cfg.name}_search",
                    description=f"Search {cfg.name}: {cfg.description}",
                    params_json_schema=_SEARCH_PARAMS_SCHEMA,
                    on_invoke_tool=_make_vectorstore_on_invoke(instance, cfg.name),
                    strict_json_schema=False,
                )
            )
        elif isinstance(cfg, HierarchicalDocumentToolConfig):
            instance = _require_instance(instances, cfg.name, cfg.type)
            tools.append(
                SDKFunctionTool(
                    name=f"{cfg.name}_search",
                    description=f"Search {cfg.name}: {cfg.description}",
                    params_json_schema=_SEARCH_PARAMS_SCHEMA,
                    on_invoke_tool=_make_hierarchical_on_invoke(instance, cfg.name),
                    strict_json_schema=False,
                )
            )
        elif isinstance(cfg, MCPTool):
            # MCP tools become SDK ``mcp_servers`` (built separately via
            # ``openai_agents_mcp.build_mcp_servers``), not ``FunctionTool``s, so
            # they are skipped here rather than wrapped.
            continue
        elif isinstance(cfg, SkillTool):
            # Skills become handoff-target Agents (built by
            # ``openai_agents_subagents.build_handoff_agents``), not
            # ``FunctionTool``s, so they are skipped here.
            continue
        elif isinstance(cfg, HOSTED_TOOL_CLASSES):
            tools.append(build_hosted_tool(cfg, allow_unsafe=allow_unsafe_hosted))
        elif isinstance(cfg, PromptTool):
            logger.warning(
                "Tool '%s' (type: prompt) has no runtime adapter on any backend; "
                "skipping it on the openai_agents backend.",
                cfg.name,
            )
            continue
        else:
            raise ConfigError(
                f"tools.{cfg.name}",
                f"'{cfg.type}' tools are not yet supported on the openai_agents "
                "backend.",
            )

    return tools


def sdk_tool_name_for(cfg: ToolUnion) -> str | None:
    """Return the SDK tool name a HoloDeck tool config is surfaced under.

    Function tools keep their config name; vectorstore and
    hierarchical-document tools are surfaced as ``{name}_search``; hosted
    tools surface under the SDK's fixed name for their class (``web_search``,
    ``file_search``, ``code_interpreter``, ``image_generation``,
    ``hosted_mcp``), so two hosted entries of one class collide. MCP,
    prompt, and skill configs produce no SDK tool and return ``None``.

    Args:
        cfg: A tool config from the parent's ``tools:`` list.

    Returns:
        The SDK tool name, or ``None`` for types with no function tool.
    """
    if isinstance(cfg, FunctionTool):
        return cfg.name
    if isinstance(cfg, VectorstoreTool | HierarchicalDocumentToolConfig):
        return f"{cfg.name}_search"
    if isinstance(cfg, HOSTED_TOOL_CLASSES):
        return cfg.sdk_tool_name
    return None


def _reject_sdk_name_collisions(tool_configs: list[ToolUnion] | None) -> None:
    """Fail when two configs map to one SDK tool name (before any filtering).

    Args:
        tool_configs: The agent's ``tools:`` list.

    Raises:
        ConfigError: Naming both config entries and the shared SDK name.
    """
    owners: dict[str, str] = {}
    for cfg in tool_configs or []:
        sdk_name = sdk_tool_name_for(cfg)
        if sdk_name is None:
            continue
        if sdk_name in owners:
            raise ConfigError(
                f"tools.{cfg.name}",
                f"tool '{cfg.name}' ({cfg.type}) and tool '{owners[sdk_name]}' "
                f"both surface as SDK tool '{sdk_name}' on the openai_agents "
                "backend; rename one of them.",
            )
        owners[sdk_name] = cfg.name


def _require_instance(
    instances: dict[str, Any],
    name: str,
    tool_type: str,
) -> Any:
    """Return the initialized instance for *name* or raise ``BackendInitError``."""
    instance = instances.get(name)
    if instance is None:
        raise BackendInitError(
            f"No initialized instance found for tool '{name}' (type: {tool_type}). "
            "Ensure tool initialization completed before building SDK tools."
        )
    return instance
