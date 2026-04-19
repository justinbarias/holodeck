"""Build structured `ToolInvocation` records from backend tool-call lists.

The Claude Agent SDK backend correlates `ToolUseBlock` → `ToolResultBlock`
pairs by `tool_use_id` (surfaced on the dict as `"call_id"`); Semantic Kernel
exposes parallel `tool_calls` / `tool_results` lists with no stable id, so
pairing is positional.

Design reference: `specs/031-eval-runs-dashboard/research.md` R8.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from holodeck.models.test_result import ToolInvocation

logger = logging.getLogger(__name__)

_CALL_ID_KEY = "call_id"
_NO_RESULT_ERROR = "no result received"


def _safe_json_bytes(result: Any) -> int:
    """Compute `len(json.dumps(result, default=str))`, never raising."""
    try:
        return len(json.dumps(result, default=str))
    except (TypeError, ValueError):
        return len(str(result))


def _coerce_json_safe(value: Any) -> Any:
    """Coerce a value to a JSON-safe form via `json.dumps(..., default=str)`.

    The round-trip turns e.g. `datetime` into its isoformat string, which is
    what consumers (dashboard, disk persistence) expect. Primitives and plain
    containers pass through unchanged.
    """
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _arguments_of(call: dict[str, Any]) -> dict[str, Any]:
    """Return the args dict for a tool-call record, regardless of key name."""
    raw = call.get("arguments", call.get("args", {}))
    return raw if isinstance(raw, dict) else {}


def _build_invocation(
    name: str,
    args: dict[str, Any],
    result: Any,
    error: str | None,
    duration_ms: int | None = None,
) -> ToolInvocation:
    coerced = _coerce_json_safe(result) if result is not None else None
    return ToolInvocation(
        name=name,
        args=args,
        result=coerced,
        bytes=_safe_json_bytes(coerced),
        duration_ms=duration_ms,
        error=error,
    )


def _pair_sk(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> list[ToolInvocation]:
    """Positional pairing — SK's parallel-list contract."""
    invocations: list[ToolInvocation] = []
    for idx, call in enumerate(tool_calls):
        name = str(call.get("name", ""))
        args = _arguments_of(call)
        if idx < len(tool_results):
            result_record = tool_results[idx]
            invocations.append(
                _build_invocation(
                    name=name,
                    args=args,
                    result=result_record.get("result"),
                    error=None,
                )
            )
        else:
            invocations.append(
                _build_invocation(
                    name=name,
                    args=args,
                    result=None,
                    error=_NO_RESULT_ERROR,
                )
            )
    return invocations


def _pair_claude(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> list[ToolInvocation]:
    """`call_id` pairing — Claude's guaranteed-unique correlation key."""
    results_by_id: dict[str, dict[str, Any]] = {}
    for record in tool_results:
        call_id = record.get(_CALL_ID_KEY)
        if call_id is None:
            logger.warning(
                "Claude tool_result missing %r; skipping: %r",
                _CALL_ID_KEY,
                record,
            )
            continue
        results_by_id[str(call_id)] = record

    invocations: list[ToolInvocation] = []
    matched_ids: set[str] = set()
    for call in tool_calls:
        name = str(call.get("name", ""))
        args = _arguments_of(call)
        call_id = call.get(_CALL_ID_KEY)
        result_record = results_by_id.get(str(call_id)) if call_id is not None else None
        if result_record is None:
            invocations.append(
                _build_invocation(
                    name=name, args=args, result=None, error=_NO_RESULT_ERROR
                )
            )
        else:
            matched_ids.add(str(call_id))
            invocations.append(
                _build_invocation(
                    name=name,
                    args=args,
                    result=result_record.get("result"),
                    error=None,
                )
            )

    orphaned = set(results_by_id) - matched_ids
    for orphan_id in orphaned:
        logger.warning(
            "Claude tool_result has no matching tool_call for call_id=%r; skipping",
            orphan_id,
        )

    return invocations


def pair_tool_calls(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    backend_kind: Literal["sk", "claude"],
) -> list[ToolInvocation]:
    """Pair a backend's tool_calls/tool_results into structured ToolInvocations.

    Args:
        tool_calls: Raw tool-call dicts from `ExecutionResult.tool_calls`.
        tool_results: Raw tool-result dicts from `ExecutionResult.tool_results`.
        backend_kind: ``"sk"`` for positional pairing, ``"claude"`` for
            `call_id` pairing. Chosen by the caller via `isinstance(backend, …)`.

    Returns:
        One `ToolInvocation` per item in `tool_calls`, preserving call order.
        Unpaired calls get `error="no result received"`.
    """
    if backend_kind == "claude":
        return _pair_claude(tool_calls, tool_results)
    return _pair_sk(tool_calls, tool_results)
