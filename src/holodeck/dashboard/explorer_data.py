"""Explorer data-assembly helpers (US5 Phase 3).

Projects :class:`EvalRun` data into flat dataclasses the Dash Explorer view
consumes directly.

Data-source precedence inside :func:`build_case_detail` (matches
handoff ``explorer.js::CaseDetail``):

1. **Real-run mode (primary)** — when ``TestResult.tool_invocations`` is
   populated (US1 Migration B), project each ``ToolInvocation`` 1:1 into a
   :class:`ToolCallView`. This is authoritative.
2. **Seed / dev enrichment** — when the caller supplies a
   ``conversations_map`` entry for the case (e.g. the handoff's
   ``sampleConversation`` payload ported as ``SEED_CONVERSATIONS``), use it
   for ``user``/``assistant``/``tool_calls``. Matches ``explorer.js:197``
   which always renders ``sampleConversation[case]`` when present.
3. **Legacy real-run fallback** — when ``tool_invocations`` is empty and no
   seed entry exists but name-only ``tool_calls: list[str]`` is populated
   (runs persisted before US1 Migration B shipped), emit one placeholder
   panel per name.
4. **Empty** — none of the above; conversation renders user/assistant text
   only (possibly empty strings).

Secrets-bearing fields on the snapshotted ``Agent`` are expected to already
be redacted upstream (US3 T218–T220); this module does not re-redact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from holodeck.models.eval_run import EvalRun
from holodeck.models.test_result import (
    MetricResult,
    TestResult,
    ToolInvocation,
    TurnResult,
)

LARGE_TOOL_RESULT_BYTES: int = 500
"""Tool-result panels collapse by default when their size exceeds this many
bytes. Matches handoff ``explorer.js:156`` — intentionally overrides spec
FR-032's 4KB threshold per T410 note."""


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseSummary:
    """Row rendered in the Explorer's cases column."""

    name: str
    passed: bool
    geval_score: float | None
    rag_avg_score: float | None
    tools_called_count: int
    turns_total: int | None = None
    turns_passed: int | None = None
    turns_failed: int | None = None


@dataclass(frozen=True)
class ToolCallView:
    """One entry in the conversation thread for a tool invocation."""

    name: str
    args: dict[str, Any]
    result: Any
    result_size_bytes: int
    duration_ms: int | None
    error: str | None

    @property
    def large(self) -> bool:
        return self.result_size_bytes > LARGE_TOOL_RESULT_BYTES


@dataclass(frozen=True)
class TurnView:
    """Per-turn projection inside a multi-turn conversation (US5).

    Stub populated by :func:`_turn_view_from_result` when
    ``TestResult.turns`` is present. Legacy single-turn cases leave
    ``ConversationView.turns`` as ``None``.
    """

    turn_index: int = 0
    input: str = ""
    response: str | None = None
    tool_invocations: list[ToolCallView] = field(default_factory=list)
    metric_results: list[MetricRow] = field(default_factory=list)
    tools_matched: bool | None = None
    arg_match_details: list[dict[str, Any]] | None = None
    errors: list[str] = field(default_factory=list)
    skipped: bool = False
    execution_time_ms: int = 0
    token_usage: dict[str, int] | None = None
    state: str = "ran"
    expected_tools: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationView:
    """User / tool-calls / assistant projection for the detail panel."""

    user: str
    assistant: str
    tool_calls: list[ToolCallView]
    turns: list[TurnView] | None = None


@dataclass(frozen=True)
class CaseHeader:
    """Top-of-detail header — pass/fail + run metadata badges."""

    case_name: str
    passed: bool
    run_timestamp: str
    prompt_version: str
    model_name: str
    temperature: float | None
    git_commit: str | None


@dataclass(frozen=True)
class AgentSnapshot:
    """Key:value pairs for the "Agent config snapshot" section."""

    model_provider: str
    model_name: str
    model_temperature: float | None
    model_max_tokens: int | None
    embedding_provider: str | None
    embedding_name: str | None
    claude_extended_thinking: bool
    prompt_version: str
    prompt_author: str | None
    prompt_file_path: str | None
    prompt_source: str
    prompt_tags: list[str]
    tools: list[tuple[str, str]]  # list of (kind, name)
    raw_config: dict[str, Any]  # for the "View raw JSON" drawer


@dataclass(frozen=True)
class ExpectedToolsCoverage:
    """Expected-tools coverage panel payload."""

    total: int
    matched: int
    missed: int
    unexpected: int
    rows: list[tuple[str, bool]]  # (tool_name, was_called)
    per_turn: list[TurnCoverage] = field(default_factory=list)


@dataclass(frozen=True)
class TurnCoverage:
    """Per-turn expected-vs-actual tool coverage (multi-turn cases)."""

    turn_index: int
    expected: list[tuple[str, bool]]  # (name, was_matched_this_turn)
    actual: list[str]


@dataclass(frozen=True)
class MetricRow:
    """One row in the "Evaluations" section."""

    kind: str
    name: str
    score: float
    threshold: float | None
    passed: bool
    reasoning: str | None


@dataclass(frozen=True)
class CaseDetail:
    """Full payload for the Explorer's right-hand detail panel."""

    header: CaseHeader
    agent_snapshot: AgentSnapshot
    conversation: ConversationView
    expected_tools_coverage: ExpectedToolsCoverage
    evaluations: dict[str, list[MetricRow]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Selection / listing helpers                                                 #
# --------------------------------------------------------------------------- #


def select_run(runs: list[EvalRun], run_id: str | None) -> EvalRun | None:
    """Return the run whose id matches ``run_id``.

    ``run_id`` is matched against ``report.timestamp`` (the id the dashboard
    uses to address a run). Returns ``None`` when no match is found or when
    ``run_id`` is ``None``.
    """

    if not run_id:
        return None
    for run in runs:
        if run.report.timestamp == run_id:
            return run
    return None


def _geval_score(case: TestResult) -> float | None:
    for m in case.metric_results:
        if m.kind == "geval":
            return float(m.score)
    return None


def _rag_avg(case: TestResult) -> float | None:
    scores = [float(m.score) for m in case.metric_results if m.kind == "rag"]
    if not scores:
        return None
    return sum(scores) / len(scores)


def list_case_summaries(run: EvalRun) -> list[CaseSummary]:
    """One row per case in the run's report, ready for the cases column."""

    out: list[CaseSummary] = []
    for case in run.report.results:
        turns_total: int | None = None
        turns_passed: int | None = None
        turns_failed: int | None = None
        if case.turns is not None:
            turns_total = len(case.turns)
            turns_passed = sum(1 for t in case.turns if t.passed)
            turns_failed = turns_total - turns_passed
        out.append(
            CaseSummary(
                name=case.test_name or "",
                passed=case.passed,
                geval_score=_geval_score(case),
                rag_avg_score=_rag_avg(case),
                tools_called_count=len(case.tool_calls or []),
                turns_total=turns_total,
                turns_passed=turns_passed,
                turns_failed=turns_failed,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Case detail assembly                                                        #
# --------------------------------------------------------------------------- #


def _tool_view_from_invocation(inv: ToolInvocation) -> ToolCallView:
    return ToolCallView(
        name=inv.name,
        args=dict(inv.args or {}),
        result=inv.result,
        result_size_bytes=int(inv.bytes),
        duration_ms=inv.duration_ms,
        error=inv.error,
    )


def _tool_view_from_seed(entry: dict[str, Any]) -> ToolCallView:
    args = dict(entry.get("args") or {})
    result = entry.get("result")
    try:
        size = len(json.dumps(result, default=str))
    except (TypeError, ValueError):
        size = 0
    return ToolCallView(
        name=str(entry.get("name", "")),
        args=args,
        result=result,
        result_size_bytes=size,
        duration_ms=entry.get("duration_ms"),
        error=entry.get("error"),
    )


def _legacy_tool_placeholder(name: str) -> ToolCallView:
    return ToolCallView(
        name=name,
        args={},
        result=None,
        result_size_bytes=0,
        duration_ms=None,
        error=None,
    )


def _build_conversation(
    case: TestResult,
    case_name: str,
    conversations_map: dict[str, Any],
) -> ConversationView:
    turns: list[TurnView] | None = None
    if case.turns is not None:
        turns = [_turn_view_from_result(t) for t in case.turns]

    # Precedence 1: real-run mode with ToolInvocation list (authoritative).
    if case.tool_invocations:
        return ConversationView(
            user=case.test_input or "",
            assistant=case.agent_response or "",
            tool_calls=[_tool_view_from_invocation(i) for i in case.tool_invocations],
            turns=turns,
        )

    # Precedence 2: seed / dev enrichment via conversations_map.
    entry = conversations_map.get(case_name)
    if entry is None and conversations_map:
        entry = conversations_map.get("refund_eligible_standard")
    if entry is not None:
        tool_entries = entry.get("tool_calls") or []
        return ConversationView(
            user=str(entry.get("user", case.test_input or "")),
            assistant=str(entry.get("assistant", case.agent_response or "")),
            tool_calls=[_tool_view_from_seed(t) for t in tool_entries],
            turns=turns,
        )

    # Precedence 3: legacy real-run with name-only tool_calls.
    if case.tool_calls:
        return ConversationView(
            user=case.test_input or "",
            assistant=case.agent_response or "",
            tool_calls=[_legacy_tool_placeholder(n) for n in case.tool_calls],
            turns=turns,
        )

    # Precedence 4: nothing at all — still surface input/response if present.
    return ConversationView(
        user=case.test_input or "",
        assistant=case.agent_response or "",
        tool_calls=[],
        turns=turns,
    )


def _turn_view_from_result(turn: TurnResult) -> TurnView:
    """Project a :class:`TurnResult` into the flat :class:`TurnView` the
    Dash renderer consumes. Reuses existing serializers so the per-turn
    block renders through the same helpers as the single-turn path."""

    tool_views = [_tool_view_from_invocation(inv) for inv in turn.tool_invocations]
    metric_rows = [_metric_row(m) for m in turn.metric_results]
    token_usage: dict[str, int] | None = None
    if turn.token_usage is not None:
        tu = turn.token_usage
        token_usage = {
            "prompt_tokens": tu.prompt_tokens,
            "completion_tokens": tu.completion_tokens,
            "total_tokens": tu.total_tokens,
            "cache_creation_tokens": tu.cache_creation_tokens,
            "cache_read_tokens": tu.cache_read_tokens,
        }
    return TurnView(
        turn_index=turn.turn_index,
        input=turn.input,
        response=turn.response,
        tool_invocations=tool_views,
        metric_results=metric_rows,
        tools_matched=turn.tools_matched,
        arg_match_details=turn.arg_match_details,
        errors=list(turn.errors),
        skipped=turn.skipped,
        execution_time_ms=turn.execution_time_ms,
        token_usage=token_usage,
        state="skipped" if turn.skipped else "ran",
        expected_tools=_expected_tool_names(turn.expected_tools),
        tool_calls=list(turn.tool_calls or []),
    )


def _expected_tool_names(expected: Any) -> list[str]:
    """Serialize `expected_tools` (mixed str / ExpectedTool / dicts) to names."""
    if not expected:
        return []
    out: list[str] = []
    for entry in expected:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict) and "name" in entry:
            out.append(str(entry["name"]))
        elif hasattr(entry, "name"):
            out.append(str(entry.name))
    return out


def _tool_name_matches(expected: str, actual: str) -> bool:
    """Case-sensitive substring match matching executor semantics."""
    return expected in actual


def _build_expected_tools_coverage(case: TestResult) -> ExpectedToolsCoverage:
    """Build the Explorer 'Tool-call coverage' payload.

    Uses case-sensitive substring match (``expected in actual``) so MCP-
    wrapped names like ``mcp__holodeck_tools__legislation_search_search``
    resolve to the configured ``legislation_search``, matching the
    executor's ``tool_name_matches`` contract.

    Multi-turn cases aggregate over per-turn ``expected_tools``, because
    ``TestResult.expected_tools`` is intentionally ``None`` on multi-turn
    results — each turn's assertion is scoped to that turn only.
    """
    if case.turns:
        per_turn: list[TurnCoverage] = []
        aggregate_expected: list[str] = []
        aggregate_matched: set[int] = set()
        for turn in case.turns:
            names = _expected_tool_names(turn.expected_tools)
            actual = list(turn.tool_calls or [])
            rows: list[tuple[str, bool]] = []
            for name in names:
                hit = any(_tool_name_matches(name, a) for a in actual)
                rows.append((name, hit))
                idx = len(aggregate_expected)
                aggregate_expected.append(name)
                if hit:
                    aggregate_matched.add(idx)
            per_turn.append(
                TurnCoverage(
                    turn_index=turn.turn_index,
                    expected=rows,
                    actual=actual,
                )
            )
        total = len(aggregate_expected)
        matched = len(aggregate_matched)
        flat_rows = [
            (aggregate_expected[i], i in aggregate_matched) for i in range(total)
        ]
        return ExpectedToolsCoverage(
            total=total,
            matched=matched,
            missed=total - matched,
            unexpected=0,
            rows=flat_rows,
            per_turn=per_turn,
        )

    expected = _expected_tool_names(case.expected_tools)
    called = list(case.tool_calls or [])

    rows = [(e, any(_tool_name_matches(e, c) for c in called)) for e in expected]
    matched = sum(1 for _, ok in rows if ok)
    missed = len(expected) - matched
    unexpected = sum(
        1 for c in called if not any(_tool_name_matches(e, c) for e in expected)
    )
    return ExpectedToolsCoverage(
        total=len(expected),
        matched=matched,
        missed=missed,
        unexpected=unexpected,
        rows=rows,
        per_turn=[],
    )


def _metric_row(m: MetricResult) -> MetricRow:
    return MetricRow(
        kind=m.kind,
        name=m.metric_name,
        score=float(m.score),
        threshold=float(m.threshold) if m.threshold is not None else None,
        passed=bool(m.passed) if m.passed is not None else False,
        reasoning=m.reasoning,
    )


def _group_evaluations(case: TestResult) -> dict[str, list[MetricRow]]:
    ordered: dict[str, list[MetricRow]] = {}
    for kind in ("geval", "rag", "standard"):
        rows = [_metric_row(m) for m in case.metric_results if m.kind == kind]
        if rows:
            ordered[kind] = rows
    return ordered


def _agent_snapshot(run: EvalRun) -> AgentSnapshot:
    cfg = run.metadata.agent_config
    pv = run.metadata.prompt_version

    embedding = cfg.embedding_provider
    claude_ext = False
    if cfg.claude is not None and cfg.claude.extended_thinking is not None:
        claude_ext = bool(getattr(cfg.claude.extended_thinking, "enabled", False))

    tools: list[tuple[str, str]] = []
    for t in cfg.tools or []:
        kind = getattr(t, "type", "") or getattr(t, "kind", "")
        tools.append((str(kind), getattr(t, "name", "")))

    raw = cfg.model_dump(mode="json")

    return AgentSnapshot(
        model_provider=str(cfg.model.provider),
        model_name=cfg.model.name,
        model_temperature=cfg.model.temperature,
        model_max_tokens=cfg.model.max_tokens,
        embedding_provider=(str(embedding.provider) if embedding else None),
        embedding_name=(embedding.name if embedding else None),
        claude_extended_thinking=claude_ext,
        prompt_version=pv.version,
        prompt_author=pv.author,
        prompt_file_path=pv.file_path,
        prompt_source=pv.source,
        prompt_tags=list(pv.tags or []),
        tools=tools,
        raw_config=raw,
    )


def _case_header(run: EvalRun, case: TestResult) -> CaseHeader:
    cfg = run.metadata.agent_config
    return CaseHeader(
        case_name=case.test_name or "",
        passed=case.passed,
        run_timestamp=run.report.timestamp,
        prompt_version=run.metadata.prompt_version.version,
        model_name=cfg.model.name,
        temperature=cfg.model.temperature,
        git_commit=run.metadata.git_commit,
    )


def build_case_detail(
    run: EvalRun,
    case_name: str,
    conversations_map: dict[str, Any],
) -> CaseDetail | None:
    """Assemble the full detail-panel payload for a case.

    Returns ``None`` when ``case_name`` isn't a case on the run.
    """

    case = next((c for c in run.report.results if c.test_name == case_name), None)
    if case is None:
        return None

    return CaseDetail(
        header=_case_header(run, case),
        agent_snapshot=_agent_snapshot(run),
        conversation=_build_conversation(case, case_name, conversations_map),
        expected_tools_coverage=_build_expected_tools_coverage(case),
        evaluations=_group_evaluations(case),
    )
