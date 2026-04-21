"""Seed data for dashboard development.

Loads the golden fixture `tests/fixtures/dashboard/seed_runs.json` — a
snapshot of the handoff's `data.js` dataset (24 runs × 12 cases × 7 prompt
versions × trajectory 0.58→0.93) deserialised into `EvalRun` instances.

The fixture lets every UI task iterate without real `results/<slug>/` files
on disk; it also backs `AppTest`-style smoke tests.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from holodeck.models.eval_run import EvalRun
from holodeck.models.test_result import (
    MetricResult,
    ReportSummary,
    TestReport,
    TestResult,
    ToolInvocation,
    TurnResult,
)

SEED_AGENT_DISPLAY_NAME = "customer-support"


def _fixture_path() -> Path:
    """Resolve the repo-committed seed fixture.

    Walks up from this file until we find `tests/fixtures/dashboard/seed_runs.json`.
    This works in both source checkouts and the wheel-installed layout (where
    `tests/` sits alongside `src/`).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tests" / "fixtures" / "dashboard" / "seed_runs.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "seed_runs.json not found; expected under tests/fixtures/dashboard/"
    )


@lru_cache(maxsize=1)
def build_seed_runs() -> list[EvalRun]:
    """Return the 24-run seed dataset as validated EvalRun instances."""
    raw = json.loads(_fixture_path().read_text(encoding="utf-8"))
    return [EvalRun.model_validate(item) for item in raw]


def _legacy_single_turn_case() -> TestResult:
    """A classic single-turn test case (no ``turns`` field)."""

    return TestResult(
        test_name="single_turn_legacy",
        test_input="What is the refund window?",
        agent_response="Our standard return window is 30 days.",
        tool_calls=[],
        tool_invocations=[],
        expected_tools=None,
        tools_matched=None,
        metric_results=[],
        ground_truth=None,
        passed=True,
        execution_time_ms=42,
        errors=[],
        timestamp="2026-04-20T00:00:00+00:00",
        turns=None,
    )


def _passing_three_turn_case() -> TestResult:
    """A 3-turn chit-chat case where every turn passes."""

    turns = [
        TurnResult(
            turn_index=i,
            input=inp,
            response=resp,
            metric_results=[],
            passed=True,
            execution_time_ms=100 + i,
            errors=[],
            skipped=False,
        )
        for i, (inp, resp) in enumerate(
            [
                ("Hi there", "Hello! How can I help?"),
                ("What's the weather?", "Sunny and 72F."),
                ("Tell me a joke", "Why did the chicken cross the road?"),
            ]
        )
    ]
    return TestResult(
        test_name="three_turn_chit_chat",
        test_input="Hi there\n---\nWhat's the weather?\n---\nTell me a joke",
        agent_response="Why did the chicken cross the road?",
        tool_calls=[],
        tool_invocations=[],
        expected_tools=None,
        tools_matched=None,
        metric_results=[],
        ground_truth=None,
        passed=True,
        execution_time_ms=sum(t.execution_time_ms for t in turns),
        errors=[],
        timestamp="2026-04-20T00:00:01+00:00",
        turns=turns,
    )


def _four_turn_failing_case() -> TestResult:
    """A 4-turn case where turn 2 fails (wrong tool + failing metric)."""

    t0 = TurnResult(
        turn_index=0,
        input="Look up order A-8844",
        response="Got it.",
        tool_calls=["lookup_order"],
        tool_invocations=[
            ToolInvocation(
                name="lookup_order",
                args={"order_id": "A-8844"},
                result={"status": "delivered"},
                bytes=32,
                duration_ms=41,
            )
        ],
        tools_matched=True,
        metric_results=[],
        passed=True,
        execution_time_ms=120,
    )
    t1 = TurnResult(
        turn_index=1,
        input="What is the total?",
        response="$249.00",
        metric_results=[
            MetricResult(
                metric_name="numeric",
                kind="code",
                score=1.0,
                threshold=None,
                passed=True,
                scale="0-1",
            )
        ],
        passed=True,
        execution_time_ms=80,
    )
    t2 = TurnResult(
        turn_index=2,
        input="Subtract 181001 from it",
        response="65587",
        ground_truth="25587",
        tool_calls=["lookup"],
        expected_tools=["subtract"],
        tools_matched=False,
        metric_results=[
            MetricResult(
                metric_name="numeric",
                kind="code",
                score=0.0,
                threshold=0.5,
                passed=False,
                scale="0-1",
                reasoning="Expected 25587, got 65587",
            )
        ],
        passed=False,
        execution_time_ms=90,
        errors=["expected tool(s) not called in this turn: subtract"],
    )
    t3 = TurnResult(
        turn_index=3,
        input="Thanks, anything else?",
        response="You're welcome!",
        metric_results=[],
        passed=True,
        execution_time_ms=70,
    )
    turns = [t0, t1, t2, t3]
    return TestResult(
        test_name="four_turn_math_failing",
        test_input="\n---\n".join(t.input for t in turns),
        agent_response=turns[-1].response,
        tool_calls=["lookup_order", "lookup"],
        tool_invocations=list(t0.tool_invocations),
        expected_tools=["subtract"],
        tools_matched=False,
        metric_results=[],
        ground_truth=None,
        passed=False,
        execution_time_ms=sum(t.execution_time_ms for t in turns),
        errors=["[turn 2] expected tool(s) not called in this turn: subtract"],
        timestamp="2026-04-20T00:00:02+00:00",
        turns=turns,
    )


def build_multi_turn_seed_case() -> EvalRun:
    """Return a mixed seed run: legacy + passing 3-turn + failing 4-turn (US5).

    Reuses the metadata block from :func:`build_seed_runs` so the
    dashboard layout has a valid :class:`EvalRun` instance to render.
    """

    base = build_seed_runs()[0]
    cases = [
        _legacy_single_turn_case(),
        _passing_three_turn_case(),
        _four_turn_failing_case(),
    ]
    total = len(cases)
    passed = sum(1 for c in cases if c.passed)
    report = TestReport(
        agent_name=base.report.agent_name,
        agent_config_path=base.report.agent_config_path,
        results=cases,
        summary=ReportSummary(
            total_tests=total,
            passed=passed,
            failed=total - passed,
            pass_rate=100.0 * passed / total,
            total_duration_ms=sum(c.execution_time_ms for c in cases),
        ),
        timestamp="2026-04-20T00:00:03+00:00",
        holodeck_version=base.report.holodeck_version,
    )
    return EvalRun(report=report, metadata=base.metadata)


SEED_CONVERSATIONS: dict[str, dict] = {
    # Ported verbatim from the design handoff's `data.js:160–178`
    # (`window.HD_DATA.sampleConversation`). Consumed by the Explorer view
    # when a real run lacks structured `ToolInvocation` records — see
    # `explorer_data.py::_build_conversation` precedence rules.
    "refund_eligible_standard": {
        "user": (
            "Hi, I ordered noise-canceling headphones on April 2 (order "
            "#A-8844) and they arrived yesterday but the right cup "
            "rattles. I'd like a refund please."
        ),
        "assistant": (
            "I'm sorry the headphones arrived faulty — I can take care of "
            "that right away. I've pulled up order #A-8844 and confirmed "
            "it shipped on April 3, so you're well within the 30-day "
            "return window. I've issued a full refund of $249.00 to your "
            "original payment method; it should appear in 3–5 business "
            "days. I've also emailed you a prepaid return label. Anything "
            "else?"
        ),
        "tool_calls": [
            {
                "name": "lookup_order",
                "args": {"order_id": "A-8844"},
                "result": {
                    "id": "A-8844",
                    "status": "delivered",
                    "total": 249.00,
                    "ship_date": "2026-04-03",
                    "items": [{"sku": "HP-NC-770", "name": "NC-770 Headphones"}],
                },
            },
            {
                "name": "issue_refund",
                "args": {
                    "order_id": "A-8844",
                    "amount": 249.00,
                    "reason": "faulty_item",
                },
                "result": {
                    "refund_id": "ref_01HX9K2PQ",
                    "status": "processed",
                    "eta_business_days": 5,
                },
            },
            {
                "name": "send_email",
                "args": {
                    "to": "dana@example.com",
                    "template": "return_label",
                    "order_id": "A-8844",
                },
                "result": {"sent": True, "message_id": "msg_a8f83c"},
            },
        ],
    },
    "refund_outside_window": {
        "user": (
            "I bought a desk lamp from you last November. It's now "
            "flickering. Can I get a refund?"
        ),
        "assistant": (
            "Thanks for flagging this. I looked up the order and the "
            "purchase was on November 12, 2025, which puts it well "
            "outside our 30-day return window. Per our return policy "
            "(section 4.2), items purchased more than 30 days ago aren't "
            "eligible for a refund, but the lamp carries a 12-month "
            "manufacturer warranty which may cover the flickering. I can "
            "point you to the warranty claim form — want me to?"
        ),
        "tool_calls": [
            {
                "name": "lookup_order",
                "args": {"query": "desk lamp november"},
                "result": {
                    "id": "B-2210",
                    "status": "delivered",
                    "ship_date": "2025-11-12",
                },
            },
            {
                "name": "policy_lookup",
                "args": {"topic": "return_window"},
                "result": {
                    "section": "4.2",
                    "window_days": 30,
                    "exceptions": ["manufacturer_warranty"],
                },
            },
        ],
    },
}
