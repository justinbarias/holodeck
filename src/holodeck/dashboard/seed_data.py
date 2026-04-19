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
