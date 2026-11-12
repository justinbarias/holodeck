"""The resolve_edge replay probe builds prompts and scores them offline (spec 043)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.models import ChatPrompt, Message

from holodeck.lib.memory.backends.graphiti_prompts import RESOLVE_EDGE_SYSTEM

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "memory_resolve_edge_replay.py"
CASES = ROOT / "scripts" / "memory_resolve_edge_cases.json"

VISA_OLD = (
    "A Subclass 117 (Orphan Relative) visa is excluded from the permanent visa "
    "category that makes an individual subject to the newly arrived resident's "
    "waiting period under section 61AA."
)
SECRETARY_NEW = (
    "Human Services Secretary means the Chief Executive Officer of Services Australia."
)


@pytest.fixture(scope="module")
def probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("memory_resolve_edge_replay", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixture_parses_with_both_labels_and_the_live_cases(probe: ModuleType) -> None:
    cases = probe.load_cases(CASES)

    expected = [case.expected for case in cases]
    assert expected.count("not_contradicted") >= 6
    assert expected.count("contradicted") >= 4
    olds = {case.old_fact for case in cases}
    news = {case.new_fact for case in cases}
    assert VISA_OLD in olds
    assert SECRETARY_NEW in news
    assert probe.DEFAULT_CASES == CASES


def test_prompt_puts_the_old_fact_as_candidate_zero(probe: ModuleType) -> None:
    case = probe.Case(old_fact="old-text", new_fact="new-text", expected="contradicted")

    prompt = probe.build_prompt(case)

    assert isinstance(prompt, ChatPrompt)
    assert prompt.system.content == RESOLVE_EDGE_SYSTEM
    user = prompt.user.content
    assert "<EXISTING FACTS>\n[]\n</EXISTING FACTS>" in user
    assert "[{'idx': 0, 'fact': 'old-text'}]" in user
    assert "<NEW FACT>\nnew-text\n</NEW FACT>" in user


@pytest.mark.asyncio
async def test_replay_scores_cases_and_lists_the_wrong_ones(probe: ModuleType) -> None:
    cases = probe.load_cases(CASES)
    seen: list[str] = []

    async def always_contradicted(prompt: ChatPrompt) -> EdgeDuplicate:
        seen.append(prompt.user.content)
        return EdgeDuplicate(duplicate_facts=[], contradicted_facts=[0])

    outcomes = await probe.replay(cases, always_contradicted, concurrency=2)

    assert len(seen) == len(cases)
    assert [o.case for o in outcomes] == cases
    table = probe.confusion(outcomes)
    wrong = sum(1 for c in cases if c.expected == "not_contradicted")
    assert table[("not_contradicted", "contradicted")] == wrong
    assert table[("contradicted", "contradicted")] == len(cases) - wrong
    assert table[("contradicted", "not_contradicted")] == 0
    text = probe.render(outcomes)
    assert f"{len(cases) - wrong} of {len(cases)} cases match" in text
    assert VISA_OLD in text
    assert SECRETARY_NEW not in text


@pytest.mark.asyncio
async def test_openai_decider_calls_graphiti_client_shape_without_network(
    probe: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-real-probe")
    calls: list[dict[str, Any]] = []

    async def generate_response(
        self: object, messages: list[Message], **kwargs: Any
    ) -> dict[str, list[int]]:
        calls.append({"messages": messages, **kwargs})
        return {"duplicate_facts": [], "contradicted_facts": []}

    from graphiti_core.llm_client.openai_client import OpenAIClient

    monkeypatch.setattr(OpenAIClient, "generate_response", generate_response)
    decide = probe.openai_decider("gpt-test")
    case = probe.Case(old_fact="old", new_fact="new", expected="not_contradicted")

    result = await decide(probe.build_prompt(case))

    assert result == EdgeDuplicate(duplicate_facts=[], contradicted_facts=[])
    (call,) = calls
    assert call["response_model"] is EdgeDuplicate
    assert call["model_size"] == ModelSize.small
    assert call["prompt_name"] == "dedupe_edges.resolve_edge"
    system, user = call["messages"]
    assert system.role == "system"
    assert system.content.startswith(RESOLVE_EDGE_SYSTEM)
    assert "'fact': 'old'" in user.content
