#!/usr/bin/env python3
"""Replay labelled contradiction cases through HoloDeck's resolve_edge prompt.

Spec 043 live probe. It measures the contradiction prompt
(``holodeck.lib.memory.backends.graphiti_prompts.resolve_edge``) on real
fact pairs without ingesting anything. Each case puts the old fact in FACT
INVALIDATION CANDIDATES as idx 0, with no EXISTING FACTS, which is how
Graphiti presents a fact between other entities. The case is predicted
"contradicted" when the model returns idx 0 in ``contradicted_facts``.

The call goes through the same Graphiti LLM client HoloDeck builds for an
OpenAI extraction model (``build_llm_client``), with Graphiti's response
model, message shaping and model size, so the prompt reaches the model as it
does during ingestion.

Usage (needs the ``graphiti`` extra and OPENAI_API_KEY in the environment):

    uv run python scripts/memory_resolve_edge_replay.py gpt-4.1-mini
    uv run python scripts/memory_resolve_edge_replay.py gpt-4.1-mini \\
        --cases scripts/memory_resolve_edge_cases.json --concurrency 4

Exit status: 0 when every case matches its label, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.models import ChatPrompt
from pydantic import BaseModel, ConfigDict, Field

from holodeck.lib.memory.backends.graphiti_clients import build_llm_client
from holodeck.lib.memory.backends.graphiti_prompts import resolve_edge
from holodeck.models.llm import LLMProvider, ProviderEnum

DEFAULT_CASES = Path(__file__).with_name("memory_resolve_edge_cases.json")
PROMPT_NAME = "dedupe_edges.resolve_edge"

Label = Literal["contradicted", "not_contradicted"]
Decider = Callable[[ChatPrompt], Awaitable[EdgeDuplicate]]


class Case(BaseModel):
    """One labelled pair: does ``new_fact`` contradict ``old_fact``?"""

    model_config = ConfigDict(extra="forbid")

    old_fact: str = Field(min_length=1)
    new_fact: str = Field(min_length=1)
    expected: Label
    note: str = ""


class CaseFile(BaseModel):
    """The fixture: a list of cases."""

    model_config = ConfigDict(extra="forbid")

    cases: list[Case] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class Outcome:
    case: Case
    predicted: Label
    raw: EdgeDuplicate

    @property
    def correct(self) -> bool:
        return self.predicted == self.case.expected


def load_cases(path: Path) -> list[Case]:
    """Read and validate the fixture."""
    return CaseFile.model_validate_json(path.read_text(encoding="utf-8")).cases


def build_prompt(case: Case) -> ChatPrompt:
    """The prompt Graphiti would build for this pair (old fact as idx 0)."""
    return resolve_edge(
        {
            "existing_edges": [],
            "edge_invalidation_candidates": [{"idx": 0, "fact": case.old_fact}],
            "new_edge": case.new_fact,
        }
    )


def graphiti_decider(client: LLMClient) -> Decider:
    """Call ``client`` the way Graphiti calls it for ``resolve_edge``."""

    async def decide(prompt: ChatPrompt) -> EdgeDuplicate:
        response = await client.generate_response(
            prompt.as_messages(),
            response_model=EdgeDuplicate,
            model_size=ModelSize.small,
            prompt_name=PROMPT_NAME,
        )
        return EdgeDuplicate.model_validate(response)

    return decide


def openai_decider(model: str) -> Decider:
    """A decider over HoloDeck's OpenAI client; the key comes from the env."""
    provider = LLMProvider(
        provider=ProviderEnum.OPENAI, name=model, api_key=None, endpoint=None
    )
    return graphiti_decider(build_llm_client(provider))


async def replay(
    cases: Sequence[Case], decide: Decider, *, concurrency: int = 4
) -> list[Outcome]:
    """Run every case; outcomes come back in case order."""
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(case: Case) -> Outcome:
        async with gate:
            raw = await decide(build_prompt(case))
        predicted: Label = (
            "contradicted" if 0 in raw.contradicted_facts else "not_contradicted"
        )
        return Outcome(case=case, predicted=predicted, raw=raw)

    return list(await asyncio.gather(*(one(case) for case in cases)))


def confusion(outcomes: Sequence[Outcome]) -> dict[tuple[Label, Label], int]:
    """Counts keyed by (expected, predicted)."""
    labels: tuple[Label, Label] = ("contradicted", "not_contradicted")
    table = {(e, p): 0 for e in labels for p in labels}
    for outcome in outcomes:
        table[(outcome.case.expected, outcome.predicted)] += 1
    return table


def render(outcomes: Sequence[Outcome]) -> str:
    """The confusion table, then every wrong case."""
    table = confusion(outcomes)
    right = sum(1 for o in outcomes if o.correct)
    corner = "expected / predicted"
    lines = [
        f"{right} of {len(outcomes)} cases match their label.",
        "",
        f"{corner:<24}{'contradicted':>14}{'not_contradicted':>18}",
    ]
    for expected in ("contradicted", "not_contradicted"):
        lines.append(
            f"{expected:<24}{table[(expected, 'contradicted')]:>14}"
            f"{table[(expected, 'not_contradicted')]:>18}"
        )
    wrong = [o for o in outcomes if not o.correct]
    if wrong:
        lines += ["", "Wrong cases:"]
    for n, outcome in enumerate(wrong, start=1):
        case = outcome.case
        lines += [
            "",
            f"{n}. expected {case.expected}, got {outcome.predicted} "
            f"(duplicate_facts={outcome.raw.duplicate_facts}, "
            f"contradicted_facts={outcome.raw.contradicted_facts})",
            f"   note: {case.note}" if case.note else "   note: -",
            f"   old:  {case.old_fact}",
            f"   new:  {case.new_fact}",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="OpenAI model name, for example gpt-4.1-mini")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    outcomes = asyncio.run(
        replay(cases, openai_decider(args.model), concurrency=args.concurrency)
    )
    print(f"model: {args.model}  cases: {args.cases}\n")
    print(render(outcomes))
    return 0 if all(o.correct for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
