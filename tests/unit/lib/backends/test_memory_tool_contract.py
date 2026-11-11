"""Contract test of the memory tool across both bindings (spec 043, T14).

The same cases run against the Claude binding (``TemporalMemoryToolAdapter``,
which feeds ``build_holodeck_sdk_server``) and the OpenAI Agents binding
(``build_sdk_tools``). The memory is real: ``HierarchicalDocumentTool`` on the
in-memory store with keyword search, and ``FakeBackend``. No network, model or
database.

Layers. Each case calls the binding where the host calls it: the ``handler`` of
the Claude ``SdkMcpTool`` and the ``on_invoke_tool`` of the OpenAI
``FunctionTool``. On the live Claude path the SDK checks arguments against the
``input_schema`` before the handler runs, so an argument that the schema
rejects never reaches the handler there. The OpenAI Agents SDK does not check
arguments of a low-level function tool, so every call reaches the handler. The
tests below call the handler directly on both sides. They show that an
argument the schema rejects still gets the same ``invalid_argument`` result
from the handler layer on both backends, and the ``schema_accepts`` column says
which cases the live Claude SDK would stop first.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import pytest_asyncio

from holodeck.lib.backends.openai_agents_tool_adapters import build_sdk_tools
from holodeck.lib.backends.tool_adapters import (
    TemporalMemoryToolAdapter,
    build_holodeck_sdk_server,
)
from holodeck.lib.memory.backends.fake import FakeBackend
from holodeck.lib.memory.tool_api import (
    MEMORY_FUNCTION_NAMES,
    MEMORY_PROVENANCE,
    MEMORY_SEARCH,
    MEMORY_TIMELINE,
)
from holodeck.models.agent import Agent
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import (
    HierarchicalDocumentToolConfig,
    SearchMode,
    TemporalMemoryToolConfig,
)
from holodeck.tools import temporal_memory as composition
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.temporal_memory_tool import TemporalMemoryTool

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

OPENAI = LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o")
SCHEMA_DIR = Path(__file__).resolve().parents[4] / "schemas" / "memory"
CLAUDE = "claude"
OPENAI_AGENTS = "openai_agents"

# Several sections per document: keyword search (BM25) scores a term that
# appears in half of a two-chunk corpus at 0 and returns nothing.
DOCS = {
    "alpha": (
        "---\neffective_date: 2020-01-01\n---\n# Rates\n\n"
        "## Pension\n\nThe pension rate is 100 dollars per fortnight.\n\n"
        "## Carer\n\nThe carer payment is paid weekly to a carer.\n\n"
        "## Youth\n\nYouth support is paid to students under age limits.\n"
    ),
    "beta": (
        "---\neffective_date: 2022-07-01\n---\n# Allowance\n\n"
        "## Rent\n\nThe rent allowance is 50 dollars per fortnight.\n\n"
        "## Travel\n\nTravel concessions apply to public transport.\n\n"
        "## Phone\n\nA phone allowance is paid each quarter.\n"
    ),
}

# Placeholders in a case's arguments, filled from a real fact in the memory.
FACT_TEXT = "$fact_text"
FACT_ID = "$fact_id"


@dataclass
class Rig:
    """A started memory tool, its fake backend and both bindings over it."""

    tool: TemporalMemoryTool
    backend: FakeBackend
    fact_text: str
    fact_id: str
    claude_tools: dict[str, Any]
    openai_tools: dict[str, Any]


@dataclass(frozen=True)
class Binding:
    """One host's view of the three functions."""

    name: str
    function_names: Callable[[], list[str]]
    schema: Callable[[str], dict[str, Any]]
    call: Callable[[str, dict[str, Any]], Awaitable[str]]


def _memory_agent() -> Agent:
    tools: list[dict[str, Any]] = [
        {
            "name": name,
            "type": "hierarchical_document",
            "description": name,
            "source": f"{name}/",
            "search_mode": "keyword",
        }
        for name in DOCS
    ]
    tools.append(
        {
            "name": "law_memory",
            "type": "temporal_memory",
            "description": "Family law memory.",
            "source_tools": list(DOCS),
            "backend": {"driver": "kuzu"},
            "ingest_on_start": True,
        }
    )
    return Agent(name="a", model=OPENAI, instructions={"inline": "hi"}, tools=tools)


async def _source_tool(root: Path, name: str) -> HierarchicalDocumentTool:
    folder = root / name
    folder.mkdir()
    (folder / "doc.md").write_text(DOCS[name])
    tool = HierarchicalDocumentTool(
        HierarchicalDocumentToolConfig(
            name=name,
            description=name,
            source=str(folder),
            search_mode=SearchMode.KEYWORD,
        )
    )
    await tool.initialize()
    return tool


@pytest_asyncio.fixture
async def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Rig]:
    backend = FakeBackend()
    monkeypatch.setattr(composition, "create_backend", lambda *args: backend)
    monkeypatch.setattr(composition, "create_embedding_service", lambda agent: None)
    sources = {name: await _source_tool(tmp_path, name) for name in DOCS}
    agent = _memory_agent()
    config = agent.tools[-1]
    assert isinstance(config, TemporalMemoryToolConfig)
    tool = TemporalMemoryTool(agent, config, sources, embedding_service=None)
    await tool.initialize()

    found = json.loads(await tool.call(MEMORY_SEARCH, {"query": "pension rate"}))
    facts = [hit for hit in found["hits"] if hit["kind"] == "fact"]
    assert facts, found

    claude = TemporalMemoryToolAdapter(config, tool).to_sdk_tools()
    openai = build_sdk_tools([config], None, {config.name: tool})
    yield Rig(
        tool=tool,
        backend=backend,
        fact_text=facts[0]["text"],
        fact_id=facts[0]["fact_id"],
        claude_tools={t.name: t for t in claude},
        openai_tools={t.name: t for t in openai},
    )
    await tool.close()
    for source in sources.values():
        await source.close()


def _binding(rig: Rig, host: str) -> Binding:
    if host == CLAUDE:

        async def call_claude(name: str, arguments: dict[str, Any]) -> str:
            result = await rig.claude_tools[name].handler(arguments)
            blocks = result["content"]
            assert len(blocks) == 1
            assert blocks[0]["type"] == "text"
            assert "is_error" not in result
            text: str = blocks[0]["text"]
            return text

        return Binding(
            CLAUDE,
            lambda: list(rig.claude_tools),
            lambda name: rig.claude_tools[name].input_schema,
            call_claude,
        )

    async def call_openai(name: str, arguments: dict[str, Any]) -> str:
        out = await rig.openai_tools[name].on_invoke_tool(None, json.dumps(arguments))
        assert isinstance(out, str)
        return out

    return Binding(
        OPENAI_AGENTS,
        lambda: list(rig.openai_tools),
        lambda name: rig.openai_tools[name].params_json_schema,
        call_openai,
    )


@pytest.fixture(params=[CLAUDE, OPENAI_AGENTS])
def binding(request: pytest.FixtureRequest, rig: Rig) -> Binding:
    return _binding(rig, request.param)


def _fill(rig: Rig, arguments: dict[str, Any]) -> dict[str, Any]:
    values = {FACT_TEXT: rig.fact_text, FACT_ID: rig.fact_id}
    return {key: values.get(value, value) for key, value in arguments.items()}


def _result_schema(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (SCHEMA_DIR / f"{name}.result.schema.json").read_text()
    )
    return loaded


def _error_schema() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((SCHEMA_DIR / "error.schema.json").read_text())
    return loaded


# Both instants are fixed in every search case: ``known_at`` defaults to the
# clock, which would make the text differ between two calls.
FIXED = {"valid_at": "2024-01-01", "known_at": "2030-01-01"}


@dataclass(frozen=True)
class Case:
    """One argument set and what every binding must do with it."""

    id: str
    function: str
    arguments: dict[str, Any]
    # False: the live Claude SDK rejects it against ``input_schema`` before the
    # handler runs. The handler is called directly here, so the case still
    # runs on both bindings.
    schema_accepts: bool
    # None: success. Otherwise the ``invalid_argument`` message names this.
    invalid_field: str | None


CASES = [
    Case(
        "search_valid_at",
        MEMORY_SEARCH,
        {"query": "pension rate", **FIXED},
        True,
        None,
    ),
    Case(
        "search_hops_2",
        MEMORY_SEARCH,
        {"query": "pension rate", "hops": 2, **FIXED},
        True,
        None,
    ),
    Case(
        "search_hops_3",
        MEMORY_SEARCH,
        {"query": "pension rate", "hops": 3, **FIXED},
        False,
        "hops",
    ),
    Case("search_empty_query", MEMORY_SEARCH, {"query": "", **FIXED}, False, "query"),
    Case(
        "search_blank_query", MEMORY_SEARCH, {"query": "   ", **FIXED}, False, "query"
    ),
    Case(
        "search_bad_date",
        MEMORY_SEARCH,
        {"query": "pension rate", "valid_at": "last July", "known_at": "2030-01-01"},
        True,  # a string: the schema does not check the date format
        "valid_at",
    ),
    Case(
        "search_unknown_argument",
        MEMORY_SEARCH,
        {"query": "pension rate", "top_k": 5, **FIXED},
        False,
        "unknown field",
    ),
    Case("timeline_valid", MEMORY_TIMELINE, {"entity": FACT_TEXT}, True, None),
    Case("timeline_empty_entity", MEMORY_TIMELINE, {"entity": ""}, False, "entity"),
    Case("provenance_valid", MEMORY_PROVENANCE, {"fact_id": FACT_ID}, True, None),
    Case(
        "provenance_unknown_fact", MEMORY_PROVENANCE, {"fact_id": "no-such"}, True, None
    ),
    Case("provenance_missing_id", MEMORY_PROVENANCE, {}, False, "fact_id"),
]
CASE_IDS = [case.id for case in CASES]
SUCCESS_CASES = [case for case in CASES if case.invalid_field is None]
INVALID_CASES = [case for case in CASES if case.invalid_field is not None]


# --- Names and schemas -------------------------------------------------------


async def test_hst_01_exposes_exactly_the_three_function_names(
    binding: Binding,
) -> None:
    assert binding.function_names() == [
        "memory_search",
        "memory_timeline",
        "memory_provenance",
    ]
    assert binding.function_names() == list(MEMORY_FUNCTION_NAMES)


async def test_hst_01_claude_allowed_tools_carry_no_tool_name_entry(rig: Rig) -> None:
    config = rig.tool.config
    _, allowed = build_holodeck_sdk_server(
        [TemporalMemoryToolAdapter(config, rig.tool)]
    )

    assert allowed == [f"mcp__holodeck_tools__{name}" for name in MEMORY_FUNCTION_NAMES]
    assert not any(config.name in name for name in allowed)


async def test_both_bindings_publish_the_same_schema_and_description(
    rig: Rig,
) -> None:
    for name in MEMORY_FUNCTION_NAMES:
        claude = rig.claude_tools[name]
        openai = rig.openai_tools[name]

        assert claude.input_schema == openai.params_json_schema
        assert claude.description == openai.description
        assert claude.description.startswith("Family law memory.")


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
async def test_hst_01_schema_verdict_is_the_same_on_both_bindings(
    rig: Rig, case: Case
) -> None:
    arguments = _fill(rig, case.arguments)
    verdicts = []
    for host in (CLAUDE, OPENAI_AGENTS):
        validator = jsonschema.Draft202012Validator(
            _binding(rig, host).schema(case.function)
        )
        verdicts.append(validator.is_valid(arguments))

    assert verdicts == [case.schema_accepts, case.schema_accepts]


# --- Same text on both bindings ---------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
async def test_both_bindings_return_the_same_text(rig: Rig, case: Case) -> None:
    arguments = _fill(rig, case.arguments)

    claude = await _binding(rig, CLAUDE).call(case.function, arguments)
    openai = await _binding(rig, OPENAI_AGENTS).call(case.function, arguments)

    assert claude == openai
    # Neither binding adds anything to what the one handler layer returns.
    assert claude == await rig.tool.call(case.function, arguments)


# --- Success results ---------------------------------------------------------


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[c.id for c in SUCCESS_CASES])
async def test_hst_01_success_result_matches_its_schema_file(
    rig: Rig, binding: Binding, case: Case
) -> None:
    result = json.loads(await binding.call(case.function, _fill(rig, case.arguments)))

    assert "error" not in result
    jsonschema.validate(result, _result_schema(case.function))


async def test_search_returns_fact_and_chunk_hits(rig: Rig, binding: Binding) -> None:
    result = json.loads(
        await binding.call(MEMORY_SEARCH, {"query": "pension rate", **FIXED})
    )

    assert result["hits"], result
    assert {hit["kind"] for hit in result["hits"]} == {"fact", "chunk"}
    assert result["degraded"] is False
    assert result["valid_at"] == "2024-01-01T00:00:00Z"
    assert result["known_at"] == "2030-01-01T00:00:00Z"


async def test_timeline_lists_the_fact_with_its_sources(
    rig: Rig, binding: Binding
) -> None:
    result = json.loads(await binding.call(MEMORY_TIMELINE, {"entity": rig.fact_text}))

    assert result["entity"] == rig.fact_text
    assert result["entries"], result
    assert rig.fact_id in {entry["fact_id"] for entry in result["entries"]}
    assert all(entry["sources"] for entry in result["entries"])


async def test_provenance_returns_the_fact_and_its_source_text(
    rig: Rig, binding: Binding
) -> None:
    result = json.loads(await binding.call(MEMORY_PROVENANCE, {"fact_id": rig.fact_id}))

    assert result["found"] is True
    assert result["fact"]["fact_id"] == rig.fact_id
    assert result["sources"], result
    assert all(source["text"] for source in result["sources"])


async def test_provenance_of_an_unknown_fact_is_not_found_and_not_an_error(
    rig: Rig, binding: Binding
) -> None:
    result = json.loads(await binding.call(MEMORY_PROVENANCE, {"fact_id": "no-such"}))

    assert result == {"found": False, "fact": None, "sources": []}


# --- Invalid arguments -------------------------------------------------------


@pytest.mark.parametrize("case", INVALID_CASES, ids=[c.id for c in INVALID_CASES])
async def test_invalid_argument_is_an_error_result_with_no_backend_call(
    rig: Rig, binding: Binding, case: Case
) -> None:
    assert case.invalid_field is not None
    before = dict(rig.backend.calls)

    result = json.loads(await binding.call(case.function, _fill(rig, case.arguments)))

    assert result["error"]["code"] == "invalid_argument"
    assert case.invalid_field in result["error"]["message"]
    jsonschema.validate(result, _error_schema())
    assert dict(rig.backend.calls) == before


# --- Where the two hosts differ ---------------------------------------------


async def test_the_openai_binding_turns_non_object_text_into_an_error_result(
    rig: Rig,
) -> None:
    # The OpenAI Agents SDK hands a function tool the raw argument text. The
    # Claude SDK parses it first and the handler gets a dict, so this case has
    # no Claude counterpart.
    before = dict(rig.backend.calls)

    for raw in ("not json", "[1, 2]"):
        out = await rig.openai_tools[MEMORY_SEARCH].on_invoke_tool(None, raw)

        assert json.loads(out)["error"]["code"] == "invalid_argument"
    assert dict(rig.backend.calls) == before


@pytest.mark.parametrize(
    "raw",
    [
        # More than 4,300 digits: json.loads raises ValueError.
        '{"query": "rate", "hops": ' + "1" * 5000 + "}",
        # Nested far past the recursion limit: json.loads raises RecursionError.
        "[" * 200_000 + "]" * 200_000,
        '{"query": ' + "[" * 200_000 + "]" * 200_000 + "}",
    ],
    ids=["huge_integer", "deep_array", "deep_value"],
)
async def test_the_openai_binding_survives_json_the_parser_cannot_take(
    rig: Rig, raw: str
) -> None:
    before = dict(rig.backend.calls)

    out = await rig.openai_tools[MEMORY_SEARCH].on_invoke_tool(None, raw)

    result = json.loads(out)
    assert result["error"]["code"] == "invalid_argument"
    jsonschema.validate(result, _error_schema())
    assert dict(rig.backend.calls) == before


async def test_a_schema_invalid_argument_differs_between_the_hosts(rig: Rig) -> None:
    """Accepted difference: the Claude SDK checks ``input_schema`` first.

    The same ``hops: 3`` is stopped by the SDK's own validation on Claude, with
    an SDK error result and no call to the memory. On OpenAI Agents it reaches
    our handler, which returns ``invalid_argument``. Neither calls the backend.
    The Claude call goes through the SDK server's tool dispatch (the layer that
    validates), in memory and with no subprocess.
    """
    import mcp

    if not hasattr(mcp, "Client"):
        pytest.skip("this test drives the in-memory mcp.Client of mcp 2.x")
    config = rig.tool.config
    server_config, _ = build_holodeck_sdk_server(
        [TemporalMemoryToolAdapter(config, rig.tool)]
    )
    before = dict(rig.backend.calls)
    arguments = {"query": "rate", "hops": 3}

    async with mcp.Client(server_config["instance"]) as client:
        claude = await client.call_tool(MEMORY_SEARCH, arguments)
        accepted = await client.call_tool(MEMORY_SEARCH, {"query": "pension rate"})
    openai = await rig.openai_tools[MEMORY_SEARCH].on_invoke_tool(
        None, json.dumps(arguments)
    )

    claude_dump = claude.model_dump(by_alias=True)
    assert claude_dump["isError"] is True
    [block] = claude_dump["content"]
    assert block["text"] == "Input validation error: 3 is greater than the maximum of 2"
    assert "invalid_argument" not in block["text"]
    assert json.loads(openai) == {
        "error": {"code": "invalid_argument", "message": "hops must be 1 or 2"}
    }
    # The dispatch does reach the handler for a valid call, so the check above
    # stopped the invalid one and was not a broken server.
    assert accepted.model_dump(by_alias=True)["isError"] is False
    assert dict(rig.backend.calls) != before


async def test_the_schema_is_passed_through_without_strict_mode(rig: Rig) -> None:
    # Strict mode would need every property required, which would make the
    # optional search arguments mandatory.
    search = rig.openai_tools[MEMORY_SEARCH]

    assert search.strict_json_schema is False
    assert search.params_json_schema["required"] == ["query"]
    assert set(search.params_json_schema["properties"]) == {
        "query",
        "valid_at",
        "known_at",
        "hops",
    }
