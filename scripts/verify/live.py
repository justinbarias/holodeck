"""Live verify tier: real providers, opt-in by credential.

For each backend whose credential is set in the environment:

* Claude Agent SDK: ``CLAUDE_CODE_OAUTH_TOKEN``, agent
  ``tests/fixtures/verify/live-full-claude.yaml`` (``auth_provider: oauth_token``).
* OpenAI Agents SDK: ``OPENAI_API_KEY``, agent
  ``tests/fixtures/verify/live-full-openai.yaml``.

Each agent exercises every tool type its backend supports (function, stdio
MCP, a vectorstore tool on every supported vector store provider,
hierarchical_document on Qdrant, Postgres and Weaviate, subagents, skills on
OpenAI) with one ``type: code`` graded test case per tool, and exports traces
to a local Aspire dashboard. The script:

1. starts or reuses the Docker containers ``holodeck-verify-qdrant``,
   ``-chroma``, ``-postgres``, ``-weaviate``, ``-cosmos`` (the Cosmos DB Linux
   emulator) and ``-aspire`` (skip with ``--no-infra``);
2. runs ``holodeck test run`` and reports PASS/FAIL/SKIP per test case from
   the results JSON. The vector tools need ``OPENAI_API_KEY`` (embeddings).
   Pinecone and Azure AI Search have no local emulator: their tools run only
   when ``PINECONE_API_KEY`` or ``AZURE_AI_SEARCH_ENDPOINT`` plus
   ``AZURE_AI_SEARCH_API_KEY`` are set. Any tool whose prerequisite is missing
   is removed from the run copy of the agent and its cases are reported as SKIP
   with the reason;
3. proves each vector store was really used, not the in-memory fallback: every
   tool gets a uniquely named collection in the run copy, and afterwards the
   script asks the provider for that collection and its record count, then
   deletes it;
4. starts ``holodeck serve`` and drives 2 threads x 2 turns over AG-UI
   ``POST /awp``: thread A turn 2 must call ``smoke_echo``; thread B turn 2
   must recall B's code word and not A's;
5. checks that no credential value was written to the run directory.

``--only providers`` runs only the vector store matrix (steps 1 to 3, no AG-UI).
Transcripts and logs go to ``.verify/live/<UTC ts>/``. Traces are for human
inspection in Aspire; the script does not read them back.

Exit codes: 0 all run backends passed, 1 any failure (including a backend
with its credential set and no passing check), 2 every backend skipped.

Usage::

    uv run python scripts/verify/live.py [--backend claude|openai] [--limit N]
        [--no-infra] [--only providers] [--env-file PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import subprocess  # noqa: S404 - asks the uv CLI for its directories
import sys
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

if not __package__:
    # Run as a file path (PEP 366): import the package so relative imports work.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.verify"  # noqa: A001

import yaml  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from holodeck.config.defaults import get_embedding_dimensions  # noqa: E402
from holodeck.lib.retrieval import CollectionSchema, create_adapter  # noqa: E402
from holodeck.lib.retrieval.schemas import (  # noqa: E402
    document_schema,
    hierarchical_schema,
)
from holodeck.models.test_result import TestReport, TestResult  # noqa: E402

from .agui import (  # noqa: E402
    TurnSpec,
    check_turn,
    drive,
    recall_problems,
    save_transcript,
)
from .infra import (  # noqa: E402
    ASPIRE,
    ASPIRE_TRACES_URL,
    CHROMA,
    COSMOS,
    POSTGRES,
    QDRANT,
    WEAVIATE,
    InfraStatus,
    Service,
    ensure,
    stop_hint,
)
from .procs import (  # noqa: E402
    ROOT,
    Check,
    print_table,
    run_cli,
    scrubbed_env,
    serve,
    tail,
)

FIXTURES = ROOT / "tests" / "fixtures" / "verify"
LIVE_ENV = ROOT / ".verify" / "live.env"
EMBEDDING_CREDENTIAL = "OPENAI_API_KEY"
RAG_TOOL_TYPES = frozenset({"vectorstore", "hierarchical_document"})
AGUI_TOOL = "smoke_echo"
ONLY_CHOICES = ("providers",)
SETTLE_SECONDS = 30.0  # hosted stores can lag between a write and a count

YamlMap = dict[str, object]


@dataclass(frozen=True)
class VectorProvider:
    """One vector store provider and what it takes to run its cases.

    A provider with a ``service`` runs against a local container. One without
    has no faithful emulator: it runs only when every name in ``credentials``
    is set, and its cases are SKIP otherwise.
    """

    name: str  # database.provider in the agent YAML
    service: Service | None = None
    credentials: tuple[str, ...] = ()


PROVIDERS = (
    VectorProvider("qdrant", service=QDRANT),
    VectorProvider("chromadb", service=CHROMA),
    VectorProvider("postgres", service=POSTGRES),
    VectorProvider("weaviate", service=WEAVIATE),
    VectorProvider("azure-cosmos-nosql", service=COSMOS),
    VectorProvider("pinecone", credentials=("PINECONE_API_KEY",)),
    VectorProvider(
        "azure-ai-search",
        credentials=("AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_API_KEY"),
    ),
)
LOCAL_SERVICES = tuple(p.service for p in PROVIDERS if p.service is not None)


@dataclass(frozen=True)
class LiveBackend:
    """One backend with its credential and agent fixture."""

    name: str
    credential: str
    fixture: str  # agent file in tests/fixtures/verify


BACKENDS = (
    LiveBackend(
        name="claude",
        credential="CLAUDE_CODE_OAUTH_TOKEN",
        fixture="live-full-claude.yaml",
    ),
    LiveBackend(
        name="openai",
        credential="OPENAI_API_KEY",
        fixture="live-full-openai.yaml",
    ),
)


def load_live_env(path: Path) -> list[str]:
    """Load ``KEY=value`` lines from *path* into ``os.environ``.

    Variables already set in the environment win. The file lives under the
    gitignored ``.verify/`` directory; values are never printed.

    Returns:
        The names that were loaded.
    """
    if not path.exists():
        return []
    loaded: list[str] = []
    for key, value in dotenv_values(path).items():
        if value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


# ---------------------------------------------------------------------------
# Agent config preparation
# ---------------------------------------------------------------------------


def _maps(value: object) -> list[YamlMap]:
    """Return the mapping items of a YAML list (other items are dropped)."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _expected_names(case: YamlMap) -> list[str]:
    """Return every expected tool name in a test case (all turns)."""
    entries: list[object] = []
    for holder in [case, *_maps(case.get("turns"))]:
        expected = holder.get("expected_tools")
        if isinstance(expected, list):
            entries.extend(expected)
    names: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.append(str(entry["name"]))
    return names


def _without(names: object, dropped: set[str]) -> object:
    """Remove *dropped* names from a YAML list of tool names."""
    if not isinstance(names, list):
        return names
    return [name for name in names if name not in dropped]


def _rag_names(data: YamlMap, provider: str | None = None) -> set[str]:
    """Return the names of vectorstore and hierarchical_document tools.

    With *provider*, only those whose ``database.provider`` is that provider.
    """
    names: set[str] = set()
    for tool in _maps(data.get("tools")):
        if tool.get("type") not in RAG_TOOL_TYPES:
            continue
        database = tool.get("database")
        found = database.get("provider") if isinstance(database, dict) else None
        if provider is None or found == provider:
            names.add(str(tool.get("name")))
    return names


def _expects(case: YamlMap, names: set[str]) -> bool:
    """Return whether a test case expects a call to one of the tools *names*.

    Matching is by substring, like ``expected_tools`` at run time, so the OpenAI
    ``<tool>_search`` and Claude ``mcp__<server>__<tool>`` names match too.
    """
    return any(r in name for name in _expected_names(case) for r in names)


def _drop_tools(out: YamlMap, dropped: set[str]) -> list[str]:
    """Remove tools *dropped* from *out* in place; return the dropped case names.

    Test cases that expect one of them and their names in skill and subagent
    tool lists go with them.
    """
    tools = _maps(out.get("tools"))
    kept_tools = [t for t in tools if str(t.get("name")) not in dropped]
    for tool in kept_tools:
        if "allowed_tools" in tool:
            tool["allowed_tools"] = _without(tool["allowed_tools"], dropped)
    out["tools"] = kept_tools
    kept_cases: list[YamlMap] = []
    dropped_cases: list[str] = []
    for case in _maps(out.get("test_cases")):
        if _expects(case, dropped):
            dropped_cases.append(str(case.get("name")))
        else:
            kept_cases.append(case)
    out["test_cases"] = kept_cases
    for section in ("claude", "openai"):
        block = out.get(section)
        agents = block.get("agents") if isinstance(block, dict) else None
        if isinstance(agents, dict):
            for spec in agents.values():
                if isinstance(spec, dict) and "tools" in spec:
                    spec["tools"] = _without(spec["tools"], dropped)
    return dropped_cases


def prepare_config(
    data: YamlMap, *, drop_rag: bool, traces: bool
) -> tuple[YamlMap, list[str]]:
    """Return the run copy of an agent config and the names of dropped cases.

    With *drop_rag*, vectorstore and hierarchical_document tools are removed
    together with ``embedding_provider``, every test case that expects one of
    them, and their names in skill and subagent tool lists. Without *traces*,
    observability is disabled so no exporter targets a missing collector.
    """
    out = copy.deepcopy(data)
    dropped_cases: list[str] = []
    if drop_rag:
        dropped_cases = _drop_tools(out, _rag_names(out))
        out.pop("embedding_provider", None)
    if not traces:
        observability = out.get("observability")
        if isinstance(observability, dict):
            observability["enabled"] = False
    return out, dropped_cases


def drop_providers(
    data: YamlMap, reasons: Mapping[str, str]
) -> tuple[YamlMap, dict[str, str]]:
    """Return a copy without the tools of the providers in *reasons*.

    Returns:
        The config and, for each dropped test case, the reason its provider
        cannot run.
    """
    out = copy.deepcopy(data)
    skipped: dict[str, str] = {}
    for provider, reason in reasons.items():
        for case in _drop_tools(out, _rag_names(out, provider)):
            skipped[case] = reason
    return out, skipped


def provider_cases_only(data: YamlMap) -> YamlMap:
    """Return a copy that keeps only the test cases for vector store tools."""
    out = copy.deepcopy(data)
    rag = _rag_names(out)
    out["test_cases"] = [c for c in _maps(out.get("test_cases")) if _expects(c, rag)]
    return out


@dataclass(frozen=True)
class RagTarget:
    """One vector store tool's collection, as the tool itself addresses it."""

    tool: str
    provider: str
    schema: CollectionSchema
    connection: Mapping[str, object]


def unique_collections(data: YamlMap, token: str) -> YamlMap:
    """Return a copy where every remote RAG tool has its own collection.

    A ``collection_name`` unique to this run means a collection that exists
    afterwards was created by this run's tool, never by an earlier run, and
    can be deleted without touching anything else.
    """
    out = copy.deepcopy(data)
    for tool in _maps(out.get("tools")):
        database = tool.get("database")
        if (
            tool.get("type") in RAG_TOOL_TYPES
            and isinstance(database, dict)
            and database.get("provider") != "in-memory"
        ):
            short = str(tool["name"]).removeprefix("verify_")
            database["collection_name"] = f"verify_{token}_{short}"
    return out


def rag_targets(data: YamlMap) -> list[RagTarget]:
    """Return the collection of every remote RAG tool in the config.

    Names and settings come from the same functions the tools use, so a
    collection found here is the one the tool wrote.
    """
    embedding = data.get("embedding_provider")
    model = embedding.get("name") if isinstance(embedding, dict) else None
    provider = embedding.get("provider") if isinstance(embedding, dict) else None
    dimensions = get_embedding_dimensions(
        str(model) if model else None, provider=str(provider or "openai")
    )
    targets: list[RagTarget] = []
    for tool in _maps(data.get("tools")):
        database = tool.get("database")
        if tool.get("type") not in RAG_TOOL_TYPES or not isinstance(database, dict):
            continue
        if database.get("provider") == "in-memory":
            continue
        connection = {k: v for k, v in database.items() if k != "provider"}
        name = str(tool["name"])
        schema = (
            hierarchical_schema(dimensions, name, connection)
            if tool["type"] == "hierarchical_document"
            else document_schema(dimensions, connection)
        )
        targets.append(RagTarget(name, str(database["provider"]), schema, connection))
    return targets


def service_name(data: YamlMap) -> str | None:
    """Return ``observability.service_name`` when tracing is enabled."""
    observability = data.get("observability")
    if not isinstance(observability, dict) or not observability.get("enabled"):
        return None
    name = observability.get("service_name")
    return name if isinstance(name, str) else None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def load_report(agent_dir: Path) -> TestReport | None:
    """Return the newest test report the runner wrote under ``results/``."""
    files = sorted(
        (agent_dir / "results").glob("*/*.json"), key=lambda p: p.stat().st_mtime
    )
    if not files:
        return None
    try:
        data: object = json.loads(files[-1].read_text(encoding="utf-8"))
        report = data.get("report") if isinstance(data, dict) else None
        return TestReport.model_validate(report)
    except (OSError, ValueError, ValidationError):
        return None


def case_problem(result: TestResult) -> str:
    """Return the first reason a test case failed, in one line."""
    errors: list[str] = [str(err) for err in result.errors]
    for turn in result.turns or []:
        errors.extend(f"turn {turn.turn_index}: {err}" for err in turn.errors)
    if errors:
        return errors[0]
    for turn in result.turns or []:
        if turn.tools_matched is False:
            return (
                f"turn {turn.turn_index}: expected tools {turn.expected_tools} "
                f"not called (called: {turn.tool_calls})"
            )
        failed = [m.metric_name for m in turn.metric_results if m.passed is False]
        if failed:
            return f"turn {turn.turn_index}: grader failed: {', '.join(failed)}"
    if result.tools_matched is False:
        return f"expected tools {result.expected_tools} not called"
    failed = [m.metric_name for m in result.metric_results if m.passed is False]
    if failed:
        return f"grader failed: {', '.join(failed)}"
    return "failed"


def case_checks(prefix: str, report: TestReport) -> list[Check]:
    """Return one PASS/FAIL row per test case in *report*."""
    checks: list[Check] = []
    for index, result in enumerate(report.results, start=1):
        name = result.test_name or f"case {index}"
        calls = sorted(
            {c for turn in result.turns or [] for c in turn.tool_calls}
            | set(result.tool_calls)
        )
        checks.append(
            Check(
                f"{prefix}: case {name}",
                "PASS" if result.passed else "FAIL",
                (
                    f"tools: {', '.join(calls) or 'none'}"
                    if result.passed
                    else case_problem(result)
                ),
            )
        )
    return checks


def leaked_credentials(root: Path, names: Iterable[str]) -> list[str]:
    """Return the names of credentials whose value occurs in a file under *root*.

    Values are compared as bytes and never returned or printed.
    """
    secrets = {
        name: value.encode()
        for name in names
        if len(value := os.environ.get(name, "")) >= 8
    }
    leaked: set[str] = set()
    for path in root.rglob("*"):
        if not secrets.keys() - leaked:
            break
        if not path.is_file():
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        leaked.update(n for n, v in secrets.items() if v in content)
    return sorted(leaked)


# ---------------------------------------------------------------------------
# Vector store providers
# ---------------------------------------------------------------------------


def provider_skip_reasons(statuses: Mapping[str, InfraStatus]) -> dict[str, str]:
    """Return why each provider cannot run; a provider that can run is absent.

    *statuses* maps container names to their ``InfraStatus``. A local provider
    needs its container ready; a hosted one needs every credential set.
    """
    reasons: dict[str, str] = {}
    for provider in PROVIDERS:
        if provider.service is not None:
            status = statuses.get(provider.service.container)
            if status is None or not status.ready:
                detail = "not started" if status is None else status.detail
                reasons[provider.name] = (
                    f"{provider.service.label} unavailable: {detail}"
                )
            continue
        missing = [n for n in provider.credentials if not os.environ.get(n)]
        if missing:
            reasons[provider.name] = (
                f"{', '.join(missing)} not set (hosted service, no local emulator)"
            )
    return reasons


def embedding_skip_reason() -> str | None:
    """Return why the vector tools cannot run at all, or ``None``."""
    if not os.environ.get(EMBEDDING_CREDENTIAL):
        return f"{EMBEDDING_CREDENTIAL} not set (OpenAI embeddings)"
    return None


def exercised_targets(
    data: YamlMap, targets: Sequence[RagTarget], ran: set[str]
) -> list[RagTarget]:
    """Return the targets whose tool a test case that ran was meant to call."""
    cases = [c for c in _maps(data.get("test_cases")) if str(c.get("name")) in ran]
    return [t for t in targets if any(_expects(c, {t.tool}) for c in cases)]


async def _probe(target: RagTarget, *, check: bool) -> Check | None:
    """Check that *target* holds records, then delete its collection.

    Without *check* the collection is only deleted. Returns ``None`` when there
    is nothing to report (not checked and nothing to delete).
    """
    row = f"provider {target.provider} ({target.tool}) holds records"
    name = target.schema.name
    adapter = None
    status, detail = "PASS", ""
    try:
        adapter = create_adapter(target.provider, target.schema, target.connection)
        exists = await adapter.collection_exists()
        if not exists:
            if not check:
                return None
            return Check(
                row,
                "FAIL",
                f"collection {name} does not exist: the tool fell back to "
                "in-memory storage or never ingested",
            )
        if check:
            deadline = time.monotonic() + SETTLE_SECONDS
            count = await adapter.count()
            while count == 0 and time.monotonic() < deadline:
                await asyncio.sleep(1.0)
                count = await adapter.count()
            if count == 0:
                status, detail = "FAIL", f"collection {name} is empty"
            else:
                detail = f"{count} records in {name}"
        await adapter.delete_collection()
        detail = f"{detail}; deleted" if detail else f"deleted {name}"
    except Exception as error:  # noqa: BLE001 - any failure is a FAIL row
        status, detail = "FAIL", f"{type(error).__name__}: {str(error)[:300]}"
        if not check:
            row = f"provider {target.provider} ({target.tool}) cleanup"
    finally:
        if adapter is not None:
            await adapter.close()
    return Check(row, status, detail)


async def _probe_all(
    targets: Sequence[RagTarget], checked: set[str]
) -> list[Check | None]:
    return [await _probe(t, check=t.tool in checked) for t in targets]


def provider_checks(
    prefix: str, targets: Sequence[RagTarget], checked: Sequence[RagTarget]
) -> list[Check]:
    """Return a row per exercised target, then delete every target's collection.

    A row is PASS only when the provider itself holds the run's collection
    with records: the tools fall back to in-memory storage when a provider
    cannot be created, and then every other check could still pass.
    """
    rows = asyncio.run(_probe_all(targets, {t.tool for t in checked}))
    return [
        Check(f"{prefix}: {row.name}", row.status, row.detail)
        for row in rows
        if row is not None
    ]


# ---------------------------------------------------------------------------
# One backend
# ---------------------------------------------------------------------------


def conversation() -> tuple[list[TurnSpec], str, str]:
    """Return the 2x2 turn plan and the two threads' code words."""
    word_a = f"ALPHA{uuid.uuid4().hex[:6]}"
    word_b = f"BRAVO{uuid.uuid4().hex[:6]}"
    return (
        [
            TurnSpec("A", f"Hi, my code word is {word_a}. Reply with a greeting."),
            TurnSpec("B", f"Hi, my code word is {word_b}. Reply with a greeting."),
            TurnSpec(
                "A",
                f"Please use the smoke tool ({AGUI_TOOL}) with text live-ping.",
                expect_tool=True,
                tool_name=AGUI_TOOL,
            ),
            TurnSpec("B", "What code word did I give you earlier? Reply with it only."),
        ],
        word_a,
        word_b,
    )


def uv_dirs() -> dict[str, str]:
    """Return the developer's uv cache and Python dirs as ``UV_*`` variables.

    The MCP tool starts through ``uvx``; with ``HOME`` isolated it would
    otherwise use an empty cache and download its wheels and Python again.
    """
    uv = shutil.which("uv")
    found: dict[str, str] = {}
    for key, command in (
        ("UV_CACHE_DIR", "cache"),
        ("UV_PYTHON_INSTALL_DIR", "python"),
    ):
        if os.environ.get(key):
            found[key] = os.environ[key]
            continue
        if uv is None:
            continue
        done = subprocess.run(  # noqa: S603
            [uv, command, "dir"], capture_output=True, text=True, check=False
        )
        if done.returncode == 0 and done.stdout.strip():
            found[key] = done.stdout.strip()
    return found


def isolated_home(home: Path) -> dict[str, str]:
    """Return env overrides that keep a live run off developer credentials.

    ``HOME`` points at an empty per-run directory: the holodeck CLI loads
    ``~/.holodeck/.env`` (which could add, say, ``ANTHROPIC_API_KEY``, which
    the Claude CLI prefers over the OAuth token) and the Claude CLI reads its
    stored login from its config dir. Either would let a run pass on
    credentials other than the one under test.
    """
    home.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        **uv_dirs(),
    }


def agui_checks(
    prefix: str, config: Path, env: dict[str, str], workdir: Path
) -> list[Check]:
    """Start ``holodeck serve`` and return the AG-UI 2 threads x 2 turns rows."""
    checks: list[Check] = []
    turns, word_a, word_b = conversation()
    serve_log = workdir / "serve.log"
    with serve(config, env=env, log=serve_log, timeout=600) as (base_url, error):
        if error is not None:
            checks.append(
                Check(f"{prefix}: serve ready", "FAIL", f"{error}; see {serve_log}")
            )
            results = []
        else:
            results = drive(base_url, turns, timeout=600)
    if results:
        save_transcript(results, workdir / "agui.jsonl")

    for result in results:
        problems = (
            [result.transport_error]
            if result.transport_error
            else check_turn(
                result.events,
                expect_tool=result.spec.expect_tool,
                expect_tool_name=result.spec.tool_name,
            )
        )
        if result.spec.thread == "B" and result.index == 2:
            problems += recall_problems(result.text, expected=word_b, forbidden=word_a)
        elif result.spec.thread == "B" and word_a.casefold() in result.text.casefold():
            problems.append("thread B reply contains thread A's code word")
        tools = sum(e.get("type") == "TOOL_CALL_START" for e in result.events)
        detail = "; ".join(problems) or f"{len(result.events)} events, {tools} tools"
        checks.append(
            Check(
                f"{prefix}: agui thread {result.spec.thread} turn {result.index}",
                "FAIL" if problems else "PASS",
                detail,
            )
        )
    return checks


def live_backend(
    backend: LiveBackend,
    rundir: Path,
    limit: int | None,
    provider_reasons: Mapping[str, str],
    aspire: InfraStatus,
    only: str | None = None,
) -> tuple[list[Check], str | None]:
    """Run the live checks for one backend whose credential is set.

    *provider_reasons* maps each provider that cannot run to why; its tools
    are removed from the run copy and its cases reported as SKIP.

    Returns:
        The checks and the trace service name (``None`` when not traced).
    """
    prefix = backend.name
    if shutil.which("uvx") is None:
        return [Check(f"{prefix}: setup", "FAIL", "uvx not found (MCP tool)")], None

    workdir = rundir / prefix
    agent_dir = workdir / "agent"
    shutil.copytree(
        FIXTURES,
        agent_dir,
        ignore=shutil.ignore_patterns("results", ".*", "__pycache__", "*.env*"),
    )
    raw: object = yaml.safe_load((FIXTURES / backend.fixture).read_text("utf-8"))
    if not isinstance(raw, dict):
        return [Check(f"{prefix}: setup", "FAIL", f"{backend.fixture} not a map")], None
    embedding_reason = embedding_skip_reason()
    data, dropped = prepare_config(
        raw, drop_rag=embedding_reason is not None, traces=aspire.ready
    )
    skipped = {name: str(embedding_reason) for name in dropped}
    data, by_provider = drop_providers(data, provider_reasons)
    skipped.update(by_provider)
    if only == "providers":
        data = provider_cases_only(data)
    data = unique_collections(data, f"{backend.name}{uuid.uuid4().hex[:8]}")
    targets = rag_targets(data) if embedding_reason is None else []
    config = agent_dir / "agent.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    checks = [
        Check(f"{prefix}: case {name}", "SKIP", reason)
        for name, reason in skipped.items()
    ]
    traced = service_name(data)
    if traced is None:
        checks.append(
            Check(f"{prefix}: traces", "SKIP", f"Aspire unavailable: {aspire.detail}")
        )

    # Keep this backend's credential, the embedding key and the credentials of
    # the hosted vector stores that will run; drop every other credential and
    # the fake endpoints from other tiers.
    hosted = [
        name
        for provider in PROVIDERS
        if provider.name not in provider_reasons
        for name in provider.credentials
    ]
    kept = tuple(dict.fromkeys((backend.credential, EMBEDDING_CREDENTIAL, *hosted)))
    env = scrubbed_env(isolated_home(workdir / "home"), keep=kept)

    test_log = workdir / "test-run.log"
    args = ["test", "run", str(config)]
    if limit is not None:
        args += ["--limit", str(limit)]
    code = run_cli(args, cwd=agent_dir, env=env, log=test_log, timeout=1800)
    report = load_report(agent_dir)
    if report is None or not report.results:
        checks.append(
            Check(
                f"{prefix}: holodeck test run",
                "FAIL",
                f"exit {code}, no results; see {test_log}\n{tail(test_log)}",
            )
        )
    else:
        checks.extend(case_checks(prefix, report))
        if code != 0 and all(c.status != "FAIL" for c in checks):
            checks.append(
                Check(
                    f"{prefix}: holodeck test run",
                    "FAIL",
                    f"exit {code}; see {test_log}\n{tail(test_log)}",
                )
            )

    if only is None:
        checks.extend(agui_checks(prefix, config, env, workdir))

    ran = {r.test_name for r in report.results if r.test_name} if report else set()
    checks.extend(
        provider_checks(prefix, targets, exercised_targets(data, targets, ran))
    )

    leaked = leaked_credentials(workdir, kept)
    checks.append(
        Check(
            f"{prefix}: no credential in artifacts",
            "FAIL" if leaked else "PASS",
            f"value of {', '.join(leaked)} found under {workdir}" if leaked else "",
        )
    )
    return checks, traced


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def infra_checks(statuses: list[InfraStatus]) -> list[Check]:
    """Return one row per infrastructure service."""
    return [
        Check(
            f"infra: {s.service.label}",
            "PASS" if s.ready else "SKIP",
            s.detail,
        )
        for s in statuses
    ]


def no_pass_failures(checks: list[Check], ran: list[str]) -> list[Check]:
    """Return a FAIL row for each run backend with no passing case or turn."""
    return [
        Check(f"{name}: overall", "FAIL", "credential set but no check passed")
        for name in ran
        if not any(
            c.status == "PASS"
            and c.name.startswith((f"{name}: case ", f"{name}: agui "))
            for c in checks
        )
    ]


def print_trace_help(services: list[str], aspire: InfraStatus) -> None:
    """Print where the traces are and what to check in Aspire."""
    print(f"\nTraces: {ASPIRE_TRACES_URL} ({aspire.detail})")
    print(
        f"  Check: services {', '.join(services)} each show invoke_agent and "
        "execute_tool spans for smoke_echo, verify_marker and leak_probe."
    )
    print(
        "  Check: the leak_probe span's tool output shows "
        "[REDACTED:github-token], never the ghp_ token."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the live tier and return the process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend", choices=[b.name for b in BACKENDS], action="append"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N test cases"
    )
    parser.add_argument(
        "--no-infra",
        action="store_true",
        help="Do not start Docker containers; local vector store and trace "
        "checks SKIP (hosted providers still run when their credentials are set)",
    )
    parser.add_argument(
        "--only",
        choices=ONLY_CHOICES,
        default=None,
        help="providers: only the vector store matrix (one case per provider "
        "and hierarchical_document case), no AG-UI turns",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=LIVE_ENV,
        help="KEY=value credentials file (default: .verify/live.env)",
    )
    args = parser.parse_args(argv)
    selected = [b for b in BACKENDS if not args.backend or b.name in args.backend]
    loaded = load_live_env(args.env_file)
    if loaded:
        print(f"verify-live: loaded {', '.join(sorted(loaded))} from {args.env_file}")

    checks: list[Check] = []
    runnable: list[LiveBackend] = []
    for backend in selected:
        if os.environ.get(backend.credential):
            runnable.append(backend)
        else:
            checks.append(
                Check(f"{backend.name}: all", "SKIP", f"{backend.credential} not set")
            )
    if not runnable:
        print_table("verify-live", checks)
        names = ", ".join(b.credential for b in selected)
        print(f"\nverify-live: SKIPPED (no credentials; set {names} to run)")
        return 2

    services = [*LOCAL_SERVICES, ASPIRE]
    if args.no_infra:
        statuses = [InfraStatus(s, False, "--no-infra") for s in services]
    else:
        print(
            "verify-live: starting "
            f"{', '.join(s.label for s in services)} (Docker; the first start "
            "pulls images and Cosmos DB takes a minute)..."
        )
        with ThreadPoolExecutor(max_workers=len(services)) as pool:
            statuses = list(pool.map(ensure, services))
    by_container = {s.service.container: s for s in statuses}
    aspire = by_container[ASPIRE.container]
    reasons = provider_skip_reasons(by_container)
    checks.extend(infra_checks(statuses))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rundir = ROOT / ".verify" / "live" / stamp
    rundir.mkdir(parents=True, exist_ok=True)
    traced: list[str] = []
    for backend in runnable:
        print(f"verify-live: running {backend.name} ({backend.fixture})...")
        backend_checks, service = live_backend(
            backend, rundir, args.limit, reasons, aspire, args.only
        )
        checks.extend(backend_checks)
        if service is not None:
            traced.append(service)
    checks.extend(no_pass_failures(checks, [b.name for b in runnable]))

    print_table(f"verify-live ({rundir.relative_to(ROOT)})", checks)
    failed = [c for c in checks if c.status == "FAIL"]
    for check in failed:
        if "\n" in check.detail:
            print(f"\n--- {check.name} ---\n{check.detail}")
    if traced:
        print_trace_help(traced, aspire)
    hint = stop_hint(statuses)
    if hint:
        print(f"\nContainers keep running for trace browsing. Stop them: {hint}")
    print(
        f"\nverify-live: {'FAILED' if failed else 'PASSED'} "
        f"({sum(c.status == 'PASS' for c in checks)} passed, {len(failed)} failed, "
        f"{sum(c.status == 'SKIP' for c in checks)} skipped)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
