"""Live verify tier: real providers, opt-in by credential.

For each backend whose credential is set in the environment:

* Claude Agent SDK: ``CLAUDE_CODE_OAUTH_TOKEN``, agent
  ``tests/fixtures/verify/live-full-claude.yaml`` (``auth_provider: oauth_token``).
* OpenAI Agents SDK: ``OPENAI_API_KEY``, agent
  ``tests/fixtures/verify/live-full-openai.yaml``.

Each agent exercises every tool type its backend supports (function, stdio
MCP, Qdrant and ChromaDB vectorstores, hierarchical_document, subagents, skills on
OpenAI) with one ``type: code`` graded test case per type, and exports traces
to a local Aspire dashboard. The script:

1. starts or reuses the ``holodeck-verify-qdrant``, ``holodeck-verify-chroma``
   and ``holodeck-verify-aspire`` containers (skip with ``--no-infra``);
2. runs ``holodeck test run`` and reports PASS/FAIL/SKIP per test case from
   the results JSON. The vector tools need ``OPENAI_API_KEY`` (embeddings)
   and both vector stores; without them, they are removed from the run copy of the
   agent and their cases are reported as SKIP;
3. starts ``holodeck serve`` and drives 2 threads x 2 turns over AG-UI
   ``POST /awp``: thread A turn 2 must call ``smoke_echo``; thread B turn 2
   must recall B's code word and not A's;
4. checks that no credential value was written to the run directory.

Transcripts and logs go to ``.verify/live/<UTC ts>/``. Traces are for human
inspection in Aspire; the script does not read them back.

Exit codes: 0 all run backends passed, 1 any failure (including a backend
with its credential set and no passing check), 2 every backend skipped.

Usage::

    uv run python scripts/verify/live.py [--backend claude|openai] [--limit N]
        [--no-infra] [--env-file PATH]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess  # noqa: S404 - asks the uv CLI for its directories
import sys
import uuid
from collections.abc import Iterable, Sequence
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
    QDRANT,
    InfraStatus,
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

YamlMap = dict[str, object]


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
        tools = _maps(out.get("tools"))
        rag = {str(t.get("name")) for t in tools if t.get("type") in RAG_TOOL_TYPES}
        kept_tools = [t for t in tools if str(t.get("name")) not in rag]
        for tool in kept_tools:
            if "allowed_tools" in tool:
                tool["allowed_tools"] = _without(tool["allowed_tools"], rag)
        out["tools"] = kept_tools
        out.pop("embedding_provider", None)
        kept_cases: list[YamlMap] = []
        for case in _maps(out.get("test_cases")):
            expected = _expected_names(case)
            if any(r in name for name in expected for r in rag):
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
                        spec["tools"] = _without(spec["tools"], rag)
    if not traces:
        observability = out.get("observability")
        if isinstance(observability, dict):
            observability["enabled"] = False
    return out, dropped_cases


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


def rag_skip_reason(stores: Sequence[InfraStatus]) -> str | None:
    """Return why the vector tools cannot run, or ``None`` when they can."""
    if not os.environ.get(EMBEDDING_CREDENTIAL):
        return f"{EMBEDDING_CREDENTIAL} not set (OpenAI embeddings)"
    for store in stores:
        if not store.ready:
            return f"{store.service.label} unavailable: {store.detail}"
    return None


def live_backend(
    backend: LiveBackend,
    rundir: Path,
    limit: int | None,
    stores: Sequence[InfraStatus],
    aspire: InfraStatus,
) -> tuple[list[Check], str | None]:
    """Run the live checks for one backend whose credential is set.

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
    skip_reason = rag_skip_reason(stores)
    data, dropped = prepare_config(
        raw, drop_rag=skip_reason is not None, traces=aspire.ready
    )
    config = agent_dir / "agent.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    checks = [
        Check(f"{prefix}: case {name}", "SKIP", str(skip_reason)) for name in dropped
    ]
    traced = service_name(data)
    if traced is None:
        checks.append(
            Check(f"{prefix}: traces", "SKIP", f"Aspire unavailable: {aspire.detail}")
        )

    # Keep this backend's credential (and the embedding key, when set); drop
    # every other credential and the fake endpoints from other tiers.
    kept = tuple(dict.fromkeys((backend.credential, EMBEDDING_CREDENTIAL)))
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
        help="Do not start Docker containers; vector and trace checks SKIP",
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

    if args.no_infra:
        statuses = [
            InfraStatus(s, False, "--no-infra") for s in (QDRANT, CHROMA, ASPIRE)
        ]
    else:
        print(
            "verify-live: starting Qdrant, Chroma and the Aspire dashboard "
            "(Docker)..."
        )
        statuses = [ensure(QDRANT), ensure(CHROMA), ensure(ASPIRE)]
    qdrant, chroma, aspire = statuses
    checks.extend(infra_checks(statuses))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rundir = ROOT / ".verify" / "live" / stamp
    rundir.mkdir(parents=True, exist_ok=True)
    traced: list[str] = []
    for backend in runnable:
        print(f"verify-live: running {backend.name} ({backend.fixture})...")
        backend_checks, service = live_backend(
            backend, rundir, args.limit, [qdrant, chroma], aspire
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
