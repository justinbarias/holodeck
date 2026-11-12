"""The memory contract: types, errors and the port (spec 043, T2).

Remediation for the boundary guard: a Graphiti or Neo4j import belongs in
``holodeck/lib/memory/backends/graphiti*.py`` only. Everything else talks to
a backend through ``MemoryBackend`` and the types in ``holodeck.lib.memory``.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from holodeck.lib.memory import (
    EpisodeIn,
    MemoryBackend,
    MemoryCapabilities,
    MemoryNotSupportedError,
    MemoryOperation,
    MemoryQuery,
    MemoryValueError,
    TemporalMemoryError,
    validate_namespace,
)
from holodeck.lib.memory.backends.fake import FakeBackend

MEMORY_SRC = Path(__file__).resolve().parents[4] / "src" / "holodeck" / "lib" / "memory"
BACKEND_CLIENTS = ("graphiti_core", "neo4j", "falkordb", "kuzu")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _client_imports(path: Path) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        found.extend(
            f"line {node.lineno}: {module}"
            for module in modules
            if any(module == c or module.startswith(f"{c}.") for c in BACKEND_CLIENTS)
        )
    return found


def test_mb_01_fake_backend_satisfies_the_port() -> None:
    assert isinstance(FakeBackend(), MemoryBackend)


@pytest.mark.parametrize(
    "path",
    [p for p in sorted(MEMORY_SRC.rglob("*.py")) if not p.name.startswith("graphiti")],
    ids=lambda p: p.name,
)
def test_mb_01_only_the_graphiti_module_imports_backend_clients(path: Path) -> None:
    violations = _client_imports(path)
    assert not violations, (
        f"{path.name} imports a backend client: {violations}. "
        "See this module's docstring."
    )


def test_mb_03_require_raises_a_named_error_for_an_unsupported_operation() -> None:
    capabilities = MemoryCapabilities(
        backend="graphiti", supported=frozenset({MemoryOperation.SEARCH})
    )
    capabilities.require(MemoryOperation.SEARCH)
    assert capabilities.supports(MemoryOperation.REVIEW_QUEUE) is False

    with pytest.raises(MemoryNotSupportedError) as raised:
        capabilities.require(MemoryOperation.REVIEW_QUEUE, "It is a Phase 2 item.")

    assert raised.value.backend == "graphiti"
    assert raised.value.operation == "pending-merge and low-confidence review"
    assert str(raised.value) == (
        "Memory backend 'graphiti' does not support pending-merge and "
        "low-confidence review. It is a Phase 2 item."
    )


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=10))),
    ],
    ids=["naive", "offset"],
)
def test_datetimes_that_are_not_utc_are_rejected(value: datetime) -> None:
    with pytest.raises(MemoryValueError, match="valid_at must be a timezone-aware UTC"):
        MemoryQuery(text="q", valid_at=value, known_at=NOW)
    with pytest.raises(MemoryValueError, match="reference_time must be"):
        EpisodeIn(
            episode_id=uuid5(NAMESPACE_URL, "e"),
            source_tool="docs",
            chunk_id="c1",
            source_uri="a.md",
            text="t",
            reference_time=value,
            reference_time_source="mtime",
        )


def test_ing_09_an_episode_cannot_be_its_own_context() -> None:
    episode_id = uuid5(NAMESPACE_URL, "e")
    with pytest.raises(MemoryValueError, match="cannot be its own context"):
        EpisodeIn(
            episode_id=episode_id,
            source_tool="docs",
            chunk_id="c1",
            source_uri="a.md",
            text="t",
            reference_time=NOW,
            reference_time_source="mtime",
            context_episode_ids=(episode_id,),
        )


@pytest.mark.parametrize("hops", [0, 3])
def test_ret_04_hops_outside_one_or_two_are_rejected(hops: int) -> None:
    with pytest.raises(MemoryValueError, match="hops must be 1 or 2"):
        MemoryQuery(text="q", valid_at=NOW, known_at=NOW, hops=hops)


def test_limit_below_one_is_rejected() -> None:
    with pytest.raises(MemoryValueError, match="limit must be at least 1"):
        MemoryQuery(text="q", valid_at=NOW, known_at=NOW, limit=0)


def test_mb_02_namespace_accepts_only_portable_characters() -> None:
    assert validate_namespace("family-assistance_1") == "family-assistance_1"
    for bad in ("", "family assistance", "a/b", "a.b"):
        with pytest.raises(MemoryValueError, match="is not valid"):
            validate_namespace(bad)


def test_value_errors_are_memory_errors_and_value_errors() -> None:
    assert issubclass(MemoryValueError, TemporalMemoryError)
    assert issubclass(MemoryValueError, ValueError)
