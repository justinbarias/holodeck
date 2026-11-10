"""Guard: tools reach vector stores only through the adapter contract (042 T4).

Remediation: when this fails, route the store call through
``holodeck.lib.retrieval`` (``create_adapter`` and ``VectorStoreAdapter``)
instead of importing a store client in a tool. Provider logic and store
clients belong in an adapter under ``holodeck/lib/retrieval/adapters/``.
Semantic Kernel itself is guarded repository-wide by
``tests/unit/test_no_semantic_kernel.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[4] / "src" / "holodeck"
GUARDED = sorted((SRC / "tools").rglob("*.py")) + [SRC / "lib" / "keyword_search.py"]
STORE_CLIENTS = {
    "qdrant_client",
    "chromadb",
    "psycopg",
    "psycopg_pool",
    "pgvector",
    "pinecone",
    "weaviate",
    "azure.search",
    "azure.cosmos",
}


def _is_store_client(module: str) -> bool:
    return any(
        module == name or module.startswith(f"{name}.") for name in STORE_CLIENTS
    )


def _violations(path: Path) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_store_client(alias.name):
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and _is_store_client(node.module)
        ):
            found.append(f"line {node.lineno}: from {node.module} import ...")
    return found


@pytest.mark.parametrize("path", GUARDED, ids=lambda p: p.name)
def test_tools_do_not_import_store_clients(path: Path) -> None:
    violations = _violations(path)
    assert not violations, (
        f"{path.relative_to(SRC.parent)} bypasses the retrieval adapter contract: "
        f"{violations}. Use holodeck.lib.retrieval instead (see module docstring)."
    )


def test_collection_filter_module_is_gone() -> None:
    assert not (SRC / "lib" / "collection_filter.py").exists(), (
        "holodeck/lib/collection_filter.py was replaced by adapter.find/"
        "list_page/delete_where/replace_source; do not reintroduce it."
    )
