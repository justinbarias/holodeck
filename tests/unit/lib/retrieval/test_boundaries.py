"""Guard: tools reach vector stores only through the adapter contract (042 T4).

Remediation: when this fails, route the store call through
``holodeck.lib.retrieval`` (``create_adapter`` and ``VectorStoreAdapter``)
instead of importing Semantic Kernel or the SK record factories. Provider
logic belongs in an adapter under ``holodeck/lib/retrieval/adapters/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[4] / "src" / "holodeck"
GUARDED = sorted((SRC / "tools").rglob("*.py")) + [SRC / "lib" / "keyword_search.py"]
RECORD_FACTORIES = {
    "create_document_record_class",
    "create_structured_record_class",
    "create_hierarchical_document_record_class",
    "get_collection_factory",
    "get_collection_class",
    "convert_document_to_query_result",
    "DocumentRecord",
    "HierarchicalDocumentRecord",
}


def _violations(path: Path) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "semantic_kernel":
                    found.append(f"line {node.lineno}: import {alias.name}")
                if alias.name == "holodeck.lib.vector_store":
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "semantic_kernel":
                found.append(f"line {node.lineno}: from {node.module} import ...")
            if node.module == "holodeck.lib.vector_store":
                names = {alias.name for alias in node.names}
                for name in sorted(names & RECORD_FACTORIES | names & {"*"}):
                    found.append(f"line {node.lineno}: {name} from vector_store")
    return found


@pytest.mark.parametrize("path", GUARDED, ids=lambda p: p.name)
def test_tools_do_not_import_semantic_kernel_or_record_factories(path: Path) -> None:
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
