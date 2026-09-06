"""Regression coverage for the repository harness checks."""

from pathlib import Path

import pytest

from scripts.check_harness import (
    DIRECTORIES,
    GENERATED,
    REQUIRED,
    check,
    render_inventory,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def harness(tmp_path: Path) -> Path:
    """Build a small valid repository with real relative links and schema data."""
    for directory in DIRECTORIES:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Document\n", encoding="utf-8")
    for name in (
        "src/holodeck/lib/eval_run/writer.py",
        "src/holodeck/deploy/state.py",
        "docs/guides/vector-stores.md",
        "docs/guides/temporal.md",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas/agent.schema.json").write_text(
        '{"title": "Agent", "properties": {"name": {"type": "string"}}}',
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text(
        "# Map\n"
        + "\n".join(
            f"- [{name}]({name})"
            for name in REQUIRED
            if name not in {"AGENTS.md", "CLAUDE.md"}
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
    (tmp_path / GENERATED).write_text(render_inventory(tmp_path), encoding="utf-8")
    return tmp_path


def test_valid_harness(harness: Path) -> None:
    """A complete, linked harness passes without loading HoloDeck dependencies."""
    assert check(harness) == []


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("CLAUDE.md", "@AGENTS.md\nExtra rules\n", "must contain only"),
        ("AGENTS.md", "\n" * 101, "exceeds 100 lines"),
        ("docs/DESIGN.md", "[Missing](missing.md)\n", "broken local link"),
        ("docs/DESIGN.md", "[Outside](../../outside.md)\n", "broken local link"),
        (GENERATED, "outdated\n", "stale inventory"),
    ],
)
def test_invalid_harness(harness: Path, name: str, content: str, message: str) -> None:
    """Policy violations fail with a remediation message."""
    (harness / name).write_text(content, encoding="utf-8")
    assert any(message in error for error in check(harness))


def test_schema_content_drift(harness: Path) -> None:
    """A nested schema change requires regeneration even with the same field count."""
    (harness / "schemas/agent.schema.json").write_text(
        '{"title": "Agent", "properties": {"name": {"type": "integer"}}}',
        encoding="utf-8",
    )
    assert any("stale inventory" in error for error in check(harness))


def test_missing_layout(harness: Path) -> None:
    """Missing required documents and empty plan directories fail."""
    (harness / "docs/DESIGN.md").unlink()
    (harness / "docs/exec-plans/active").rmdir()
    errors = check(harness)
    assert any("missing harness document" in error for error in errors)
    assert any("missing directory" in error for error in errors)


def test_orphaned_plan(harness: Path) -> None:
    """New plans must be reachable through the knowledge map."""
    plan = harness / "docs/exec-plans/active/new-plan.md"
    plan.write_text("# New plan\n", encoding="utf-8")
    assert any("new-plan.md: not reachable" in error for error in check(harness))


def test_examples_urls_and_nested_links(harness: Path) -> None:
    """Examples and external URLs are ignored, while nested relative links resolve."""
    (harness / "docs/DESIGN.md").write_text(
        "[External](https://example.invalid/missing)\n"
        "[Heading](#local-heading)\n"
        "```markdown\n[Example](not-a-file.md)\n```\n"
        "[Plan](exec-plans/active/plan%20one.md#objective)\n",
        encoding="utf-8",
    )
    (harness / "docs/exec-plans/active/plan one.md").write_text(
        "# Objective\n[Architecture](../../../ARCHITECTURE.md)\n", encoding="utf-8"
    )
    assert check(harness) == []
