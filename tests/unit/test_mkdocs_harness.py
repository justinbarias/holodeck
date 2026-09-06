"""Verify that repository knowledge links also work in published documentation."""

from pathlib import Path

import pytest
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page

from scripts.mkdocs_hooks import on_page_markdown, on_pre_build

pytestmark = pytest.mark.unit


@pytest.fixture
def site(tmp_path: Path) -> tuple[MkDocsConfig, Page, Files]:
    """Create a minimal MkDocs context with a nested source document."""
    docs = tmp_path / "docs"
    (docs / "design-docs").mkdir(parents=True)
    (docs / "DESIGN.md").write_text("# Design\n", encoding="utf-8")
    (tmp_path / "ARCHITECTURE.md").write_text("# Architecture\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas/agent.schema.json").write_text("{}", encoding="utf-8")
    config = MkDocsConfig(config_file_path=str(tmp_path / "mkdocs.yml"))
    config.docs_dir = str(docs)
    config.repo_url = "https://github.com/example/holodeck"
    config.edit_uri = "edit/main/docs/"
    file = File("design-docs/index.md", str(docs), str(tmp_path / "site"), True)
    return config, Page("Index", file, config), Files([file])


def test_published_repository_links(site: tuple[MkDocsConfig, Page, Files]) -> None:
    """Files, directories, and fragments become source links on the configured ref."""
    config, page, files = site
    config.edit_uri = "edit/release/docs-update/docs/"
    markdown = "[Map](../../ARCHITECTURE.md#flow)\n[Code](../../src/)\n"
    result = on_page_markdown(markdown, page, config, files)
    assert result == (
        "[Map](https://github.com/example/holodeck/blob/"
        "release/docs-update/ARCHITECTURE.md#flow)\n"
        "[Code](https://github.com/example/holodeck/tree/release/docs-update/src)\n"
    )


def test_preserve_doc_links_and_examples(
    site: tuple[MkDocsConfig, Page, Files],
) -> None:
    """Docsite links, missing targets, remote links, and examples remain unchanged."""
    config, page, files = site
    markdown = (
        "[Design](../DESIGN.md)\n[Missing](../../missing.md)\n"
        "[External](https://example.com/)\n[Heading](#here)\n"
        "```markdown\n[Example](../../ARCHITECTURE.md)\n```\n"
        "~~~text\n[Example](../../ARCHITECTURE.md)\n~~~\n"
    )
    assert on_page_markdown(markdown, page, config, files) == markdown


def test_schema_publication_preserved(site: tuple[MkDocsConfig, Page, Files]) -> None:
    """The existing schema publication hook still writes both public aliases."""
    config, _, _ = site
    on_pre_build(config)
    for name in ("schema.json", "agent.schema.json"):
        assert (Path(config.docs_dir) / "schemas" / name).read_text() == "{}"


def test_task_labels_render_without_api_references(
    site: tuple[MkDocsConfig, Page, Files],
) -> None:
    """Escape task labels while preserving checkboxes, links, and literal examples."""
    config, _, _ = site
    file = File("exec-plans/active/tasks.md", config.docs_dir, "site", True)
    page = Page("Tasks", file, config)
    markdown = (
        "- [x] T001 [P] [US1] Build feature.\n"
        "- [ ] T002 [US3/4] Check feature.\n"
        "Labels: `[P] [US1]` and ``[P] [US2]``.\n"
        "[US1](#story) and [P][parallel]\n"
        "```text\n[P] [US1]\n```\n"
    )
    result = on_page_markdown(markdown, page, config, Files([file]))
    assert result == (
        "- [x] T001 \\[P\\] \\[US1\\] Build feature.\n"
        "- [ ] T002 \\[US3/4\\] Check feature.\n"
        "Labels: `[P] [US1]` and ``[P] [US2]``.\n"
        "[US1](#story) and [P][parallel]\n"
        "```text\n[P] [US1]\n```\n"
    )
