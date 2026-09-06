"""MkDocs build hooks.

Currently:
- ``on_pre_build`` copies ``schemas/agent.schema.json`` into ``docs/schemas/``
  so the JSON Schema is published as a raw static asset at
  ``https://docs.useholodeck.ai/schemas/schema.json`` (canonical) and
  ``/schemas/agent.schema.json`` (legacy alias). This lets any editor that
  speaks ``yaml-language-server`` resolve the schema directly from the docs
  site for auto-complete and validation.
- ``on_page_markdown`` publishes links to repository files outside ``docs/``
  as GitHub source links, keeping repository-local Markdown navigable.

The copy targets are git-ignored — the upstream source of truth remains
``schemas/agent.schema.json`` at the repo root.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

SCHEMA_SOURCE = Path("schemas/agent.schema.json")
PUBLISHED_NAMES = ("schema.json", "agent.schema.json")


def on_pre_build(config: MkDocsConfig) -> None:
    """Copy the agent JSON Schema into the docs tree before the build."""
    repo_root = Path(config["config_file_path"]).parent
    src = repo_root / SCHEMA_SOURCE
    if not src.exists():
        return
    target_dir = Path(config["docs_dir"]) / "schemas"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED_NAMES:
        shutil.copy2(src, target_dir / name)


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files
) -> str:
    """Publish repository-relative links outside docs as GitHub source links.

    Keep local Markdown usable by agents. Resolve only existing repository paths
    outside the docsite, leaving missing targets visible to MkDocs validation.
    The repository's edit_uri supplies the branch and docs path.
    """
    if not page.file.abs_src_path or not config.repo_url or not config.edit_uri:
        return markdown
    repo_root = Path(config.config_file_path).parent.resolve()
    docs_root = Path(config.docs_dir).resolve()
    source = Path(page.file.abs_src_path)
    edit_base = f"{config.repo_url.rstrip('/')}/{config.edit_uri.strip('/')}"
    # edit_uri is edit/<ref>/<docs path>; preserve refs containing slashes.
    docs_suffix = "/" + docs_root.relative_to(repo_root).as_posix()
    if not edit_base.endswith(docs_suffix):
        return markdown
    source_base = edit_base.removesuffix(docs_suffix).replace("/edit/", "/blob/", 1)

    def replace_link(match: re.Match[str]) -> str:
        destination = urlsplit(match[2])
        if destination.scheme or destination.netloc or not destination.path:
            return match[0]
        target = (source.parent / unquote(destination.path)).resolve()
        if (
            not target.exists()
            or not target.is_relative_to(repo_root)
            or target.is_relative_to(docs_root)
        ):
            return match[0]
        base = (
            source_base.replace("/blob/", "/tree/", 1)
            if target.is_dir()
            else source_base
        )
        url = urlsplit(f"{base}/{quote(target.relative_to(repo_root).as_posix())}")
        resolved = urlunsplit(
            (url.scheme, url.netloc, url.path, destination.query, destination.fragment)
        )
        return f"{match[1]}{resolved}{match[3]}"

    # Fenced examples must remain literal. Inline links use the same simple syntax
    # as the harness checker; heading anchors and reference-style links stay intact.
    rendered = []
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            rendered.append(line)
        elif fence is not None:
            rendered.append(line)
        else:
            # Imported task labels are prose, not API cross-references.
            # Escape adjacent labels for autorefs without editing historical tasks.
            if source.is_relative_to(docs_root / "exec-plans"):
                parts = re.split(r"(`+[^`]*`+)", line)
                for index in range(0, len(parts), 2):
                    parts[index] = re.sub(
                        r"(?<!\\)\[(P|US\d+(?:/\d+)*)\](?![\[(])",
                        r"\\[\1\\]",
                        parts[index],
                    )
                line = "".join(parts)
            rendered.append(
                re.sub(r"(\[[^\]\n]+\]\()([^\s)]+)(\))", replace_link, line)
            )
    return "".join(rendered)
