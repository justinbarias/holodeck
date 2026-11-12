"""Fetch and convert the Acts for the temporal-memory sample.

Downloads each compilation in ``manifest.json`` once from the Federal
Register of Legislation, checks its SHA-256, and converts it to one markdown
file under ``docs/<tool>/``. The first download of an entry records its hash,
size and date in the manifest; later runs verify the cached file against it.

    uv run python fetch_corpus.py           # download if needed, convert, report
    uv run python fetch_corpus.py --check   # verify cached files only

The converter reads the Word paragraph styles of the Office of Parliamentary
Counsel templates (``ActHead 2`` for a Part, ``ActHead 5`` for a section, and
so on) from ``word/document.xml``. It uses the standard library only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import UUID
from xml.etree import ElementTree as ET  # nosec B405 - files are hash-pinned

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
CACHE = HERE / ".cache"
DOCS = HERE / "docs"
AGENT_YAML = HERE / "agent.yaml"

USER_AGENT = "holodeck-sample-corpus-fetch/1.0 (temporal-memory sample corpus fetch)"
PAUSE_SECONDS = 11.0  # robots.txt on www.legislation.gov.au asks for 10
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"

# Office of Parliamentary Counsel heading styles and their markdown level in
# the body of an Act: Part "#", Division "##", Subdivision "###", section
# "####". ActHead 1 is a Chapter or a Schedule (these Acts put their
# Schedules in ActHead 1); it is "#", and every heading after the first one
# moves one level down, so a Schedule's Parts nest under the Schedule and
# its clauses are "#####". ActHead 6 to 9 are the older Schedule styles.
HEADING_LEVELS = {
    "ActHead 2": 1,
    "ActHead 3": 2,
    "ActHead 4": 3,
    "ActHead 5": 4,
    "ActHead 7": 1,
    "ActHead 8": 2,
    "ActHead 9": 4,
}
CONTAINER_STYLES = frozenset({"ActHead 1", "ActHead 6"})
SECTION_STYLES = frozenset({"ActHead 5", "ActHead 9"})
ENDNOTES_STYLE_PREFIX = "ENotesHeading"
# Older compilations print formulas as pictures with no text description.
NO_DESCRIPTION = "picture with no text description in the source"


def is_heading(style: str) -> bool:
    """True for an OPC heading style."""
    return style in HEADING_LEVELS or style in CONTAINER_STYLES


class FetchError(Exception):
    """A download, hash or conversion problem, with the compilation named."""


@dataclass
class Compilation:
    """One manifest entry."""

    tool: str
    title: str
    series_id: str
    register_id: str
    compilation_number: int
    start_date: str
    selected_as: str
    volume: int | None
    url: str
    retrieved: str | None
    size_bytes: int | None
    sha256: str | None

    @property
    def label(self) -> str:
        """Short name for messages."""
        volume = f" volume {self.volume}" if self.volume else ""
        return (
            f"{self.series_id} compilation {self.compilation_number}{volume} "
            f"({self.start_date})"
        )

    @property
    def stem(self) -> str:
        """File name stem shared by the cached file and the markdown."""
        volume = f"-vol{self.volume}" if self.volume else ""
        return f"{self.series_id}-c{self.compilation_number:03d}{volume}"

    @property
    def cache_path(self) -> Path:
        """The cached Word file."""
        return CACHE / f"{self.stem}.docx"

    @property
    def markdown_path(self) -> Path:
        """The converted markdown file."""
        return DOCS / self.tool / f"{self.start_date}-{self.stem}.md"


def load_manifest(path: Path = MANIFEST) -> tuple[dict[str, object], list[Compilation]]:
    """Return the raw manifest and its compilations."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, [Compilation(**entry) for entry in raw["compilations"]]


def save_manifest(
    raw: dict[str, object], compilations: Sequence[Compilation], path: Path = MANIFEST
) -> None:
    """Write the manifest back with the recorded hashes."""
    raw["compilations"] = [vars(c) for c in compilations]
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", "utf-8")


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(compilation: Compilation) -> None:
    """Check the cached file against the manifest hash.

    Raises:
        FetchError: The file is missing, has no recorded hash, or differs.
    """
    path = compilation.cache_path
    if not path.is_file():
        raise FetchError(f"{compilation.label}: not downloaded ({path.name})")
    if not compilation.sha256:
        raise FetchError(f"{compilation.label}: no sha256 recorded in the manifest")
    actual = sha256_of(path)
    if actual != compilation.sha256:
        raise FetchError(
            f"{compilation.label}: SHA-256 mismatch for {path.name}: manifest "
            f"{compilation.sha256}, file {actual}. Delete the file to download "
            "it again, or update the manifest if the Register replaced it."
        )


def download(compilations: Sequence[Compilation]) -> bool:
    """Download every compilation that is not cached; return True if any was.

    One request at a time, with a pause between requests, as the Register's
    robots.txt asks. A file already cached is verified and not fetched again.
    """
    import httpx

    CACHE.mkdir(parents=True, exist_ok=True)
    fetched = False
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=120.0
    ) as client:
        for compilation in compilations:
            if compilation.cache_path.is_file():
                if compilation.sha256 is None:
                    record(compilation)
                    fetched = True
                verify(compilation)
                continue
            if fetched:
                time.sleep(PAUSE_SECONDS)
            print(f"downloading {compilation.label} from {compilation.url}")
            response = client.get(compilation.url)
            if response.status_code != 200:
                raise FetchError(
                    f"{compilation.label}: HTTP {response.status_code} from "
                    f"{compilation.url}"
                )
            if not response.content.startswith(b"PK"):
                raise FetchError(f"{compilation.label}: response is not a Word file")
            partial = compilation.cache_path.with_suffix(".part")
            partial.write_bytes(response.content)
            if compilation.sha256 is not None:
                actual = sha256_of(partial)
                if actual != compilation.sha256:
                    partial.unlink()
                    raise FetchError(
                        f"{compilation.label}: SHA-256 mismatch: manifest "
                        f"{compilation.sha256}, download {actual}"
                    )
            partial.replace(compilation.cache_path)
            if compilation.sha256 is None:
                record(compilation)
            fetched = True
    return fetched


def record(compilation: Compilation) -> None:
    """Record hash, size and retrieval date of a first download."""
    path = compilation.cache_path
    compilation.sha256 = sha256_of(path)
    compilation.size_bytes = path.stat().st_size
    compilation.retrieved = date.today().isoformat()


# --- conversion -------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """A paragraph (with its style name) or a table, in document order."""

    style: str
    text: str
    rows: tuple[tuple[str, ...], ...] = ()

    @property
    def is_table(self) -> bool:
        """True for a table."""
        return bool(self.rows)


def _parse_xml(data: bytes) -> ET.Element:
    """Parse a part of a Word file.

    The files come from the Register and are checked against the SHA-256 in
    the manifest before they are parsed; the standard parser does not expand
    external entities.
    """
    return ET.fromstring(data)  # noqa: S314  # nosec B314


def _style_names(archive: zipfile.ZipFile) -> dict[str, str]:
    """Map style ids to style names from ``word/styles.xml``."""
    try:
        root = _parse_xml(archive.read("word/styles.xml"))
    except KeyError:
        return {}
    names: dict[str, str] = {}
    for style in root.iter(f"{W}style"):
        style_id = style.get(f"{W}styleId")
        name = style.find(f"{W}name")
        if style_id and name is not None:
            names[style_id] = name.get(f"{W}val", style_id)
    return names


def _paragraph_text(paragraph: ET.Element) -> str:
    """Return the visible text of a paragraph.

    Field instructions (``w:instrText``) and deleted text are not visible and
    are left out. Tabs and line breaks become spaces. The Acts print their
    formulas as pictures with a text description; the description is kept
    as ``[Formula: ...]``.
    """
    parts: list[str] = []
    for node in paragraph.iter():
        tag = node.tag
        if tag == f"{W}t":
            parts.append(node.text or "")
        elif tag in (f"{W}tab", f"{W}br", f"{W}cr"):
            parts.append(" ")
        elif tag == f"{W}noBreakHyphen":
            parts.append("-")
        elif tag == f"{WP}docPr":
            description = normalise(node.get("descr", ""))
            parts.append(f" [Formula: {description or NO_DESCRIPTION}] ")
    return normalise("".join(parts))


def normalise(text: str) -> str:
    """Collapse white space (including no-break spaces) to single spaces."""
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def _table_rows(table: ET.Element) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for row in table.findall(f"{W}tr"):
        cells: list[str] = []
        for cell in row.findall(f"{W}tc"):
            texts = [_paragraph_text(p) for p in cell.iter(f"{W}p")]
            cells.append(" ".join(t for t in texts if t))
        if any(cells):
            rows.append(tuple(cells))
    return tuple(rows)


def read_blocks(docx: Path) -> list[Block]:
    """Return the body paragraphs and tables of a Word file, in order."""
    with zipfile.ZipFile(docx) as archive:
        styles = _style_names(archive)
        root = _parse_xml(archive.read("word/document.xml"))
    body = root.find(f"{W}body")
    if body is None:
        raise FetchError(f"{docx.name}: no document body")
    blocks: list[Block] = []
    for child in _body_children(body):
        if child.tag == f"{W}p":
            style_el = child.find(f"{W}pPr/{W}pStyle")
            style_id = style_el.get(f"{W}val", "") if style_el is not None else ""
            blocks.append(Block(styles.get(style_id, style_id), _paragraph_text(child)))
        elif child.tag == f"{W}tbl":
            blocks.append(Block("table", "", _table_rows(child)))
    return blocks


def _body_children(element: ET.Element) -> Iterator[ET.Element]:
    """Yield paragraphs and tables, looking inside content controls."""
    for child in element:
        if child.tag in (f"{W}p", f"{W}tbl"):
            yield child
        elif child.tag == f"{W}sdt":
            content = child.find(f"{W}sdtContent")
            if content is not None:
                yield from _body_children(content)


def _escape(text: str) -> str:
    """Keep a body line from being read as a heading or front matter."""
    if text.startswith(("#", "---")):
        return "\\" + text
    return text


def _table_markdown(rows: tuple[tuple[str, ...], ...]) -> str:
    width = max(len(row) for row in rows)
    lines = []
    for index, row in enumerate(rows):
        cells = [cell.replace("|", "\\|") for cell in row]
        cells += [""] * (width - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
        if index == 0:
            lines.append("|" + " --- |" * width)
    return "\n".join(lines)


@dataclass(frozen=True)
class Converted:
    """A converted compilation."""

    markdown: str
    sections: int
    words: int


def body_blocks(blocks: Sequence[Block]) -> list[Block]:
    """Return the Act body: from the first heading to the endnotes.

    The cover, "About this compilation" and the table of contents come before
    the first ``ActHead`` paragraph (contents lines use ``TOC`` styles); the
    endnotes start at the first ``ENotesHeading`` paragraph.
    """
    start = next(
        (i for i, b in enumerate(blocks) if is_heading(b.style) and b.text),
        None,
    )
    if start is None:
        raise FetchError("no ActHead heading found: not an OPC compilation?")
    end = next(
        (
            i
            for i, b in enumerate(blocks)
            if i > start and b.style.startswith(ENDNOTES_STYLE_PREFIX)
        ),
        len(blocks),
    )
    return list(blocks[start:end])


def front_matter(compilation: Compilation) -> str:
    """Return the YAML front matter for a compilation."""
    retrieved = compilation.retrieved or "unknown date"
    attribution = (
        f"Based on content from the Federal Register of Legislation at "
        f"{_long_date(retrieved)}. For the latest information on Australian "
        "Government legislation please go to https://www.legislation.gov.au. "
        f"Licensed under CC BY 4.0 ({LICENCE_URL}). Changed: converted from "
        "Word to markdown; the cover, contents, front pages and endnotes are "
        "removed and headings are reformatted."
    )
    volume = f"volume {compilation.volume}" if compilation.volume else "all"
    fields = [
        ("title", compilation.title),
        ("series_id", compilation.series_id),
        ("register_id", compilation.register_id),
        ("compilation", str(compilation.compilation_number)),
        ("volume", volume),
        ("source_url", compilation.url),
        ("licence", LICENCE_URL),
        ("attribution", attribution),
    ]
    lines = ["---", f"effective_date: {compilation.start_date}"]
    lines += [
        f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in fields
    ]
    lines.append("---")
    return "\n".join(lines)


def _long_date(iso: str) -> str:
    try:
        day = date.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{day.day} {day.strftime('%B %Y')}"


def convert_blocks(blocks: Sequence[Block], header: str) -> Converted:
    """Convert body blocks to markdown under ``header`` (front matter)."""
    out: list[str] = [header]
    sections = 0
    words = 0
    shift = 0
    for block in body_blocks(blocks):
        if block.is_table:
            out.append(_table_markdown(block.rows))
            words += sum(len(cell.split()) for row in block.rows for cell in row)
            continue
        if not block.text:
            continue
        words += len(block.text.split())
        if block.style in CONTAINER_STYLES:
            shift = 1
            out.append(f"# {block.text}")
        elif block.style in HEADING_LEVELS:
            if block.style in SECTION_STYLES:
                sections += 1
            out.append(f"{'#' * (HEADING_LEVELS[block.style] + shift)} {block.text}")
        else:
            out.append(_escape(block.text))
    return Converted("\n\n".join(out) + "\n", sections, words)


def convert(compilation: Compilation) -> Converted:
    """Convert a cached compilation to markdown."""
    try:
        blocks = read_blocks(compilation.cache_path)
        return convert_blocks(blocks, front_matter(compilation))
    except FetchError as exc:
        raise FetchError(f"{compilation.label}: {exc}") from exc


# --- report -----------------------------------------------------------------


@dataclass(frozen=True)
class ChunkReport:
    """Chunks per markdown file and the episode ids they would have."""

    chunks: dict[Path, list[tuple[str, str]]]
    namespace: str


def preview_chunks(tools: Sequence[str]) -> ChunkReport:
    """Chunk the converted files with the sample's real tool settings.

    Builds each ``hierarchical_document`` tool from agent.yaml (no
    environment substitution is needed: nothing connects) and lists chunks
    the way ``holodeck memory ingest --dry-run`` does. No embedding, no store.
    """
    import yaml

    from holodeck.models.tool import (
        HierarchicalDocumentToolConfig,
        TemporalMemoryToolConfig,
    )
    from holodeck.tools.hierarchical_chunk_source import HierarchicalChunkSource
    from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool

    agent = yaml.safe_load(AGENT_YAML.read_text(encoding="utf-8"))
    by_name = {tool["name"]: tool for tool in agent["tools"]}
    memory = next(t for t in agent["tools"] if t.get("type") == "temporal_memory")
    namespace = TemporalMemoryToolConfig.model_validate(memory).resolved_namespace

    async def collect() -> dict[Path, list[tuple[str, str]]]:
        found: dict[Path, list[tuple[str, str]]] = {}
        for name in tools:
            config = HierarchicalDocumentToolConfig.model_validate(by_name[name])
            source = HierarchicalChunkSource(
                HierarchicalDocumentTool(config, base_dir=str(HERE)), preview=True
            )
            for document in await source.documents():
                refs = await source.chunks(document)
                found[document.path.resolve()] = [
                    (ref.section_id, ref.text) for ref in refs
                ]
        return found

    return ChunkReport(asyncio.run(collect()), namespace)


def report(compilations: Sequence[Compilation], stats: dict[str, Converted]) -> None:
    """Print sections, words and chunks per compilation and the overlap."""
    from holodeck.lib.memory.service import episode_id_for

    tool_names = {
        "family-assistance-act": "family_assistance_act",
        "family-assistance-admin-act": "family_assistance_admin_act",
    }
    chunks = preview_chunks(sorted({tool_names[c.tool] for c in compilations}))
    episodes: dict[str, set[UUID]] = {}
    total: set[UUID] = set()
    print()
    print(f"{'compilation':<52} {'sections':>8} {'words':>8} {'chunks':>7}")
    for compilation in compilations:
        refs = chunks.chunks.get(compilation.markdown_path.resolve(), [])
        ids = {episode_id_for(chunks.namespace, s, t) for s, t in refs}
        episodes[compilation.stem] = ids
        total |= ids
        converted = stats[compilation.stem]
        print(
            f"{compilation.label:<52} {converted.sections:>8} "
            f"{converted.words:>8} {len(refs):>7}"
        )
    print()
    by_tool: dict[str, list[Compilation]] = {}
    for compilation in compilations:
        by_tool.setdefault(compilation.tool, []).append(compilation)
    for tool, group in by_tool.items():
        group = sorted(group, key=lambda c: c.start_date)
        for older, newer in zip(group, group[1:], strict=False):
            same = episodes[older.stem] & episodes[newer.stem]
            print(
                f"{tool}: {len(same)} of {len(episodes[newer.stem])} chunks of "
                f"{newer.start_date} are identical to {older.start_date} "
                "(skipped by a second ingest)"
            )
    print(f"distinct episodes for a first ingest of all compilations: {len(total)}")


# --- main -------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fetch, check or conversion."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the cached files against the manifest and write nothing",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="skip the chunk report (it imports HoloDeck)",
    )
    args = parser.parse_args(argv)
    raw, compilations = load_manifest()
    try:
        if args.check:
            for compilation in compilations:
                verify(compilation)
                print(f"ok {compilation.label} {compilation.sha256}")
            return 0
        if download(compilations):
            save_manifest(raw, compilations)
        stats: dict[str, Converted] = {}
        for compilation in compilations:
            verify(compilation)
            converted = convert(compilation)
            compilation.markdown_path.parent.mkdir(parents=True, exist_ok=True)
            compilation.markdown_path.write_text(converted.markdown, "utf-8")
            stats[compilation.stem] = converted
            print(f"wrote {compilation.markdown_path.relative_to(HERE)}")
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.no_report:
        report(compilations, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
