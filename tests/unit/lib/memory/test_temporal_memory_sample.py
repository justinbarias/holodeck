"""The temporal-memory sample: its Word converter and its agent.yaml.

The converter runs on a small Word file built here, in the Office of
Parliamentary Counsel style, so no Act text is in the repository and no test
needs the network. The agent loads through the real loader with dummy
environment values.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.memory.effective_date import strip_front_matter
from holodeck.models.agent import Agent
from holodeck.models.evaluation import GEvalMetric
from holodeck.models.tool import (
    HierarchicalDocumentToolConfig,
    TemporalMemoryToolConfig,
)

if TYPE_CHECKING:
    # The sample module is loaded from its file path at run time; the type
    # checker sees the real classes through this import.
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[4] / "sample" / "temporal-memory")
    )
    from fetch_corpus import Compilation, Converted

pytestmark = pytest.mark.unit

SAMPLE = Path(__file__).resolve().parents[4] / "sample" / "temporal-memory"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"

# Style ids differ from style names on purpose: the converter must map them
# through word/styles.xml.
STYLES = {
    "ShortT": "ShortT",
    "TOC2": "toc 2",
    "TOC5": "toc 5",
    "LongT": "LongT",
    "ActHead1": "ActHead 1",
    "ActHead2": "ActHead 2",
    "ActHead3": "ActHead 3",
    "ActHead5": "ActHead 5",
    "subsection": "subsection",
    "paragraph": "paragraph",
    "ENotesHeading1": "ENotesHeading 1",
    "ENotesHeading2": "ENotesHeading 2",
}


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "temporal_memory_fetch_corpus", SAMPLE / "fetch_corpus.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fetch = _load_module()


def _p(style: str, *runs: str) -> str:
    """A paragraph; a run of "\t" becomes a tab element."""
    body = "".join(
        "<w:r><w:tab/></w:r>" if run == "\t" else f"<w:r><w:t>{run}</w:t></w:r>"
        for run in runs
    )
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{body}</w:p>'


def _table(*rows: tuple[str, ...]) -> str:
    cells = "".join(
        "<w:tr>"
        + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in row)
        + "</w:tr>"
        for row in rows
    )
    return f"<w:tbl>{cells}</w:tbl>"


def _formula(description: str) -> str:
    return (
        '<w:p><w:pPr><w:pStyle w:val="subsection"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline><wp:docPr id="1" name="Picture 1" descr="{description}"/>'
        "</wp:inline></w:drawing></w:r></w:p>"
    )


BODY = [
    _p("ShortT", "Example Benefits Act 2001"),
    _p("subsection", "About this compilation"),
    _p("subsection", "This compilation shows the text as amended."),
    _p("TOC2", "Part 1—Preliminary", "\t", "1"),
    _p("TOC5", "1", "\t", "Definitions", "\t", "2"),
    _p("LongT", "An Act about benefits"),
    _p("ActHead2", "Part 1—Preliminary"),
    _p("ActHead3", "Division 1—Definitions"),
    _p("ActHead5", "1", "\t", "Definitions"),
    _p("subsection", "In this Act:"),
    _p("subsection", "benefit means a payment under section 2."),
    _p("ActHead5", "2", "\t", "Rate of benefit"),
    _p("subsection", "(1)", "\t", "The rate is set out in the table."),
    _table(("Item", "Rate"), ("1", "$100 | per week")),
    _formula("Start formula Rate times 52 end formula"),
    _p("paragraph", "# not a heading"),
    _p("ActHead1", "Schedule 1—Rates"),
    _p("ActHead2", "Part 1—Standard rate"),
    _p("ActHead5", "1", "\t", "Standard rate"),
    _p("subsection", "The standard rate is $5."),
    _p("ENotesHeading1", "Endnotes"),
    _p("ENotesHeading2", "Endnote 1—About the endnotes"),
    _p("subsection", "Amendment history text."),
]


def _write_docx(path: Path) -> Path:
    document = (
        f'<w:document xmlns:w="{W_NS}" xmlns:wp="{WP_NS}"><w:body>'
        + "".join(BODY)
        + "</w:body></w:document>"
    )
    styles = (
        f'<w:styles xmlns:w="{W_NS}">'
        + "".join(
            f'<w:style w:type="paragraph" w:styleId="{sid}">'
            f'<w:name w:val="{name}"/></w:style>'
            for sid, name in STYLES.items()
        )
        + "</w:styles>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
    return path


def _compilation() -> Compilation:
    """A manifest entry; the sample module is loaded from its file path."""
    return fetch.Compilation(
        tool="example-act",
        title="Example Benefits Act 2001",
        series_id="C2001A00001",
        register_id="C2020C00001",
        compilation_number=7,
        start_date="2020-07-01",
        selected_as="in force on 2020-07-01",
        volume=None,
        url="https://example.invalid/act/word",
        retrieved="2026-11-11",
        size_bytes=1,
        sha256="0" * 64,
    )


def _convert(docx: Path) -> Converted:
    blocks = fetch.read_blocks(docx)
    return fetch.convert_blocks(blocks, fetch.front_matter(_compilation()))


class TestConverter:
    def test_front_matter_has_effective_date_and_attribution(
        self, tmp_path: Path
    ) -> None:
        markdown = _convert(_write_docx(tmp_path / "act.docx")).markdown

        head = markdown.split("\n---\n", 1)[0]
        assert head.startswith("---\neffective_date: 2020-07-01\n")
        assert 'title: "Example Benefits Act 2001"' in head
        assert 'compilation: "7"' in head
        assert 'source_url: "https://example.invalid/act/word"' in head
        assert "Federal Register of Legislation at 11 November 2026" in head
        assert "creativecommons.org/licenses/by/4.0" in head

    def test_headings_follow_word_styles(self, tmp_path: Path) -> None:
        markdown = _convert(_write_docx(tmp_path / "act.docx")).markdown

        headings = [line for line in markdown.splitlines() if line.startswith("#")]
        assert headings == [
            "# Part 1—Preliminary",
            "## Division 1—Definitions",
            "#### 1 Definitions",
            "#### 2 Rate of benefit",
            "# Schedule 1—Rates",
            "## Part 1—Standard rate",
            "##### 1 Standard rate",
        ]

    def test_contents_front_pages_and_endnotes_are_removed(
        self, tmp_path: Path
    ) -> None:
        markdown = _convert(_write_docx(tmp_path / "act.docx")).markdown

        for removed in (
            "About this compilation",
            "An Act about benefits",
            "Endnote",
            "Amendment history text.",
            "\t",
        ):
            assert removed not in markdown
        assert markdown.count("Part 1—Preliminary") == 1

    def test_body_text_tables_and_formulas_are_kept(self, tmp_path: Path) -> None:
        converted = _convert(_write_docx(tmp_path / "act.docx"))
        markdown = converted.markdown

        assert "(1) The rate is set out in the table." in markdown
        assert "| Item | Rate |\n| --- | --- |\n| 1 | $100 \\| per week |" in markdown
        assert "[Formula: Start formula Rate times 52 end formula]" in markdown
        assert "\\# not a heading" in markdown
        assert converted.sections == 3

    def test_output_is_byte_identical_on_a_second_run(self, tmp_path: Path) -> None:
        docx = _write_docx(tmp_path / "act.docx")

        first = _convert(docx).markdown.encode()
        second = _convert(docx).markdown.encode()

        assert first == second

    def test_front_matter_is_the_only_preamble(self, tmp_path: Path) -> None:
        markdown = _convert(_write_docx(tmp_path / "act.docx")).markdown

        assert strip_front_matter(markdown).startswith("# Part 1—Preliminary")

    def test_hash_mismatch_names_the_compilation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fetch, "CACHE", tmp_path)
        compilation = _compilation()
        _write_docx(compilation.cache_path)

        with pytest.raises(fetch.FetchError, match="C2001A00001 compilation 7"):
            fetch.verify(compilation)


class TestManifest:
    def test_two_compilations_of_each_act_are_pinned(self) -> None:
        _, compilations = fetch.load_manifest()

        by_series = {(c.series_id, c.start_date) for c in compilations}
        assert by_series == {
            ("C2004A00490", "2020-07-01"),
            ("C2004A00490", "2026-07-01"),
            ("C2004A00491", "2020-07-01"),
            ("C2004A00491", "2026-07-01"),
        }
        for compilation in compilations:
            assert compilation.url.startswith("https://www.legislation.gov.au/")
            assert compilation.sha256 and len(compilation.sha256) == 64
            if compilation.series_id == "C2004A00491":
                assert compilation.volume == 1


class TestAgentYaml:
    @pytest.fixture
    def agent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Agent:
        # Keep the developer's ~/.holodeck/config.yaml out of the test.
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "unit-not-real")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
        monkeypatch.setenv("NEO4J_USER", "neo4j")
        monkeypatch.setenv("NEO4J_PASSWORD", "unit-not-real")
        return ConfigLoader().load_agent_yaml(str(SAMPLE / "agent.yaml"))

    def test_tools(self, agent: Agent) -> None:
        tools = {tool.name: tool for tool in agent.tools or []}
        assert set(tools) == {
            "family_assistance_act",
            "family_assistance_admin_act",
            "benefit_memory",
        }
        memory = tools["benefit_memory"]
        assert isinstance(memory, TemporalMemoryToolConfig)
        assert memory.resolved_namespace == "family-assistance"
        assert memory.ingest_on_start is False
        assert memory.backend.driver == "neo4j"
        assert memory.source_tools == [
            "family_assistance_act",
            "family_assistance_admin_act",
        ]

    def test_source_tools_use_qdrant_without_contextual_embeddings(
        self, agent: Agent
    ) -> None:
        sources = [
            tool
            for tool in agent.tools or []
            if isinstance(tool, HierarchicalDocumentToolConfig)
        ]
        assert len(sources) == 2
        for tool in sources:
            assert not isinstance(tool.database, str) and tool.database is not None
            assert tool.database.provider == "qdrant"
            assert tool.contextual_embeddings is False

    def test_test_cases_assert_memory_tools(self, agent: Agent) -> None:
        cases = agent.test_cases or []
        assert len(cases) == 5
        for case in cases:
            assert case.expected_tools
            assert case.ground_truth

    def test_test_cases_grade_the_answer_against_ground_truth(
        self, agent: Agent
    ) -> None:
        # A tool-name-only case passes when the agent answers "not found".
        for case in agent.test_cases or []:
            graders = [
                metric
                for metric in case.evaluations or []
                if isinstance(metric, GEvalMetric)
            ]
            assert graders, f"{case.name}: no ground-truth grader"
            for metric in graders:
                assert "expected_output" in metric.evaluation_params
                assert metric.threshold is not None and metric.threshold > 0
                assert metric.fail_on_error is True
