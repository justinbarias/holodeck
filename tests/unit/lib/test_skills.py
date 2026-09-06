"""Tests for ``holodeck.lib.skills`` (SKILL.md parsing, spec 023 FR-023)."""

from __future__ import annotations

from pathlib import Path

import pytest

from holodeck.lib.skills import (
    SkillLoadError,
    load_skill_definition,
    resolve_skill_dir,
)


def _write_skill(root: Path, text: str, name: str = "my-skill") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    return skill_dir


VALID = """---
name: research-assistant
description: Finds and summarises sources.
allowed-tools: Bash
---
# Research

Search the knowledge base, then summarise.
"""


@pytest.mark.unit
class TestLoadSkillDefinition:
    def test_parses_frontmatter_and_body(self, tmp_path: Path) -> None:
        skill_dir = _write_skill(tmp_path, VALID)
        definition = load_skill_definition(skill_dir)
        assert definition.name == "research-assistant"
        assert definition.description == "Finds and summarises sources."
        assert definition.instructions == (
            "# Research\n\nSearch the knowledge base, then summarise."
        )
        assert definition.path == skill_dir

    def test_indented_dashes_inside_block_scalar_are_content(
        self, tmp_path: Path
    ) -> None:
        text = (
            "---\n"
            "name: x\n"
            "description: |\n"
            "  first line\n"
            "  ---\n"
            "  still description\n"
            "---\n"
            "body\n"
        )
        definition = load_skill_definition(_write_skill(tmp_path, text))
        assert definition.description == "first line\n---\nstill description"
        assert definition.instructions == "body"

    def test_trailing_whitespace_on_delimiter_tolerated(self, tmp_path: Path) -> None:
        text = "---  \nname: x\ndescription: y\n---\t\nbody\n"
        assert (
            load_skill_definition(_write_skill(tmp_path, text)).instructions == "body"
        )

    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(SkillLoadError, match="not a directory"):
            load_skill_definition(tmp_path / "nope")

    def test_missing_skill_file(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(SkillLoadError, match="has no SKILL.md"):
            load_skill_definition(tmp_path / "empty")

    def test_missing_frontmatter_block(self, tmp_path: Path) -> None:
        skill_dir = _write_skill(tmp_path, "# No frontmatter\nbody")
        with pytest.raises(SkillLoadError, match="must start with a '---'"):
            load_skill_definition(skill_dir)

    def test_unterminated_frontmatter(self, tmp_path: Path) -> None:
        skill_dir = _write_skill(tmp_path, "---\nname: x\ndescription: y\nbody")
        with pytest.raises(SkillLoadError, match="not terminated"):
            load_skill_definition(skill_dir)

    def test_missing_required_fields_are_listed(self, tmp_path: Path) -> None:
        skill_dir = _write_skill(tmp_path, "---\nversion: 1\n---\nbody")
        with pytest.raises(SkillLoadError, match="name, description"):
            load_skill_definition(skill_dir)

    def test_blank_description_counts_as_missing(self, tmp_path: Path) -> None:
        skill_dir = _write_skill(tmp_path, "---\nname: x\ndescription: '  '\n---\nbody")
        with pytest.raises(SkillLoadError, match="field\\(s\\): description"):
            load_skill_definition(skill_dir)

    def test_invalid_yaml_frontmatter(self, tmp_path: Path) -> None:
        skill_dir = _write_skill(tmp_path, "---\nname: [unclosed\n---\nbody")
        with pytest.raises(SkillLoadError, match="invalid SKILL.md frontmatter"):
            load_skill_definition(skill_dir)

    def test_non_mapping_frontmatter(self, tmp_path: Path) -> None:
        skill_dir = _write_skill(tmp_path, "---\n- a\n- b\n---\nbody")
        with pytest.raises(SkillLoadError, match="must be a mapping"):
            load_skill_definition(skill_dir)

    def test_empty_body_rejected(self, tmp_path: Path) -> None:
        skill_dir = _write_skill(tmp_path, "---\nname: x\ndescription: y\n---\n\n")
        with pytest.raises(SkillLoadError, match="body \\(instructions\\) is empty"):
            load_skill_definition(skill_dir)


@pytest.mark.unit
class TestResolveSkillDir:
    def test_relative_to_base_dir(self, tmp_path: Path) -> None:
        assert resolve_skill_dir("skills/x", tmp_path) == tmp_path / "skills/x"

    def test_absolute_path_unchanged(self, tmp_path: Path) -> None:
        assert resolve_skill_dir(str(tmp_path), Path("/elsewhere")) == tmp_path

    def test_relative_without_base_uses_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_skill_dir("s", None) == tmp_path / "s"
