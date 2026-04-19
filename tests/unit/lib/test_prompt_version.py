"""Unit tests for :func:`resolve_prompt_version` (031-eval-runs-dashboard US2).

Covers T105–T112 and T114:
- T105: inline short-circuits frontmatter parsing.
- T106: full frontmatter round-trip.
- T107: no ``version:`` key → ``auto-<sha256[:8]>`` stable.
- T108: no frontmatter at all → no error + defaults.
- T109: one-char edit changes hash (SC-004).
- T110: unknown keys preserved in ``extra``; recognised keys not duplicated.
- T111: malformed YAML raises :class:`ConfigError`.
- T112: relative path resolved against ``base_dir``.
- T114: ``resolve_instructions()`` body unaffected; additive module invariant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent

import pytest

from holodeck.lib.errors import ConfigError
from holodeck.lib.instruction_resolver import resolve_instructions
from holodeck.lib.prompt_version import resolve_prompt_version
from holodeck.models.agent import Instructions


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.unit
class TestResolvePromptVersionInline:
    """T105 — inline branch."""

    def test_inline_skips_frontmatter(self) -> None:
        text = "You are a helpful test agent."
        pv = resolve_prompt_version(Instructions(inline=text), base_dir=None)
        expected_hash = _sha256(text)
        assert pv.source == "inline"
        assert pv.file_path is None
        assert pv.body_hash == expected_hash
        assert pv.version == f"auto-{expected_hash[:8]}"
        assert pv.author is None
        assert pv.description is None
        assert pv.tags == []
        assert pv.extra == {}

    def test_inline_looking_frontmatter_is_not_parsed(self) -> None:
        """Even if the inline text contains ``---`` fences, they are NOT parsed
        as YAML frontmatter (FR-014)."""
        text = dedent(
            """\
            ---
            version: "1.2"
            ---
            Body here.
            """
        )
        pv = resolve_prompt_version(Instructions(inline=text), base_dir=None)
        # The entire inline string (fences + body) is hashed as-is.
        assert pv.version == f"auto-{_sha256(text)[:8]}"
        assert pv.body_hash == _sha256(text)


@pytest.mark.unit
class TestResolvePromptVersionFileFullFrontmatter:
    """T106 — full frontmatter mapping."""

    def test_all_recognised_keys_returned_verbatim(self, tmp_path: Path) -> None:
        body = "You are a helpful agent.\n"
        content = (
            dedent(
                """\
            ---
            version: "1.2"
            author: jane
            description: d
            tags:
              - a
              - b
            ---
            """
            )
            + body
        )
        md = tmp_path / "instructions.md"
        md.write_text(content)

        pv = resolve_prompt_version(
            Instructions(file="instructions.md"), base_dir=tmp_path
        )
        assert pv.version == "1.2"
        assert pv.author == "jane"
        assert pv.description == "d"
        assert pv.tags == ["a", "b"]
        assert pv.source == "file"
        assert pv.file_path == str(tmp_path / "instructions.md")
        # body_hash is SHA-256 of the body only (post-frontmatter-strip).
        assert pv.body_hash == _sha256(body.rstrip("\n"))


@pytest.mark.unit
class TestResolvePromptVersionAutoVersion:
    """T107 — auto-hash fallback."""

    def test_no_version_key_yields_auto_hash(self, tmp_path: Path) -> None:
        body = "Body content.\n"
        content = (
            dedent(
                """\
            ---
            author: jane
            ---
            """
            )
            + body
        )
        md = tmp_path / "instructions.md"
        md.write_text(content)

        pv1 = resolve_prompt_version(
            Instructions(file="instructions.md"), base_dir=tmp_path
        )
        pv2 = resolve_prompt_version(
            Instructions(file="instructions.md"), base_dir=tmp_path
        )
        assert pv1.version.startswith("auto-")
        assert pv1.version == pv2.version
        assert pv1.body_hash == pv2.body_hash
        assert pv1.version == f"auto-{pv1.body_hash[:8]}"


@pytest.mark.unit
class TestResolvePromptVersionNoFrontmatter:
    """T108 — no frontmatter at all is not an error."""

    def test_no_frontmatter_defaults_returned(self, tmp_path: Path) -> None:
        body = "Just a plain markdown body without frontmatter."
        md = tmp_path / "instructions.md"
        md.write_text(body)

        pv = resolve_prompt_version(
            Instructions(file="instructions.md"), base_dir=tmp_path
        )
        assert pv.version == f"auto-{_sha256(body)[:8]}"
        assert pv.author is None
        assert pv.description is None
        assert pv.tags == []
        assert pv.extra == {}
        assert pv.source == "file"


@pytest.mark.unit
class TestResolvePromptVersionBodyEditChangesHash:
    """T109 — one-char edit changes the auto version."""

    def test_one_char_edit_changes_version(self, tmp_path: Path) -> None:
        md = tmp_path / "instructions.md"
        md.write_text("You are a helpful agent.")
        pv_before = resolve_prompt_version(
            Instructions(file="instructions.md"), base_dir=tmp_path
        )

        md.write_text("You are a helpful Agent.")  # capitalised A
        pv_after = resolve_prompt_version(
            Instructions(file="instructions.md"), base_dir=tmp_path
        )

        assert pv_before.version != pv_after.version
        assert pv_before.body_hash != pv_after.body_hash


@pytest.mark.unit
class TestResolvePromptVersionExtraKeys:
    """T110 — unknown frontmatter keys go to ``extra``."""

    def test_unknown_keys_preserved_in_extra(self, tmp_path: Path) -> None:
        content = dedent(
            """\
            ---
            version: "1.0"
            author: jane
            custom_field: value
            another: 42
            ---
            body
            """
        )
        md = tmp_path / "instructions.md"
        md.write_text(content)

        pv = resolve_prompt_version(
            Instructions(file="instructions.md"), base_dir=tmp_path
        )
        assert pv.extra == {"custom_field": "value", "another": 42}
        # Recognised keys MUST NOT be duplicated into extra.
        assert "version" not in pv.extra
        assert "author" not in pv.extra
        assert "description" not in pv.extra
        assert "tags" not in pv.extra


@pytest.mark.unit
class TestResolvePromptVersionMalformedYAML:
    """T111 — malformed frontmatter → ConfigError."""

    def test_malformed_yaml_raises_config_error(self, tmp_path: Path) -> None:
        content = dedent(
            """\
            ---
            tags: [unclosed
            ---
            body
            """
        )
        md = tmp_path / "instructions.md"
        md.write_text(content)

        with pytest.raises(ConfigError) as excinfo:
            resolve_prompt_version(
                Instructions(file="instructions.md"), base_dir=tmp_path
            )
        assert excinfo.value.field == "instructions.file"
        # Error message must reference the file path.
        assert "instructions.md" in excinfo.value.message


@pytest.mark.unit
class TestResolvePromptVersionRelativePathResolution:
    """T112 — relative ``instructions.file`` resolved against ``base_dir``."""

    def test_relative_path_resolves_against_base_dir(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        body = "Relative body"
        (subdir / "instructions.md").write_text(body)

        pv = resolve_prompt_version(
            Instructions(file="instructions.md"), base_dir=subdir
        )
        assert pv.file_path == str(subdir / "instructions.md")
        assert pv.body_hash == _sha256(body)


@pytest.mark.unit
class TestResolveInstructionsUnaffected:
    """T114 — additive-module invariant.

    ``resolve_instructions()`` MUST return the same string as before, including
    frontmatter (it does NOT strip it). ``resolve_prompt_version()`` is a
    SEPARATE call on the same ``Instructions`` object.
    """

    def test_resolve_instructions_returns_full_file_unchanged(
        self, tmp_path: Path
    ) -> None:
        raw = dedent(
            """\
            ---
            version: "1.2"
            author: jane
            ---
            Body goes here.
            """
        )
        md = tmp_path / "instructions.md"
        md.write_text(raw)

        body = resolve_instructions(
            Instructions(file="instructions.md"), base_dir=tmp_path
        )
        # Byte-for-byte identical to file contents — frontmatter is NOT
        # stripped at resolve time.
        assert body == raw

    def test_resolve_instructions_signature_unchanged(self) -> None:
        """Guard against accidental signature changes."""
        import inspect

        sig = inspect.signature(resolve_instructions)
        params = list(sig.parameters.keys())
        assert params == ["instructions", "base_dir"]
        assert sig.return_annotation is str
