"""Unit tests for :class:`PromptVersion` model (031-eval-runs-dashboard US2).

Covers T103 (field coverage) and T104 (cross-field validation).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from holodeck.models.eval_run import PromptVersion

_VALID_HASH = "a" * 64


@pytest.mark.unit
class TestPromptVersionFieldCoverage:
    """T103 — field inventory for :class:`PromptVersion`."""

    def test_minimal_inline_instance_valid(self) -> None:
        pv = PromptVersion(
            version="1.0",
            source="inline",
            body_hash=_VALID_HASH,
        )
        assert pv.version == "1.0"
        assert pv.author is None
        assert pv.description is None
        assert pv.tags == []
        assert pv.source == "inline"
        assert pv.file_path is None
        assert pv.body_hash == _VALID_HASH
        assert pv.extra == {}

    def test_all_fields_populated(self) -> None:
        pv = PromptVersion(
            version="1.2",
            author="jane",
            description="d",
            tags=["a", "b"],
            source="file",
            file_path="/abs/path/instructions.md",
            body_hash=_VALID_HASH,
            extra={"custom_field": "value"},
        )
        assert pv.tags == ["a", "b"]
        assert pv.file_path == "/abs/path/instructions.md"
        assert pv.extra == {"custom_field": "value"}

    def test_tags_default_empty_list(self) -> None:
        pv = PromptVersion(version="v1", source="inline", body_hash=_VALID_HASH)
        assert pv.tags == []

    def test_extra_default_empty_dict(self) -> None:
        pv = PromptVersion(version="v1", source="inline", body_hash=_VALID_HASH)
        assert pv.extra == {}

    def test_version_required(self) -> None:
        with pytest.raises(ValidationError):
            PromptVersion(  # type: ignore[call-arg]
                source="inline",
                body_hash=_VALID_HASH,
            )

    def test_source_required(self) -> None:
        with pytest.raises(ValidationError):
            PromptVersion(  # type: ignore[call-arg]
                version="1.0",
                body_hash=_VALID_HASH,
            )

    def test_body_hash_required(self) -> None:
        with pytest.raises(ValidationError):
            PromptVersion(  # type: ignore[call-arg]
                version="1.0",
                source="inline",
            )

    def test_model_has_extra_forbid(self) -> None:
        """Unknown model-level kwargs must be rejected; frontmatter extras go
        into the ``extra`` dict, not as additional model fields."""
        with pytest.raises(ValidationError):
            PromptVersion(
                version="1.0",
                source="inline",
                body_hash=_VALID_HASH,
                unknown_field="nope",  # type: ignore[call-arg]
            )


@pytest.mark.unit
class TestPromptVersionValidation:
    """T104 — validator behaviour."""

    def test_inline_with_file_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromptVersion(
                version="1.0",
                source="inline",
                file_path="/abs/path/instructions.md",
                body_hash=_VALID_HASH,
            )

    def test_file_without_file_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromptVersion(
                version="1.0",
                source="file",
                body_hash=_VALID_HASH,
            )

    def test_file_with_empty_file_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromptVersion(
                version="1.0",
                source="file",
                file_path="",
                body_hash=_VALID_HASH,
            )

    def test_file_with_file_path_valid(self) -> None:
        pv = PromptVersion(
            version="1.0",
            source="file",
            file_path="instructions.md",
            body_hash=_VALID_HASH,
        )
        assert pv.source == "file"
        assert pv.file_path == "instructions.md"

    def test_body_hash_wrong_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromptVersion(
                version="1.0",
                source="inline",
                body_hash="a" * 10,
            )

    def test_body_hash_non_hex_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromptVersion(
                version="1.0",
                source="inline",
                body_hash="z" * 64,
            )

    def test_body_hash_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromptVersion(
                version="1.0",
                source="inline",
                body_hash="A" * 64,
            )
