"""Test case models for agent configuration.

This module defines test case and file input models used in agent.yaml
configuration for specifying test scenarios.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from holodeck.lib.errors import ConfigError
from holodeck.models.evaluation import (
    CodeMetric,
    EvaluationMetric,
    GEvalMetric,
    RAGMetric,
)

# ----------------------------------------------------------------------
# ArgMatcher — discriminated by shape (data-model.md §3)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LiteralMatcher:
    """Literal value — exact equality (int↔float numeric equivalence)."""

    value: Any


@dataclass(frozen=True)
class FuzzyMatcher:
    """Case/whitespace/separator-tolerant, numeric-aware match."""

    pattern: str


@dataclass(frozen=True)
class RegexMatcher:
    """Anchored regex full-match over `str(actual)`."""

    compiled: re.Pattern[str]


ArgMatcher = LiteralMatcher | FuzzyMatcher | RegexMatcher


def _coerce_arg_matcher(value: Any, field_path: str) -> ArgMatcher:
    """Coerce a raw YAML/JSON value to an ArgMatcher.

    Shape-discriminated (not key-discriminated) — a dict with exactly one of
    `fuzzy` / `regex` becomes a FuzzyMatcher / RegexMatcher; anything else
    (scalar, list, dict-without-matcher-keys) is a LiteralMatcher.

    Args:
        value: Raw argument specifier.
        field_path: Dotted path used in error messages.

    Returns:
        The resolved matcher instance.

    Raises:
        ConfigError: If the dict has both matcher keys, an unknown key, or an
            uncompilable regex.
    """
    if isinstance(value, (LiteralMatcher, FuzzyMatcher, RegexMatcher)):
        return value
    if isinstance(value, dict):
        keys = set(value.keys())
        matcher_keys = keys & {"fuzzy", "regex"}
        if "fuzzy" in keys and "regex" in keys:
            raise ConfigError(
                field_path,
                "matcher dict must have exactly one of 'fuzzy' or 'regex', not both",
            )
        if matcher_keys:
            # Exactly one matcher key — reject any extra keys.
            other_keys = keys - matcher_keys
            if other_keys:
                raise ConfigError(
                    field_path,
                    f"matcher dict has unknown keys: {sorted(other_keys)}",
                )
            if "fuzzy" in keys:
                pattern = value["fuzzy"]
                if not isinstance(pattern, str):
                    raise ConfigError(
                        f"{field_path}.fuzzy",
                        f"fuzzy pattern must be a string, got {type(pattern).__name__}",
                    )
                return FuzzyMatcher(pattern=pattern)
            # regex
            pattern = value["regex"]
            if not isinstance(pattern, str):
                raise ConfigError(
                    f"{field_path}.regex",
                    f"regex pattern must be a string, got {type(pattern).__name__}",
                )
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ConfigError(
                    f"{field_path}.regex",
                    f"cannot compile regex: {exc}",
                ) from exc
            return RegexMatcher(compiled=compiled)
    # Literal — scalar, list, dict without matcher keys.
    return LiteralMatcher(value=value)


# ----------------------------------------------------------------------
# ExpectedTool (data-model.md §2)
# ----------------------------------------------------------------------


class ExpectedTool(BaseModel):
    """Object form of an expected tool assertion (FR-003).

    A bare string like `"subtract"` is promoted to
    `ExpectedTool(name="subtract", args=None, count=1)` by the list
    pre-validator on the enclosing model, but this class itself only holds
    the object shape.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(..., description="Tool name — case-sensitive substring match")
    args: dict[str, Any] | None = Field(
        None, description="Per-arg matcher map (FR-004)"
    )
    count: int = Field(1, ge=1, description="Minimum number of matching calls")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @model_validator(mode="after")
    def _coerce_args(self) -> ExpectedTool:
        if self.args is None:
            return self
        coerced: dict[str, ArgMatcher] = {}
        for key, raw in self.args.items():
            coerced[key] = _coerce_arg_matcher(raw, f"args.{key}")
        # Mutate in place — args is Any-typed on the model, but stores
        # ArgMatcher dataclass values after validation.
        object.__setattr__(self, "args", coerced)
        return self


# ----------------------------------------------------------------------
# FileInput (unchanged)
# ----------------------------------------------------------------------


class FileInput(BaseModel):
    """File input for multimodal test cases.

    Represents a single file reference for test case inputs, supporting
    both local files and remote URLs with optional extraction parameters.
    """

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(None, description="Local file path")
    url: str | None = Field(None, description="Remote URL")
    type: str = Field(
        ..., description="File type: image, pdf, text, excel, word, powerpoint, csv"
    )
    description: str | None = Field(None, description="File description")
    pages: list[int] | None = Field(
        None, description="Specific pages/slides to extract"
    )
    sheet: str | None = Field(None, description="Excel sheet name")
    range: str | None = Field(None, description="Excel cell range (e.g., A1:E100)")
    cache: bool | None = Field(
        None, description="Cache remote files (default true for URLs)"
    )

    @field_validator("path", "url", mode="before")
    @classmethod
    def check_path_or_url(cls, v: Any, info: Any) -> Any:
        """Validate that exactly one of path or url is provided."""
        # This runs before validation, so we check in root_validator
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate file type is supported."""
        valid_types = {"image", "pdf", "text", "excel", "word", "powerpoint", "csv"}
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got {v}")
        return v

    @field_validator("pages")
    @classmethod
    def validate_pages(cls, v: list[int] | None) -> list[int] | None:
        """Validate pages are positive integers."""
        if v is not None and not all(isinstance(p, int) and p > 0 for p in v):
            raise ValueError("pages must be positive integers")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate path and url mutual exclusivity after initialization."""
        if self.path and self.url:
            raise ValueError("Cannot provide both 'path' and 'url'")
        if not self.path and not self.url:
            raise ValueError("Must provide either 'path' or 'url'")


# ----------------------------------------------------------------------
# expected_tools list normalization
# ----------------------------------------------------------------------


def _normalize_expected_tools_list(
    value: Any, location: str
) -> list[str | ExpectedTool] | None:
    """Coerce a raw `expected_tools` list into its normalized form.

    Each element is kept as a `str` (legacy) or parsed into `ExpectedTool`.
    Error messages include the provided `location` (e.g. test-case name +
    field path).
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("expected_tools must be a list")
    out: list[str | ExpectedTool] = []
    for idx, raw in enumerate(value):
        path = f"{location}[{idx}]"
        if isinstance(raw, (str, ExpectedTool)):
            out.append(raw)
        elif isinstance(raw, dict):
            try:
                et = _build_expected_tool(raw, path)
            except ConfigError:
                raise
            except Exception as exc:
                raise ConfigError(path, str(exc)) from exc
            out.append(et)
        else:
            raise ConfigError(
                path,
                f"expected_tools entries must be str or object, "
                f"got {type(raw).__name__}",
            )
    return out


def _build_expected_tool(raw: dict[str, Any], path: str) -> ExpectedTool:
    """Build an ExpectedTool, translating inner failures into path-prefixed
    ConfigErrors."""
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{path}.name", "name must be a non-empty string")
    count = raw.get("count", 1)
    if not isinstance(count, int) or count < 1:
        raise ConfigError(f"{path}.count", "count must be an integer >= 1")
    args_raw = raw.get("args")
    args: dict[str, ArgMatcher] | None = None
    if args_raw is not None:
        if not isinstance(args_raw, dict):
            raise ConfigError(f"{path}.args", "args must be a mapping")
        args = {}
        for key, raw_val in args_raw.items():
            args[key] = _coerce_arg_matcher(raw_val, f"{path}.args.{key}")
    # Allowed keys check (extra="forbid" equivalent) before building.
    allowed = {"name", "args", "count"}
    extra = set(raw.keys()) - allowed
    if extra:
        raise ConfigError(path, f"unknown keys on ExpectedTool: {sorted(extra)}")
    # Build bypassing args re-coercion.
    et = ExpectedTool.model_construct(name=name, args=args, count=count)
    return et


def _serialize_expected_tools(
    value: list[str | ExpectedTool] | None,
) -> list[Any] | None:
    """Dump to wire form — bare strings emit as `str`, object forms as dict.

    Legacy-promoted entries (`args is None and count == 1`) emitted via
    pre-validator path are stored as bare strings already (FR-024 guard in
    T014a).
    """
    if value is None:
        return None
    out: list[Any] = []
    for entry in value:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, ExpectedTool):
            obj: dict[str, Any] = {"name": entry.name}
            if entry.args is not None:
                obj["args"] = _dump_args(entry.args)
            if entry.count != 1:
                obj["count"] = entry.count
            out.append(obj)
    return out


def _dump_args(args: dict[str, Any]) -> dict[str, Any]:
    """Dump args dict — ArgMatcher dataclasses → wire shapes."""
    out: dict[str, Any] = {}
    for key, val in args.items():
        if isinstance(val, LiteralMatcher):
            out[key] = val.value
        elif isinstance(val, FuzzyMatcher):
            out[key] = {"fuzzy": val.pattern}
        elif isinstance(val, RegexMatcher):
            out[key] = {"regex": val.compiled.pattern}
        else:
            out[key] = val
    return out


# ----------------------------------------------------------------------
# Turn (data-model.md §1)
# ----------------------------------------------------------------------


class Turn(BaseModel):
    """Single exchange within a multi-turn test case (data-model.md §1).

    A multi-turn test case is an ordered list of turns driven through a
    single AgentSession. Each turn owns its own input, optional ground
    truth, expected tools, files, retrieval context, and metric overrides.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    input: str = Field(..., description="User query for this turn")
    ground_truth: str | None = Field(
        None, description="Expected output for this turn (non-empty if provided)"
    )
    expected_tools: list[str | ExpectedTool] | None = Field(
        None,
        description=(
            "Tools expected to be called during this turn. Mixed list of "
            "bare strings (legacy name-only) and ExpectedTool objects (US3)."
        ),
    )
    files: list[FileInput] | None = Field(
        None, description="Multimodal file inputs for this turn"
    )
    retrieval_context: list[str] | None = Field(
        None,
        description="Retrieved text chunks for RAG evaluation on this turn",
    )
    evaluations: (
        list[EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric] | None
    ) = Field(
        None,
        description="Per-turn metric overrides (FR-023)",
    )

    @field_validator("input")
    @classmethod
    def _validate_input(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("input must be a non-empty string")
        return v

    @field_validator("ground_truth")
    @classmethod
    def _validate_ground_truth(cls, v: str | None) -> str | None:
        if v is not None and (not v or not v.strip()):
            raise ValueError("ground_truth must be non-empty if provided")
        return v

    @field_validator("files")
    @classmethod
    def _validate_files(cls, v: list[FileInput] | None) -> list[FileInput] | None:
        if v is not None and len(v) > 10:
            raise ValueError("Maximum 10 files per turn")
        return v

    @field_validator("expected_tools", mode="before")
    @classmethod
    def _normalize_expected_tools(cls, v: Any) -> Any:
        return _normalize_expected_tools_list(v, "expected_tools")

    @field_serializer("expected_tools")
    def _dump_expected_tools(
        self, value: list[str | ExpectedTool] | None
    ) -> list[Any] | None:
        return _serialize_expected_tools(value)


# ----------------------------------------------------------------------
# TestCaseModel
# ----------------------------------------------------------------------


class TestCaseModel(BaseModel):
    """Test case for agent evaluation.

    Represents a single test scenario. Either legacy single-turn (with
    `input`) or multi-turn (with `turns`), but not both.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str | None = Field(None, description="Test case identifier")
    input: str | None = Field(
        None,
        description="User query or prompt (required when `turns` is absent)",
    )
    expected_tools: list[str | ExpectedTool] | None = Field(
        None, description="Tools expected to be called"
    )
    ground_truth: str | None = Field(None, description="Expected output for comparison")
    files: list[FileInput] | None = Field(None, description="Multimodal file inputs")
    retrieval_context: list[str] | None = Field(
        None, description="Retrieved text chunks for RAG evaluation metrics"
    )
    evaluations: (
        list[EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric] | None
    ) = Field(
        None,
        description="Per-test metric overrides (standard, GEval, RAG, or code)",
    )
    turns: list[Turn] | None = Field(
        None,
        description=(
            "Ordered multi-turn conversation (feature 032). Mutually exclusive "
            "with `input` / `ground_truth` / top-level `expected_tools` / `files` "
            "/ `retrieval_context`."
        ),
    )

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str | None) -> str | None:
        """Input, when supplied, must be non-empty."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("input must be a non-empty string")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        if v is not None and (not v or not v.strip()):
            raise ValueError("name must be non-empty if provided")
        return v

    @field_validator("ground_truth")
    @classmethod
    def validate_ground_truth(cls, v: str | None) -> str | None:
        if v is not None and (not v or not v.strip()):
            raise ValueError("ground_truth must be non-empty if provided")
        return v

    @field_validator("files")
    @classmethod
    def validate_files(cls, v: list[FileInput] | None) -> list[FileInput] | None:
        if v is not None and len(v) > 10:
            raise ValueError("Maximum 10 files per test case")
        return v

    @field_validator("expected_tools", mode="before")
    @classmethod
    def _normalize_top_level_expected_tools(cls, v: Any) -> Any:
        return _normalize_expected_tools_list(v, "expected_tools")

    @field_serializer("expected_tools")
    def _dump_expected_tools(
        self, value: list[str | ExpectedTool] | None
    ) -> list[Any] | None:
        return _serialize_expected_tools(value)

    @model_validator(mode="before")
    @classmethod
    def _prefix_path_for_turn_tools(cls, data: Any) -> Any:
        """Enrich nested-turn validation errors with test-case name + full
        `turns[i].expected_tools[j]...` path (FR-025).
        """
        if not isinstance(data, dict):
            return data
        name = data.get("name")
        turns = data.get("turns")
        if not isinstance(turns, list):
            return data
        # Walk turns and normalize expected_tools with precise paths so
        # errors include the surrounding test case.
        tc_label = name if isinstance(name, str) and name.strip() else None
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            et = t.get("expected_tools")
            if et is None:
                continue
            field_base = f"turns[{i}].expected_tools"
            try:
                _normalize_expected_tools_list(et, field_base)
            except ConfigError as exc:
                # Re-raise with test-case context embedded in the message.
                prefix = f"[{tc_label}] " if tc_label else ""
                raise ConfigError(
                    exc.field,
                    f"{prefix}{exc.message}",
                ) from exc
        return data

    @model_validator(mode="after")
    def _validate_single_vs_multi_turn(self) -> TestCaseModel:
        """Enforce mutual-exclusion rules from test-case-schema.md §2."""
        if self.turns is not None:
            if len(self.turns) == 0:
                raise ValueError("turns must be a non-empty list if provided")
            conflicts: list[str] = []
            if self.input is not None:
                conflicts.append("input")
            if self.ground_truth is not None:
                conflicts.append("ground_truth")
            if self.expected_tools is not None:
                conflicts.append("expected_tools")
            if self.files is not None:
                conflicts.append("files")
            if self.retrieval_context is not None:
                conflicts.append("retrieval_context")
            if conflicts:
                raise ValueError(
                    "multi-turn test case (with `turns`) cannot also set "
                    f"top-level {conflicts}; move these fields onto each turn"
                )
        else:
            if self.input is None:
                raise ValueError(
                    "test case requires either `input` (single-turn) or "
                    "`turns` (multi-turn)"
                )
        return self


# Alias for backward compatibility
TestCase = TestCaseModel
