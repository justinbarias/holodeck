# Phase 1 Data Model: Multi-Turn Test Cases

**Feature**: 032-multi-turn-test-cases
**Date**: 2026-04-20

This document is the authoritative description of the new and modified Pydantic models. Implementation lives under `src/holodeck/models/`; this file defines the shape, validation rules, and relationships.

All new fields use Pydantic v2 `ConfigDict(extra="forbid")` — consistent with the rest of HoloDeck. All validators run at config load time (FR-025).

---

## 1. `Turn` (new, `src/holodeck/models/test_case.py`)

Represents one exchange within a multi-turn test case.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `input` | `str` | Yes | Non-empty; trimmed. |
| `ground_truth` | `str \| None` | No | Non-empty if provided. |
| `expected_tools` | `list[str \| ExpectedTool] \| None` | No | Mixed strings (legacy name-only) and `ExpectedTool` objects allowed (FR-003). |
| `files` | `list[FileInput] \| None` | No | Same schema as today's `TestCaseModel.files`. Max 10 files per turn (matches existing limit). |
| `retrieval_context` | `list[str] \| None` | No | RAG metric context for this turn only (FR-002, FR-009). |
| `evaluations` | `list[MetricType] \| None` | No | Per-turn metric overrides (FR-023). |

**Validation rules**:
- `input` must be non-empty (`field_validator`).
- `ground_truth`, if provided, must be non-empty.
- `expected_tools` elements are discriminated: if an element is a `dict` with a `name` key, it's parsed as `ExpectedTool`; if a string, kept as-is.

---

## 2. `ExpectedTool` (new, `src/holodeck/models/test_case.py`)

Object form of an expected tool assertion (FR-003).

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `name` | `str` | Yes | — | Case-sensitive substring-matched against actual tool names (FR-011). |
| `args` | `dict[str, ArgMatcher] \| None` | No | `None` | Per-arg matcher map (FR-004). |
| `count` | `int` | No | `1` | Minimum number of matching calls required in this turn (A5). |

**Validation rules**:
- `name` non-empty.
- `count >= 1` (0 would be "must not be called", out of scope per FR and Assumptions).
- `args` keys are plain arg names (no dots, no indexing); values go through `ArgMatcher` discriminator.

---

## 3. `ArgMatcher` (new, `src/holodeck/models/test_case.py`)

Discriminated union over three matcher kinds (FR-004). Stored as `ArgMatcher = Annotated[Union[...], Field(discriminator='kind')]` or, for YAML ergonomics, as a custom validator that inspects shape.

**Accepted YAML shapes**:

1. **Literal** (exact, normalized) — any scalar, list, or dict:
   ```yaml
   args:
     a: 206588                  # int
     b: 181001.0                # float — coerces to int equivalence (FR-013, A7)
     name: "operating cash"     # string — exact match, no normalization
   ```

2. **Fuzzy**:
   ```yaml
   args:
     a: { fuzzy: "206588" }     # tolerates "206,588", "206588.0", case, whitespace
   ```

3. **Regex**:
   ```yaml
   args:
     cell: { regex: "^2009.*cash.*$" }     # anchored full-match; compiled at load
   ```

**Validation rules** (FR-025):
- A dict with exactly one of `fuzzy` / `regex` keys is parsed as that matcher kind.
- A dict with any other keys (or both `fuzzy` and `regex`) is a `ConfigError`.
- `regex` pattern is compiled at load via `re.compile(pattern)`; compile failures raise `ConfigError` naming the test case, turn index, tool name, and arg key.
- Literal is everything else (scalar, list, dict-without-matcher-keys).

**Internal representation**:

```python
@dataclass(frozen=True)
class LiteralMatcher: value: Any
@dataclass(frozen=True)
class FuzzyMatcher: pattern: str
@dataclass(frozen=True)
class RegexMatcher: compiled: re.Pattern[str]

ArgMatcher = Union[LiteralMatcher, FuzzyMatcher, RegexMatcher]
```

---

## 4. `TestCaseModel` (modified, `src/holodeck/models/test_case.py`)

Adds multi-turn surface while preserving the legacy single-turn shape (FR-001, FR-005).

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | `str \| None` | No | Unchanged. |
| `input` | `str \| None` | No (conditional) | **Required iff `turns` is absent.** Mutually exclusive with `turns`. |
| `expected_tools` | `list[str \| ExpectedTool] \| None` | No | Same mixed-shape union as on `Turn` (FR-003, FR-024). |
| `ground_truth` | `str \| None` | No | Mutually exclusive with `turns`. |
| `files` | `list[FileInput] \| None` | No | Unchanged. |
| `retrieval_context` | `list[str] \| None` | No | Unchanged. |
| `evaluations` | `list[MetricType] \| None` | No | Unchanged (now admits `equality`/`numeric`/`code` variants). |
| `turns` | `list[Turn] \| None` | No | **New** — presence switches the executor to multi-turn mode. |

**Validation rules** (FR-001):
- `turns` and any of `input` / `ground_truth` / top-level `expected_tools` (on test case) are mutually exclusive; both present → `ConfigError` with the test-case `name` or index.
- `turns` when present must be non-empty; `turns: []` → `ConfigError`.
- When `turns` is absent, `input` is required (preserves legacy contract).
- Top-level `expected_tools` is still legal for single-turn cases (back-compat, FR-024).

---

## 5. `EvaluationMetric` (modified, `src/holodeck/models/evaluation.py`)

Adds `equality` and `numeric` as new values of `metric` (FR-018). Existing fields unchanged.

**New optional fields** (only meaningful when `metric in {"equality", "numeric"}`):

| Field | Type | Applies to | Default | Notes |
|-------|------|-----------|---------|-------|
| `case_insensitive` | `bool` | equality | `false` | Lowercase both sides before compare. |
| `strip_whitespace` | `bool` | equality | `false` | Collapse runs of whitespace, trim. |
| `strip_punctuation` | `bool` | equality | `false` | Remove `string.punctuation` before compare. |
| `absolute_tolerance` | `float` | numeric | `1e-6` | Pass when `abs(actual - expected) <= absolute_tolerance`. |
| `relative_tolerance` | `float` | numeric | `0.0` | Pass when `abs(actual - expected) <= relative_tolerance * abs(expected)`. |
| `accept_percent` | `bool` | numeric | `false` | Parse trailing `%` as `/100`. |
| `accept_thousands_separators` | `bool` | numeric | `false` | Strip `,`, `_`, narrow NBSP before parse. |

**Validation rules**:
- `equality` and `numeric` evaluators never accept an LLM `model` — presence is a no-op (tolerated for uniformity, warning-logged).
- `absolute_tolerance >= 0`, `relative_tolerance >= 0`.
- All flags default to `false` (strict); tolerances default per clarification #5.

---

## 6. `CodeMetric` (new, `src/holodeck/models/evaluation.py`)

New top-level variant in the `MetricType` discriminated union (FR-019).

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `type` | `Literal["code"]` | Yes | `"code"` | Discriminator. |
| `grader` | `str` | Yes | — | `"module.path:callable_name"` — resolved at load time. |
| `threshold` | `float \| None` | No | `None` | Applied only if grader returns a float score without explicit `passed`. |
| `enabled` | `bool` | No | `true` | |
| `fail_on_error` | `bool` | No | `false` | When `true`, exception in grader marks whole test case failed; default `false` keeps the per-turn fail semantic (FR-022). |
| `name` | `str \| None` | No | Derived from `grader` | Used as `metric_name` in the report; falls back to the callable name. |

**Validation rules** (FR-025):
- `grader` must match `^[\w.]+:[\w_]+$`; mismatched → `ConfigError`.
- `importlib.import_module(module)` executed at load; `ImportError` / `ModuleNotFoundError` → `ConfigError` naming the test case / turn index / grader path.
- `getattr(module, callable_name)` must succeed and resolve to a callable; failure → `ConfigError`.
- Resolved callable is cached on the model instance so each turn doesn't re-import.

---

## 7. `MetricType` union (modified)

```python
MetricType = Annotated[
    Union[EvaluationMetric, GEvalMetric, RAGMetric, CodeMetric],
    Field(discriminator="type"),
]
```

Additive change — existing consumers matching on `isinstance(m, EvaluationMetric)` continue to work; new consumers handle `CodeMetric` explicitly.

## 7a. `MetricResult.kind` (modified)

The runtime `MetricResult.kind` field in `src/holodeck/models/test_result.py` is extended from `Literal["standard", "rag", "geval"]` to `Literal["standard", "rag", "geval", "code"]`. Additive; legacy reports continue to parse. The dashboard Compare-view and metric-trend panels filter on this field, so code-grader results receive a distinct bucket rather than masquerading as `"standard"`.

The `_metric_kind()` helper in `src/holodeck/lib/test_runner/executor.py` is updated in lock-step:

```python
def _metric_kind(
    metric_config: EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric,
) -> Literal["standard", "rag", "geval", "code"]:
    return metric_config.type
```

---

## 8. `TurnResult` (new, `src/holodeck/models/test_result.py`)

Per-turn outcome stored on `TestResult.turns`.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `turn_index` | `int` | Yes | 0-based. |
| `input` | `str` | Yes | Verbatim turn input. |
| `response` | `str \| None` | No | Agent response, or `None` if the turn erred before responding. |
| `ground_truth` | `str \| None` | No | Copied from turn config. |
| `expected_tools` | `list[str \| dict] \| None` | No | Serialized config form (normalized objects for readers). |
| `tool_calls` | `list[str]` | No | Names only (legacy shape, for dashboards). |
| `tool_invocations` | `list[ToolInvocation]` | No | Structured, per feature 031. |
| `tools_matched` | `bool \| None` | No | `None` if no `expected_tools` on this turn. |
| `arg_match_details` | `list[dict] \| None` | No | Per-assertion record: `{tool, args_asserted, matched_call_index, unmatched_reason}`. |
| `metric_results` | `list[MetricResult]` | Yes | Per-turn metric scores. |
| `passed` | `bool` | Yes | `all(metric passed) and tools_matched != False and not errors`. |
| `execution_time_ms` | `int` | Yes | Duration of this turn's `send()`. |
| `token_usage` | `TokenUsage \| None` | No | Usage for this turn (sums into test-case rollup). |
| `errors` | `list[str]` | No | Per-turn errors (timeout, backend error, grader exception). |
| `skipped` | `bool` | No | `true` if the session became unrecoverable before this turn ran. |
| `grader_details` | `dict[str, Any] \| None` | No | Populated from `GraderResult.details`, keyed by `metric_name`. Surfaces grader-specific payload (e.g., program-tree diff) to dashboards. Default `None`. See contracts/code-grader-contract.md §7. |

---

## 9. `TestResult` (modified)

Adds one optional field; all others unchanged.

| Field (changes only) | Type | Notes |
|----------------------|------|-------|
| `turns` | `list[TurnResult] \| None` (default `None`) | Only populated for multi-turn cases. Legacy single-turn runs leave it `None`. |

**Roll-up contract for multi-turn cases** (FR-015):

| Test-case field | Derivation from turns |
|-----------------|------------------------|
| `test_input` | `"\n---\n".join(turn.input for turn in turns)` |
| `agent_response` | `turns[-1].response` (final turn) |
| `tool_calls` | Flattened in turn order (legacy field) |
| `tool_invocations` | Flattened in turn order |
| `processed_files` | Flattened in turn order; per-turn `files` inputs processed via `_prepare_agent_input` and concatenated for test-case-level display. |
| `expected_tools` | Union of all turn `expected_tools` (for dashboard display) |
| `tools_matched` | `all(turn.tools_matched is not False for turn in turns)` — `None` if no turn asserted tools |
| `token_usage` | Element-wise sum of `turn.token_usage` (FR-007) |
| `passed` | `all(turn.passed for turn in turns)` (FR-016) |
| `ground_truth` | `None` at test-case level for multi-turn (use `turn.ground_truth`) |
| `metric_results` | Per-metric rollup — each `metric_name` appears once with `score = mean(turn_scores)` and `passed = all(turn_passed)`. Turns that skipped the metric do not contribute. If every turn skipped a metric, the metric is omitted entirely. |
| `errors` | Flattened list of per-turn errors, prefixed `[turn N] ...`. |

---

## 10. `ExecutionConfig` (modified, `src/holodeck/models/config.py`)

| New field | Type | Default | Notes |
|-----------|------|---------|-------|
| `parallel_test_cases` | `int` | `1` | Max concurrent multi-turn test cases (FR-009a). Resolved through CLI > YAML > env > defaults. `>= 1` enforced. |

`src/holodeck/config/defaults.py` — the `DEFAULT_EXECUTION_CONFIG` dict is updated to carry `"parallel_test_cases": 1` so the resolver has a final fallback after CLI > YAML > project > user > env.

## 10a. `TokenUsage` (modified, `src/holodeck/models/token_usage.py`)

Additive fields to carry provider-reported cache accounting so the feature 031 cost contract composes correctly across multi-turn runs.

| New field | Type | Default | Notes |
|-----------|------|---------|-------|
| `cache_creation_tokens` | `int` (≥ 0) | `0` | Tokens written to provider cache on this call (Anthropic / compatible). |
| `cache_read_tokens` | `int` (≥ 0) | `0` | Tokens served from provider cache on this call. |

**Validator change**: the `total_tokens` check is relaxed from strict equality (`total == prompt + completion`) to `total >= prompt + completion`. Cache reads do not always double-count against prompt tokens at every provider, so strict equality would reject valid responses. `__add__` sums all four countable fields element-wise; `zero()` returns every field as `0`.

**Back-compat**: legacy JSON lacking `cache_creation_tokens` / `cache_read_tokens` deserializes with the defaults. All existing readers that only touch `prompt_tokens` / `completion_tokens` / `total_tokens` continue to work unchanged. Cost-computation paths in feature 031 should begin to incorporate the new fields opt-in; until then they read as zero.

---

## 11. `ReportSummary.validate_test_counts` (unchanged — FR-015)

Explicitly preserved: `passed + failed == total_tests` still counts **test cases**, not turns. Turns are reporting detail, not summary units.

---

## Relationships

```
Agent
 └── test_cases: list[TestCaseModel]
      ├── (legacy) input + ground_truth + expected_tools
      └── (new)  turns: list[Turn]
                  └── each Turn
                      ├── input
                      ├── ground_truth
                      ├── expected_tools: list[str | ExpectedTool]
                      │    └── ExpectedTool.args: dict[str, ArgMatcher]
                      │         └── ArgMatcher: Literal | Fuzzy | Regex
                      ├── files
                      ├── retrieval_context
                      └── evaluations: list[MetricType]
                                        └── EvaluationMetric (incl. equality/numeric)
                                        └── GEvalMetric
                                        └── RAGMetric
                                        └── CodeMetric (NEW)
```

At runtime:

```
TestReport
 └── results: list[TestResult]
      └── turns: list[TurnResult] | None        ← new, populated for multi-turn only
           └── metric_results: list[MetricResult]   (same shape as TestResult.metric_results)
```

---

## Back-Compat Guarantees

- No existing field is removed, renamed, or given new required status.
- All new fields are optional with safe defaults.
- Discriminator union is extended (not narrowed).
- `TestResult.turns=None` is indistinguishable on-the-wire from legacy JSON missing the field (Pydantic default).
- `ExecutionConfig.parallel_test_cases=1` preserves the current sequential behavior byte-for-byte.
