# Contract: `TestResult.turns` JSON Schema

**Feature**: 032-multi-turn-test-cases
**Consumers**: Dashboard (feature 031), markdown/JSON reporter, CI summarizers.

Describes the on-the-wire shape of the new `TestResult.turns` field and the rules the runner uses to populate the existing test-case-level fields from per-turn data. Any reader that already parsed `TestResult` continues to work — new readers unlock per-turn detail by reading `turns`.

## 1. `TestResult.turns` field

```json
{
  "turns": [
    { "turn_index": 0, "input": "...", "response": "...", "ground_truth": "206588",
      "tool_calls": ["lookup"], "tool_invocations": [ { "name": "lookup", "args": {...}, "result": 206588.0, "bytes": 32, "duration_ms": 41 } ],
      "tools_matched": true, "arg_match_details": null,
      "metric_results": [ { "metric_name": "numeric", "kind": "standard", "score": 1.0, "threshold": null, "passed": true, "scale": "0-1", "error": null, "retry_count": 0, "evaluation_time_ms": 0, "model_used": null, "reasoning": null } ],
      "passed": true, "execution_time_ms": 412, "token_usage": { "prompt_tokens": 180, "completion_tokens": 12, "total_tokens": 192 },
      "errors": [], "skipped": false }
  ]
}
```

- Optional field. Present only for multi-turn test cases.
- When absent or `null`, readers treat the result as single-turn (legacy).
- When present, `len(turns) >= 1` — the runner never writes an empty list.

## 2. `TurnResult` object

| Field | JSON Type | Required | Notes |
|-------|-----------|----------|-------|
| `turn_index` | integer | yes | 0-based position in the original `turns` list. |
| `input` | string | yes | Verbatim turn input. |
| `response` | string \| null | no | `null` if the turn failed before producing a response. |
| `ground_truth` | string \| null | no | Copied from turn config. |
| `expected_tools` | array \| null | no | Normalized form: every element is either a bare string or `{name, args?, count}`. |
| `tool_calls` | array of string | yes | Names only. Empty list if no tools called. |
| `tool_invocations` | array of `ToolInvocation` | yes | Same shape as on `TestResult` (feature 031). |
| `tools_matched` | bool \| null | no | `null` when no tool assertion on the turn. |
| `arg_match_details` | array \| null | no | Per-assertion record — see §2.1. |
| `metric_results` | array of `MetricResult` | yes | Same shape as on `TestResult`. |
| `passed` | bool | yes | Composition rule in §3. |
| `execution_time_ms` | integer | yes | Wall-clock duration of this turn's `send()`. |
| `token_usage` | `TokenUsage` \| null | no | Per-turn usage. |
| `errors` | array of string | yes | Per-turn errors. Empty list if none. |
| `skipped` | bool | no | `true` if the session became unrecoverable before this turn. When `true`, `response`, `tool_calls`, and `metric_results` may be empty and `errors` will name the reason. |

### 2.1 `arg_match_details` entry

For each `ExpectedTool` with `args` on this turn:

```json
{
  "expected_tool": "subtract",
  "args_asserted": { "a": 206588, "b": { "fuzzy": "181001" } },
  "matched_call_index": 0,                    // index into tool_invocations; -1 if no match
  "unmatched_reason": null                     // string iff matched_call_index == -1
}
```

When every matcher on every asserted tool finds a satisfying call, `matched_call_index` is the index of the first satisfying call and `unmatched_reason` is `null`. Otherwise `matched_call_index: -1` and `unmatched_reason` explains (e.g. `"no call to 'subtract' found"`, `"arg 'a' mismatch: expected 206588, got 180000"`).

## 3. Turn pass/fail rule

```
turn.passed =
     turn.errors == []
  && turn.skipped == false
  && (turn.tools_matched is null or turn.tools_matched == true)
  && all(m.passed != false for m in turn.metric_results)
```

A turn is **failed** if any of the following:
- An error was captured during `send()` or grader execution.
- It was skipped due to unrecoverable session state (counts as failed on the enclosing test case per FR-008).
- `tools_matched` is explicitly `false` (name substring or arg assertion failed).
- Any configured metric has `passed == false`.

## 4. Test-case-level roll-up (FR-015)

The runner populates `TestResult` fields (for multi-turn cases) from the turns:

| Field | Rule |
|-------|------|
| `test_input` | `"\n---\n".join(turn.input for turn in turns)` |
| `agent_response` | `turns[-1].response` (may be `null` if the final turn failed) |
| `processed_files` | Concatenated in turn order (file metadata retained). |
| `tool_calls` | Flattened across all turns in order. |
| `tool_invocations` | Flattened across all turns in order. |
| `expected_tools` | Union of turn-level assertions (dashboards render as "across turns"). |
| `tools_matched` | `null` if no turn asserted tools; otherwise `all(t.tools_matched in (True, None) for t in turns)`. |
| `ground_truth` | `null` (inspect per turn). |
| `metric_results` | Per-metric rollup: each `metric_name` appears once; `score = mean(turn_scores for that metric, skipping turns that didn't run it)`; `passed = all(turn_passed for that metric)`. If every turn skipped a metric, the metric is omitted. |
| `passed` | `all(t.passed for t in turns)` (FR-016). |
| `execution_time_ms` | `sum(t.execution_time_ms for t in turns)` — does not include evaluation overhead. |
| `token_usage` | Element-wise sum across turns of all four countable fields — `prompt_tokens`, `completion_tokens`, `total_tokens`, `cache_creation_tokens`, `cache_read_tokens` (FR-007). See data-model.md §10a for the expanded `TokenUsage` schema. |
| `errors` | `[f"[turn {i}] {msg}" for i, t in enumerate(turns) for msg in t.errors]`. |
| `timestamp` | ISO-8601 at the moment the test case *started* (turn 0). |

## 5. Summary invariant (FR-015, preserved)

`ReportSummary`:
- `total_tests = len(results)` — counts test cases, not turns.
- `passed + failed == total_tests` still holds.
- `metrics_evaluated[name]` is the count of **test cases** where the rolled-up metric appears (unchanged semantics).
- `average_scores[name]` is the mean of rolled-up test-case scores (not per-turn) — preserves the dashboard metric-average contract.
