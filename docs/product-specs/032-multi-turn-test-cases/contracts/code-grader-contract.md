# Contract: Code Grader (`type: code` Evaluator)

**Feature**: 032-multi-turn-test-cases
**Consumers**: Grader authors (user-land), `src/holodeck/lib/test_runner/code_grader.py`, executor, config loader.

This is the exception surface for Principle I (No-Code-First). See `spec.md §Complexity Tracking` for the governance log. Keep the documented surface small and stable.

---

## 1. YAML declaration (FR-019)

```yaml
evaluations:
  - type: code
    grader: "my_benchmarks.convfinqa:program_equivalence"
    threshold: 1.0             # optional; only used if grader returns bare float without `passed`
    enabled: true              # default true
    fail_on_error: false       # default false; true escalates grader exceptions to test-case fatal
    name: "program_equivalence"  # optional display name; defaults to the callable name
```

**Grader path format**: `"<module.path>:<callable_name>"`. Exactly one `:` separator. Module part must be importable via `importlib.import_module`.

## 2. Load-time resolution (FR-019, FR-025)

At config load:

1. Validate `grader` matches `^[\w.]+:[\w_]+$`.
2. `module = importlib.import_module("my_benchmarks.convfinqa")`.
3. `fn = getattr(module, "program_equivalence")`.
4. Assert `callable(fn)`.
5. Store `fn` on the `CodeMetric` instance for per-turn reuse.

Any failure (`ImportError`, `AttributeError`, not-callable) is a `ConfigError` naming:
- Test case name / index.
- Turn index (if per-turn).
- Full grader string.
- Underlying exception class and message.

Never raised at runtime — always at load.

## 3. Callable signature

```python
from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult

def program_equivalence(ctx: GraderContext) -> GraderResult: ...
```

`GraderContext` and `GraderResult` are public, importable dataclasses. Graders live in the user's package; they simply import these types.

### 3.1 `GraderContext` (read-only input)

```python
from dataclasses import dataclass
from typing import Any
from holodeck.models.test_result import ToolInvocation

@dataclass(frozen=True)
class GraderContext:
    turn_input: str                                  # turn's user message
    agent_response: str                              # agent's text reply (may be "")
    ground_truth: str | None                         # turn config's ground_truth
    tool_invocations: tuple[ToolInvocation, ...]     # ordered, immutable
    retrieval_context: tuple[str, ...] | None        # ordered, immutable, None if unset
    turn_index: int                                  # 0-based
    test_case_name: str | None
    turn_config: dict[str, Any]                      # raw per-turn YAML dict, for grader-specific keys
```

Frozen + tuples ensure graders cannot mutate them (FR-020). A grader reaching into HoloDeck internals outside this contract is unsupported.

### 3.2 `GraderResult` (return value)

```python
@dataclass(frozen=True)
class GraderResult:
    score: float                            # in [0.0, 1.0]; enforced at post-processing
    passed: bool | None = None              # if None, derived (see §5)
    reason: str | None = None               # human-readable one-liner
    details: dict[str, Any] | None = None   # arbitrary JSON-safe payload for the report
```

**Permissible return shortcuts** (auto-lifted by the runner):

| Return value | Normalized to |
|--------------|---------------|
| `True` | `GraderResult(score=1.0, passed=True)` |
| `False` | `GraderResult(score=0.0, passed=False)` |
| `0.7` (`float`) | `GraderResult(score=0.7, passed=None)` → `passed` derived |
| `GraderResult(...)` | used as-is |
| anything else | treated as grader error (see §6) |

## 4. Execution semantics

- Called **once per turn** where the `CodeMetric` is active.
- Called **synchronously**. Async graders are unsupported in v1 (see research.md §5).
- Wrapped in `try/except Exception` by the runner; exceptions never propagate (unless `fail_on_error=true`).
- Timeout: the grader is not independently timed out in v1. Graders should return quickly (< 1s typical). A grader that hangs will hang its test case — users should not write blocking-I/O graders.

## 5. Pass derivation

```
if result.passed is not None:
    passed = result.passed
elif metric.threshold is not None:
    passed = result.score >= metric.threshold
else:
    passed = result.score >= 0.5    # default gate when user provided no threshold
```

## 6. Error handling (FR-022)

If the grader raises any `Exception`:

- Captured as `MetricResult.error = f"{ExceptionClass.__name__}: {msg}"`.
- `MetricResult.score = 0.0`, `passed = False`.
- The turn's `passed` becomes `False` (standard pass rule still applies).
- Remaining turns continue to execute (per FR-008).
- Other metrics on the same turn are unaffected — exceptions are scoped to the one grader.
- If `fail_on_error=true`, the exception is still caught, but the **whole test case** is marked failed immediately and subsequent turns are not executed. Rare; opt-in.

## 7. Recording in the report

A grader's result is serialized into a regular `MetricResult`:

```json
{
  "metric_name": "program_equivalence",
  "kind": "code",                            // new literal value on MetricResult.kind (see data-model.md §7a)
  "score": 1.0,
  "threshold": 1.0,
  "passed": true,
  "scale": "0-1",
  "error": null,
  "retry_count": 0,
  "evaluation_time_ms": 3,
  "model_used": null,
  "reasoning": "program matches turn_program with #0 back-ref substitution"   // populated from GraderResult.reason
}
```

`kind: "code"` lets the dashboard Compare view and metric-trend filters distinguish grader results from NLP / RAG / G-Eval results — essential once benchmarks routinely mix them.

`details` (if provided) is written to a separate per-turn extension on `TurnResult.metric_results` (outside the strict `MetricResult` schema — Pydantic `ConfigDict(extra="forbid")` prevents dropping it, so the runner persists `details` on `TurnResult.grader_details: dict[str, Any]` for dashboards).

## 8. Example grader (for ConvFinQA `turn_program`)

```python
# my_benchmarks/convfinqa.py
from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult

def program_equivalence(ctx: GraderContext) -> GraderResult:
    """Return PASS if agent's tool-call DAG reproduces turn_program."""
    expected_program = ctx.turn_config.get("turn_program")
    if not expected_program:
        return GraderResult(score=0.0, passed=False, reason="turn_program missing in config")

    # User-land logic: parse expected_program, resolve #N back-refs using prior-turn results
    # from ctx.turn_config["_prior_turn_results"] (if the author wires them), then compare
    # against ctx.tool_invocations ...
    matches = _compare_program(expected_program, ctx.tool_invocations)
    return GraderResult(
        score=1.0 if matches else 0.0,
        passed=matches,
        reason="program matched" if matches else "program diverged",
        details={"expected": expected_program, "actual_tools": [t.name for t in ctx.tool_invocations]},
    )
```

## 9. Security (FR-026, A9)

- Graders run in-process with full Python privileges — identical trust to tool plugins and MCP servers that already execute in HoloDeck.
- Users MUST NOT load grader code from untrusted sources.
- HoloDeck docs MUST state this explicitly next to the `type: code` surface.

## 10. Versioning

`GraderContext` and `GraderResult` are additive-only going forward. New optional fields may be added; no existing field is ever renamed or removed. A grader written against v1 must continue to work against v2+.
