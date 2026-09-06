# Quickstart: Running a ConvFinQA-Shaped Multi-Turn Test Case

**Feature**: 032-multi-turn-test-cases
**Audience**: Benchmark author who wants to run their first 4-turn conversational test end-to-end against HoloDeck. Target: under 15 minutes (SC-001).

This walk-through uses one example lifted directly from `/Users/justinbarias/Documents/Git/python/justinbarias/data/convfinqa_dataset.json` (train item `Single_JKHY/2009/page_28.pdf-3`).

---

## 1. Install / activate

```bash
cd my-benchmark                    # any directory with its own agent.yaml
source .venv/bin/activate
holodeck --version                 # should print >= the 0.x that ships this feature
```

## 2. Author `agent.yaml`

```yaml
name: convfinqa-bench
description: "Runs ConvFinQA dialogues against a financial reasoning agent"

model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.0

instructions:
  inline: |
    You are a financial analyst. Use the provided tools (lookup, subtract, divide)
    to answer the user's questions about an SEC filing. Return numeric answers as
    plain numbers (no units, no commas).

tools:
  - name: subtract
    type: function
    description: "Compute a - b"
  - name: divide
    type: function
    description: "Compute a / b"
  - name: lookup
    type: function
    description: "Look up a scalar value from the document by description"

evaluations:
  metrics:
    # Standard deterministic — no LLM needed, cheap and exact
    - type: standard
      metric: numeric
      absolute_tolerance: 0.5            # tolerate ±0.5 on integer answers
      accept_percent: true               # let "14.14%" parse as 0.1414
      accept_thousands_separators: true

execution:
  parallel_test_cases: 4                 # run 4 multi-turn cases concurrently

test_cases:
  - name: "Single_JKHY_2009_page_28"
    turns:
      - input: "what is the net cash from operating activities in 2009?"
        ground_truth: "206588"
        expected_tools: ["lookup"]

      - input: "what about in 2008?"
        ground_truth: "181001"
        expected_tools: ["lookup"]

      - input: "what is the difference?"
        ground_truth: "25587"
        expected_tools:
          - name: "subtract"
            args:
              a: { fuzzy: "206588" }
              b: { fuzzy: "181001" }

      - input: "what percentage change does this represent?"
        ground_truth: "0.14136"
        expected_tools:
          - name: "divide"
            args:
              a: { fuzzy: "25587" }
              b: { fuzzy: "181001" }
        evaluations:
          - type: standard
            metric: numeric
            absolute_tolerance: 0.005     # tighter tolerance for the ratio
            accept_percent: true
```

**What the three new surfaces buy you**:

- `turns` makes the dialogue stateful — turn 2's "what about in 2008?" arrives after the agent has already seen turn 1's response.
- `expected_tools: [{name, args: {...}}]` asserts both name *and* args. Fuzzy matchers tolerate `206588` vs `"206,588"` vs `206588.0` drift.
- `metric: numeric` grades scalar answers honestly — BLEU/ROUGE on `"206588"` would be nonsensical.

## 3. Run

```bash
holodeck test
```

Expected CLI output (abridged):

```
Running 1 test case with parallel_test_cases=4
  ✓ Single_JKHY_2009_page_28 (4 turns)
    turn 0: ✓ numeric=1.0  tools=[lookup]
    turn 1: ✓ numeric=1.0  tools=[lookup]
    turn 2: ✓ numeric=1.0  tools=[subtract(a=206588, b=181001)]
    turn 3: ✓ numeric=1.0  tools=[divide(a=25587, b=181001)]

1/1 passed (100%)   4 turns passed, 0 failed   1.8s
```

If a turn fails, the report pinpoints exactly which one (SC-004):

```
  ✗ Single_JKHY_2009_page_28 (4 turns, 3 passed, 1 failed)
    turn 3: ✗ numeric=0.0  expected 0.14136, got 0.145 (outside absolute_tolerance=0.005)
```

## 4. Inspect the JSON report

```bash
holodeck test --output results/report.json --format json
```

`results/report.json` now contains a `turns` array on the multi-turn test result:

```json
{
  "test_name": "Single_JKHY_2009_page_28",
  "passed": true,
  "turns": [
    { "turn_index": 0, "input": "what is the net cash ...",
      "response": "206588", "ground_truth": "206588",
      "tool_invocations": [{"name": "lookup", "args": {"description": "2009 net cash from operating activities"}, "result": 206588}],
      "tools_matched": true,
      "metric_results": [{"metric_name": "numeric", "score": 1.0, "passed": true, ...}],
      "passed": true, "execution_time_ms": 420,
      "token_usage": {"prompt_tokens": 180, "completion_tokens": 8, "total_tokens": 188}
    },
    ...
  ],
  "token_usage": { "total_tokens": 1104 }
}
```

The test-case-level `token_usage.total_tokens` is the element-wise sum across turns (FR-007).

## 5. (Optional) Add a code grader for `turn_program` equivalence

The ConvFinQA dataset ships a symbolic `turn_program` per turn (e.g. `"subtract(206588, 181001), divide(#0, 181001)"`). No built-in can grade this — `#0` back-references require a graph walk. Write a code grader:

```python
# my_benchmarks.py
from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult

def program_equivalence(ctx: GraderContext) -> GraderResult:
    expected = ctx.turn_config.get("turn_program", "")
    actual_sequence = [f"{t.name}({','.join(map(str, t.args.values()))})"
                       for t in ctx.tool_invocations]
    actual = ", ".join(actual_sequence)
    matches = _compare_program(expected, actual, ctx.turn_index)
    return GraderResult(
        score=1.0 if matches else 0.0,
        passed=matches,
        reason=f"expected `{expected}`, got `{actual}`",
    )
```

Then wire it into turn 3 of `agent.yaml`:

```yaml
      - input: "what is the difference?"
        ground_truth: "25587"
        turn_program: "subtract(206588, 181001)"       # passed through via turn_config
        evaluations:
          - type: code
            grader: "my_benchmarks:program_equivalence"
```

Run again — HoloDeck imports `my_benchmarks` at load time (import error would surface *before* any agent call, per FR-025), runs your grader per turn, and records its result alongside the built-in `numeric` metric.

## 6. Back-compat sanity check (SC-002)

If you already have single-turn test cases in this `agent.yaml`:

```yaml
test_cases:
  - name: "greeting"
    input: "hello"
    ground_truth: "hi there"
    expected_tools: ["greet"]
```

...they run unchanged. The report for those entries does **not** include a `turns` array. No config edit required.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `ConfigError: test_cases[0] has both 'turns' and 'input'` | FR-001 — remove one. |
| `ConfigError: invalid regex ...` at load | Malformed pattern; fix the regex. |
| `ConfigError: cannot import 'my_benchmarks'` | Grader module not on PYTHONPATH; `pip install -e .` your benchmark repo. |
| Turn 2 asks about "2008" but agent replies about 2009 | Your backend may be ignoring conversation state. Verify with SK+Claude dual smoke per SC-010. |
| `parallel_test_cases=8` but test still runs serially | Check that `ExecutionConfig.parallel_test_cases` appears in the CLI > YAML > env resolution — e.g. pass `--parallel-test-cases 8`. |

## What's new at a glance

| Surface | Old behavior | New behavior |
|---------|--------------|--------------|
| `turns` | n/a | Stateful multi-turn dialogue. |
| `expected_tools` object form | strings only | `{name, args, count}` with literal/fuzzy/regex matchers. |
| `metric: equality` / `numeric` | not available | Built-in deterministic, no LLM. |
| `type: code` | not available | User-supplied Python grader, import-path referenced. |
| `parallel_test_cases` | always 1 | Concurrent test cases; turns inside a case still sequential. |
| `TestResult.turns` | absent | Per-turn detail in JSON report; dashboard renders as expandable rows. |
