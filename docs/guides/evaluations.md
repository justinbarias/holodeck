# Evaluations

HoloDeck grades your agent automatically: you declare **metrics** in `agent.yaml`, attach **test cases**, and run `holodeck test run`. Each metric scores the agent's response (0.0–1.0) and passes or fails against a threshold.

## Quick start

Add one deterministic `numeric` metric and one test case to `agent.yaml`, then run the suite. This case asserts the agent returns `60.94` within a 1% relative tolerance:

```yaml
# agent.yaml
evaluations:
  metrics:
    - type: standard
      metric: numeric
      relative_tolerance: 0.01

test_cases:
  - name: "Exercise price"
    input: "What was the weighted average exercise price per share in 2007?"
    ground_truth: "60.94"
```

```bash
holodeck test run agent.yaml
```

```
Test: "Exercise price"
Metrics:
  ✓ numeric: 1.00 (threshold: —)
Result: PASS

1 passed, 0 failed
```

`numeric` runs locally with **no LLM call** — fast, free, and reproducible. Swap in `metric: equality` for exact string answers, or `type: geval` for LLM-judged quality (see below).

## How it works

You attach `metrics` under `evaluations` and a list of `test_cases` (each with an `input` and optional `ground_truth`). When you run `holodeck test run agent.yaml`, HoloDeck invokes the agent on each test input, records its response and tool calls, then runs every enabled metric. Metrics fall into four families: **deterministic / NLP** (`type: standard` — no LLM, scores token overlap or compares numbers/strings), **DeepEval** (`type: geval` and `type: rag` — LLM-as-a-judge), **code graders** (`type: code` — your Python callable with full tool-call access), and **legacy Azure AI** metrics (deprecated). A metric **passes** when its score meets its `threshold`; a test case passes when all its metrics pass. The evaluation LLM (for `geval`/`rag`) is provider-agnostic — configure it via a `model:` block (see [Evaluation model configuration](#evaluation-model-configuration)).

## Choosing a metric

| Family | `type` | LLM needed? | Use for |
|--------|--------|-------------|---------|
| Deterministic | `standard` (`equality`, `numeric`) | No | Exact strings, numbers, slot-filling, classification labels |
| NLP overlap | `standard` (`f1_score`, `bleu`, `rouge`, `meteor`) | No | Token / n-gram similarity to a reference |
| DeepEval GEval | `geval` | Yes | Custom natural-language quality criteria |
| DeepEval RAG | `rag` | Yes | Hallucination / retrieval quality in RAG pipelines |
| Code grader | `code` | No | Tool-call assertions, domain logic over tool results |
| Legacy (deprecated) | `standard` (`groundedness`, etc.) | Yes | Migrate to `geval`/`rag` instead |

---

## Deterministic & NLP metrics (`type: standard`)

Standard metrics compare the agent response to `ground_truth` without an LLM call. They emit a `score` in `0.0`–`1.0` and use a `metric:` discriminator. Implemented in `src/holodeck/lib/evaluators/deterministic.py`.

### Numeric

Number compare with absolute and/or relative tolerance. Score is `1.0` if **either** tolerance passes, `0.0` otherwise. Requires `ground_truth` (must parse to a number).

```yaml
- type: standard
  metric: numeric
  absolute_tolerance: 0.5
  relative_tolerance: 0.0
  accept_percent: true
  accept_thousands_separators: true
```

| Flag | Default | Effect |
|------|---------|--------|
| `absolute_tolerance` | `1e-6` | Pass when `abs(actual - expected) <= absolute_tolerance` (inclusive) |
| `relative_tolerance` | `0.0` | Pass when `abs(actual - expected) <= relative_tolerance * abs(expected)` |
| `accept_percent` | `false` | Parse a trailing `%` as `/100` (so `"35%"` ↔ `0.35`) |
| `accept_thousands_separators` | `false` | Strip `,`, `_`, and NBSP before parsing |
| `response_path` | — | Extract a leaf from a JSON response envelope before parsing (e.g. `answer` for `{"answer": 60.94}`) |

**When to use**: financial / quantitative answers. Common pairing: enable both `accept_percent` and `accept_thousands_separators` for filings-style data (`"1,234.56"`, `"35.8%"`). See `sample/financial-assistant/claude/agent.yaml` for a working example that grades the `answer` leaf of a structured-output envelope via `response_path`.

### Equality

Strict string equality with optional normalization. Score is `1.0` on match, `0.0` otherwise. Requires `ground_truth`.

```yaml
- type: standard
  metric: equality
  case_insensitive: true
  strip_whitespace: true
  strip_punctuation: false
```

| Flag | Effect (all default `false`) |
|------|--------|
| `case_insensitive` | Lowercase both sides before compare |
| `strip_whitespace` | Collapse whitespace runs and trim before compare |
| `strip_punctuation` | Remove `string.punctuation` before compare |

**When to use**: closed-form answers where the agent must return an exact token (slot-filling, classification labels, "yes"/"no", enum values). Setting a `model` override on this metric is a no-op (a warning is logged).

### NLP overlap metrics

Token / n-gram overlap against `ground_truth`. All score `0.0`–`1.0` (higher is better).

| `metric` | Measures | Use for |
|----------|----------|---------|
| `f1_score` | Token-level precision/recall balance | Exact word matching matters |
| `bleu` | N-gram similarity; penalizes brevity | Translation, paraphrase |
| `rouge` | Recall of n-grams; reference coverage | Summarization |
| `meteor` | N-gram match with synonyms + word order | Translation/paraphrase with synonyms |

```yaml
- type: standard
  metric: f1_score
  threshold: 0.8
```

---

## DeepEval metrics (`type: geval`, `type: rag`)

DeepEval provides LLM-as-a-judge evaluation. It works with any configured provider (Ollama for free local eval, or OpenAI / Anthropic / Azure OpenAI). Set `temperature: 0.0` for deterministic scoring.

### GEval — custom criteria

GEval uses the G-Eval algorithm with chain-of-thought prompting to score responses against natural-language criteria.

```yaml
- type: geval
  name: "Technical Accuracy"
  criteria: |
    Evaluate whether the response provides accurate technical information
    that correctly addresses the user's question.
  evaluation_steps:              # Optional: auto-generated if omitted
    - "Check if the response directly addresses the user's question"
    - "Verify technical accuracy of any code or commands provided"
  evaluation_params:             # Which test-case fields to feed in
    - actual_output
    - input
  threshold: 0.8
  strict_mode: false             # Binary scoring (1.0 / 0.0) when true
  model:                         # Optional per-metric model override
    provider: openai
    name: gpt-4
```

**GEval config fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | `"geval"` |
| `name` | string | Yes | Metric name (e.g. "Coherence") |
| `criteria` | string | Yes | Natural-language evaluation criteria |
| `evaluation_steps` | list | No | Step-by-step instructions (auto-generated if omitted) |
| `evaluation_params` | list | No | Test-case fields to use (default `["actual_output"]`) |
| `threshold` | float | No | Minimum passing score (0–1) |
| `strict_mode` | bool | No | Binary scoring when `true` (default `false`) |
| `enabled` | bool | No | Default `true` |
| `fail_on_error` | bool | No | Fail test on evaluation error (default `false`) |
| `model` | object | No | Per-metric model override |

**Evaluation parameters**:

| Parameter | Description | When to use |
|-----------|-------------|-------------|
| `actual_output` | Agent's response | Always (required) |
| `input` | User's query | When relevance to query matters |
| `expected_output` | Ground-truth answer | When comparing to expected response |
| `context` | Additional context provided | When evaluating context usage |
| `retrieval_context` | Retrieved documents | For RAG evaluation |

### RAG metrics

RAG metrics evaluate responses generated from retrieved context.

```yaml
- type: rag
  metric_type: faithfulness
  threshold: 0.8
  include_reason: true           # Include reasoning in results
  model:                         # Optional per-metric override
    provider: openai
    name: gpt-4
```

| `metric_type` | Detects / measures | Required params |
|---------------|--------------------|-----------------|
| `faithfulness` | Hallucinations (claims unsupported by context) | input, actual_output, retrieval_context |
| `answer_relevancy` | Response relevance to query | input, actual_output |
| `contextual_relevancy` | Relevance of retrieved chunks | input, actual_output, retrieval_context |
| `contextual_precision` | Chunk ranking quality (most-relevant ranked highest) | input, actual_output, expected_output, retrieval_context |
| `contextual_recall` | Retrieval completeness for the expected answer | input, actual_output, expected_output, retrieval_context |

The test runner automatically extracts `retrieval_context` from vectorstore tool results, or you can supply it per test case.

---

## Code graders (`type: code`)

Code graders are user-supplied Python callables that score a turn's behavior with the **full per-turn context**: input, response, ground truth, **all tool invocations (args + results)**, and a free-form `turn_config`. They run in-process — free, deterministic, and as fast as your Python.

**When to use**: verifying tool-call ordering or argument shapes, domain logic over actual tool-returned values (program equivalence, schema-aware diff), or anything that doesn't reduce to a single LLM-judgeable string.

```yaml
- type: code
  grader: "graders.exact_tool_call:called_subtract_with"
  threshold: 0.7         # Optional — used when grader returns a bare float
  enabled: true          # Optional — default true
  fail_on_error: false   # Optional — true = grader exception fails the case
  name: "Turn Program"   # Optional — defaults to the callable name
```

The `grader` is a `"module.path:callable"` reference resolved at config-load time (a missing module / attribute / non-callable raises `ConfigError` *before* any agent runs). The module must be importable from the working directory (e.g. a `graders/` package next to `agent.yaml`).

### Grader contract

```python
from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult

def my_grader(ctx: GraderContext) -> GraderResult | bool | float:
    ...
```

`GraderContext` (frozen dataclass) fields: `turn_input`, `agent_response`, `ground_truth`, `tool_invocations` (immutable tuple; each exposes `name`, `args`, `result`, `bytes`, `duration_ms`, `error`), `retrieval_context`, `turn_index`, `test_case_name`, `turn_config`.

`GraderResult` fields: `score` (`float`, normalized to `[0,1]`), `passed` (`bool | None` — `None` defers to `threshold`, or `>= 0.5` if unset), `reason` (`str | None`), `details` (`dict | None`).

**Shortcut returns**: `True` → `score=1.0, passed=True`; `False` → `score=0.0, passed=False`; bare `float`/`int` → `score=value, passed=None` (caller derives `passed` from `threshold`).

### Example

```python
# graders/exact_tool_call.py
from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult

def called_subtract_with(ctx: GraderContext) -> GraderResult:
    expected_a = ctx.turn_config.get("expected_a")
    expected_b = ctx.turn_config.get("expected_b")
    for inv in ctx.tool_invocations:
        if inv.name == "subtract" and inv.args.get("a") == expected_a and inv.args.get("b") == expected_b:
            return GraderResult(score=1.0, passed=True, reason="exact match")
    return GraderResult(score=0.0, passed=False, reason="no matching subtract call")
```

A production-style example lives at `sample/financial-assistant/claude/graders/turn_program_equivalence.py`, which walks ConvFinQA `turn_program` strings against the actual `subtract` / `divide` tool calls.

---

## Test cases

### Per-test-case metric overrides

A test case can specify its own `evaluations:` list, overriding the agent-wide metrics for that case:

```yaml
test_cases:
  - name: "Fact check test"
    input: "What's our company's founding date?"
    expected_tools: [search_kb]
    ground_truth: "Founded in 2010"
    evaluations:
      - type: rag
        metric_type: faithfulness
        threshold: 0.85
      - type: rag
        metric_type: answer_relevancy
        threshold: 0.7
```

### Multi-turn test cases

A test case is **multi-turn** when it carries a `turns:` list instead of a top-level `input` / `ground_truth` / `expected_tools`. Each turn drives one exchange through the same agent session, so retrieval context, history, and tool state carry forward.

```yaml
test_cases:
  - name: "ConvFinQA · MRO 2007"
    turns:
      - input: "What was the weighted average exercise price per share in 2007?"
        ground_truth: "60.94"
        turn_config:
          turn_program: "60.94"
      - input: "And in 2005?"
        ground_truth: "25.14"
        turn_config:
          turn_program: "25.14"
      - input: "What was the change over the years?"
        ground_truth: "35.8"
        turn_config:
          turn_program: "subtract(60.94, 25.14)"
        expected_tools:
          - name: subtract
            args:
              a: { fuzzy: "60.94" }
              b: { fuzzy: "25.14" }
        evaluations:
          - type: code
            grader: "graders.turn_program_equivalence:turn_program_equivalence"
```

**Per-turn fields** (`holodeck.models.test_case.Turn`):

| Field | Type | Notes |
|-------|------|-------|
| `input` | string | Required, non-empty |
| `ground_truth` | string | Optional expected reply for this turn |
| `expected_tools` | list[str \| ExpectedTool] | Per-turn tool assertions; bare strings → `ExpectedTool(name=..., count=1)` |
| `files` | list[FileInput] | Up to 10 multimodal inputs per turn |
| `retrieval_context` | list[str] | RAG context for this turn |
| `evaluations` | list[metric] | Per-turn metric overrides (same `type` discriminators) |
| `turn_config` | dict | Free-form payload passed to code graders as `ctx.turn_config` |

**Mutual exclusivity**: when `turns:` is present, the top-level `input`, `ground_truth`, `expected_tools`, `files`, and `retrieval_context` keys must not be set.

**Argument matchers** in `expected_tools[*].args`:

| YAML shape | Matcher | Behavior |
|------------|---------|----------|
| `key: 42` (scalar/list/dict) | `LiteralMatcher` | Exact equality (int↔float numeric equivalence) |
| `key: { fuzzy: "60.94" }` | `FuzzyMatcher` | Case / whitespace / separator-tolerant, numeric-aware |
| `key: { regex: "^\\d+$" }` | `RegexMatcher` | Anchored full-match against `str(actual)` |

### External test-case files (`test_cases_file`)

For large or shared suites, lift `test_cases:` out of `agent.yaml` and reference an external file:

```yaml
# agent.yaml
evaluations:
  metrics:
    - type: standard
      metric: numeric
      relative_tolerance: 0.01

test_cases_file: data/convfinqa_subset.yaml
```

The referenced file may be either a top-level **list** of cases or a **dict** with a `test_cases:` key. Resolution rules (`src/holodeck/config/loader.py:_resolve_test_cases_file`):

- Resolved relative to the directory containing `agent.yaml` (absolute paths also work).
- Parsed as YAML with the same `${VAR}` environment-variable substitution applied to `agent.yaml`.
- Mutually exclusive with an inline `test_cases:` block — specifying both raises `ConfigError` at load time.
- Missing files, parse errors, and non-list/dict payloads raise `ConfigError` before any test runs.

**When to use**: large suites (hundreds of cases), generated fixtures, or sharing one corpus across multiple agent variants.

---

## Evaluation model configuration

LLM-backed metrics (`geval`, `rag`) need an evaluation model. It is **provider-agnostic** — configure it with the same `model:` block shape used elsewhere, at any of three levels (highest priority first):

1. **Per-metric** — a `model:` on the metric itself, used for that metric only.
2. **Evaluation-wide** — `evaluations.model`, the default for all metrics without an override.
3. **Agent model** — falls back to the agent's top-level `model` if neither above is set.

```yaml
evaluations:
  model:                          # Level 2: default for all metrics
    provider: ollama              # Free local inference
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    - type: rag
      metric_type: faithfulness
      threshold: 0.9
      model:                      # Level 1: override for this critical metric
        provider: openai
        name: gpt-4
    - type: geval
      name: "Coherence"
      criteria: "Evaluate response clarity."
      threshold: 0.75             # Uses the Level 2 model above
```

**Model fields**: `provider` (`ollama` / `openai` / `azure_openai` / `anthropic`) and `name` are required; `temperature` (recommend `0.0`), `max_tokens`, and `top_p` are optional. Backends route automatically: `openai` / `azure_openai` use the [OpenAI Agents backend](openai-backend.md); `anthropic` / `ollama` use the [Claude backend](claude-backend.md). The eval model is independent of which backend runs the agent.

### Cost optimization

- Default to free local Ollama; override to a paid model only for critical metrics (per-metric `model:`).
- Prefer deterministic / NLP `type: standard` metrics where possible — they make **no LLM calls** at all.
- For LLM metrics, a smaller model (e.g. `gpt-4o-mini`) is often enough and far cheaper than `gpt-4`.

---

## Metric options reference

These fields apply to most metric types:

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `threshold` | float (0–1) | none | Minimum score to pass. Without it, the metric is informational. |
| `enabled` | bool | `true` | Temporarily disable a metric without removing it. |
| `fail_on_error` | bool | `false` | Whether an evaluation error fails the test (soft failure by default). |
| `model` | object | inherited | Per-metric LLM override (LLM-backed metrics only). |

---

## Legacy AI metrics (deprecated)

!!! warning "Deprecated"
    Azure AI-based metrics (`groundedness`, `relevance`, `coherence`, `safety`) are deprecated and will be removed in a future version. They remain supported for backwards compatibility only.

| Legacy metric | Migrate to |
|---------------|------------|
| `groundedness` | `type: rag`, `metric_type: faithfulness` |
| `relevance` | `type: rag`, `metric_type: answer_relevancy` |
| `coherence` | `type: geval` with custom `criteria` |
| `safety` | `type: geval` with safety `criteria` |

```yaml
# Before (deprecated)            # After
- type: standard                 - type: rag
  metric: groundedness             metric_type: faithfulness
  threshold: 0.8                   threshold: 0.8
```

---

## Test execution

`holodeck test run agent.yaml` (or the shorthand `holodeck test agent.yaml`) for each test case:

1. Executes the agent with the test input.
2. Records which tools were called and validates them against `expected_tools`.
3. Runs each enabled metric and compares against thresholds.
4. Reports pass/fail per metric, then an overall `N passed, M failed` summary.

The command exits non-zero if any test fails (CI-friendly). Use `holodeck test view` to launch the dashboard over past runs — see the [Dashboard guide](dashboard.md).

```
Test: "Password Reset"
Input: "How do I reset my password?"
Tools called: [search_kb] ✓
Metrics:
  ✓ Faithfulness: 0.92 (threshold: 0.8)
  ✓ Answer Relevancy: 0.88 (threshold: 0.75)
  ✓ F1 Score: 0.81 (threshold: 0.7)
Result: PASS
```

---

## Troubleshooting

**Error: "invalid metric type"** — valid `type` values are `geval`, `rag`, `standard`, `code`. For `standard`: `f1_score`, `bleu`, `rouge`, `meteor`, `equality`, `numeric`. For `rag`: `faithfulness`, `answer_relevancy`, `contextual_relevancy`, `contextual_precision`, `contextual_recall`. For `code`, provide a `grader: "module.path:callable"`.

**Metric always fails** — check the evaluation model is working; try without a threshold first; for RAG metrics ensure the required parameters are available.

**LLM evaluation too slow** — use a local Ollama model, a faster model (`gpt-4o-mini`), or switch to NLP / deterministic metrics (free and fast).

**Inconsistent evaluation results** — set `temperature: 0.0`; use a more powerful model for complex criteria; add `evaluation_steps` to GEval for more stable scoring.

**RAG metric missing `retrieval_context`** — ensure your agent uses a vectorstore tool (the runner auto-extracts context from tool results), or supply manual `retrieval_context` in the test case.

---

## Next steps

- [Optimizer guide](optimizer.md) — `holodeck test optimize` automates the tune-and-rerun loop as a coordinate-descent optimizer over your eval suite.
- [Agent Configuration Guide](agent-configuration.md) — full `agent.yaml` reference.
- [Dashboard guide](dashboard.md) — visualize eval-run history.
- [API Reference](../api/evaluators.md) — evaluator class details.
