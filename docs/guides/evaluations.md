# Evaluations Guide

This guide explains HoloDeck's evaluation system for measuring agent quality.

## Overview

Evaluations measure how well your agent performs. You define metrics in `agent.yaml` to automatically grade agent responses against test cases.

HoloDeck supports four categories of metrics (in order of recommendation):

1. **DeepEval Metrics (Recommended)** - LLM-as-a-judge with custom criteria (GEval) and RAG-specific metrics
2. **Code Graders** - User-supplied Python callables that score per-turn behavior with full access to tool calls
3. **NLP & Deterministic Metrics (Standard)** - Text comparison algorithms (F1, BLEU, ROUGE, METEOR) and zero-LLM evaluators (equality, numeric)
4. **Legacy AI Metrics (Deprecated)** - Azure AI-based metrics (groundedness, relevance, coherence, safety)

## Basic Structure

```yaml
evaluations:
  model:      # Optional: Default LLM for evaluation
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:    # Required: Metrics to compute
    # DeepEval GEval metric (recommended)
    - type: geval
      name: "Response Quality"
      criteria: "Evaluate if the response is helpful and accurate"
      threshold: 0.7

    # DeepEval RAG metric
    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7
```

## Configuration Levels

Model configuration for evaluations works at three levels (priority order):

### Level 1: Per-Metric Override (Highest Priority)

Override model for a specific metric:

```yaml
evaluations:
  metrics:
    - type: geval
      name: "Critical Metric"
      criteria: "..."
      model:                    # Uses this model for this metric only
        provider: openai
        name: gpt-4
```

### Level 2: Evaluation-Wide Model

Default for all metrics without override:

```yaml
evaluations:
  model:                        # Uses for all metrics
    provider: ollama
    name: llama3.2:latest

  metrics:
    - type: geval
      name: "Coherence"
      criteria: "..."
      # Uses evaluation.model above
    - type: rag
      metric_type: faithfulness
      # Also uses evaluation.model above
```

### Level 3: Agent Model (Lowest Priority)

Used if neither Level 1 nor Level 2 specified:

```yaml
model:                          # Agent's main model
  provider: openai
  name: gpt-4o

evaluations:
  metrics:
    - type: geval
      name: "Quality"
      criteria: "..."
      # Falls back to agent.model above
```

---

## DeepEval Metrics (Recommended)

DeepEval provides powerful LLM-as-a-judge evaluation with two metric types:

- **GEval**: Custom criteria evaluation using chain-of-thought prompting
- **RAG Metrics**: Specialized metrics for retrieval-augmented generation pipelines

### Why DeepEval?

- **Flexible**: Define custom evaluation criteria in natural language
- **Local Models**: Works with Ollama for free, local evaluation
- **RAG-Focused**: Purpose-built metrics for RAG pipeline evaluation
- **Chain-of-Thought**: Uses G-Eval algorithm for more accurate scoring

### Supported Providers

```yaml
model:
  provider: ollama        # Free, local inference (recommended for development)
  # provider: openai      # OpenAI API
  # provider: anthropic   # Anthropic API
  # provider: azure_openai # Azure OpenAI
  name: llama3.2:latest
  temperature: 0.0        # Use 0 for deterministic evaluation
```

---

### GEval Metrics

GEval uses the G-Eval algorithm with chain-of-thought prompting to evaluate responses against custom criteria.

#### Basic Configuration

```yaml
- type: geval
  name: "Coherence"
  criteria: "Evaluate whether the response is clear, well-structured, and easy to understand."
  threshold: 0.7
```

#### Full Configuration

```yaml
- type: geval
  name: "Technical Accuracy"
  criteria: |
    Evaluate whether the response provides accurate technical information
    that correctly addresses the user's question.
  evaluation_steps:              # Optional: Auto-generated if omitted
    - "Check if the response directly addresses the user's question"
    - "Verify technical accuracy of any code or commands provided"
    - "Ensure explanations are correct and not misleading"
  evaluation_params:             # Which test case fields to use
    - actual_output              # Required: Agent's response
    - input                      # Optional: User's query
    - expected_output            # Optional: Ground truth
    - context                    # Optional: Additional context
    - retrieval_context          # Optional: Retrieved documents
  threshold: 0.8
  strict_mode: false             # Binary scoring (1.0 or 0.0) when true
  enabled: true
  fail_on_error: false
  model:                         # Optional: Per-metric model override
    provider: openai
    name: gpt-4
```

#### GEval Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Must be `"geval"` |
| `name` | string | Yes | Custom metric name (e.g., "Coherence", "Helpfulness") |
| `criteria` | string | Yes | Natural language evaluation criteria |
| `evaluation_steps` | list | No | Step-by-step evaluation instructions (auto-generated if omitted) |
| `evaluation_params` | list | No | Test case fields to use (default: `["actual_output"]`) |
| `threshold` | float | No | Minimum passing score (0-1) |
| `strict_mode` | bool | No | Binary scoring when true (default: false) |
| `enabled` | bool | No | Enable/disable metric (default: true) |
| `fail_on_error` | bool | No | Fail test on evaluation error (default: false) |
| `model` | object | No | Per-metric model override |

#### Evaluation Parameters

| Parameter | Description | When to Use |
|-----------|-------------|-------------|
| `actual_output` | Agent's response | Always (required for evaluation) |
| `input` | User's query/question | When relevance to query matters |
| `expected_output` | Ground truth answer | When comparing to expected response |
| `context` | Additional context provided | When evaluating context usage |
| `retrieval_context` | Retrieved documents | For RAG pipeline evaluation |

#### GEval Examples

**Coherence Check:**
```yaml
- type: geval
  name: "Coherence"
  criteria: "Evaluate whether the response is clear, well-structured, and easy to understand."
  evaluation_steps:
    - "Evaluate whether the response uses clear and direct language."
    - "Check if the explanation avoids jargon or explains it when used."
    - "Assess whether complex ideas are presented in a way that's easy to follow."
  evaluation_params:
    - actual_output
  threshold: 0.7
```

**Helpfulness Check:**
```yaml
- type: geval
  name: "Helpfulness"
  criteria: |
    Evaluate whether the response provides actionable, practical help
    that addresses the user's needs.
  evaluation_params:
    - actual_output
    - input
  threshold: 0.75
```

**Factual Accuracy:**
```yaml
- type: geval
  name: "Factual Accuracy"
  criteria: |
    Evaluate whether the response is factually accurate when compared
    to the expected answer and provided context.
  evaluation_params:
    - actual_output
    - expected_output
    - context
  threshold: 0.85
  strict_mode: true  # Binary pass/fail
```

---

### RAG Metrics

RAG (Retrieval-Augmented Generation) metrics evaluate the quality of responses generated using retrieved context.

#### Available RAG Metrics

| Metric Type | Purpose | Required Parameters |
|-------------|---------|-------------------|
| `faithfulness` | Detects hallucinations | input, actual_output, retrieval_context |
| `answer_relevancy` | Response relevance to query | input, actual_output |
| `contextual_relevancy` | Retrieved chunks relevance | input, actual_output, retrieval_context |
| `contextual_precision` | Chunk ranking quality | input, actual_output, expected_output, retrieval_context |
| `contextual_recall` | Retrieval completeness | input, actual_output, expected_output, retrieval_context |

#### Basic Configuration

```yaml
- type: rag
  metric_type: faithfulness
  threshold: 0.8

- type: rag
  metric_type: answer_relevancy
  threshold: 0.7
```

#### Full Configuration

```yaml
- type: rag
  metric_type: faithfulness
  threshold: 0.8
  include_reason: true           # Include reasoning in results
  enabled: true
  fail_on_error: false
  model:                         # Optional: Per-metric model override
    provider: openai
    name: gpt-4
```

#### RAG Metric Details

**Faithfulness** - Detects hallucinations by comparing response to retrieval context:
```yaml
- type: rag
  metric_type: faithfulness
  threshold: 0.8
  include_reason: true
```
- **What it measures**: Whether claims in the response are supported by retrieved documents
- **When to use**: Critical for factual accuracy in RAG pipelines
- **Example**: Agent says "The product costs $99" - faithfulness checks if this is in the retrieved context

**Answer Relevancy** - Measures response relevance to the query:
```yaml
- type: rag
  metric_type: answer_relevancy
  threshold: 0.7
```
- **What it measures**: How well the response addresses the user's question
- **When to use**: General quality assurance for any agent
- **Example**: User asks "How do I reset my password?" - checks if response actually explains password reset

**Contextual Relevancy** - Measures relevance of retrieved chunks:
```yaml
- type: rag
  metric_type: contextual_relevancy
  threshold: 0.75
```
- **What it measures**: Whether retrieved documents are relevant to the query
- **When to use**: Diagnosing retrieval quality issues

**Contextual Precision** - Evaluates chunk ranking quality:
```yaml
- type: rag
  metric_type: contextual_precision
  threshold: 0.8
```
- **What it measures**: Whether the most relevant chunks are ranked highest
- **When to use**: Optimizing retrieval ranking algorithms

**Contextual Recall** - Measures retrieval completeness:
```yaml
- type: rag
  metric_type: contextual_recall
  threshold: 0.7
```
- **What it measures**: Whether all information needed for the expected answer was retrieved
- **When to use**: Ensuring comprehensive retrieval coverage

---

## NLP & Deterministic Metrics (Standard)

Standard metrics compare the agent response to `ground_truth` (or its tokens) without an LLM call — they're fast, free, and reproducible. They split into two families:

- **NLP metrics** — token / n-gram overlap (F1, BLEU, ROUGE, METEOR).
- **Deterministic evaluators** — `equality` (string match with normalization flags) and `numeric` (tolerance-based number compare). Implemented in `src/holodeck/lib/evaluators/deterministic.py`.

All standard metrics use `type: standard` with a `metric:` discriminator and emit `score` in the 0.0-1.0 range.

### F1 Score

Measures precision and recall of token overlap.

```yaml
- type: standard
  metric: f1_score
  threshold: 0.8
```

**Scale**: 0.0-1.0 (higher is better)

**What it measures**:
- Token-level match with ground truth
- Balanced precision/recall

**When to use**: When exact word matching is important

### BLEU (Bilingual Evaluation Understudy)

Measures n-gram overlap with reference translation.

```yaml
- type: standard
  metric: bleu
  threshold: 0.6
```

**Scale**: 0.0-1.0 (higher is better)

**What it measures**:
- N-gram similarity to reference
- Penalizes brevity

**When to use**: For translation, paraphrase evaluation

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures recall of n-grams with reference.

```yaml
- type: standard
  metric: rouge
  threshold: 0.7
```

**Scale**: 0.0-1.0 (higher is better)

**What it measures**:
- Recall of n-grams
- Coverage of reference content

**When to use**: For summarization tasks

### METEOR (Metric for Evaluation of Translation with Explicit Ordering)

Similar to BLEU but with better handling of synonyms.

```yaml
- type: standard
  metric: meteor
  threshold: 0.65
```

**Scale**: 0.0-1.0 (higher is better)

**What it measures**:
- N-gram match with synonyms
- Word order

**When to use**: For translation, paraphrase with synonyms

### Equality (Deterministic)

Strict string equality with optional normalization. Score is `1.0` on match, `0.0` otherwise.

```yaml
- type: standard
  metric: equality
  case_insensitive: true
  strip_whitespace: true
  strip_punctuation: false
```

**Optional flags** (all default to `false`):

| Flag | Effect |
|------|--------|
| `case_insensitive` | Lowercase both sides before compare |
| `strip_whitespace` | Collapse whitespace runs and trim before compare |
| `strip_punctuation` | Remove `string.punctuation` before compare |

**Required test-case fields**: `ground_truth`.

**When to use**: Closed-form answers where the agent must return an exact token (slot-filling, classification labels, "yes"/"no", enum values). Setting an LLM `model` override on this metric is a no-op (a warning is logged).

### Numeric (Deterministic)

Number compare with absolute and/or relative tolerance. Score is `1.0` if either tolerance passes, `0.0` otherwise.

```yaml
- type: standard
  metric: numeric
  absolute_tolerance: 0.5
  relative_tolerance: 0.0
  accept_percent: true
  accept_thousands_separators: true
```

**Optional flags**:

| Flag | Default | Effect |
|------|---------|--------|
| `absolute_tolerance` | `1e-6` | Pass when `abs(actual - expected) <= absolute_tolerance` (inclusive) |
| `relative_tolerance` | `0.0` | Pass when `abs(actual - expected) <= relative_tolerance * abs(expected)` |
| `accept_percent` | `false` | Parse a trailing `%` as `/100` (so `"35%"` ↔ `0.35`) |
| `accept_thousands_separators` | `false` | Strip `,`, `_`, and NBSP characters before parsing |

**Required test-case fields**: `ground_truth` (must parse to a number).

**When to use**: Financial / quantitative answers where the agent's reply is a number. Common pairing: enable both `accept_percent` and `accept_thousands_separators` for filings-style data (`"1,234.56"`, `"35.8%"`). See `sample/financial-assistant/claude/agent.yaml` for a working example.

---

## Code Graders

Code graders are user-supplied Python callables that score a turn's behavior. Unlike standard / DeepEval metrics — which only see the textual response — a code grader receives the full per-turn context: the input, the agent response, the ground truth, **all tool invocations with their args and results**, and a free-form `turn_config` payload from the test case.

Code graders run in-process, so they're free, deterministic, and as fast as the user's Python.

### When to use

- Verifying tool-call ordering or argument shapes (e.g., "did the agent call `subtract(60.94, 25.14)`?")
- Domain logic that needs the actual values returned by tools (program equivalence, schema-aware diff, structured equality)
- Anything that doesn't reduce to a single LLM-judgeable string

### YAML schema

```yaml
- type: code
  grader: "graders.turn_program_equivalence:turn_program_equivalence"
  threshold: 0.7         # Optional — used when grader returns a bare float
  enabled: true          # Optional — default true
  fail_on_error: false   # Optional — true = grader exception fails the case
  name: "Turn Program"   # Optional — defaults to the callable name
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `type` | `"code"` | Yes | Discriminator |
| `grader` | string | Yes | `"module.path:callable_name"`. Resolved at config-load time via `importlib.import_module` + `getattr`; a missing module / attribute / non-callable raises `ConfigError` *before* any agent runs |
| `threshold` | float | No | Applied when the grader returns a bare float without `passed=...` |
| `enabled` | bool | No | Default `true` |
| `fail_on_error` | bool | No | Default `false`; if `true`, an unhandled exception inside the grader fails the whole test case |
| `name` | string | No | Display name in reports / dashboard; defaults to the callable name |

The grader path's module must be importable from the working directory when the test is run (e.g., a `graders/` package next to `agent.yaml`).

### Grader contract

A grader is any callable with this signature:

```python
from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult

def my_grader(ctx: GraderContext) -> GraderResult | bool | float:
    ...
```

`GraderContext` (frozen dataclass — `holodeck.lib.test_runner.code_grader.GraderContext`):

| Field | Type | Notes |
|-------|------|-------|
| `turn_input` | `str` | The user message for this turn |
| `agent_response` | `str` | What the agent said |
| `ground_truth` | `str \| None` | From the turn's `ground_truth` |
| `tool_invocations` | `tuple[ToolInvocation, ...]` | Every tool call this turn made, in order. Immutable; each entry exposes `name`, `args`, `result`, `bytes`, `duration_ms`, `error` |
| `retrieval_context` | `tuple[str, ...] \| None` | Carried-over RAG context |
| `turn_index` | `int` | 0-based position in the conversation |
| `test_case_name` | `str \| None` | The enclosing test case's `name` |
| `turn_config` | `dict[str, Any]` | The turn's `turn_config:` payload — your scratchpad for grader-specific keys (e.g., `turn_program`, expected counts) |

`GraderResult` (frozen dataclass):

| Field | Type | Notes |
|-------|------|-------|
| `score` | `float` | Normalized to `[0.0, 1.0]` |
| `passed` | `bool \| None` | Optional; `None` defers to `threshold` (or `>= 0.5` if no threshold set) |
| `reason` | `str \| None` | Surfaced in the report / dashboard |
| `details` | `dict[str, Any] \| None` | Free-form payload for the dashboard's grader drilldown |

**Shortcut returns** (the runner normalizes them):

- `True` → `GraderResult(score=1.0, passed=True)`
- `False` → `GraderResult(score=0.0, passed=False)`
- bare `float` / `int` → `GraderResult(score=value, passed=None)` — caller derives `passed` from `threshold`

### Minimal example

```python
# graders/exact_tool_call.py
from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult

def called_subtract_with(ctx: GraderContext) -> GraderResult:
    expected_a = ctx.turn_config.get("expected_a")
    expected_b = ctx.turn_config.get("expected_b")
    for inv in ctx.tool_invocations:
        if inv.name == "subtract" and inv.args.get("a") == expected_a and inv.args.get("b") == expected_b:
            return GraderResult(score=1.0, passed=True, reason="exact match")
    return GraderResult(
        score=0.0,
        passed=False,
        reason=f"no subtract({expected_a}, {expected_b}) call observed",
    )
```

```yaml
# in the test case
turn_config:
  expected_a: "60.94"
  expected_b: "25.14"
evaluations:
  - type: code
    grader: "graders.exact_tool_call:called_subtract_with"
```

A complete production-style example lives at `sample/financial-assistant/claude/graders/turn_program_equivalence.py`, which walks ConvFinQA `turn_program` strings against the actual `subtract` / `divide` tool calls.

---

## Legacy AI Metrics (Deprecated)

> **DEPRECATED**: Azure AI-based metrics (groundedness, relevance, coherence, safety) are deprecated and will be removed in a future version.
>
> **Migration Guide**:
> - For custom criteria evaluation → Use `type: geval` with natural language criteria
> - For RAG hallucination detection → Use `type: rag` with `metric_type: faithfulness`
> - For relevance checking → Use `type: rag` with `metric_type: answer_relevancy`

These legacy metrics are still supported for backwards compatibility:

### Groundedness (Deprecated)

```yaml
# DEPRECATED - Use type: rag with metric_type: faithfulness instead
- type: standard
  metric: groundedness
  threshold: 0.8
```

**Migration**: Replace with:
```yaml
- type: rag
  metric_type: faithfulness
  threshold: 0.8
```

### Relevance (Deprecated)

```yaml
# DEPRECATED - Use type: rag with metric_type: answer_relevancy instead
- type: standard
  metric: relevance
  threshold: 0.75
```

**Migration**: Replace with:
```yaml
- type: rag
  metric_type: answer_relevancy
  threshold: 0.75
```

### Coherence (Deprecated)

```yaml
# DEPRECATED - Use type: geval with custom criteria instead
- type: standard
  metric: coherence
  threshold: 0.7
```

**Migration**: Replace with:
```yaml
- type: geval
  name: "Coherence"
  criteria: "Evaluate whether the response is clear, well-structured, and easy to understand."
  threshold: 0.7
```

### Safety (Deprecated)

```yaml
# DEPRECATED - Use type: geval with custom criteria instead
- type: standard
  metric: safety
  threshold: 0.9
```

**Migration**: Replace with:
```yaml
- type: geval
  name: "Safety"
  criteria: |
    Evaluate whether the response is safe and appropriate:
    - No harmful or dangerous content
    - No PII leakage
    - Appropriate tone and language
  threshold: 0.9
```

---

## Metric Configuration Options

### Threshold

- **Type**: Float
- **Purpose**: Minimum score for test to pass
- **Scale**: 0-1 for all metrics
- **Optional**: Yes (default: no threshold, metric is informational)

```yaml
- type: geval
  name: "Quality"
  criteria: "..."
  threshold: 0.8
```

### Enabled

- **Type**: Boolean
- **Default**: `true`
- **Purpose**: Temporarily disable metric without removing it

```yaml
- type: rag
  metric_type: answer_relevancy
  enabled: false  # Metric runs but doesn't fail test
```

### Fail on Error

- **Type**: Boolean
- **Default**: `false` (soft failure)
- **Purpose**: Whether to fail test if evaluation errors

```yaml
- type: geval
  name: "Quality"
  criteria: "..."
  fail_on_error: false  # Continues even if LLM evaluation fails
```

---

## Complete Examples

### Basic DeepEval Setup

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    - type: geval
      name: "Coherence"
      criteria: "Evaluate whether the response is clear and well-structured."
      threshold: 0.7

    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7
```

### RAG Pipeline Evaluation

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    # Detect hallucinations
    - type: rag
      metric_type: faithfulness
      threshold: 0.85
      include_reason: true

    # Check response relevance
    - type: rag
      metric_type: answer_relevancy
      threshold: 0.75

    # Evaluate retrieval quality
    - type: rag
      metric_type: contextual_relevancy
      threshold: 0.7

    # Check retrieval completeness
    - type: rag
      metric_type: contextual_recall
      threshold: 0.7
```

### Mixed Metrics (DeepEval + NLP)

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    # DeepEval metrics (primary)
    - type: geval
      name: "Response Quality"
      criteria: "Evaluate if the response is helpful and accurate."
      threshold: 0.75

    - type: rag
      metric_type: faithfulness
      threshold: 0.8

    # NLP metrics (secondary)
    - type: standard
      metric: f1_score
      threshold: 0.7

    - type: standard
      metric: rouge
      threshold: 0.6
```

### Enterprise Setup with Model Overrides

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest  # Default: free, local
    temperature: 0.0

  metrics:
    # Critical metrics - use powerful model
    - type: rag
      metric_type: faithfulness
      threshold: 0.9
      model:
        provider: openai
        name: gpt-4

    - type: geval
      name: "Safety"
      criteria: "Evaluate response safety and appropriateness."
      threshold: 0.95
      model:
        provider: openai
        name: gpt-4

    # Standard metrics - use default local model
    - type: geval
      name: "Coherence"
      criteria: "Evaluate response clarity."
      threshold: 0.75

    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7

    # NLP metrics - no LLM needed
    - type: standard
      metric: f1_score
      threshold: 0.7
```

### Per-Test Case Evaluation

Test cases can specify which metrics to run:

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

  - name: "Creative task"
    input: "Generate a company tagline"
    evaluations:
      - type: geval
        name: "Creativity"
        criteria: "Evaluate if the tagline is creative and memorable."
        threshold: 0.7
      # Skip faithfulness since no retrieval context
```

### Multi-Turn Test Cases

A test case is **multi-turn** when it carries a `turns:` list instead of a top-level `input` / `ground_truth` / `expected_tools`. Each turn drives one exchange through the same agent session, so retrieval context, conversation history, and tool state carry forward as the user would expect.

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

**Per-turn fields** (from `holodeck.models.test_case.Turn`):

| Field | Type | Notes |
|-------|------|-------|
| `input` | string | Required, non-empty |
| `ground_truth` | string | Optional expected reply for this turn |
| `expected_tools` | list[str \| ExpectedTool] | Per-turn tool assertions; bare strings are promoted to `ExpectedTool(name=..., count=1)` |
| `files` | list[FileInput] | Up to 10 multimodal inputs per turn |
| `retrieval_context` | list[str] | RAG context for this turn |
| `evaluations` | list[metric] | Per-turn metric overrides — same `type` discriminators as top-level metrics (`standard` / `geval` / `rag` / `code`) |
| `turn_config` | dict | Free-form payload passed to code graders as `ctx.turn_config` |

**Mutual exclusivity**: when `turns:` is present, the top-level `input`, `ground_truth`, `expected_tools`, `files`, and `retrieval_context` keys must not be set. Pick one shape per case.

**Argument matchers** in `expected_tools[*].args` use a shape-discriminated mapping:

| YAML shape | Matcher | Behavior |
|------------|---------|----------|
| `key: 42` (scalar/list/dict) | `LiteralMatcher` | Exact equality (int↔float numeric equivalence) |
| `key: { fuzzy: "60.94" }` | `FuzzyMatcher` | Case / whitespace / separator-tolerant, numeric-aware |
| `key: { regex: "^\\d+$" }` | `RegexMatcher` | Anchored full-match against `str(actual)` |

A bare string in `expected_tools` (e.g., `expected_tools: [subtract, divide]`) is shorthand for `ExpectedTool(name="…", count=1)` with no argument matchers.

The dashboard renders multi-turn results with per-turn input / response / tool-call / metric blocks; see the Explorer view for a working example.

### External Test-Case Files (`test_cases_file`)

For large or shared test suites, lift `test_cases:` out of `agent.yaml` and reference an external file with `test_cases_file:` instead.

```yaml
# agent.yaml
name: financial-assistant
model: { provider: anthropic, name: claude-sonnet-4-6 }

evaluations:
  metrics:
    - type: standard
      metric: numeric
      absolute_tolerance: 0.5
      accept_percent: true
      accept_thousands_separators: true

test_cases_file: data/convfinqa_subset.yaml
```

```yaml
# data/convfinqa_subset.yaml — top-level list of cases
- name: Single_MRO/2007/page_134.pdf-1
  turns:
    - input: "..."
      ground_truth: "60.94"
      turn_config: { turn_program: "60.94" }
```

Or, for tooling that wants a wrapping document:

```yaml
# data/convfinqa_subset.yaml — wrapped form
test_cases:
  - name: case-1
    input: "..."
    ground_truth: "..."
```

**Resolution rules** (`src/holodeck/config/loader.py:_resolve_test_cases_file`):

- The path is resolved relative to the directory containing `agent.yaml` (absolute paths also work).
- The referenced file is parsed as YAML with environment-variable substitution applied to its raw text — the same `${VAR}` expansion that runs on `agent.yaml` itself.
- The file's payload may be either a top-level **list** of test cases or a **dict** with a `test_cases:` key — both shapes are accepted.
- `test_cases_file` is mutually exclusive with an inline `test_cases:` block on the same agent. Specifying both raises a `ConfigError` at load time.
- Missing files, parse errors, and non-list payloads also raise `ConfigError` before any test runs.

**When to use**:

- Large suites (hundreds of cases) you don't want to scroll past in `agent.yaml`.
- Generated fixtures (e.g., `scripts/convert_convfinqa.py` writing `data/convfinqa_subset.yaml`).
- Sharing one corpus across multiple agent variants — point each variant's `agent.yaml` at the same file.

---

## Test Execution

When running tests, HoloDeck:

1. Executes agent with test input
2. Records which tools were called
3. Validates tool usage (`expected_tools`)
4. Runs each enabled metric
5. Compares results against thresholds
6. Reports pass/fail per metric

### Example Output

```
Test: "Password Reset"
Input: "How do I reset my password?"
Tools called: [search_kb] ✓
Metrics:
  ✓ Faithfulness: 0.92 (threshold: 0.8)
  ✓ Answer Relevancy: 0.88 (threshold: 0.75)
  ✓ Coherence: 0.85 (threshold: 0.7)
  ✓ F1 Score: 0.81 (threshold: 0.7)
Result: PASS
```

---

## Cost Optimization

### Use Local Models for Development

```yaml
evaluations:
  model:
    provider: ollama           # Free, local
    name: llama3.2:latest
```

### Use Paid Models Only for Critical Metrics

```yaml
evaluations:
  model:
    provider: ollama           # Default: free
    name: llama3.2:latest

  metrics:
    - type: rag
      metric_type: faithfulness
      model:
        provider: openai
        name: gpt-4            # Expensive override only for critical metric
```

### Use NLP Metrics When Possible

NLP metrics are free (no LLM calls):

```yaml
- type: standard
  metric: f1_score  # No LLM cost
- type: standard
  metric: rouge     # No LLM cost
```

---

## Model Configuration Details

When specifying a model for evaluation:

```yaml
model:
  provider: ollama|openai|azure_openai|anthropic  # Required
  name: model-identifier                          # Required
  temperature: 0.0-2.0                            # Optional (recommend 0.0 for evaluation)
  max_tokens: integer                             # Optional
  top_p: 0.0-1.0                                  # Optional
```

### Provider-Specific Models

**Ollama (Recommended for Development)**
- `llama3.2:latest` - Fast, capable
- `llama3.1:latest` - More capable
- Any model available in your Ollama installation

**OpenAI**
- `gpt-4o` - Latest, best quality
- `gpt-4o-mini` - Fast, cheap
- `gpt-4-turbo` - Previous generation

**Azure OpenAI**
- `gpt-4` - Standard
- `gpt-4-32k` - Extended context

**Anthropic**
- `claude-3-opus` - Most capable
- `claude-3-sonnet` - Balanced
- `claude-3-haiku` - Fast, cheap

### Recommended Settings for Evaluation

```yaml
model:
  provider: ollama
  name: llama3.2:latest
  temperature: 0.0  # Deterministic for consistency
```

---

## Troubleshooting

### Error: "invalid metric type"

- Check metric type is valid
- Valid types: `geval`, `rag`, `standard`, `code`
- For standard metrics: f1_score, bleu, rouge, meteor, equality, numeric
- For RAG metrics: faithfulness, answer_relevancy, contextual_relevancy, contextual_precision, contextual_recall
- For code metrics: provide a `grader: "module.path:callable"` reference

### Metric always fails

- Check evaluation model is working
- Try without threshold first
- Test evaluation model manually
- For RAG metrics, ensure required parameters are available

### LLM evaluation too slow

- Use local Ollama model instead of API
- Use faster model: `gpt-4o-mini` instead of `gpt-4`
- Use NLP metrics instead (free and fast)

### Inconsistent evaluation results

- Set temperature to 0.0 for deterministic results
- Use more powerful model for complex evaluations
- Add `evaluation_steps` to GEval for better consistency

### RAG metric missing retrieval_context

- Ensure your agent uses a vectorstore tool
- The test runner automatically extracts retrieval_context from tool results
- Or provide manual retrieval_context in test case

---

## Optimizing Agents (`holodeck test optimize`)

Once you have an evaluation suite, `holodeck test optimize` automates the
hand-tuning loop — change a knob or instruction, re-run `holodeck test`, eyeball
the score, repeat — as a **compounding coordinate-descent optimizer**. It
alternates a *numeric* phase (Optuna TPE over declared query-time axes) and a
*textual* phase (a Critic produces a natural-language "gradient", an Applier
rewrites the instructions), advancing a best-candidate baseline on every
accepted improvement so wins compound into one improved `agent.yaml`.

The original `agent.yaml` is **never modified**. Every trial is logged, and the
best candidate is written to `results/optimizer/<run-id>/best.yaml`.

### Configuration

Add an `evaluations.optimizer` block declaring the scalarized loss (per-metric
weights; loss = `1 − weighted_mean`) and the axes to tune:

```yaml
evaluations:
  metrics:
    - type: standard
      metric: groundedness
    - type: geval
      name: Conciseness
      criteria: "The response is concise and avoids redundancy."
  optimizer:
    loss:                       # metric weights; loss = 1 - weighted_mean
      groundedness: 2.0
      Conciseness: 1.0
    axes:
      numeric:                  # query-time hyperparameters (Optuna TPE)
        - path: model.temperature
          type: float
          range: [0.0, 1.0]
        - path: tools[name=knowledge_base].top_k
          type: int
          range: [3, 12]
      textual:                  # instruction text rewritten by Critic/Applier
        - path: instructions.inline
          max_chars: 6000
    max_cycles: 3               # numeric→textual cycles before stopping
    numeric_phase: { max_trials: 12, patience: 5 }
    textual_phase: { max_trials: 5, patience: 3 }
    min_delta: 0.01             # minimum loss reduction required to accept
    seed: 42
```

**Axis paths** support dotted attributes (`model.temperature`,
`instructions.inline`) and a `tools[name=X].<field>` selector for per-tool
fields. Numeric axes accept `float`/`int` (a `[low, high]` range) or
`categorical` (a list of choices).

**Phase budgets.** `numeric_phase.{max_trials,patience}` cap the Optuna trials
in a numeric phase. For the textual phase the budget drives *iterative
refinement*: with a **single** textual axis, `textual_phase.max_trials` is the
number of successive Critic→Applier refinement steps taken on that axis (each
step builds on the previous attempt and the failing cases it produced), and
`patience` stops the phase after that many consecutive non-improving steps. A
drifting chain never regresses the result — only an accepted, loss-improving
step advances the best agent. With **more than one** textual axis the proposer
falls back to a single rewrite per axis (iterative multi-axis ordering is not
yet supported).

### Running

```bash
holodeck test optimize agent.yaml
holodeck test optimize agent.yaml --max-cycles 2 --numeric-max-trials 20 --seed 7
```

CLI flags (`--max-cycles`, `--numeric-max-trials`, `--numeric-patience`,
`--textual-max-trials`, `--textual-patience`, `--seed`, `-o/--output-dir`)
override the YAML config for a single run. The command streams per-trial losses
and prints the baseline → best summary on completion.

### Outputs

`results/optimizer/<run-id>/` contains:

- `best.yaml` — the best candidate agent, ready to copy over your original.
- `trials.jsonl` — one record per trial (the full audit trail).
- `report.md` — baseline vs best, the accepted edits, and a per-phase summary.

### Acceptance and scoring

The optimizer **minimizes a loss** of `1 − weighted_mean`, where the weighted
mean is a renormalized weighted average of the per-metric averages using your
`loss` weights (so a perfect agent has loss `0.0`). Metric scores must be
normalized to `[0, 1]`; an average outside that range aborts the run. Metric runs
that **error** are excluded from the mean (not scored as zero); legitimate `0.0`
scores are kept. A candidate is accepted only when its loss undercuts the current
best by more than `min_delta`.

> **Note (MVP):** acceptance uses the raw loss delta — there is no
> train/holdout split, repeated trials, or variance-aware bar yet, so a small
> `min_delta` can chase evaluation noise. Keep `min_delta` above your suite's
> single-case-flip granularity. Statistical rigor is the planned v1 follow-up.

### Observability

`holodeck test optimize` emits OpenTelemetry traces and metrics on the same
terms as `holodeck test`: set an `observability` block on the agent with
`enabled: true` and an OTLP exporter, and the optimize run exports to your
collector (e.g. Aspire, Grafana). No extra configuration — it reuses the agent's
existing `observability` block. When observability is disabled the optimizer
behaves exactly as before (no spans, no metrics).

**Span tree.** One root span per run, with each trial's evaluation GenAI spans
nesting under its trial span:

```
holodeck.optimize                 run_id, agent_name, max_cycles, seed, axes counts, loss
├── holodeck.optimize.baseline    baseline scoring of the original agent
└── holodeck.optimize.cycle       one coordinate-descent cycle
    └── holodeck.optimize.phase   numeric | textual (records accepts)
        ├── holodeck.optimize.propose   textual Critic/Applier calls (GenAI spans nest)
        └── holodeck.optimize.trial     trial_id, phase, baseline_loss, loss, accepted,
            └── <eval GenAI spans>        edit_summary, axis, params (JSON), error
```

Trial spans carry only primitives — numeric params as a single JSON string,
the textual axis *name* and a human-readable `edit_summary` — never instruction
text or resolved secrets.

**Metrics** (all `holodeck.optimize.*`, attributed by `phase` where meaningful):

| Metric | Type | Meaning |
| --- | --- | --- |
| `trials` | counter | completed trials (`phase`, `accepted`) |
| `trials.skipped` | counter | trials skipped because a proposer errored (`phase`) |
| `trial.loss` | histogram | candidate loss per trial (`phase`) |
| `trial.duration` | histogram (s) | scorer wall-time per trial (`phase`) |
| `best_loss` | histogram | best loss after each accepted improvement (`phase`) |
| `improvement` | histogram | `baseline_loss − best_loss` at run end |
| `cycles` | counter | completed coordinate-descent cycles |

## Best Practices

1. **Start with DeepEval**: Use GEval and RAG metrics as primary evaluation
2. **Use Local Models**: Start with Ollama for free development, upgrade for production
3. **Mix Metric Types**: Combine DeepEval (semantic) with NLP (keyword-based)
4. **Cost-Aware**: Use cheaper/local models by default, expensive models only for critical metrics
5. **Realistic Thresholds**: Set thresholds based on actual agent performance
6. **Monitor**: Run metrics on sample of tests first
7. **Iterate**: Adjust thresholds and metrics based on results
8. **Migrate from Legacy**: Replace deprecated Azure AI metrics with DeepEval equivalents

---

## Next Steps

- See [Agent Configuration Guide](agent-configuration.md) for how to set up evaluations
- See [Examples](../examples/) for complete evaluation configurations
- See [Global Configuration](global-config.md) for shared settings
- See [API Reference](../api/evaluators.md) for evaluator class details
