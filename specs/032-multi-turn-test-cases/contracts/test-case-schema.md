# Contract: Multi-Turn Test Case YAML Schema

**Feature**: 032-multi-turn-test-cases
**Consumers**: `holodeck test`, config loader, agent authors.

This contract specifies the YAML shape that `agent.yaml` accepts for multi-turn test cases. The existing single-turn shape remains fully supported; this document only describes the additions.

## 1. Grammar

```yaml
test_cases:
  # (A) Legacy single-turn (unchanged)
  - name: "greeting"
    input: "hello"
    ground_truth: "hi there"
    expected_tools: ["greet"]

  # (B) Multi-turn — turn-level assertions only
  - name: "convfinqa-0"
    turns:
      - input: "what is the net cash from operating activities in 2009?"
        ground_truth: "206588"
      - input: "what about in 2008?"
        ground_truth: "181001"
      - input: "what is the difference?"
        ground_truth: "25587"
        expected_tools: ["subtract"]
      - input: "what percentage change does this represent?"
        ground_truth: "14.1%"
        expected_tools: ["divide"]

  # (C) Multi-turn — object-form expected_tools with arg matchers
  - name: "convfinqa-0-strict"
    turns:
      - input: "what is the net cash from operating activities in 2009?"
        ground_truth: "206588"
      - input: "what is the difference between 2009 and 2008?"
        expected_tools:
          - name: "subtract"
            args:
              a: 206588                         # literal (int ↔ float equivalence)
              b: { fuzzy: "181001" }             # fuzzy (thousands separators ok)
        ground_truth: "25587"
        evaluations:
          - type: standard
            metric: numeric
            absolute_tolerance: 0.5
```

## 2. Mutual-Exclusion Rules (FR-001)

| Top-level field set | `turns` set | Result |
|---------------------|-------------|--------|
| `input` (+ optional `ground_truth` / `expected_tools`) | absent | Legacy single-turn test case — unchanged behavior. |
| — (no `input`) | present, non-empty | Multi-turn test case. |
| `input` or `ground_truth` | present | **ConfigError** at load time. |
| none | `turns: []` | **ConfigError** at load time (empty turns). |

`files` and `retrieval_context` may appear on the test case *or* on individual turns, but for multi-turn cases they belong on turns (FR-009). Placing them at the test-case level when `turns` is set → `ConfigError` with a hint.

## 3. `expected_tools` — mixed list (FR-003)

At either the test-case level (legacy) or the turn level, `expected_tools` accepts a list whose elements are individually one of:

- `str` — bare tool name (case-sensitive substring match, no arg check).
- `dict` — object of shape `{name: str, args?: dict, count?: int}` (FR-003, FR-011).

A single list may mix both forms:

```yaml
expected_tools:
  - "lookup"                                 # name-only
  - name: "subtract"                          # object form
    args: { a: 206588, b: 181001 }
  - name: "divide"
    count: 1
```

## 4. `ArgMatcher` shapes (FR-004)

Each value inside `expected_tools[*].args` is one of:

| Shape | Example | Semantics |
|-------|---------|-----------|
| **Literal scalar** | `a: 206588` | Exact after numeric normalization (`int == float`). |
| **Literal list** | `items: [1, 2]` | Exact list equality after element normalization. |
| **Literal dict** | `opts: {mode: "fast"}` | Exact dict equality. |
| **Fuzzy** | `a: { fuzzy: "206588" }` | Case/whitespace-insensitive, numeric-aware, separator-tolerant. |
| **Regex** | `a: { regex: "^2009.*$" }` | Anchored `re.fullmatch` over `str(actual)`. |

A dict with both `fuzzy` and `regex` keys, or with any unknown top-level key, is rejected at load time with the field path.

## 5. Per-turn metric overrides (FR-023)

Each turn may carry its own `evaluations` list. Precedence (highest wins):

1. `turns[i].evaluations` — per-turn override.
2. `test_cases[j].evaluations` — per-test-case override.
3. `evaluations.metrics` — agent-level default.

New evaluator types that may appear at any level:

```yaml
# Built-in deterministic (no LLM)
- type: standard
  metric: equality
  case_insensitive: true
  strip_whitespace: true

- type: standard
  metric: numeric
  absolute_tolerance: 0.01
  accept_percent: true
  accept_thousands_separators: true

# User-supplied Python grader (escape hatch — spec §Complexity Tracking)
- type: code
  grader: "my_benchmarks.convfinqa:program_equivalence"
```

## 6. Files per turn (FR-009)

Per-turn `files` are processed and included with that turn's input *only* — they are not replayed on subsequent turns. The existing `FileInput` schema is reused unchanged:

```yaml
turns:
  - input: "summarize the attached image"
    files:
      - path: "./data/chart.png"
        type: image
```

## 7. Validation error messages (FR-025)

Load-time errors MUST include:
- The test-case `name` (or index if unnamed).
- The field path (`turns[2].expected_tools[1].args.a.regex`).
- A one-line cause (`"cannot compile regex: unbalanced parenthesis at position 7"`).

## 8. Examples

### 8.1 Minimal multi-turn (Story 1)
```yaml
test_cases:
  - name: "chit-chat"
    turns:
      - input: "hi"
      - input: "who are you?"
      - input: "tell me a joke"
```

### 8.2 Per-turn ground truth + tool assertion (Story 2)
```yaml
test_cases:
  - name: "convfinqa-0"
    turns:
      - input: "what is the net cash from operating activities in 2009?"
        ground_truth: "206588"
      - input: "what is the difference from 2008's 181001?"
        ground_truth: "25587"
        expected_tools: ["subtract"]
```

### 8.3 Fuzzy + regex args (Story 3)
```yaml
test_cases:
  - name: "convfinqa-0-argcheck"
    turns:
      - input: "subtract 206588 from 181001"
        expected_tools:
          - name: "subtract"
            args:
              a: { fuzzy: "206588" }
              b: { regex: "^181001(\\.0+)?$" }
```

### 8.4 Numeric + code evaluator (Story 4)
```yaml
test_cases:
  - name: "convfinqa-0-code-grader"
    turns:
      - input: "what percentage change does this represent?"
        ground_truth: "0.14136"
        evaluations:
          - type: standard
            metric: numeric
            absolute_tolerance: 0.001
          - type: code
            grader: "my_benchmarks.convfinqa:program_equivalence"
```
