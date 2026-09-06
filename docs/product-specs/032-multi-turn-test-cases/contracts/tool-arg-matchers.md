# Contract: Tool Argument Matchers

**Feature**: 032-multi-turn-test-cases
**Consumers**: `src/holodeck/lib/test_runner/tool_arg_matcher.py`, executor turn-assertion path, config loader.

Defines the exact semantics of the three matcher kinds and the call-selection rule. This is the authoritative reference for SC-006 acceptance pairs.

---

## 1. Call-selection (per tool name)

Within a single turn, for each `ExpectedTool(name=T, args=A, count=C)`:

1. Collect all `tool_invocations` on the turn whose name **contains** `T` as a case-sensitive substring (preserves the existing legacy contract of `validate_tool_calls`). Preserve the order they were invoked.
2. Define "matching call" as one where **every** asserted arg key `k` in `A` has a value `v_actual` such that `match(A[k], v_actual) == True`. Unasserted keys on the actual call are ignored (FR-012, A4).
3. If ≥ `C` matching calls exist, the assertion passes (default `C = 1`, A5). Otherwise it fails.
4. First-match-wins is used only to populate `arg_match_details.matched_call_index` in the report; selection itself is order-independent (FR-011 scenario 6).

If `A` is `None` (name-only assertion), step 2 collapses to "any call with a matching name".

## 2. Literal matcher

`match(expected=LiteralMatcher(v), actual)`:

```python
def match_literal(v, actual):
    # Numeric int/float equivalence (A7)
    if isinstance(v, (int, float)) and isinstance(actual, (int, float)) and not isinstance(v, bool) and not isinstance(actual, bool):
        return float(v) == float(actual)
    # Bool strict
    if isinstance(v, bool) or isinstance(actual, bool):
        return type(v) == type(actual) and v == actual
    # None
    if v is None or actual is None:
        return v is None and actual is None
    # Strings / lists / dicts: deep equality, no normalization
    return v == actual
```

**No string normalization** (case, whitespace, punctuation) on literal. No separator stripping. That's the fuzzy matcher's job.

## 3. Fuzzy matcher

`match(expected=FuzzyMatcher(pattern), actual)` — case-insensitive, whitespace-normalized, numeric-aware, separator-tolerant.

```python
def match_fuzzy(pattern, actual):
    s_expected = _normalize(pattern)
    s_actual   = _normalize(str(actual))
    # Try numeric equivalence first
    n_e = _try_parse_number(s_expected)
    n_a = _try_parse_number(s_actual)
    if n_e is not None and n_a is not None:
        return math.isclose(n_e, n_a, rel_tol=0.0, abs_tol=1e-9)
    return s_expected == s_actual
```

Where:

```python
def _normalize(s: str) -> str:
    # lowercase, collapse runs of whitespace, strip thousands separators
    s = s.strip().lower()
    s = re.sub(r"[,_\u202f]", "", s)        # comma, underscore, narrow NBSP
    s = re.sub(r"\s+", " ", s)
    return s

def _try_parse_number(s: str) -> float | None:
    # handle trailing percent
    pct = False
    if s.endswith("%"):
        s = s[:-1].strip()
        pct = True
    try:
        v = float(s)
        return v / 100 if pct else v
    except ValueError:
        return None
```

**Consequences** (verified in SC-006 matrix):
- `fuzzy("206588")` matches `206588`, `206588.0`, `"206588"`, `"206,588"`, `"206 588"`, `" 206588 "`.
- `fuzzy("14.14%")` matches `0.1414` (percent parsing).
- `fuzzy("YES")` matches `"yes"`, `" yes "`.
- `fuzzy("206588")` does **not** match `206000` (no distance tolerance).

## 4. Regex matcher

`match(expected=RegexMatcher(compiled), actual)`:

```python
def match_regex(compiled, actual):
    return compiled.fullmatch(str(actual)) is not None
```

- Always compared against `str(actual)` (FR-014).
- `fullmatch` semantics — equivalent to `^…$` anchoring without requiring the user to write them.
- Pattern compile failures are caught at load time (FR-025), never here.

## 5. Missing vs extra args

| Situation | Outcome |
|-----------|---------|
| Asserted arg key absent on actual call | Turn fails; `arg_match_details.unmatched_reason = "arg '<k>' missing"`. |
| Unasserted arg key present on actual call | Ignored (A4). Never causes failure. |
| Actual call has no arguments and no args asserted | Passes name-only assertion. |
| Asserted arg has matcher but actual value is `None` | Match is kind-specific: literal requires expected `None`; fuzzy compares `"none"`; regex compares `"None"` (Python `str(None)`). |

## 6. Count semantics

- Default `count = 1` (A5). `count >= 1` enforced at load time.
- Turn passes when ≥ `count` calls in the turn satisfy both the name substring match *and* the arg matchers simultaneously (FR-011).
- Calls across different turns never contribute to a single turn's count.

## 7. Acceptance matrix (SC-006, 20+ pairs)

| # | Expected matcher | Actual arg value | Expected result |
|---|------------------|------------------|-----------------|
| 1 | `a: 206588` | `206588` | PASS |
| 2 | `a: 206588` | `206588.0` | PASS (int↔float) |
| 3 | `a: 206588` | `206000` | FAIL |
| 4 | `a: 206588.0` | `206588` | PASS |
| 5 | `a: {fuzzy: "206588"}` | `"206,588"` | PASS (separator tol.) |
| 6 | `a: {fuzzy: "206588"}` | `"206 588"` | PASS (whitespace tol.) |
| 7 | `a: {fuzzy: "206588"}` | `206588.0` | PASS |
| 8 | `a: {fuzzy: "206588"}` | `205000` | FAIL |
| 9 | `a: {fuzzy: "14.14%"}` | `0.1414` | PASS (percent) |
| 10 | `a: {fuzzy: "YES"}` | `"yes"` | PASS (case) |
| 11 | `a: {fuzzy: "YES"}` | `" yes "` | PASS (whitespace trim) |
| 12 | `a: {regex: "^2009.*$"}` | `"2009 cash flow"` | PASS (anchored fullmatch) |
| 13 | `a: {regex: "^2009.*$"}` | `"report 2009 cash flow"` | FAIL (not anchored to start) |
| 14 | `a: {regex: "^181001(\\.0+)?$"}` | `"181001.0"` | PASS |
| 15 | `a: {regex: "^181001(\\.0+)?$"}` | `"181001.5"` | FAIL |
| 16 | `a: 206588` (asserted) | arg `a` missing | FAIL (missing) |
| 17 | `a: 206588` (asserted), extra `b: "x"` | actual `{a: 206588, b: "x", c: "y"}` | PASS (extras ignored) |
| 18 | `a: true` | `True` | PASS (bool) |
| 19 | `a: true` | `1` | FAIL (bool≠int) |
| 20 | `a: [1, 2]` | `[1, 2]` | PASS (list eq) |
| 21 | `a: [1, 2]` | `[1.0, 2.0]` | PASS (element numeric eq) |
| 22 | `a: {mode: "fast"}` | `{mode: "fast"}` | PASS (dict eq) |
| 23 | multi-call, `a: {fuzzy: "206588"}` | turn has two `subtract` calls; second matches | PASS (any-call wins, FR-011.6) |
