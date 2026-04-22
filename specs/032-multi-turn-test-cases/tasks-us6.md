---
description: "Tasks for User Story 6 — Sample agent: Financial Assistant backed by ConvFinQA (P3)"
---

# Tasks: User Story 6 — Sample Agent: Financial Assistant on ConvFinQA (P3)

**Input**: `/specs/032-multi-turn-test-cases/` — spec.md (US6), quickstart.md (ConvFinQA example), contracts/test-case-schema.md, contracts/code-grader-contract.md. Source dataset: `/Users/justinbarias/Documents/Git/python/justinbarias/data/convfinqa_dataset.json` (21 MB, `{"train": [...], "dev": [...]}` shape — verified).
**Approach**: TDD where meaningful (converter utility, function tools, grader stub); example-driven authoring for the agent configuration + README.

**Story Goal**: Ship `sample/financial-assistant/claude/` that demonstrates multi-turn test cases end-to-end on a real benchmark. User can `cd sample/financial-assistant/claude && holodeck test` from a fresh checkout and see 10 multi-turn ConvFinQA-shaped cases execute with per-turn reports, per-turn tool assertions (fuzzy arg matchers), deterministic `numeric` scoring, and one illustrative `type: code` grader.

**Independent Test**: From a fresh checkout with `ANTHROPIC_API_KEY` set: `cd sample/financial-assistant/claude && holodeck test`. Expect 10 test cases, 30–50 turns total, all complete, per-turn breakdown in the markdown report, exit code reflects pass/fail honestly. Setup-to-first-run ≤ 15 minutes following `README.md` (SC-001 on real data).

**Depends on**: US1 (multi-turn execution), US2 (per-turn assertions), US3 (arg matchers — fuzzy matchers on `subtract`/`divide` args), US4 (`numeric` built-in + `type: code` grader). US5 is independent but complements (the sample's runs are dashboard fodder).

**Not in scope**: Committing the full 21 MB `convfinqa_dataset.json` — only a curated 10-example subset lands under `data/`. The converter utility regenerates the subset from the upstream dataset path.

---

## Phase 0: FunctionTool runtime support — **[COMPLETE]**

Shipped in commit `16d0d44 feat(tools): FunctionTool runtime on SK + Claude backends`. Verified present in tree:

- `src/holodeck/lib/function_tool_loader.py` — `load_function_tool(tool, base_dir)` callable resolver.
- `src/holodeck/lib/backends/tool_adapters.py` — `FunctionToolAdapter` + dispatch branch in `create_tool_adapters` (lines 178–204, 297–299).
- `src/holodeck/lib/test_runner/agent_factory.py` — SK-side `FunctionTool` dispatch (lines 1013–1054).
- `tests/fixtures/agents/tools/orders.py` — fixture stub present; `tests/fixtures/agents/valid_agent.yaml` loads cleanly.

US6 consumes this runtime as-is; no further Phase 0 work is required.

---

## Phase 0.5: Model prerequisite (pre-US6)

- [x] T000 [US6] Add a `turn_config: dict[str, Any] | None = None` field to `Turn` in `src/holodeck/models/test_case.py` (currently `ConfigDict(extra="forbid")` with no such field — verified at line 345). Thread it through `turn.model_dump(mode="python")` so the existing `executor.py:1372` call surfaces it to `build_grader_context` unchanged. Data-model.md and `contracts/code-grader-contract.md` §3.1 both reference `ctx.turn_config` as the channel for grader-specific per-turn keys (e.g., `turn_program`) — this task makes that channel real. Add unit test covering `Turn(**{"input": "x", "turn_config": {"turn_program": "subtract(a,b)"}})` round-trip + a test that an unknown top-level key still fails forbid.

---

## Phase 1: Setup + scaffold

- [x] T001 Read spec US6 + quickstart.md §1–§6. Confirm US1 + US2 foundational tasks are merged (multi-turn execution exists end-to-end); US3/US4 ideally merged (arg matchers + `numeric` + `code` evaluator) — sample can scaffold in parallel with US3/US4 but the full demo needs both landed. Phase 0 (FunctionTool runtime) is already merged (`16d0d44`).
- [x] T001a [US6] Verify-only gate before starting Phase 5: confirm (a) `ExecutionConfig.parallel_test_cases` is a defined field on `src/holodeck/models/config.py` (currently line 73) and (b) `Turn.turn_config: dict[str, Any] | None` exists after T000. If either is missing or was renamed, pause Phase 5 and reconcile before `agent.yaml` / grader wiring drifts.
- [x] T002 Create directory structure under the repo root, mirroring the existing multi-backend sample layout (`sample/legal-assistant/claude/...`, `sample/content-moderation/claude/...`):
  ```
  sample/financial-assistant/
  └── claude/
      ├── agent.yaml
      ├── .env.example
      ├── README.md
      ├── instructions/
      │   └── system-prompt.md
      ├── data/
      │   └── convfinqa_subset.yaml        # created by T007; referenced via test_cases_file
      ├── tools/
      │   └── financial_tools.py
      ├── graders/
      │   ├── __init__.py
      │   └── turn_program_equivalence.py
      └── scripts/
          └── convert_convfinqa.py
  ```
  Add the directory without modifying root `pyproject.toml` — sample agents follow the existing convention (see `sample/research/`, `sample/support/`). Ollama / OpenAI variants may be added later as siblings under `sample/financial-assistant/` (e.g., `sample/financial-assistant/ollama/`) without restructuring.

  **Note on grader import path**: The sample dir name `financial-assistant` contains a hyphen and cannot appear in a dotted Python module path. The grader reference used throughout Phase 4/5 is therefore `graders.turn_program_equivalence:turn_program_equivalence` — i.e., resolved relative to the agent-yaml directory (`sample/financial-assistant/claude/`), which `holodeck test` adds to `sys.path` at invocation. T022 verifies this resolution works; if it does not, raise the gap as a HoloDeck grader-resolver improvement rather than work around it by renaming the sample. Do **not** write `sample.financial_assistant.graders...`.

---

## Phase 2: Converter utility (TDD)

### Tests first

- [x] T003 [P] [US6] Write `tests/unit/sample/test_convert_convfinqa.py::test_converts_single_dialogue_to_multi_turn_shape` — given a fixture ConvFinQA record (`id`, `doc`, `dialogue` with `conv_questions`, `conv_answers`, `turn_program`, `executed_answers`), the converter emits a dict matching `TestCaseModel` multi-turn shape: `name=<id>`, `turns=[{input, ground_truth, turn_config: {turn_program}}]` with `len(turns) == len(conv_questions)`.
- [x] T004 [P] [US6] Add `test_convert_convfinqa.py::test_preserves_turn_program_via_turn_config` — `turn_program` (e.g., `"subtract(206588, 181001)"`) is attached on each turn as `turn_config.turn_program` so the code grader can read it via `ctx.turn_config["turn_program"]` (per contracts/code-grader-contract.md §3.1).
- [x] T005 [P] [US6] Add `test_convert_convfinqa.py::test_handles_turn_program_with_backrefs` — ConvFinQA's later turns use `#N` back-refs (e.g., `"subtract(206588, 181001), divide(#0, 181001)"`). Converter preserves the raw string faithfully; does not try to resolve `#N` (that's the grader's job).
- [x] T006 [P] [US6] Add `test_convert_convfinqa.py::test_output_validates_against_testcase_model` — converter output, when loaded through `TestCaseModel(**data)`, validates without error (guards against schema drift).

### Implementation

- [x] T007 [US6] Write `sample/financial-assistant/claude/scripts/convert_convfinqa.py`:
  ```
  Usage: python scripts/convert_convfinqa.py \
           --source /path/to/convfinqa_dataset.json \
           --split dev \
           --n 10 \
           --out data/convfinqa_subset.yaml
  ```
  - Reads `{"train": [...], "dev": [...]}`; picks first N examples from `--split`.
  - Emits `data/convfinqa_subset.yaml` in **HoloDeck's native test_cases format** — a YAML list of `{name, turns}` entries that the external `test_cases_file:` loader (commit `5f0c84a feat(loader): support external test_cases_file reference in agent.yaml`) parses directly. No custom YAML directives (no `!include`).
  - Each turn: `{input: conv_questions[i], ground_truth: str(executed_answers[i]), turn_config: {turn_program: turn_program[i]}}`. The `turn_program` lands under `turn_config` so the code grader can read it via `ctx.turn_config["turn_program"]` per contracts/code-grader-contract.md §3.1. Relies on T000 having landed the `Turn.turn_config` field.
  - Uses PyYAML (already a HoloDeck runtime dep) plus stdlib.
  - Emits a one-line summary: `"Converted N dialogues → <path>"`.
  - Rerunnable — overwrites `--out` on each run. `agent.yaml` itself is never modified; the `test_cases_file:` pointer reaches the generated file.
- [x] T008 [US6] Run the converter once against the upstream dataset to produce `sample/financial-assistant/claude/data/convfinqa_subset.yaml`. Commit the artifact; expect ≤ 150 KB at N=10 (budget allowance for full SEC document tables in each dialogue's `doc`). No changes to `agent.yaml` here — the pointer is wired in T017.

---

## Phase 3: Function tools (TDD)

### Tests first

- [x] T009 [P] [US6] Write `tests/unit/sample/test_financial_tools.py::test_subtract_basic` — `subtract(a=206588, b=181001) == 25587`; also tests negative result, float args, and `subtract(a=1.0, b=1) == 0.0` (int/float parity).
- [x] T010 [P] [US6] Add `test_financial_tools.py::test_divide_basic_and_zero_guard` — `divide(a=25587, b=181001) ≈ 0.14136` (abs tol 1e-5); `divide(a=1, b=0)` raises `ZeroDivisionError` (or returns a documented sentinel — pick one and document in the tool docstring).
- [x] T011 [P] [US6] Add `test_financial_tools.py::test_lookup_returns_scalar` — `lookup(document, description)` where `document` is the ConvFinQA `doc` structure (with `pre_text`, `post_text`, `table`), returns the numeric scalar matching `description`. Exact implementation of `lookup` can be "find by description in the table" — verify for a fixture row.

### Implementation

- [x] T012 [US6] Write `sample/financial-assistant/claude/tools/financial_tools.py` with three pure-Python functions:
  ```python
  def subtract(a: float | int, b: float | int) -> float: ...
  def divide(a: float | int, b: float | int) -> float: ...
  def lookup(document: dict, description: str) -> float | None: ...
  ```
  `lookup` implementation: case-insensitive substring search across `document["table"]` row/column headers; returns the first matching numeric cell. Docstrings required for all three (HoloDeck `type: function` tools auto-use the docstring as the tool description).

---

## Phase 4: Optional code grader stub

### Tests first

- [x] T013 [P] [US6] Write `tests/unit/sample/test_turn_program_equivalence.py::test_matching_program_passes` — import `ToolInvocation` from `holodeck.models.test_result` and `GraderContext, GraderResult` from `holodeck.lib.test_runner.code_grader`. Given `ctx.turn_config["turn_program"] = "subtract(206588, 181001)"` and `ctx.tool_invocations = (ToolInvocation(name="subtract", args={"a":206588, "b":181001}, ...),)` (note: tuple, per frozen-dataclass contract §3.1), the grader returns `GraderResult(score=1.0, passed=True)`.
- [x] T013a [P] [US6] Add `test_turn_program_equivalence.py::test_diverging_program_fails` — same `turn_program` but agent called `divide` instead; grader returns `GraderResult(score=0.0, passed=False)` with a reason string.

### Implementation

- [x] T014 [US6] Write `sample/financial-assistant/claude/graders/turn_program_equivalence.py`:
  ```python
  from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult

  def turn_program_equivalence(ctx: GraderContext) -> GraderResult:
      expected = ctx.turn_config.get("turn_program", "")
      # Minimal prototype: compare op names; detailed DAG walk with #N
      # backref substitution is left as an exercise (linked in README).
      ...
  ```
  Keep the implementation minimal — this is an illustration of the grader interface, not a production program-equivalence checker. README points users to quickstart.md §5 for a fuller treatment.
- [x] T015 [US6] Add `sample/financial-assistant/claude/graders/__init__.py` (empty). The grader reference format used in `agent.yaml` is `"graders.turn_program_equivalence:turn_program_equivalence"` — resolved relative to the agent-yaml directory (`sample/financial-assistant/claude/`), which HoloDeck's test runner adds to `sys.path` at invocation. No `sample.financial_assistant.` prefix is used (the hyphen in `financial-assistant` precludes dotted imports). T022 verifies a misspelled grader path surfaces as a load-time `ConfigError` (FR-025); if the relative resolution does not work out of the box, raise it as a HoloDeck grader-resolver improvement rather than renaming the sample directory.

---

## Phase 5: Agent configuration

- [x] T016 [US6] Write `sample/financial-assistant/claude/instructions/system-prompt.md` — financial-analyst persona (≤ 300 words): "You are a financial analyst. Use `lookup`, `subtract`, `divide` to answer numeric questions about an SEC filing. Return plain numbers (no commas, no percent signs unless explicitly asked). Treat each user turn as a follow-up to the previous exchanges; resolve anaphoric references like 'what about in 2008?' using prior context." Mirror the tone of `sample/research/instructions/system-prompt.md`.
- [x] T017 [US6] Write `sample/financial-assistant/claude/agent.yaml` using **Claude Agent SDK as the default backend** (HoloDeck first-class per CLAUDE.md) and the real `FunctionTool` schema (`{name, description, type: function, file, function}` — verified against `src/holodeck/models/tool.py:422-457` and `tests/fixtures/agents/valid_agent.yaml:22-27`):
  ```yaml
  name: financial-assistant
  description: "ConvFinQA-driven financial analyst demonstrating multi-turn test cases"
  model:
    provider: anthropic
    name: claude-sonnet-4-6
    temperature: 0.0
    auth_provider: api_key
  instructions:
    file: instructions/system-prompt.md
  tools:
    - name: lookup
      type: function
      description: "Look up a scalar value from the SEC document by description (e.g., '2009 net cash from operating activities')."
      file: tools/financial_tools.py
      function: lookup
    - name: subtract
      type: function
      description: "Compute a - b for two numeric operands."
      file: tools/financial_tools.py
      function: subtract
    - name: divide
      type: function
      description: "Compute a / b for two numeric operands. Raises ZeroDivisionError when b is zero."
      file: tools/financial_tools.py
      function: divide
  evaluations:
    metrics:
      - type: standard
        metric: numeric
        absolute_tolerance: 0.5
        accept_percent: true
        accept_thousands_separators: true
  execution:
    parallel_test_cases: 4
  test_cases_file: data/convfinqa_subset.yaml
  ```
  Test cases are loaded from an external YAML file via the `test_cases_file:` pointer (commit `5f0c84a feat(loader): support external test_cases_file reference in agent.yaml`) — no inline block, no sentinel comments, no converter injection mode. Backend-switch instructions live in `README.md` (T020) rather than inline comments: users can flip `model.provider` to `openai` or `ollama` and set the matching env var.
- [x] T018 [US6] Enrich **at least two** test cases inside `data/convfinqa_subset.yaml` with:
  - `expected_tools` using `{name, args: {a: {fuzzy: "..."}, b: {fuzzy: "..."}}}` object form (exercises US3 matchers on real ConvFinQA numeric values).
  - An `evaluations` block adding `{type: code, grader: "graders.turn_program_equivalence:turn_program_equivalence"}` on one turn that has a `turn_program` (exercises US4 code grader). Note: relative (no `sample.` prefix) per T015.
  These augmentations happen via a second pass over `convfinqa_subset.yaml` — extend the converter T007 with an `--augment` mode OR edit the two cases by hand after conversion and commit.
- [x] T019 [US6] Write `sample/financial-assistant/claude/.env.example` with stubs for:
  ```
  # Default backend: Claude Agent SDK (Anthropic) — required for first run
  ANTHROPIC_API_KEY=

  # Optional alternate backends — set these and switch `model.provider` in agent.yaml
  # OPENAI_API_KEY=
  # OLLAMA_BASE_URL=http://localhost:11434
  ```

---

## Phase 6: README + quickstart

- [x] T020 [US6] Write `sample/financial-assistant/claude/README.md` (target ≤ 350 words — SC-001 time budget):
  - Prereqs: HoloDeck installed (`pip install -e .` from repo root) and an `ANTHROPIC_API_KEY` with access to `claude-sonnet-4-6`.
  - Step 1: `cp .env.example .env` and fill in `ANTHROPIC_API_KEY`.
  - Step 2: `cd sample/financial-assistant/claude && holodeck test` — `agent.yaml` loads test cases from `data/convfinqa_subset.yaml` via `test_cases_file:`. Expect per-turn output like quickstart §3.
  - Step 3 (optional regen): `python scripts/convert_convfinqa.py --source <path-to-convfinqa_dataset.json> --split dev --n 10 --out data/convfinqa_subset.yaml` to refresh the subset. `agent.yaml` is untouched.
  - Step 4 (optional backend swap): switch `model.provider` to `openai` (set `OPENAI_API_KEY`) or `ollama` (set `OLLAMA_BASE_URL`; pull a capable local model). Dual-backend coverage is the counterpart to SC-010 on real data.
  - Troubleshooting table: `ANTHROPIC_API_KEY` missing, grader import error, dataset path typo, `test_cases_file:` points at a missing YAML (re-run the converter).
  - Cross-reference `specs/032-multi-turn-test-cases/quickstart.md` for the full feature tutorial.

---

## Phase 7: End-to-end smoke + integration

- [x] T021 [P] [US6] Add `tests/integration/test_sample_financial_assistant.py::test_sample_loads_and_runs_on_stubbed_backend` — loads `sample/financial-assistant/claude/agent.yaml`, stubs the backend with scripted responses matching ground truths, runs `test_runner.execute`, asserts: (a) 10 `TestResult` objects produced; (b) each has `turns` populated with `len(turns) >= 3`; (c) per-turn `metric_results` includes `numeric`; (d) at least one turn carries `arg_match_details` from the augmented test cases (T018); (e) at least one turn carries a `code`-kind `MetricResult` from the grader.
- [x] T022 [US6] Add `tests/integration/test_sample_financial_assistant.py::test_sample_loads_via_testcase_model` — simple config-validation smoke: `ConfigLoader().load("sample/financial-assistant/claude/agent.yaml")` succeeds; `Agent(**config)` validates; the external `test_cases_file: data/convfinqa_subset.yaml` is resolved and populates `agent.test_cases`. Add a negative case that corrupts the `graders.turn_program_equivalence:turn_program_equivalence` reference (e.g., typo the callable name) and asserts a `ConfigError` naming the turn and the bad path (FR-025). Catches YAML drift AND grader-path drift without invoking a backend.
- [x] T023 [P] [US6] (Optional) Add gated dual-backend tests in `test_sample_financial_assistant.py`: `::test_sample_runs_on_anthropic` under `@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"))` (the default path — smokes the committed `model.provider: anthropic`), and `::test_sample_runs_on_ollama` under `HOLODECK_IT_OLLAMA` / `::test_sample_runs_on_openai` under `HOLODECK_IT_OPENAI` for the opt-in SK-backend variants. Runs the real sample against the real backend; asserts non-zero-length responses per turn. Counterpart to SC-010 on a real dataset.

---

## Phase 8: Polish + repo-level visibility

- [x] T024 [US6] Run `make format && make lint && make type-check && pytest tests/unit/sample -n auto && pytest tests/integration/test_sample_financial_assistant.py -n auto`. All green.
- [ ] T025 [P] [US6] Manual end-to-end validation: fresh terminal, `cd sample/financial-assistant/claude && holodeck test`. Confirm ≤ 15-minute clock time from a clean checkout to first green run (walk through `README.md` as a first-time user would).
- [ ] T026 [P] [US6] If a top-level `docs/samples.md` or `README.md` index exists, add a one-line entry pointing to `sample/financial-assistant/claude/`. Do NOT create new doc files if none exists (project rule).

---

## Dependencies

- **Phase 0 (FunctionTool runtime) is already merged** (`16d0d44`). No US6-internal Phase 0 work remains.
- **Phase 0.5 (T000 — `Turn.turn_config` field) must land before Phase 5** — T007's converter output and T018's code-grader wiring both pass `turn_program` through `turn_config`.
- Phase 1 → Phase 2 (converter) → Phase 3 (tools) → Phase 4 (grader) can land in parallel with Phases 2–3.
- Phase 5 (`agent.yaml`) depends on Phases 2, 3, 4 (it references their outputs), AND on T001a's verify-gate passing (ensures `parallel_test_cases` + `Turn.turn_config` are real before we bake references to them).
- Phase 6 (README) can draft in parallel; final version depends on Phase 5.
- Phase 7 (integration) depends on Phases 2–5.
- Phase 8 is sequenced last.

Within Phase 2: T003–T006 parallel; T007–T008 serial.
Within Phase 3: T009–T011 parallel; T012 serial.
Within Phase 4: T013–T013a parallel; T014–T015 serial.
Within Phase 7: T021–T023 parallel.

## Independent test criteria (recap)

- Sample runs end-to-end in ≤ 15 minutes from a fresh checkout (SC-001 on real data — T025).
- Converter produces a subset that validates against `TestCaseModel` multi-turn (T006).
- All US3 arg-matcher surfaces exercised on at least one real turn (T018, T021).
- All US4 evaluator surfaces (`numeric` + `code`) exercised (T018, T021).
- Grader import failures surface at config-load time (T019 + FR-025 behavior — verified indirectly by T022 catching a misspelled grader path).

## Notes on constitution + out-of-scope alignment

- **FunctionTool runtime landed ahead of US6** (commit `16d0d44`). The loader, Claude adapter, and SK dispatch are already in tree; US6 consumes them as-is. No core-package expansion is in scope for this story.
- **Claude Agent SDK is the committed default backend** for this sample (consistent with CLAUDE.md "Claude Code First-Class Citizen" design principle). Ollama and OpenAI remain opt-in via `model.provider` swap + matching env var. `ANTHROPIC_API_KEY` is the only credential required for first run.
- The converter lives in `sample/financial-assistant/claude/scripts/` — **user-land**, not in the core HoloDeck package. Consistent with Assumption A10 ("No dataset-specific built-ins") and the updated §Out of Scope bullet (user-land sample-side converters are allowed; the core package stays dataset-agnostic).
- The `type: code` grader stub is illustrative. It operates on `ctx.turn_config["turn_program"]` per the documented grader-context shape; no core HoloDeck changes are needed beyond what US4 ships.
- Constitution Principle I (No-Code-First) — the sample's `agent.yaml` is pure YAML. The three function tools and one code grader are Python code in user-land files, invoked via import paths — consistent with how the canonical fixture (`tests/fixtures/agents/valid_agent.yaml`) and docs examples (`docs/examples/with_tools.yaml`) already document function tools, and with the scoped exception for `type: code` graders logged in spec §Complexity Tracking.
