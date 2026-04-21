---
description: "Tasks for User Story 6 — Sample agent: Financial Assistant backed by ConvFinQA (P3)"
---

# Tasks: User Story 6 — Sample Agent: Financial Assistant on ConvFinQA (P3)

**Input**: `/specs/032-multi-turn-test-cases/` — spec.md (US6), quickstart.md (ConvFinQA example), contracts/test-case-schema.md, contracts/code-grader-contract.md. Source dataset: `/Users/justinbarias/Documents/Git/python/justinbarias/data/convfinqa_dataset.json` (21 MB, `{"train": [...], "dev": [...]}` shape — verified).
**Approach**: TDD where meaningful (converter utility, function tools, grader stub); example-driven authoring for the agent configuration + README.

**Story Goal**: Ship `sample/financial_assistant/` that demonstrates multi-turn test cases end-to-end on a real benchmark. User can `cd sample/financial-assistant && holodeck test` from a fresh checkout and see 10 multi-turn ConvFinQA-shaped cases execute with per-turn reports, per-turn tool assertions (fuzzy arg matchers), deterministic `numeric` scoring, and one illustrative `type: code` grader.

**Independent Test**: From a fresh checkout with Ollama running: `cd sample/financial-assistant && holodeck test`. Expect 10 test cases, 30–50 turns total, all complete, per-turn breakdown in the markdown report, exit code reflects pass/fail honestly. Setup-to-first-run ≤ 15 minutes following `README.md` (SC-001 on real data).

**Depends on**: US1 (multi-turn execution), US2 (per-turn assertions), US3 (arg matchers — fuzzy matchers on `subtract`/`divide` args), US4 (`numeric` built-in + `type: code` grader). US5 is independent but complements (the sample's runs are dashboard fodder).

**Not in scope**: Committing the full 21 MB `convfinqa_dataset.json` — only a curated 10-example subset lands under `data/`. The converter utility regenerates the subset from the upstream dataset path.

---

## Phase 0: FunctionTool runtime support (prerequisite)

**Why this phase exists**: Verified via code search — `type: function` is schema-only today. `src/holodeck/lib/test_runner/agent_factory.py:1011-1021` and `src/holodeck/lib/backends/tool_adapters.py:147-199` both dispatch only on `VectorstoreTool` / `MCPTool` / `HierarchicalDocumentToolConfig`; `FunctionTool` is silently ignored. No integration test exercises a function tool end-to-end. The only fixture that declares one (`tests/fixtures/agents/valid_agent.yaml:23-27`) points at a non-existent `tools/orders.py`. US6's `subtract` / `divide` / `lookup` tools cannot ship without this runtime on both backends (SK + Claude Agent SDK).

### Tests first

- [ ] T000a [P] [US6] Add `tests/fixtures/tools/echo.py` with `def echo(message: str) -> str` that returns `message`. Shared fixture for every Phase 0 test below.
- [ ] T000b [P] [US6] Write `tests/unit/lib/test_function_tool_loader.py::test_imports_file_and_returns_callable` — given a `FunctionTool(file="tests/fixtures/tools/echo.py", function="echo")` and a `base_dir`, the loader uses `importlib.util.spec_from_file_location` to import, `getattr`s the function, asserts callable, and returns it.
- [ ] T000c [P] [US6] Add `test_function_tool_loader.py::test_missing_file_raises_config_error`, `::test_missing_function_raises_config_error`, `::test_non_callable_raises_config_error` — each raises `ConfigError` naming the tool, `file`, and underlying exception (mirrors FR-025 shape).
- [ ] T000d [P] [US6] Write `tests/unit/lib/backends/test_function_tool_sk.py::test_sk_backend_registers_function_tool` — feed the SK backend an agent config whose tools list includes the echo `FunctionTool`; assert the SK kernel has a registered callable under the tool name after `backend.initialize()`. Use the existing SK backend smoke pattern (see `test_sk_backend.py` for scaffold).
- [ ] T000e [P] [US6] Write `tests/unit/lib/backends/test_function_tool_claude.py::test_claude_adapter_wraps_function_tool` — `tool_adapters.create_tool_adapters` handed a `FunctionTool` returns an SDK-MCP-compatible adapter carrying a JSON schema derived from the callable's type hints and a `call()` that returns the function result. Extends the existing `tool_adapters.py` test file.
- [ ] T000f [P] [US6] Write `tests/integration/test_function_tool_e2e.py::test_sk_invokes_function_tool` and `::test_claude_invokes_function_tool` — scripted agents invoke `echo`; responses carry the echoed string. Gate Claude variant on `ANTHROPIC_API_KEY`, SK variant on `HOLODECK_IT_SK` (or the existing integration env convention).

### Implementation

- [ ] T000g [US6] Create `src/holodeck/lib/function_tool_loader.py` exposing `load_function_tool(tool: FunctionTool, base_dir: Path) -> Callable`:
  - Resolves `tool.file` relative to `base_dir` (the agent project root — same one already threaded into tool initialization).
  - Imports via `importlib.util.spec_from_file_location` + `module_from_spec` + `loader.exec_module` — no `sys.path` mutation.
  - `getattr(module, tool.function)`; assert `callable`.
  - Raises `ConfigError` naming the tool name, resolved file path, and underlying exception on any failure.
- [ ] T000h [US6] Extend `src/holodeck/lib/backends/tool_adapters.py::create_tool_adapters` (currently dispatching at lines 147-199): add an `isinstance(tool, FunctionTool)` branch that calls `load_function_tool`, derives a JSON schema from the callable's type hints via `inspect.signature` + `typing.get_type_hints`, and wraps as an SDK MCP tool matching the existing vectorstore-adapter pattern. The adapter's `call()` invokes the Python callable with the agent-provided args and returns the result.
- [ ] T000i [US6] Extend SK backend registration: locate the dispatch site in `src/holodeck/lib/test_runner/agent_factory.py` around lines 1011-1021 (or the equivalent in `src/holodeck/lib/backends/sk_backend.py`) and add an `isinstance(tool, FunctionTool)` branch that calls `load_function_tool` and registers the callable as a Semantic Kernel `KernelFunction` / plugin via the SK idiom already used for MCPTool (mirror the existing tool-init pattern — do not invent a new one).
- [ ] T000j [US6] Update `tests/fixtures/agents/valid_agent.yaml` by adding a matching `tests/fixtures/agents/tools/orders.py` with a `get_order_status` stub so the declared tool can actually load. Prior to Phase 0 this fixture was schema-only; the runtime now exists.
- [ ] T000k [US6] Update `sample/support/tools/README.md` and `sample/research/tools/README.md` Function Tools examples to include the required `file:` and `function:` fields (both currently show schema-incomplete examples). Small doc fix; unblocks future copy-paste from the templates.
- [ ] T000l [US6] Run `make format && make lint && make type-check && pytest tests/unit/lib/test_function_tool_loader.py tests/unit/lib/backends -n auto`. All green. Gate T000f integration run on credentials.

---

## Phase 1: Setup + scaffold

- [ ] T001 Read spec US6 + quickstart.md §1–§6. Confirm **Phase 0 (FunctionTool runtime) is merged** AND US1 + US2 foundational tasks are merged (multi-turn execution exists end-to-end); US3/US4 ideally merged (arg matchers + `numeric` + `code` evaluator) — sample can scaffold in parallel with US3/US4 but the full demo needs both landed.
- [ ] T001a [US6] Verify-only gate before starting Phase 5: confirm `ExecutionConfig.parallel_test_cases` (FR-009a, `src/holodeck/models/config.py:51`) is a defined field and `Turn` (wherever US1 lands it) carries a mechanism for passing arbitrary per-turn keys through to `GraderContext.turn_config` (contracts/code-grader-contract.md §3.1) — either an explicit `turn_config: dict[str, Any]` field or `ConfigDict(extra="allow")`. If either is missing or was renamed during US1 review, pause US6 Phase 5 and reconcile with US1 before `agent.yaml` / grader wiring drifts.
- [ ] T002 Create directory structure under the repo root:
  ```
  sample/financial_assistant/
  ├── agent.yaml
  ├── .env.example
  ├── README.md
  ├── instructions/
  │   └── system-prompt.md
  ├── data/
  │   └── convfinqa_subset.json        # created by T007
  ├── tools/
  │   └── financial_tools.py
  ├── graders/
  │   ├── __init__.py
  │   └── turn_program_equivalence.py
  └── scripts/
      └── convert_convfinqa.py
  ```
  Add the directory to repo root without modifying root `pyproject.toml` — sample agents follow the existing convention (see `sample/research/`, `sample/support/`).

---

## Phase 2: Converter utility (TDD)

### Tests first

- [ ] T003 [P] [US6] Write `tests/unit/sample/test_convert_convfinqa.py::test_converts_single_dialogue_to_multi_turn_shape` — given a fixture ConvFinQA record (`id`, `doc`, `dialogue` with `conv_questions`, `conv_answers`, `turn_program`, `executed_answers`), the converter emits a dict matching `TestCaseModel` multi-turn shape: `name=<id>`, `turns=[{input, ground_truth, turn_config: {turn_program}}]` with `len(turns) == len(conv_questions)`.
- [ ] T004 [P] [US6] Add `test_convert_convfinqa.py::test_preserves_turn_program_via_turn_config` — `turn_program` (e.g., `"subtract(206588, 181001)"`) is attached on each turn as `turn_config.turn_program` so the code grader can read it via `ctx.turn_config["turn_program"]` (per contracts/code-grader-contract.md §3.1).
- [ ] T005 [P] [US6] Add `test_convert_convfinqa.py::test_handles_turn_program_with_backrefs` — ConvFinQA's later turns use `#N` back-refs (e.g., `"subtract(206588, 181001), divide(#0, 181001)"`). Converter preserves the raw string faithfully; does not try to resolve `#N` (that's the grader's job).
- [ ] T006 [P] [US6] Add `test_convert_convfinqa.py::test_output_validates_against_testcase_model` — converter output, when loaded through `TestCaseModel(**data)`, validates without error (guards against schema drift).

### Implementation

- [ ] T007 [US6] Write `sample/financial_assistant/scripts/convert_convfinqa.py`:
  ```
  Usage: python scripts/convert_convfinqa.py \
           --source /path/to/convfinqa_dataset.json \
           --split dev \
           --n 10 \
           --out data/convfinqa_subset.yaml \
           [--inject-into agent.yaml]
  ```
  - Reads `{"train": [...], "dev": [...]}`; picks first N examples from `--split`.
  - Emits `data/convfinqa_subset.yaml` in **HoloDeck's native test_cases format** — a YAML list of `{name, turns}` entries that `ConfigLoader` can parse directly. No custom YAML directives (no `!include` — HoloDeck's `yaml.safe_load` rejects custom tags).
  - Each turn: `{input: conv_questions[i], ground_truth: str(executed_answers[i]), turn_config: {turn_program: turn_program[i]}}`. The `turn_program` lands under `turn_config` so the code grader can read it via `ctx.turn_config["turn_program"]` per contracts/code-grader-contract.md §3.1.
  - When `--inject-into agent.yaml` is provided, the converter opens `agent.yaml`, locates sentinel comments (`# BEGIN convfinqa subset` … `# END convfinqa subset`) inside the `test_cases:` block, and replaces everything between them with the generated entries inlined. Idempotent — rerunnable with no drift. This is how test cases reach runtime; the standalone YAML file is kept for human diff review.
  - Uses PyYAML (already a HoloDeck runtime dep) plus stdlib.
  - Emits a one-line summary: `"Converted N dialogues → <path> (injected: <agent.yaml>)"` when `--inject-into` is used.
- [ ] T008 [US6] Run the converter once against the upstream dataset with `--inject-into agent.yaml` to (a) produce `sample/financial_assistant/data/convfinqa_subset.yaml` AND (b) populate the `test_cases:` block in `sample/financial_assistant/agent.yaml`. Commit both artifacts; expect ≤ 150 KB total at N=10 (budget allowance for full SEC document tables in each dialogue's `doc`).

---

## Phase 3: Function tools (TDD)

### Tests first

- [ ] T009 [P] [US6] Write `tests/unit/sample/test_financial_tools.py::test_subtract_basic` — `subtract(a=206588, b=181001) == 25587`; also tests negative result, float args, and `subtract(a=1.0, b=1) == 0.0` (int/float parity).
- [ ] T010 [P] [US6] Add `test_financial_tools.py::test_divide_basic_and_zero_guard` — `divide(a=25587, b=181001) ≈ 0.14136` (abs tol 1e-5); `divide(a=1, b=0)` raises `ZeroDivisionError` (or returns a documented sentinel — pick one and document in the tool docstring).
- [ ] T011 [P] [US6] Add `test_financial_tools.py::test_lookup_returns_scalar` — `lookup(document, description)` where `document` is the ConvFinQA `doc` structure (with `pre_text`, `post_text`, `table`), returns the numeric scalar matching `description`. Exact implementation of `lookup` can be "find by description in the table" — verify for a fixture row.

### Implementation

- [ ] T012 [US6] Write `sample/financial_assistant/tools/financial_tools.py` with three pure-Python functions:
  ```python
  def subtract(a: float | int, b: float | int) -> float: ...
  def divide(a: float | int, b: float | int) -> float: ...
  def lookup(document: dict, description: str) -> float | None: ...
  ```
  `lookup` implementation: case-insensitive substring search across `document["table"]` row/column headers; returns the first matching numeric cell. Docstrings required for all three (HoloDeck `type: function` tools auto-use the docstring as the tool description).

---

## Phase 4: Optional code grader stub

### Tests first

- [ ] T013 [P] [US6] Write `tests/unit/sample/test_turn_program_equivalence.py::test_matching_program_passes` — import `ToolInvocation` from `holodeck.models.test_result` and `GraderContext, GraderResult` from `holodeck.lib.test_runner.code_grader`. Given `ctx.turn_config["turn_program"] = "subtract(206588, 181001)"` and `ctx.tool_invocations = (ToolInvocation(name="subtract", args={"a":206588, "b":181001}, ...),)` (note: tuple, per frozen-dataclass contract §3.1), the grader returns `GraderResult(score=1.0, passed=True)`.
- [ ] T013a [P] [US6] Add `test_turn_program_equivalence.py::test_diverging_program_fails` — same `turn_program` but agent called `divide` instead; grader returns `GraderResult(score=0.0, passed=False)` with a reason string.

### Implementation

- [ ] T014 [US6] Write `sample/financial_assistant/graders/turn_program_equivalence.py`:
  ```python
  from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult

  def turn_program_equivalence(ctx: GraderContext) -> GraderResult:
      expected = ctx.turn_config.get("turn_program", "")
      # Minimal prototype: compare op names; detailed DAG walk with #N
      # backref substitution is left as an exercise (linked in README).
      ...
  ```
  Keep the implementation minimal — this is an illustration of the grader interface, not a production program-equivalence checker. README points users to quickstart.md §5 for a fuller treatment.
- [ ] T015 [US6] Add `sample/__init__.py`, `sample/financial_assistant/__init__.py`, and `sample/financial_assistant/graders/__init__.py` (all empty) so the dotted path `sample.financial_assistant.graders.turn_program_equivalence` resolves via normal Python package semantics from the repo root. HoloDeck's test runner already places cwd + repo root on `sys.path`; verify in T022 that a misspelled grader path surfaces as a load-time `ConfigError` (FR-025).

---

## Phase 5: Agent configuration

- [ ] T016 [US6] Write `sample/financial_assistant/instructions/system-prompt.md` — financial-analyst persona (≤ 300 words): "You are a financial analyst. Use `lookup`, `subtract`, `divide` to answer numeric questions about an SEC filing. Return plain numbers (no commas, no percent signs unless explicitly asked). Treat each user turn as a follow-up to the previous exchanges; resolve anaphoric references like 'what about in 2008?' using prior context." Mirror the tone of `sample/research/instructions/system-prompt.md`.
- [ ] T017 [US6] Write `sample/financial_assistant/agent.yaml` using **Claude Agent SDK as the default backend** (HoloDeck first-class per CLAUDE.md) and the real `FunctionTool` schema (`{name, description, type: function, file, function}` — verified against `src/holodeck/models/tool.py:422-457` and `tests/fixtures/agents/valid_agent.yaml:22-27`):
  ```yaml
  name: financial-assistant
  description: "ConvFinQA-driven financial analyst demonstrating multi-turn test cases"
  model:
    provider: anthropic
    name: claude-sonnet-4-5
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
  test_cases:
    # BEGIN convfinqa subset
    # (generated by scripts/convert_convfinqa.py --inject-into agent.yaml — see T008)
    # END convfinqa subset
  ```
  The `test_cases:` block is populated inline by the converter's `--inject-into` mode (T007/T008); no `!include` directive is used (HoloDeck's `yaml.safe_load` rejects custom tags). Backend-switch instructions live in `README.md` (T020) rather than inline comments: users can flip `model.provider` to `openai` or `ollama` and set the matching env var.
- [ ] T018 [US6] Enrich **at least two** test cases in the generated subset with:
  - `expected_tools` using `{name, args: {a: {fuzzy: "..."}, b: {fuzzy: "..."}}}` object form (exercises US3 matchers on real ConvFinQA numeric values).
  - An `evaluations` block adding `{type: code, grader: "sample.financial_assistant.graders.turn_program_equivalence:turn_program_equivalence"}` on one turn that has a `turn_program` (exercises US4 code grader).
  These augmentations happen via a second pass over `convfinqa_subset.json` — extend the converter T007 with an `--augment` mode OR edit the two cases by hand after conversion and commit.
- [ ] T019 [US6] Write `sample/financial_assistant/.env.example` with stubs for:
  ```
  # Default backend: Claude Agent SDK (Anthropic) — required for first run
  ANTHROPIC_API_KEY=

  # Optional alternate backends — set these and switch `model.provider` in agent.yaml
  # OPENAI_API_KEY=
  # OLLAMA_BASE_URL=http://localhost:11434
  ```

---

## Phase 6: README + quickstart

- [ ] T020 [US6] Write `sample/financial_assistant/README.md` (target ≤ 350 words — SC-001 time budget):
  - Prereqs: HoloDeck installed (`pip install -e .` from repo root) and an `ANTHROPIC_API_KEY` with access to `claude-sonnet-4-5`.
  - Step 1: `cp .env.example .env` and fill in `ANTHROPIC_API_KEY`.
  - Step 2: `holodeck test` — the committed `agent.yaml` already carries the injected `test_cases:`. Expect per-turn output like quickstart §3.
  - Step 3 (optional regen): `python scripts/convert_convfinqa.py --source <path-to-convfinqa_dataset.json> --split dev --n 10 --out data/convfinqa_subset.yaml --inject-into agent.yaml` to refresh the subset and re-inject into `agent.yaml`.
  - Step 4 (optional backend swap): switch `model.provider` to `openai` (set `OPENAI_API_KEY`) or `ollama` (set `OLLAMA_BASE_URL`; pull a capable local model). Dual-backend coverage is the counterpart to SC-010 on real data.
  - Troubleshooting table: `ANTHROPIC_API_KEY` missing, grader import error, dataset path typo, `test_cases:` block empty (re-run `--inject-into`).
  - Cross-reference `specs/032-multi-turn-test-cases/quickstart.md` for the full feature tutorial.

---

## Phase 7: End-to-end smoke + integration

- [ ] T021 [P] [US6] Add `tests/integration/test_sample_financial_assistant.py::test_sample_loads_and_runs_on_stubbed_backend` — loads `sample/financial_assistant/agent.yaml`, stubs the backend with scripted responses matching ground truths, runs `test_runner.execute`, asserts: (a) 10 `TestResult` objects produced; (b) each has `turns` populated with `len(turns) >= 3`; (c) per-turn `metric_results` includes `numeric`; (d) at least one turn carries `arg_match_details` from the augmented test cases (T018); (e) at least one turn carries a `code`-kind `MetricResult` from the grader.
- [ ] T022 [US6] Add `tests/integration/test_sample_financial_assistant.py::test_sample_loads_via_testcase_model` — simple config-validation smoke: `ConfigLoader().load("sample/financial_assistant/agent.yaml")` succeeds; `Agent(**config)` validates. Catches YAML drift without invoking a backend.
- [ ] T023 [P] [US6] (Optional) Add gated dual-backend tests in `test_sample_financial_assistant.py`: `::test_sample_runs_on_anthropic` under `@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"))` (the default path — smokes the committed `model.provider: anthropic`), and `::test_sample_runs_on_ollama` under `HOLODECK_IT_OLLAMA` / `::test_sample_runs_on_openai` under `HOLODECK_IT_OPENAI` for the opt-in SK-backend variants. Runs the real sample against the real backend; asserts non-zero-length responses per turn. Counterpart to SC-010 on a real dataset.

---

## Phase 8: Polish + repo-level visibility

- [ ] T024 [US6] Run `make format && make lint && make type-check && pytest tests/unit/sample -n auto && pytest tests/integration/test_sample_financial_assistant.py -n auto`. All green.
- [ ] T025 [P] [US6] Manual end-to-end validation: fresh terminal, `cd sample/financial-assistant && holodeck test`. Confirm ≤ 15-minute clock time from a clean checkout to first green run (walk through `README.md` as a first-time user would).
- [ ] T026 [P] [US6] If a top-level `docs/samples.md` or `README.md` index exists, add a one-line entry pointing to `sample/financial_assistant/`. Do NOT create new doc files if none exists (project rule).

---

## Dependencies

- **Phase 0 (FunctionTool runtime) is a hard prerequisite for Phase 1** — without it, Phases 3 (function tools) and 7 (end-to-end) cannot validate; `agent.yaml` would load but the agent would never actually invoke `subtract` / `divide` / `lookup`.
- Phase 1 → Phase 2 (converter) → Phase 3 (tools) → Phase 4 (grader) can land in parallel with Phases 2–3.
- Phase 5 (`agent.yaml`) depends on Phases 2, 3, 4 (it references their outputs), AND on T001a's verify-gate passing (ensures US1 schema surfaces `parallel_test_cases` + `turn_config` before we bake references to them).
- Phase 6 (README) can draft in parallel; final version depends on Phase 5.
- Phase 7 (integration) depends on Phases 2–5.
- Phase 8 is sequenced last.

Within Phase 0: T000a–T000f parallel; T000g (loader) → T000h / T000i (backend dispatch) serial per-backend; T000j / T000k / T000l serial after.
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

- **FunctionTool runtime (Phase 0) is a core-package expansion, justified by US6's charter.** `type: function` was schema-only before this work (verified in `src/holodeck/lib/backends/tool_adapters.py:147-199` and `src/holodeck/lib/test_runner/agent_factory.py:1011-1021`). US6 needs real tool invocations to exercise US3's arg matchers and US4's code grader on real data, so the runtime ships here rather than as a separate feature. Consumers beyond US6 benefit (the existing `tests/fixtures/agents/valid_agent.yaml` fixture becomes runnable, and the `sample/support` / `sample/research` READMEs that advertise function tools finally have a working runtime backing them). The dispatch is additive to both backends — no change to the `AgentBackend` / `AgentSession` protocols.
- **Claude Agent SDK is the committed default backend** for this sample (consistent with CLAUDE.md "Claude Code First-Class Citizen" design principle). Ollama and OpenAI remain opt-in via `model.provider` swap + matching env var. `ANTHROPIC_API_KEY` is the only credential required for first run.
- The converter lives in `sample/financial_assistant/scripts/` — **user-land**, not in the core HoloDeck package. Consistent with Assumption A10 ("No dataset-specific built-ins") and the updated §Out of Scope bullet (user-land sample-side converters are allowed; the core package stays dataset-agnostic).
- The `type: code` grader stub is illustrative. It operates on `ctx.turn_config["turn_program"]` per the documented grader-context shape; no core HoloDeck changes are needed beyond what US4 ships.
- Constitution Principle I (No-Code-First) — the sample's `agent.yaml` is pure YAML. The three function tools and one code grader are Python code in user-land files, invoked via import paths — consistent with how the canonical fixture (`tests/fixtures/agents/valid_agent.yaml`) and docs examples (`docs/examples/with_tools.yaml`) already document function tools, and with the scoped exception for `type: code` graders logged in spec §Complexity Tracking.
