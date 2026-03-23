# Tasks: US7 — Agent Skills Replace Prompt Tools

**Input**: Design documents from `/specs/023-choose-your-backend/`
**Prerequisites**: plan.md (required), spec.md (required), research.md (R7), data-model.md (§8)
**Tests**: TDD approach — write tests FIRST, verify they FAIL, then implement.
**Dependencies**: US1 (ADK backend) and US2 (AF backend) must be complete before backend adapter tasks.

**Organization**: Tasks are grouped by phase to enable incremental delivery.

## Format: `[ID] [P?] [US7] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US7]**: All tasks belong to User Story 7

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root

---

## Phase 2: Foundational — Remove PromptTool from ToolUnion

**Purpose**: Clean removal of the unimplemented PromptTool before introducing SkillTool.

**CRITICAL**: SkillTool implementation (Phase 3) cannot begin until PromptTool is fully removed.

- [ ] T701 [P] [US7] Remove `PromptTool` class from `src/holodeck/models/tool.py` — delete the entire `PromptTool(BaseModel)` class definition (currently at line ~555)
- [ ] T702 [P] [US7] Remove `PromptTool` from `ToolUnion` discriminated union in `src/holodeck/models/tool.py` — remove the `Annotated[PromptTool, Tag("prompt")]` variant from the union type (currently at line ~888)
- [ ] T703 [P] [US7] Remove `PromptTool` export from `src/holodeck/models/__init__.py` — remove from `__all__` and import list
- [ ] T704 [US7] Remove any `PromptTool`/`TestPromptTool` tests from `tests/unit/models/test_tool_models.py` — delete existing prompt tool test cases to prevent import errors
- [ ] T705 [US7] Run `make test` to verify PromptTool removal causes no regressions (any remaining references to PromptTool should surface as import errors)

**Checkpoint**: PromptTool fully removed — ToolUnion clean for SkillTool addition.

---

## Phase 3: US7 — SkillTool Model, Validation, Adapters, Tests

### 3a: SkillTool Pydantic Model

**Purpose**: Define the SkillTool model with all validation rules from data-model.md §8.

#### Tests (TDD — write FIRST, verify they FAIL)

- [ ] T706 [P] [US7] Write test `test_inline_skill_valid` in `tests/unit/models/test_tool_models.py` — create SkillTool with `name="sentiment-analyzer"`, `type="skill"`, `description="Analyze sentiment"`, `instructions="Analyze the given text"`, assert model validates successfully
- [ ] T707 [P] [US7] Write test `test_file_based_skill_valid` in `tests/unit/models/test_tool_models.py` — create SkillTool with `name="research-assistant"`, `type="skill"`, `path="./skills/research-assistant/"`, `description="Research tool"`, assert model validates successfully
- [ ] T708 [P] [US7] Write test `test_skill_rejects_both_instructions_and_path` in `tests/unit/models/test_tool_models.py` — provide both `instructions` and `path`, assert `ValidationError` raised
- [ ] T709 [P] [US7] Write test `test_skill_rejects_neither_instructions_nor_path` in `tests/unit/models/test_tool_models.py` — omit both `instructions` and `path`, assert `ValidationError` raised
- [ ] T710 [P] [US7] Write test `test_inline_skill_requires_description` in `tests/unit/models/test_tool_models.py` — create inline skill (with `instructions`) but no `description`, assert `ValidationError` raised
- [ ] T711 [P] [US7] Write test `test_file_based_skill_description_optional` in `tests/unit/models/test_tool_models.py` — create file-based skill (with `path`) but no `description`, assert model validates successfully (description falls back to SKILL.md frontmatter at runtime)
- [ ] T712 [P] [US7] Write test `test_skill_name_pattern_valid` in `tests/unit/models/test_tool_models.py` — test valid names: `"my-skill"`, `"a"`, `"skill-123"`, `"multi-word-skill-name"`, assert all validate
- [ ] T713 [P] [US7] Write test `test_skill_name_pattern_invalid` in `tests/unit/models/test_tool_models.py` — test invalid names: `"MySkill"` (uppercase), `"my_skill"` (underscore), `"-leading"` (leading hyphen), `"trailing-"` (trailing hyphen), `"double--hyphen"` (consecutive hyphens), `""` (empty), assert `ValidationError` for each
- [ ] T714 [P] [US7] Write test `test_skill_name_max_length` in `tests/unit/models/test_tool_models.py` — test name with 65 characters, assert `ValidationError`; test name with 64 characters, assert valid
- [ ] T715 [P] [US7] Write test `test_skill_description_max_length` in `tests/unit/models/test_tool_models.py` — test description with 1025 characters, assert `ValidationError`; test description with 1024 characters, assert valid
- [ ] T716 [P] [US7] Write test `test_skill_with_allowed_tools` in `tests/unit/models/test_tool_models.py` — create SkillTool with `allowed_tools=["knowledge_base", "web_search"]`, assert model validates and `allowed_tools` is stored correctly
- [ ] T717 [P] [US7] Write test `test_skill_in_tool_union` in `tests/unit/models/test_tool_models.py` — create a tool dict with `type: "skill"` and valid fields, parse through `ToolUnion` discriminated union, assert result is `SkillTool` instance

#### Implementation

- [ ] T718 [US7] Implement `SkillTool(BaseModel)` class in `src/holodeck/models/tool.py` with fields: `name` (str, 1-64 chars, pattern `^[a-z0-9]+(-[a-z0-9]+)*$`), `description` (str|None, max 1024 chars), `type` (Literal["skill"]), `instructions` (str|None), `path` (str|None), `allowed_tools` (list[str]|None). Add `@model_validator(mode="after")` enforcing: (1) exactly one of `instructions`/`path` set, (2) `description` required when `instructions` set
- [ ] T719 [US7] Add `SkillTool` to `ToolUnion` discriminated union in `src/holodeck/models/tool.py` — add `Annotated[SkillTool, Tag("skill")]` variant
- [ ] T720 [US7] Export `SkillTool` from `src/holodeck/models/__init__.py` — add to `__all__` and import list
- [ ] T721 [US7] Run tests T706–T717 and verify they PASS after implementation

**Checkpoint**: SkillTool model complete with all validation rules.

---

### 3b: allowed_tools Cross-Validation Against Parent Agent

**Purpose**: Validate that `allowed_tools` references exist in the parent agent's tools list at config time.

#### Tests (TDD — write FIRST, verify they FAIL)

- [ ] T722 [P] [US7] Write test `test_allowed_tools_valid_references` in `tests/unit/models/test_tool_models.py` — create an Agent with tools `[knowledge_base (vectorstore), sentiment-analyzer (skill, allowed_tools=["knowledge_base"])]`, assert config validates successfully
- [ ] T723 [P] [US7] Write test `test_allowed_tools_invalid_reference` in `tests/unit/models/test_tool_models.py` — create an Agent with tools `[sentiment-analyzer (skill, allowed_tools=["nonexistent_tool"])]`, assert `ValidationError` raised with clear error message mentioning the nonexistent tool name
- [ ] T724 [P] [US7] Write test `test_allowed_tools_none_is_valid` in `tests/unit/models/test_tool_models.py` — create a skill with `allowed_tools=None`, assert model validates (skill has no tool access)

#### Implementation

- [ ] T725 [US7] Add `@model_validator(mode="after")` to `Agent` in `src/holodeck/models/agent.py` that iterates over `tools`, finds `SkillTool` instances, and validates each `allowed_tools` entry exists as a `name` in the agent's other tools. Raise `ValidationError` with message: `"Skill '{skill.name}' references unknown tool '{tool_name}' in allowed_tools. Available tools: {available}"`
- [ ] T726 [US7] Run tests T722–T724 and verify they PASS

**Checkpoint**: Cross-validation ensures skill tool references are valid at config time.

---

### 3c: SK Backend Rejection for SkillTool

**Purpose**: SK backend raises a clear error when skill tools are configured (SK is excluded from skill support).

#### Tests (TDD — write FIRST, verify they FAIL)

- [ ] T727 [P] [US7] Write test `test_sk_backend_rejects_skill_tool` in `tests/unit/lib/backends/test_selector.py` — create an Agent with `backend: semantic_kernel` and a skill tool configured, attempt backend selection/initialization, assert error raised explaining skills are not supported on SK backend

#### Implementation

- [ ] T728 [US7] Add skill tool validation to `BackendSelector` or `SKBackend.initialize()` in `src/holodeck/lib/backends/selector.py` (or `sk_backend.py`) — check if any tool in the agent's tools list is a `SkillTool`; if so and the resolved backend is `semantic_kernel`, raise `BackendInitError` with message: `"Skill tools (type: skill) are not supported on the Semantic Kernel backend. Use google_adk, agent_framework, or claude backend instead."`
- [ ] T729 [US7] Run test T727 and verify it PASSES

**Checkpoint**: SK backend clearly rejects skill tools.

---

### 3d: Backend Skill Adapters (Depends on US1, US2)

**Purpose**: Each supported backend adapts SkillTool to its native sub-agent/skill runtime.

> **BLOCKED**: These tasks require US1 (ADK backend) and US2 (AF backend) to be complete.

#### Tests (TDD — write FIRST, verify they FAIL)

- [ ] T730 [P] [US7] Write test `test_adk_skill_adapter_inline` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — create an inline SkillTool, adapt via ADK tool adapter, assert a sub-agent is created with the skill's instructions using the parent's model, and the skill's `allowed_tools` are filtered from the parent's tool set
- [ ] T731 [P] [US7] Write test `test_adk_skill_adapter_file_based` in `tests/unit/lib/backends/test_adk_tool_adapters.py` — create a file-based SkillTool with `path`, adapt via ADK tool adapter, assert skill directory path is passed to ADK's native skill runtime
- [ ] T732 [P] [US7] Write test `test_af_skill_adapter_inline` in `tests/unit/lib/backends/test_af_tool_adapters.py` — create an inline SkillTool, adapt via AF tool adapter, assert a sub-agent is created with the skill's instructions using the parent's model, and `allowed_tools` are filtered
- [ ] T733 [P] [US7] Write test `test_af_skill_adapter_file_based` in `tests/unit/lib/backends/test_af_tool_adapters.py` — create a file-based SkillTool with `path`, adapt via AF tool adapter, assert skill directory path is passed to AF's native skill runtime
- [ ] T734 [P] [US7] Write test `test_claude_skill_adapter_inline` in `tests/unit/lib/backends/test_claude_backend.py` (or dedicated test file) — create an inline SkillTool, adapt via Claude tool adapter in `src/holodeck/lib/backends/tool_adapters.py`, assert sub-agent created using Claude SDK's native sub-agent system with the skill's instructions and filtered tools
- [ ] T735 [P] [US7] Write test `test_claude_skill_adapter_file_based` in `tests/unit/lib/backends/test_claude_backend.py` (or dedicated test file) — create a file-based SkillTool with `path`, adapt via Claude tool adapter, assert skill directory is passed to Claude SDK's native sub-agent system

#### Implementation

- [ ] T736 [US7] Add SkillTool adapter to `src/holodeck/lib/backends/adk_tool_adapters.py` — handle inline skills (create ADK sub-agent with skill instructions and filtered parent tools) and file-based skills (pass skill directory to ADK's native skill runtime). Skill inherits parent's model config
- [ ] T737 [US7] Add SkillTool adapter to `src/holodeck/lib/backends/af_tool_adapters.py` — handle inline skills (create AF sub-agent via AF's delegation system with skill instructions and filtered parent tools) and file-based skills (pass skill directory to AF's native skill runtime). Skill inherits parent's model config
- [ ] T738 [US7] Add SkillTool adapter to `src/holodeck/lib/backends/tool_adapters.py` — handle inline skills (create Claude SDK sub-agent with skill instructions and filtered parent tools) and file-based skills (pass skill directory to Claude SDK's native sub-agent system). Skill inherits parent's model config
- [ ] T739 [US7] Run tests T730–T735 and verify they PASS

**Checkpoint**: All three supported backends can adapt skill tools to their native sub-agent runtimes.

---

### 3e: SKILL.md Frontmatter Parsing (Config-Time Validation)

**Purpose**: For file-based skills, parse SKILL.md YAML frontmatter to validate required fields and provide fallback description.

#### Tests (TDD — write FIRST, verify they FAIL)

- [ ] T740 [P] [US7] Write test `test_skill_md_frontmatter_valid` in `tests/unit/models/test_tool_models.py` — create a tmp skill directory with a valid SKILL.md containing frontmatter (`name`, `description`), create a file-based SkillTool with that path, assert frontmatter is parseable and `description` fallback works
- [ ] T741 [P] [US7] Write test `test_skill_md_frontmatter_missing_name` in `tests/unit/models/test_tool_models.py` — SKILL.md missing `name` in frontmatter, assert validation error
- [ ] T742 [P] [US7] Write test `test_skill_md_missing_file` in `tests/unit/models/test_tool_models.py` — `path` points to a directory with no SKILL.md, assert validation error
- [ ] T743 [P] [US7] Write test `test_file_based_skill_fallback_description` in `tests/unit/models/test_tool_models.py` — file-based skill with no `description` in YAML but SKILL.md frontmatter has `description: "From frontmatter"`, assert the skill resolves to the frontmatter description

#### Implementation

- [ ] T744 [US7] Add SKILL.md frontmatter parsing utility — either as a private method on `SkillTool` or a standalone function in `src/holodeck/models/tool.py`. Uses PyYAML (existing dependency) to parse YAML frontmatter from SKILL.md. Extract `name` and `description` fields. Validate that `name` is present. Provide `description` fallback for file-based skills when not set in YAML
- [ ] T745 [US7] Run tests T740–T743 and verify they PASS

**Checkpoint**: File-based skills have config-time validation and description fallback from SKILL.md frontmatter.

---

## Phase 4: Polish — Documentation and Examples

**Purpose**: Update documentation and examples to reflect SkillTool replacing PromptTool.

- [ ] T746 [P] [US7] Update `docs/examples/with_tools.yaml` — replace the prompt tool example with a skill tool example showing both inline and file-based forms
- [ ] T747 [US7] Run `make format` to format all modified files with Black + Ruff
- [ ] T748 [US7] Run `make lint` and fix any Ruff + Bandit violations
- [ ] T749 [US7] Run `make type-check` and fix any MyPy errors (ensure SkillTool type annotations are correct)
- [ ] T750 [US7] Run full test suite `make test` to verify no regressions

**Checkpoint**: US7 complete — PromptTool replaced by SkillTool across model, validation, all backends, and docs.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 2 (Foundational)**: No dependencies — can start immediately
- **Phase 3a (SkillTool Model)**: Depends on Phase 2 completion (PromptTool removed)
- **Phase 3b (Cross-Validation)**: Depends on Phase 3a (SkillTool model exists)
- **Phase 3c (SK Rejection)**: Depends on Phase 3a (SkillTool model exists). Can run in parallel with 3b.
- **Phase 3d (Backend Adapters)**: Depends on Phase 3a + US1 (ADK backend) + US2 (AF backend). BLOCKED until both are complete.
- **Phase 3e (SKILL.md Parsing)**: Depends on Phase 3a. Can run in parallel with 3b, 3c, 3d.
- **Phase 4 (Polish)**: Depends on all Phase 3 sub-phases being complete

### External Dependencies

- **US1 (ADK Backend)**: Required for T730, T731, T736 (ADK skill adapter)
- **US2 (AF Backend)**: Required for T732, T733, T737 (AF skill adapter)
- Claude backend skill adapter (T734, T735, T738) has no external dependency — Claude backend already exists

### Parallel Opportunities

- T701, T702, T703 can run in parallel (different files)
- T706–T717 can run in parallel (same file, independent test functions)
- T722, T723, T724 can run in parallel (same file, independent test functions)
- T730–T735 can run in parallel (different files, independent test functions)
- T740–T743 can run in parallel (same file, independent test functions)
- Phase 3b, 3c, 3e can run in parallel after 3a completes

---

## Key Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `src/holodeck/models/tool.py` | 2, 3a | Remove PromptTool; add SkillTool class + ToolUnion variant |
| `src/holodeck/models/__init__.py` | 2, 3a | Remove PromptTool export; add SkillTool export |
| `src/holodeck/models/agent.py` | 3b | Add `@model_validator` for allowed_tools cross-validation |
| `src/holodeck/lib/backends/selector.py` or `sk_backend.py` | 3c | Reject skill tools on SK backend |
| `src/holodeck/lib/backends/adk_tool_adapters.py` | 3d | Add SkillTool → ADK sub-agent adapter |
| `src/holodeck/lib/backends/af_tool_adapters.py` | 3d | Add SkillTool → AF sub-agent adapter |
| `src/holodeck/lib/backends/tool_adapters.py` | 3d | Add SkillTool → Claude sub-agent adapter |
| `tests/unit/models/test_tool_models.py` | 2, 3a, 3b, 3e | Remove PromptTool tests; add SkillTool tests |
| `tests/unit/lib/backends/test_selector.py` | 3c | SK rejection test |
| `tests/unit/lib/backends/test_adk_tool_adapters.py` | 3d | ADK skill adapter tests |
| `tests/unit/lib/backends/test_af_tool_adapters.py` | 3d | AF skill adapter tests |
| `tests/unit/lib/backends/test_claude_backend.py` | 3d | Claude skill adapter tests |
| `docs/examples/with_tools.yaml` | 4 | Replace prompt tool example with skill example |

---

## Notes

- [P] tasks = different files, no dependencies
- [US7] label maps all tasks to User Story 7 for traceability
- TDD: Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate phase independently
- PromptTool was never implemented — removal is safe with no runtime impact
- SKILL.md frontmatter parsing uses PyYAML (existing dependency) — no new deps needed
- Each backend handles SKILL.md content natively via its own skill runtime; HoloDeck only validates frontmatter at config time
- No model override on skills — they inherit the parent agent's model entirely
- SK is excluded from skill support (planned for deprecation)
