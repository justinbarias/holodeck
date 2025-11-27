# Implementation Plan: CLI & Core Agent Engine (v0.1)

**Branch**: `001-cli-core-engine` | **Date**: 2025-10-19 | **Spec**: `/specs/001-cli-core-engine/spec.md`
**Input**: Feature specification from `/specs/001-cli-core-engine/spec.md`
**Focus**: User Story 1 - Define Agent Configuration (Priority: P1)

**Note**: This plan focuses on US1 only. US2, US3+ will be planned in subsequent `/speckit.plan` runs.

## Summary

**US1 Goal**: Enable developers to define AI agents through pure YAML configuration. The agent.yaml schema must support model providers (OpenAI, Azure, Anthropic), instructions (file or inline), tool definitions (vectorstore, function, MCP, prompt), and evaluations with flexible model configuration. Configuration MUST be validated with clear error messages. This is foundational—agents cannot execute without a valid configuration.

**Technical Approach**:

1. Design agent.yaml schema (Pydantic models) with comprehensive validation
2. Implement ConfigLoader to parse YAML and validate against schema
3. Create type-safe data structures (Agent, Tool, Evaluation, TestCase entities)
4. Implement graceful error handling with actionable messages
5. Support configuration composition (file references, inline values)

## Technical Context

**Language/Version**: Python 3.10+ (per CLAUDE.md)
**Primary Dependencies**:

- Pydantic v2 (schema validation, data models)
- PyYAML (configuration parsing)
- python-dotenv (environment variable support for API keys)
- Semantic Kernel (agent framework base)

**Storage**: File-based (agent.yaml, instructions files, data sources); no database required for v0.1
**Testing**: pytest (unit + integration tests for config loading and validation)
**Target Platform**: Linux/macOS/Windows (Python CLI, platform-agnostic)
**Project Type**: Single Python package (CLI-first)
**Performance Goals**: Configuration parsing <100ms per agent.yaml file (measured in tests)
**Constraints**:

- Zero external API calls during config validation (validation must be synchronous, fast)
- All validation errors must be human-readable (surface Pydantic errors with custom messages)
- Support for environment variable interpolation in config (e.g., `${OPENAI_API_KEY}`)

**Scale/Scope**:

- Single agent configuration per agent.yaml (no multi-agent mixing in one file)
- Support up to 50 tools per agent
- Support up to 100 test cases per agent.yaml

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

| Principle                      | Requirement                                           | US1 Status  | Justification                                                             |
| ------------------------------ | ----------------------------------------------------- | ----------- | ------------------------------------------------------------------------- |
| **I. No-Code-First**           | Agent config via YAML only, no Python code required   | ✅ PASS     | Schema driven by Pydantic models; configuration entirely declarative      |
| **II. MCP for APIs**           | API integrations use MCP servers, not custom tools    | ✅ PASS     | US1 defines tool type definitions; MCP tool type already in schema        |
| **III. Test-First Multimodal** | Test cases support multimodal inputs + expected_tools | ✅ PASS     | Test case schema includes file support, expected_tools field              |
| **IV. OpenTelemetry Native**   | Tracing/metrics instrumentation from day one          | ⚠️ DEFERRED | US1 focuses on config schema; observability integration in Phase 3+ (US3) |
| **V. Evaluation Flexibility**  | Model config at 3 levels (global/run/metric)          | ✅ PASS     | EvaluationMetric schema supports per-metric model override                |

**Gate Result**: ✅ **PASS** - No blocking violations. Observability deferred to agent execution phase (post-US1).

**Complexity Tracking**: None - all constraints can be met with straightforward Pydantic schema.

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
src/holodeck/
├── config/                           # US1: Configuration loading & validation
│   ├── __init__.py
│   ├── schema.py                     # Pydantic models (Agent, Tool, Evaluation, TestCase)
│   ├── loader.py                     # ConfigLoader (YAML parsing)
│   ├── validator.py                  # Validation logic & error handling
│   └── defaults.py                   # Default configurations & templates
├── models/                           # Data entities
│   ├── __init__.py
│   ├── agent.py                      # Agent entity
│   ├── tool.py                       # Tool entity
│   ├── evaluation.py                 # EvaluationMetric entity
│   └── test_case.py                  # TestCase entity
├── cli/                              # CLI entry point (separate from US1)
│   ├── __init__.py
│   └── main.py
└── lib/                              # Shared utilities
    ├── __init__.py
    └── errors.py                     # Custom exception types

tests/
├── unit/
│   ├── test_config_schema.py         # Pydantic model validation tests
│   ├── test_config_loader.py         # YAML parsing tests
│   ├── test_config_validation.py     # Error message tests
│   └── conftest.py
├── integration/
│   ├── test_config_end_to_end.py     # Full config loading workflows
│   └── conftest.py
└── fixtures/
    ├── agents/
    │   ├── valid_agent.yaml
    │   ├── minimal_agent.yaml
    │   └── invalid_agent.yaml
    └── tools/
```

**Structure Decision**: Single Python package (Option 1). US1 requires:

- `src/holodeck/config/`: Core configuration loading (ConfigLoader, schema validation)
- `src/holodeck/models/`: Type-safe data models (Pydantic)
- `tests/`: Unit + integration tests for config loading and validation

## Complexity Tracking

_No violations - all constraints can be met._

---

## Implementation Strategy

### Phase 0: Research & Decisions (Completed)

From clarifications session 2025-10-19:

1. **SearchResult Type** → Structured {matched_content, metadata_dict, source_reference, relevance_score}
2. **Tool Error Handling** → Graceful degradation (log + empty string), extensible via middleware
3. **Chat Memory** → Configurable (v0.1 CLI: in-memory last N messages)
4. **Session Isolation** → Via session_id, serialized per session
5. **Evaluation Failures** → Soft failure (ERROR status, log details, test continues)

All unknowns resolved. No NEEDS CLARIFICATION remaining.

### Phase 1: Design (This Document - COMPLETE)

✅ data-model.md → Complete entity definitions
✅ contracts/config_loader_interface.md → Interface contract
✅ quickstart.md → Test scenarios & acceptance criteria
✅ project structure → Directories defined
✅ Constitution Check → PASS (no violations)

### Phase 2: Implementation (Next: `/speckit.tasks`)

Tasks for ConfigLoader implementation:

1. Define Pydantic models (schema.py)
2. Implement ConfigLoader class (loader.py)
3. Error handling & validation (validator.py)
4. Unit tests (test_config_schema.py, test_config_loader.py)
5. Integration tests (test_config_end_to_end.py)
6. Fixtures (agent YAML files for testing)

---

## Next Steps

1. **Run `/speckit.tasks plan-us1`** to generate tasks.md with implementation tasks
2. **Execute tasks in order** (Phase 1 setup, Phase 2 foundational, Phase 3 US1)
3. **After US1 complete**: Run `/speckit.plan plan-us2` for User Story 2

---

## Design Decisions

| Decision            | Choice                                     | Rationale                                                   |
| ------------------- | ------------------------------------------ | ----------------------------------------------------------- |
| **Config Schema**   | Pydantic v2                                | Type-safe, built-in validation, excellent error messages    |
| **YAML Parser**     | PyYAML                                     | Industry standard, simple API, adequate performance         |
| **Error Handling**  | Custom exceptions + human messages         | Constitution requires clear, actionable errors              |
| **Tool Validation** | Type-discriminated union (Tool type field) | Clean separation of concerns, extensible for new tool types |
| **File Resolution** | Relative to agent.yaml directory           | Portable, matches industry convention (Docker, k8s, etc.)   |
| **Env Var Pattern** | `${VAR_NAME}`                              | YAML-safe, matches HashiCorp/Terraform convention           |

---

## Risks & Mitigations

| Risk                                  | Mitigation                                                     |
| ------------------------------------- | -------------------------------------------------------------- |
| Pydantic error messages too technical | Custom validator wraps Pydantic errors in human language       |
| Circular file references              | Validate file exists before loading to prevent infinite loops  |
| Large YAML files slow to parse        | Performance budget (< 100ms) covered in tests                  |
| Complex nested tool validation        | Type-discriminated union keeps validation per-tool-type simple |

---

## Appendix: File Structure Decision

```
specs/001-cli-core-engine/
├── spec.md                              # Original feature spec (read-only)
├── plan.md                              # This file (Phase 1 planning)
├── data-model.md                        # Phase 1 design output (entity definitions)
├── quickstart.md                        # Phase 1 design output (test scenarios)
├── contracts/
│   └── config_loader_interface.md      # Phase 1 design output (interface contract)
└── tasks.md                             # Phase 2 output (from /speckit.tasks)
```

This structure ensures:

- Clear separation of planning (Phase 1) from implementation (Phase 2+)
- Frozen spec.md vs. living plan.md
- Design artifacts (data-model, contracts, quickstart) before tasks
- Tasks are generated from complete design
