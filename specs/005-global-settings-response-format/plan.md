# Implementation Plan: Global Settings and Response Format Configuration

**Branch**: `005-global-settings-response-format` | **Date**: 2025-10-25 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/005-global-settings-response-format/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement three-level configuration hierarchy (user-level → project-level → agent-level) with global settings files (`config.yml`/`config.yaml`) and optional response format constraints using Basic JSON Schema. This enables configuration reuse across agents while maintaining clear inheritance and override semantics. Critical for production use cases requiring structured LLM outputs.

## Technical Context

**Language/Version**: Python 3.10+ (per CLAUDE.md)
**Primary Dependencies**:

- PyYAML (YAML parsing)
- jsonschema (JSON Schema validation - Basic draft 2020-12)
- pydantic (Configuration validation)
- pathlib (file path handling)

**Storage**: File-based (YAML files at `~/.holodeck/config.yml|config.yaml` and `config.yml|config.yaml`)
**Testing**: pytest with markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
**Target Platform**: Cross-platform (Linux, macOS, Windows via Python)
**Project Type**: Single library package (src/holodeck/)
**Performance Goals**: Configuration load < 100ms per agent; validation < 50ms per schema
**Constraints**:

- Basic JSON Schema support only (type, properties, required, additionalProperties)
- Configuration files must be valid YAML syntax
- Schema validation at config load time, not runtime
- Assume OpenAI-API compliant LLM providers

**Scale/Scope**: Support 100+ agents per project with centralized configuration; multi-project user support via user-level settings

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

### Principle Compliance Assessment

| Principle                                          | Compliance   | Notes                                                                                                                                       |
| -------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **I. No-Code-First Agent Definition**              | ✅ COMPLIANT | Configuration entirely via YAML (agent.yaml inherits from config.yml/config.yaml). No code required to define settings or response formats. |
| **II. MCP for API Integrations**                   | ✅ N/A       | This feature does not introduce new API integrations; it focuses on configuration. MCP requirements apply to future tool implementations.   |
| **III. Test-First with Multimodal Support**        | ✅ COMPLIANT | Will implement comprehensive unit and integration tests. Response format validation supports structured validation of multimodal outputs.   |
| **IV. OpenTelemetry-Native Observability**         | ✅ COMPLIANT | Configuration loading and validation will be instrumented with OpenTelemetry traces. Schema validation will emit observability signals.     |
| **V. Evaluation Flexibility with Model Overrides** | ✅ COMPLIANT | Global response_format supports per-agent overrides; no restrictions on evaluation-level model choices.                                     |

**Status**: ✅ **GATE PASSES** - Feature aligns with all five core principles. No exceptions or complexity tracking needed.

## Project Structure

### Documentation (this feature)

```
specs/005-global-settings-response-format/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file (implementation plan)
├── research.md          # Phase 0 output (resolves technical unknowns)
├── data-model.md        # Phase 1 output (configuration entities & contracts)
├── quickstart.md        # Phase 1 output (getting started guide)
├── contracts/           # Phase 1 output (configuration schema contracts)
│   └── config-schema.openapi.yaml
├── checklists/
│   └── requirements.md  # Quality checklist
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
src/holodeck/
├── config/              # NEW: Configuration management module
│   ├── __init__.py
│   ├── loader.py        # Configuration file loading (user, project, agent levels)
│   ├── models.py        # Pydantic models for configuration
│   ├── schema.py        # JSON schema validation (response_format)
│   └── merge.py         # Configuration merging & inheritance logic
├── (existing modules)

tests/
├── unit/
│   ├── config/          # NEW: Configuration module tests
│   │   ├── test_loader.py
│   │   ├── test_schema_validation.py
│   │   ├── test_inheritance.py
│   │   └── test_error_handling.py
│   └── (existing tests)
└── integration/
    ├── test_config_end_to_end.py  # NEW: Full integration tests
    └── (existing tests)
```

**Structure Decision**: Single library package with new `src/holodeck/config/` module encapsulating all configuration logic. This maintains separation of concerns and aligns with existing HoloDeck architecture.

## Complexity Tracking

_No violations - Constitution Check passes without exception. This section left empty._

---

## Phase 0: Research & Technical Decisions

### Research Completed (Clarifications Resolved)

All critical unknowns were resolved during spec clarification phase:

1. **Response Format Schema Support Level**: Basic JSON Schema (type, properties, required, additionalProperties) - sufficient for 99% of use cases while maintaining LLM compatibility
2. **LLM Provider Compatibility**: Assume OpenAI-API compliant providers with structured output support; log warning if unsupported
3. **Configuration Inheritance Model**: All-or-nothing override (agent config completely replaces global settings, not granular field-level override)
4. **Configuration File Locations**: User-level (`~/.holodeck/config.yml|config.yaml`) and project-level (`config.yml|config.yaml` at project root)
5. **File Extension Handling**: Support both `.yml` and `.yaml`; prefer `.yml` if both exist

### Technical Dependencies & Best Practices

| Component                | Technology         | Rationale                                                                               |
| ------------------------ | ------------------ | --------------------------------------------------------------------------------------- |
| YAML Parsing             | PyYAML             | Standard, battle-tested Python YAML library with wide adoption                          |
| JSON Schema Validation   | jsonschema         | Pure Python, supports draft 2020-12, no native C dependencies                           |
| Configuration Models     | Pydantic           | Type-safe configuration with automatic validation; integrates with FastAPI              |
| File Path Handling       | pathlib            | Cross-platform path handling; Python 3.10+ standard                                     |
| Configuration Precedence | Custom merge logic | Simple precedence chain (user → project → agent); no complex conflict resolution needed |

### Research Output

This technical context forms the foundation for Phase 1 design. All clarifications have been integrated into the specification.

---

## Phase 1: Design & Contracts

### Completed Artifacts

#### 1. Data Model (data-model.md)

**Defines**:

- GlobalConfig entity (reused for both user-level and project-level)
- AgentConfiguration with response_format and inheritance
- ResponseFormatSchema (Basic JSON Schema only)
- Configuration merging and precedence rules
- File discovery and loading sequence
- Error handling states

**Key Design Decisions**:

- Reuse existing GlobalConfig model (no new model)
- API keys part of LMProvider (not separate)
- Response format and tools are agent-specific only
- All-or-nothing inheritance (complete override, not granular)
- YAML/JSON file support with `.yml` preference

#### 2. Configuration Contracts (contracts/config-schema.yaml)

**Defines**:

- GlobalConfig schema (user-level and project-level)
- AgentConfig schema extensions (response_format, tools)
- LMProvider component schema
- VectorstoreConfig component schema
- DeploymentConfig component schema
- ResponseFormatSchema validation rules
- File format and extension handling
- Configuration precedence rules
- Error response format

#### 3. Quick Start Guide (quickstart.md)

**Covers**:

- 5-minute getting started tutorial
- Configuration file creation (user-level and project-level)
- Adding response formats to agents
- Inheritance examples (4 scenarios)
- Response format examples (3 patterns)
- Common usage patterns
- Troubleshooting guide

### Implementation Readiness

**Modules to Implement**:

1. **src/holodeck/config/loader.py**

   - Load GlobalConfig from user-level location
   - Load GlobalConfig from project-level location
   - Load agent configuration
   - Handle file discovery (.yml vs .yaml preference)
   - Merge configurations with proper precedence

2. **src/holodeck/config/schema.py**

   - Validate response_format JSON schema
   - Enforce Basic JSON Schema keywords only
   - Load external schema files
   - Provide detailed error messages

3. **src/holodeck/config/merge.py**

   - Merge user-level and project-level GlobalConfig
   - Merge global settings with agent config
   - Handle inherit_global flag
   - Apply override semantics

4. **Tests**:
   - test_loader.py - File discovery, loading, YAML parsing
   - test_schema_validation.py - Schema validation, errors
   - test_inheritance.py - Configuration merging, precedence
   - test_error_handling.py - Error messages, edge cases
   - test_config_end_to_end.py - Full integration scenarios

---

## Phase 2: Implementation Task Generation

**Next Command**: `/speckit.tasks`

This will generate the detailed task breakdown (tasks.md) with:

- Development tasks (one per module)
- Test tasks (comprehensive coverage)
- Documentation tasks
- Integration and acceptance test tasks
- Task dependencies and sequencing
- Effort estimates

**Task Breakdown Preview**:

1. **Load User-Level Configuration** - loader.py
2. **Load Project-Level Configuration** - loader.py
3. **Merge Global Configurations** - merge.py
4. **Validate Response Format Schema** - schema.py
5. **Load External Schema Files** - schema.py
6. **Implement Agent Config Inheritance** - merge.py
7. **Handle File Discovery & Extensions** - loader.py
8. **Comprehensive Unit Tests** - tests/unit/config/
9. **Integration Tests** - tests/integration/
10. **Error Message & Edge Cases** - all modules
11. **Documentation & Examples** - docs/
12. **Code Quality & Security** - all modules (format, lint, type-check, security)

---

## Summary

✅ **Phase 0 Complete**: All technical unknowns resolved
✅ **Phase 1 Complete**: Data model, contracts, and design documented
⏳ **Phase 2 Next**: Task generation via `/speckit.tasks`

**Readiness**: Ready for implementation. All design decisions made, no blocking ambiguities.
