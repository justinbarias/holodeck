# Implementation Plan: Ollama Endpoint Support

**Branch**: `009-ollama-endpoint-support` | **Date**: 2025-11-26 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/009-ollama-endpoint-support/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Enable HoloDeck users to configure and use Ollama endpoints (local or remote) as LLM providers for chat and test commands. Users will define Ollama configuration in agent YAML files including provider type, host URL, model name, and execution settings. The implementation will leverage Semantic Kernel's OllamaChatCompletion connector and extend HoloDeck's configuration system to support Ollama-specific schemas with lazy validation on first invocation.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: Semantic Kernel (microsoft/semantic-kernel), Pydantic 2.0+, PyYAML 6.0+, python-dotenv 1.0+
**Storage**: N/A (configuration only)
**Testing**: pytest with markers (unit, integration), pytest-cov, pytest-asyncio
**Target Platform**: Cross-platform CLI (Linux, macOS, Windows)
**Project Type**: Single project (existing HoloDeck codebase)
**Performance Goals**: Chat responses within Ollama endpoint latency + <50ms HoloDeck overhead; configuration loading <100ms
**Constraints**: No significant latency added to Ollama responses; error messages clear enough for 90% of users to self-resolve; lazy validation only on first invocation
**Scale/Scope**: Support 5+ popular Ollama models (llama3, phi3, mistral, codellama, gemma); handle both local (localhost:11434) and remote endpoints; integrate with existing agent/test CLI commands

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

### I. No-Code-First Agent Definition

✅ **PASS** - Ollama configuration is fully declarative via YAML (provider, host, model name, execution settings). No Python code required from users.

### II. MCP for API Integrations

✅ **PASS** - Ollama integration does not introduce custom API tool types. It extends LLM provider configuration only. API integrations (if needed) will continue to use MCP servers.

### III. Test-First with Multimodal Support

✅ **PASS** - Feature enables `holodeck test` command to work with Ollama endpoints. Existing multimodal test infrastructure (images, PDFs, Office docs) remains unchanged and functional.

### IV. OpenTelemetry-Native Observability

⚠️ **DEFERRED** - OpenTelemetry instrumentation is planned for the Agent Engine but not yet implemented. This feature extends configuration and will align with observability when Agent Engine implements it (FR-016 requires logging of connection attempts, errors, response times as preparation).

### V. Evaluation Flexibility with Model Overrides

✅ **PASS** - Feature enables Ollama as an evaluation model option, maintaining existing three-level model configuration hierarchy (global, per-evaluation, per-metric).

### Architecture Constraints

✅ **PASS** - Changes are isolated to configuration layer (models, validation) and will integrate with Agent Engine when implemented. No cross-engine coupling introduced.

### Code Quality & Testing Discipline

✅ **PASS** - Implementation will follow Python 3.10+, Google Style Guide, MyPy strict mode, pytest with markers, 80% coverage minimum, Black/Ruff formatting, Bandit/Safety security scanning, and pre-commit hooks as defined in CLAUDE.md.

**Overall Status**: ✅ APPROVED (1 deferred item aligns with project roadmap)

---

## Constitution Check (Post-Design Re-Evaluation)

_Re-evaluation after Phase 1 design completion_

### I. No-Code-First Agent Definition

✅ **PASS** - Design maintains fully declarative YAML configuration. No new code requirements introduced for users.

### II. MCP for API Integrations

✅ **PASS** - Design does not introduce custom API tool types. Ollama integration is LLM provider level, not tool level.

### III. Test-First with Multimodal Support

✅ **PASS** - Design preserves existing test infrastructure. Quickstart guide includes test case examples with ground_truth validation.

### IV. OpenTelemetry-Native Observability

⚠️ **DEFERRED** - Still deferred pending Agent Engine observability implementation. Error logging added as preparation (research.md Q4).

### V. Evaluation Flexibility with Model Overrides

✅ **PASS** - Design verified: Ollama models can be used at all three levels (global, per-evaluation, per-metric). Quickstart guide demonstrates this pattern.

### Architecture Constraints

✅ **PASS** - Design isolated to configuration layer and AgentFactory (Agent Engine component). No cross-engine coupling. Clean separation maintained.

### Code Quality & Testing Discipline

✅ **PASS** - Design includes comprehensive test strategy with unit tests (mocked) and optional integration tests. Follows pytest conventions with markers.

**Post-Design Status**: ✅ APPROVED - All principles maintained through design phase

## Project Structure

### Documentation (this feature)

```
specs/009-ollama-endpoint-support/
├── spec.md              # Feature specification (input)
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
├── models/
│   ├── llm.py           # MODIFIED - Add OllamaConfig model
│   ├── agent.py         # MODIFIED - Extend to support ollama provider
│   └── config.py        # EXISTING - Base configuration models
├── config/
│   ├── validator.py     # MODIFIED - Add Ollama-specific validation
│   ├── loader.py        # EXISTING - YAML configuration loader
│   └── defaults.py      # MODIFIED - Add Ollama default values
├── cli/
│   └── commands/
│       ├── chat.py      # EXISTING - Will use Ollama via Agent Engine
│       └── test.py      # EXISTING - Will use Ollama via Agent Engine
└── lib/
    └── errors.py        # MODIFIED - Add Ollama-specific error classes

tests/
├── unit/
│   ├── models/
│   │   └── test_llm_ollama.py          # NEW - Test OllamaConfig model
│   └── config/
│       └── test_validator_ollama.py     # NEW - Test Ollama validation
└── integration/
    ├── test_ollama_config_loading.py    # NEW - End-to-end config loading
    └── fixtures/
        └── ollama/
            └── agent_ollama.yaml        # NEW - Sample Ollama agent config
```

**Structure Decision**: Single project structure (Option 1) is used as HoloDeck is a unified CLI application. Ollama support extends existing configuration models and validation logic without introducing new architectural layers. Changes are isolated to configuration layer (`models/`, `config/`) with integration points in CLI commands that will be activated when Agent Engine is implemented.

## Complexity Tracking

_Fill ONLY if Constitution Check has violations that must be justified_

No violations requiring justification. All constitution principles are satisfied or appropriately deferred in alignment with project roadmap.
