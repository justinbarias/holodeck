# Implementation Plan: Interactive Init Wizard

**Branch**: `011-interactive-init-wizard` | **Date**: 2025-11-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/011-interactive-init-wizard/spec.md`

## Summary

Enhance the `holodeck init` command with an interactive wizard that guides users through configuration choices for agent name, LLM provider (Ollama with gpt-oss:20b default, OpenAI, Azure OpenAI, Anthropic), vector store (ChromaDB at http://localhost:8000 default, Redis, In-Memory), evaluation metrics (rag-faithfulness, rag-answer_relevancy default), and MCP servers (brave-search[web-search], memory, sequentialthinking default). The wizard supports both interactive and non-interactive modes with clean cancellation handling.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: Click (CLI framework), Pydantic (validation), PyYAML, InquirerPy (interactive prompts)
**Storage**: N/A (generates YAML configuration files)
**Testing**: pytest with unit/integration markers, pytest-mock for mocking
**Target Platform**: Linux/macOS/Windows terminal environments
**Project Type**: single - CLI extension to existing holodeck package
**Performance Goals**: Interactive wizard completion <60s with defaults, non-interactive <5s
**Constraints**: Terminal must support stdin for interactive mode
**Scale/Scope**: CLI enhancement affecting `src/holodeck/cli/commands/init.py` and supporting modules

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

### I. No-Code-First Agent Definition

**Status**: COMPLIANT

- Users configure agents via interactive YAML wizard prompts
- All selections translate directly to YAML configuration values
- No Python code required from users

### II. MCP for API Integrations

**Status**: COMPLIANT

- MCP servers are first-class citizens in the wizard
- Uses predefined list of common MCP servers
- Generated configuration uses standard MCP tool definitions

### III. Test-First with Multimodal Support

**Status**: COMPLIANT (not directly applicable)

- This feature focuses on project scaffolding, not agent testing
- Generated test cases in scaffolded projects will support multimodal inputs

### IV. OpenTelemetry-Native Observability

**Status**: NOT APPLICABLE

- Init wizard is a CLI scaffolding tool, not runtime agent code
- Generated projects will include observability configuration stubs

### V. Evaluation Flexibility with Model Overrides

**Status**: COMPLIANT

- Wizard includes eval selection step with sensible defaults
- Generated evaluation configuration will support model overrides

### Architecture Constraints

**Status**: COMPLIANT

- Extends CLI layer only (not Agent Engine, Evaluation Framework, or Deployment Engine)
- Uses well-defined configuration models (`LLMProvider`, `DatabaseConfig`, `MCPTool`, `EvalConfig`)

### Code Quality & Testing Discipline

**Status**: TO BE VALIDATED

- Will follow Google Python Style Guide
- Will include comprehensive unit and integration tests
- Will maintain 80%+ coverage on new code

## Project Structure

### Documentation (this feature)

```
specs/011-interactive-init-wizard/
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
├── cli/
│   ├── commands/
│   │   └── init.py           # Enhanced init command with wizard
│   ├── utils/
│   │   ├── project_init.py   # Extended initializer logic
│   │   └── wizard.py         # NEW: Interactive wizard module
│   └── exceptions.py         # Existing exception types
├── models/
│   ├── llm.py                # Existing LLM provider models
│   ├── tool.py               # Existing tool models (MCP, vectorstore)
│   ├── evaluation.py         # Existing evaluation models
│   └── wizard_config.py      # NEW: Wizard selection models
└── templates/                # Existing templates (may need updates)

tests/
├── unit/
│   ├── test_wizard.py        # NEW: Wizard unit tests
│   └── test_wizard_config.py # NEW: Config model tests
├── integration/
│   └── test_init_wizard.py   # NEW: End-to-end wizard tests
└── fixtures/
    └── wizard_defaults.py    # NEW: Default values for testing
```

**Structure Decision**: Extending single-project CLI structure. New modules added to cli/utils/ for wizard logic. Models added to models/ following existing patterns. No external registry client needed.

## Complexity Tracking

_No Constitution violations requiring justification_

| Violation | Why Needed | Simpler Alternative Rejected Because |
| --------- | ---------- | ------------------------------------ |
| N/A       | -          | -                                    |
