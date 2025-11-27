# Implementation Plan: Initialize New Agent Project

**Branch**: `004-init-agent-project` | **Date**: 2025-10-22 | **Spec**: [004-init-agent-project/spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-init-agent-project/spec.md`

## Summary

Implement the `holodeck init` CLI command to bootstrap new AI agent projects with pre-built templates. This feature provides the foundational onboarding experience for HoloDeck, enabling developers to create fully-formed project structures with example configurations, test cases, and data files in under 30 seconds. Three domain-specific templates (conversational, research, customer-support) provide starter configurations aligned to common use cases. This is a P1 feature essential for MVP viability and demonstrates HoloDeck's no-code vision through YAML-first project definition.

## Technical Context

**Language/Version**: Python 3.10+ (per Constitution)
**Primary Dependencies**: Click (CLI framework), Pyyaml (YAML parsing/generation), Jinja2 (template rendering)
**Storage**: File system only (project directory structure and files)
**Testing**: pytest with @pytest.mark.unit, @pytest.mark.integration markers
**Target Platform**: CLI tool (Linux/macOS/Windows)
**Project Type**: Single Python package (CLI command + template system)
**Performance Goals**: Project initialization completes in <30 seconds (SC-001)
**Constraints**: Projects created only in current working directory; no external network calls required (templates bundled)
**Scale/Scope**: Support 3 built-in templates; extensible design for v0.2+ custom templates

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

| Principle                                      | Status      | Verification                                                                                                                                                                                           |
| ---------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| I. No-Code-First Agent Definition              | ✅ PASS     | Project templates define agents entirely through YAML (agent.yaml, instructions/, tools/ config). Users create agents without writing code—just editing YAML and example files.                        |
| II. MCP for API Integrations                   | ✅ PASS     | This feature is CLI-only; no API integrations. Template examples reference MCP pattern but don't implement APIs. N/A for this scope.                                                                   |
| III. Test-First with Multimodal Support        | ✅ PASS     | Generated test case examples include input, expected_tools, ground_truth structure. Multimodal support deferred to v0.2+; current templates provide text-based examples.                               |
| IV. OpenTelemetry-Native Observability         | ⚠️ DEFERRED | CLI initialization logging is basic (success/error messages). Structured logging with OpenTelemetry follows in Agent Engine (001-cli-core-engine Phase 2). Current scope: file creation feedback only. |
| V. Evaluation Flexibility with Model Overrides | ✅ PASS     | Templates include evaluation examples structure (deferred details). No evaluation execution in init command itself.                                                                                    |

**Overall Gate Status**: ✅ PASS - Feature aligns with Constitution. Observability deferral is appropriate for CLI bootstrap feature (not part of core agent execution).

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
├── cli/
│   ├── __init__.py
│   ├── commands/
│   │   ├── __init__.py
│   │   └── init.py                    # holodeck init command (Click command group)
│   └── utils/
│       ├── __init__.py
│       └── project_init.py            # Core initialization logic
├── models/
│   ├── __init__.py
│   ├── project_config.py              # Pydantic AgentConfig model (from 001-cli-core-engine)
│   └── template_manifest.py           # TemplateManifest model (validates template structure)
├── templates/                         # Built-in bundled templates (resources)
│   ├── __init__.py
│   ├── conversational/
│   │   ├── agent.yaml.j2              # Jinja2 template with embedded test_cases (validates against AgentConfig)
│   │   ├── instructions/system-prompt.md.j2
│   │   ├── tools/README.md.j2
│   │   ├── data/faqs.md
│   │   └── manifest.yaml              # Template metadata (name, description, defaults)
│   ├── research/
│   │   ├── agent.yaml.j2              # Jinja2 template with embedded test_cases
│   │   ├── instructions/system-prompt.md.j2
│   │   ├── tools/README.md.j2
│   │   ├── data/papers_index.json
│   │   └── manifest.yaml
│   └── customer-support/
│       ├── agent.yaml.j2              # Jinja2 template with embedded test_cases
│       ├── instructions/system-prompt.md.j2
│       ├── tools/README.md.j2
│       ├── data/sample_issues.csv
│       └── manifest.yaml
└── lib/
    └── template_engine.py             # TemplateRenderer (validates output against AgentConfig model)

tests/
├── unit/
│   ├── test_project_init.py           # Project creation, validation logic
│   ├── test_cli_init_command.py       # CLI command parsing and execution
│   └── test_template_engine.py        # Template rendering and schema validation
├── integration/
│   ├── test_init_templates.py         # Full template rendering -> validation -> file creation
│   └── test_agent_config_compliance.py # Verify generated agent.yaml matches AgentConfig schema
└── fixtures/
    └── generated_projects/            # Temporary test projects for validation
```

**Structure Decision**: Single Python package (holodeck) with dedicated CLI module. Jinja2 templates with `.j2` extension for dynamic rendering. **Critical constraint**: TemplateRenderer validates ALL generated YAML against `AgentConfig` Pydantic model before writing to disk. Template logic cannot produce invalid configurations. Template manifests define allowed variable substitutions and template defaults. This ensures template modifications never violate the AgentConfig schema.

## Complexity Tracking

_No Constitution violations to justify - all requirements aligned_

| Violation | Why Needed                                | Simpler Alternative Rejected Because |
| --------- | ----------------------------------------- | ------------------------------------ |
| (none)    | Feature fully compliant with Constitution | N/A                                  |

---

## Phase 0 & Phase 1 Artifacts

**Phase 0 Outputs** (Research):

- `research.md`: Technical decisions documented (Click, Jinja2, bundled templates, schema validation)

**Phase 1 Outputs** (Design & Contracts):

- `data-model.md`: Entity definitions (AgentConfig, TemplateManifest, ProjectInitInput, ProjectInitResult, TemplateRenderer)
- `contracts/cli-init-command.md`: CLI command specification (arguments, options, exit codes, behavior)
- `quickstart.md`: Developer guide and implementation patterns
- Project source structure documented above

---

## Next Steps

Ready for `/speckit.tasks` to generate implementation task breakdown.

All critical design decisions made:

- ✅ Click for CLI framework
- ✅ Jinja2 for templates with strict AgentConfig validation
- ✅ Bundled templates (no network)
- ✅ Three built-in templates (conversational, research, customer-support)
- ✅ TemplateManifest validation model
- ✅ All-or-nothing project creation
- ✅ Current directory only (no --path flag in v0.1)
