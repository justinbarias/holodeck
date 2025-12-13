# Implementation Plan: MCP CLI Command Group

**Branch**: `013-mcp-cli` | **Date**: 2025-12-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/013-mcp-cli/spec.md`

## Summary

Implement a new CLI command group `holodeck mcp` with four subcommands: `search`, `list`, `add`, and `remove`. The `search` command queries the official MCP registry API to discover available servers. The `add` and `remove` commands manage MCP server configurations in either agent-level YAML files or the global `~/.holodeck/config.yaml`. The `list` command displays installed servers from both sources.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: Click (CLI), requests (HTTP), PyYAML (config parsing), Pydantic (validation)
**Storage**: YAML files (agent.yaml, ~/.holodeck/config.yaml)
**Testing**: pytest with markers (@pytest.mark.unit, @pytest.mark.integration)
**Target Platform**: Linux/macOS/Windows (cross-platform CLI)
**Project Type**: Single project - extends existing CLI infrastructure
**Performance Goals**: 5s timeout for registry API calls (fail fast)
**Constraints**: No automatic retries on API failures, stdio transport default
**Scale/Scope**: Single user CLI tool, no concurrent access concerns

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First Agent Definition | ✅ PASS | MCP servers are added via CLI, stored in YAML configs |
| II. MCP for API Integrations | ✅ PASS | This feature enhances MCP adoption - uses MCP registry |
| III. Test-First with Multimodal Support | ✅ PASS | Unit/integration tests will be written for all commands |
| IV. OpenTelemetry-Native Observability | N/A | CLI commands, no runtime instrumentation needed |
| V. Evaluation Flexibility | N/A | No evaluation component in this feature |

**Architecture Constraints**:
- ✅ CLI extension - does not modify Agent/Evaluation/Deployment engines
- ✅ Uses existing configuration infrastructure (loader, validator)

**Code Quality Gates**:
- ✅ Python 3.10+ target
- ✅ Will follow Google Python Style Guide
- ✅ MyPy strict mode compliance required
- ✅ pytest for testing with proper markers
- ✅ 80% coverage target
- ✅ Pre-commit hooks required

## Project Structure

### Documentation (this feature)

```text
specs/013-mcp-cli/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API client contract)
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/holodeck/
├── cli/
│   ├── main.py                    # MODIFY - register mcp command group
│   └── commands/
│       ├── __init__.py            # Existing
│       └── mcp.py                 # NEW - mcp command group (search, list, add, remove)
├── config/
│   └── loader.py                  # MODIFY - add MCP server merge logic to merge_configs()
├── models/
│   ├── tool.py                    # Existing - MCPTool model (reuse as-is)
│   └── config.py                  # MODIFY - add mcp_servers field to GlobalConfig
└── services/
    └── mcp_registry.py            # NEW - MCP registry API client

tests/
├── unit/
│   ├── cli/
│   │   └── commands/
│   │       └── test_mcp.py        # NEW - unit tests for mcp commands
│   └── services/
│       └── test_mcp_registry.py   # NEW - registry client tests
└── integration/
    └── cli/
        └── test_mcp_integration.py # NEW - end-to-end CLI tests
```

**Structure Decision**: Extend existing infrastructure. Key changes:
- Add `mcp_servers: list[MCPTool]` field to existing `GlobalConfig` in `models/config.py`
- Add MCP server merge logic to existing `ConfigLoader.merge_configs()` in `config/loader.py`
- Create new `mcp.py` CLI command module and `mcp_registry.py` service

## Complexity Tracking

> No violations to justify - all constitutional constraints are satisfied.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
