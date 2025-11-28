# Implementation Plan: MCP Tool Operations

**Branch**: `010-mcp-tool-operations` | **Date**: 2025-11-28 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/010-mcp-tool-operations/spec.md`

## Summary

Implement MCP (Model Context Protocol) tool operations enabling HoloDeck agents to connect to and invoke standardized MCP servers. This feature adds support for four transport types (stdio, SSE, WebSocket, streamable HTTP), automatic tool/prompt discovery, content type conversion, and robust error handling. The implementation leverages Semantic Kernel's existing MCP module as the foundation while adding HoloDeck-specific configuration and integration patterns.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- semantic-kernel>=1.37.1 (already installed - provides MCPStdioPlugin, MCPSsePlugin, MCPWebsocketPlugin, MCPStreamableHttpPlugin)
- mcp (Model Context Protocol SDK - transitive dependency via semantic-kernel)
- pydantic>=2.0.0 (configuration models)
- python-dotenv>=1.0.0 (envFile loading)

**Storage**: N/A (MCP servers manage their own state)
**Testing**: pytest with pytest-asyncio for async MCP operations
**Target Platform**: Linux/macOS/Windows (cross-platform)
**Project Type**: Single project (Python library with CLI)
**Performance Goals**:
- Tool invocations complete within 10 seconds for typical operations
- Default request_timeout: 60 seconds
**Constraints**:
- Must integrate with existing HoloDeck agent.yaml configuration
- Must follow Semantic Kernel MCPPluginBase patterns
- Environment variable resolution must fail-fast at config load time
**Scale/Scope**: Support unlimited MCP server connections per agent

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First Agent Definition | ✅ PASS | MCP tools defined via YAML in agent.yaml |
| II. MCP for API Integrations | ✅ PASS | This feature IS the MCP implementation |
| III. Test-First with Multimodal Support | ✅ PASS | MCP content types include images, audio, binary |
| IV. OpenTelemetry-Native Observability | ⚠️ DEFERRED | Instrumentation planned for observability feature |
| V. Evaluation Flexibility | N/A | Not applicable to tool infrastructure |

**Architecture Constraints**:
- ✅ Part of Agent Engine (tool execution)
- ✅ Decoupled from Evaluation Framework and Deployment Engine

**Code Quality**:
- ✅ Python 3.10+ target
- ✅ Will use pytest with async markers
- ✅ Type hints required (MyPy strict)

## Project Structure

### Documentation (this feature)

```
specs/010-mcp-tool-operations/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── mcp-config.md    # YAML configuration contract
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```
src/holodeck/
├── models/
│   └── tool.py              # UPDATE: Enhance MCPTool model with full config
├── tools/
│   ├── __init__.py
│   ├── vectorstore_tool.py  # Existing
│   └── mcp/                 # NEW: MCP tool implementation
│       ├── __init__.py
│       ├── plugin.py        # MCP plugin wrapper (wraps SK plugins)
│       ├── content.py       # Content type conversion
│       ├── errors.py        # MCP-specific errors
│       └── factory.py       # Plugin factory based on transport type
├── config/
│   └── env_loader.py        # REUSE: substitute_env_vars(), load_env_file()
└── lib/
    └── errors.py            # UPDATE: Add MCP error types

tests/
├── unit/
│   └── tools/
│       └── mcp/
│           ├── test_plugin.py
│           ├── test_config.py
│           ├── test_content.py
│           └── test_factory.py
└── integration/
    └── tools/
        └── test_mcp_integration.py
```

**Structure Decision**: Single project structure maintained. New MCP functionality added under `src/holodeck/tools/mcp/` module to keep MCP-related code organized and testable.

## Complexity Tracking

*No Constitution violations requiring justification*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | - | - |

---

## Post-Design Constitution Re-Check

*Re-evaluated after Phase 1 design completion*

| Principle | Status | Verification |
|-----------|--------|--------------|
| I. No-Code-First Agent Definition | ✅ PASS | MCP config contract (contracts/mcp-config.md) is pure YAML |
| II. MCP for API Integrations | ✅ PASS | This feature implements the MCP infrastructure |
| III. Test-First with Multimodal Support | ✅ PASS | ContentBlock supports text, image, audio, binary |
| IV. OpenTelemetry-Native Observability | ⚠️ DEFERRED | To be added in observability feature |
| V. Evaluation Flexibility | N/A | Not applicable |

**Design Artifacts Produced**:
- ✅ research.md - All technical decisions documented
- ✅ data-model.md - Entity relationships and validation rules
- ✅ contracts/mcp-config.md - YAML configuration contract
- ✅ quickstart.md - Developer onboarding guide

**Ready for Phase 2**: `/speckit.tasks` can now generate implementation tasks.
