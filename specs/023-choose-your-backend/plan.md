# Implementation Plan: Choose Your Backend

**Branch**: `023-choose-your-backend` | **Date**: 2026-03-15 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/023-choose-your-backend/spec.md`

## Summary

Add Google ADK and Microsoft Agent Framework as two new execution backends alongside existing Semantic Kernel and Claude Agent SDK. This requires: adding a `BackendEnum` for runtime selection, adding `google` to `ProviderEnum`, introducing an optional top-level `backend` field for explicit runtime selection with auto-detection fallback, creating backend-specific Pydantic config models, implementing `AgentBackend`/`AgentSession` protocols for each new backend, building tool adapters for all 5 tool types, replacing SK embedding classes with a unified LiteLLM-based `EmbeddingService` protocol, and updating the `BackendSelector` routing logic. The existing chat executor and test executor require zero changes due to proper protocol abstraction.

Additionally, this feature replaces the unimplemented `PromptTool` (type: prompt) with `SkillTool` (type: skill) following the [Agent Skills specification](https://agentskills.io/specification). Skills are sub-agent invocations scoped to a subset of the parent's tools, with their own instructions. They run on the same backend as the parent agent. Two forms are supported: inline (defined in agent.yaml) and file-based (referencing a skill directory containing SKILL.md).

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: google-adk (pinned RC), agent-framework-core (pinned v1.0.0rc4), litellm (new core dep — unified embeddings), semantic-kernel (existing, planned deprecation), claude-agent-sdk (existing)
**Storage**: N/A (in-memory session management for both new backends)
**Testing**: pytest with pytest-asyncio, pytest-mock, pytest-xdist (`-n auto`)
**Target Platform**: Linux/macOS (CLI tool)
**Project Type**: Single Python package with optional dependency groups
**Performance Goals**: Backend initialization < 5s, single-turn invocation latency dominated by upstream LLM, not adapter overhead
**Constraints**: New backends MUST NOT add import-time dependencies; lazy imports only. `litellm` is added as a core dependency (unified embedding provider, replaces SK embedding classes). `BackendEnum` defines valid runtime backends (`sk`, `claude`, `google_adk`, `agent_framework`).
**Prerequisites**: Before implementation, verify exact PyPI package names and API surfaces for both `google-adk` and `agent-framework-core` (see Prerequisite Tasks below).
**Scale/Scope**: 2 new backends + SkillTool replacement, ~14 new files, ~22 modified files, ~3500 lines of new code

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First | PASS | All backend selection via YAML `backend` field (with auto-detect from `model.provider` fallback). No Python code required from users. |
| II. MCP for API Integrations | PASS | Both new backends support MCP tools natively (ADK via `McpToolset`, AF via `MCPStdioTool`/`MCPStreamableHTTPTool`). No custom API tool types introduced. |
| III. Test-First with Multimodal | PASS | Existing test framework unchanged. New backends return `ExecutionResult` compatible with all evaluation metrics. |
| IV. OTel-Native Observability | PASS (basic) | Basic OTel span wrapping for `invoke_once()`, `send()`, and tool calls is IN SCOPE for both new backends. Full GenAI semconv parity with Claude backend is deferred to a follow-up feature. |
| V. Evaluation Flexibility | PASS | New backends return `ExecutionResult` with token_usage, tool_calls, tool_results — fully compatible with existing 3-level evaluation model. |

**Gate Result**: PASS (basic OTel span wrapping in scope; full GenAI semconv parity deferred to follow-up)

### Post-Phase 1 Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First | PASS | Data model confirms all new config is YAML-driven (`GoogleADKConfig`, `AgentFrameworkConfig` as optional Pydantic sections). Quickstart shows pure-YAML examples. |
| II. MCP for API Integrations | PASS | Tool adapter design (data-model §5-7) maps all MCP transports to native backend APIs. No custom API tool types introduced. |
| III. Test-First with Multimodal | PASS | Project structure includes unit tests for all new modules + integration tests per backend. `ExecutionResult` unchanged. |
| IV. OTel-Native Observability | PASS (basic) | Basic span instrumentation added to scope. Full GenAI semconv parity deferred. |
| V. Evaluation Flexibility | PASS | No change — `ExecutionResult` contract unchanged. |

**Post-Design Gate**: PASS

## Project Structure

### Documentation (this feature)

```text
specs/023-choose-your-backend/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
pyproject.toml                         # MODIFY: Add litellm core dep + google-adk/agent-framework optional groups

src/holodeck/
├── models/
│   ├── llm.py                          # MODIFY: Add GOOGLE to ProviderEnum; add BackendEnum
│   ├── agent.py                        # MODIFY: Add optional `backend` field; add google_adk, agent_framework config fields
│   ├── tool.py                         # MODIFY: Replace PromptTool with SkillTool in ToolUnion
│   ├── __init__.py                     # MODIFY: Replace PromptTool export with SkillTool
│   ├── google_adk_config.py            # NEW: GoogleADKConfig Pydantic model
│   └── af_config.py                    # NEW: AgentFrameworkConfig Pydantic model
│
├── lib/
│   ├── tool_initializer.py             # MODIFY: Replace SK embedding classes with EmbeddingService protocol
│   ├── embedding_protocol.py           # NEW: EmbeddingService protocol + LiteLLMEmbeddingAdapter
│   └── backends/
│       ├── __init__.py                 # MODIFY: Export new types
│       ├── base.py                     # NO CHANGE
│       ├── selector.py                 # MODIFY: Route by `backend` field instead of `model.provider`; auto-detect fallback
│       ├── sk_backend.py               # NO CHANGE
│       ├── claude_backend.py           # NO CHANGE
│       ├── adk_backend.py              # NEW: ADKBackend + ADKSession
│       ├── adk_tool_adapters.py        # NEW: HoloDeck → ADK tool conversion
│       ├── af_backend.py               # NEW: AFBackend + AFSession
│       ├── af_tool_adapters.py         # NEW: HoloDeck → AF tool conversion
│       ├── adk_otel.py                 # NEW: Basic OTel span wrapping for ADK backend
│       └── af_otel.py                  # NEW: Basic OTel span wrapping for AF backend
│
├── chat/
│   ├── session.py                      # NO CHANGE
│   └── executor.py                     # NO CHANGE
│
└── cli/
    └── utils/
        └── wizard.py                   # MODIFY: Add google to provider choices; add backend selection

tests/
├── unit/
│   ├── lib/
│   │   ├── backends/
│   │   │   ├── test_adk_backend.py     # NEW
│   │   │   ├── test_af_backend.py      # NEW
│   │   │   ├── test_adk_tool_adapters.py # NEW
│   │   │   ├── test_af_tool_adapters.py  # NEW
│   │   │   └── test_selector.py        # MODIFY: Add new provider routing tests
│   │   └── test_embedding_protocol.py  # NEW
│   └── models/
│       ├── test_google_adk_config.py   # NEW
│       ├── test_af_config.py           # NEW
│       └── test_tool_models.py         # MODIFY: Replace TestPromptTool with TestSkillTool
└── integration/
    ├── test_adk_integration.py         # NEW (requires google-adk installed)
    └── test_af_integration.py          # NEW (requires agent-framework installed)

docs/
└── examples/
    └── with_tools.yaml                 # MODIFY: Replace prompt tool example with skill example
```

**Structure Decision**: Follows existing single-project structure. New backend files are added under `lib/backends/` matching the established pattern (cf. `sk_backend.py`, `claude_backend.py`). New config models follow the `claude_config.py` pattern.

**SK Deprecation Note**: The Semantic Kernel backend is planned for deprecation. SK receives no new feature work in this feature (no SkillTool support, marked as NO CHANGE). Existing SK functionality is maintained. Deprecation is a separate future effort.

## Prerequisite Tasks

These tasks MUST be completed before implementation begins:

### P0-1: Verify PyPI Package Names and API Surfaces

**Goal**: Confirm that the exact package names and API patterns referenced in research.md are correct.

**Steps**:
1. Install `google-adk` from PyPI — verify exact package name, latest version, and that `google.adk.agents.Agent`, `Runner`, `InMemorySessionService`, `McpToolset` are importable
2. Install `agent-framework-core` from PyPI — verify exact package name (may be `microsoft-agents-sdk` or similar), version, and that `BaseChatClient.as_agent()`, `MCPStdioTool`, `FunctionTool` exist
3. Update research.md R6 with verified package names and pinned versions
4. Update pyproject.toml optional dependency groups with verified names

**Blocking**: All implementation tasks depend on this.

### P0-2: Create embedding_provider Validator in Agent Model

**Goal**: Add `@model_validator(mode='after')` to `Agent` in `agent.py` that enforces `embedding_provider` is set when resolved `backend` is `claude`, `google_adk`, or `agent_framework` and vectorstore/hierarchical_document tools are configured.

**Note**: This validator does NOT currently exist — it must be created from scratch. This fixes a pre-existing gap for the `anthropic` provider and covers both new backends.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Constitution IV: Full GenAI semconv deferred | Basic OTel spans are in scope (invoke_once, send, tool calls). Full GenAI semantic convention parity with Claude backend deferred because ADK and AF have different internal telemetry models that require dedicated research. | Adding basic spans now satisfies the MUST; full semconv is additive. |
