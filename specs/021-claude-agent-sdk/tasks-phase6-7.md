# Tasks: Phase 6 (MCP Bridge) & Phase 7 (OTel Bridge) — Claude Agent SDK

**Input**: Design documents from `/specs/021-claude-agent-sdk/`
**Prerequisites**: Phases 1–5 complete (base.py, validators.py, sk_backend.py, selector.py, tool_adapters.py)

**Tests**: TDD approach — write failing tests first, then implement.

**Organization**: Phases 6 and 7 are independent of each other (both depend only on Phase 4). They are organized here as two parallel workstreams.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US3, US4)
- Include exact file paths in descriptions

---

## Phase 6: MCP Bridge (US3 — MCP Tools Work with Claude-Native Agents)

**Goal**: Translate HoloDeck `MCPTool` configs (stdio transport) into Claude SDK `McpStdioServerConfig` dicts for `ClaudeAgentOptions.mcp_servers`, so the Claude subprocess can launch and communicate with external MCP servers natively.

**Plan Reference**: `plan.md` lines 355–373 (Phase 6: MCP Bridge)
**Spec Reference**: `spec.md` lines 61–74 (User Story 3), FR-005, FR-035
**Research Reference**: `research.md` lines 177–196 (§6 — MCP Tool Integration)
**Data Model Reference**: `data-model.md` lines 344–376 (§7 — Entity Relationships, MCP routing)
**Quickstart Reference**: `quickstart.md` lines 326–358 (§6 — MCP Tool Bridge)

**Independent Test**: Configure an agent with an MCP tool (stdio transport). Verify `build_claude_mcp_configs()` produces a valid `McpStdioServerConfig` dict with resolved env vars, correct command/args, and that non-stdio transports emit a warning and are skipped.

### Tests for Phase 6 (TDD — write first, verify they FAIL)

- [ ] T001 [P] [US3] Write unit tests for `build_claude_mcp_configs()` happy path (single stdio MCP tool) in `tests/unit/lib/backends/test_mcp_bridge.py`
  - **Ref**: `plan.md:372` ("Unit tests for config translation")
  - Construct `MCPTool(name="my_server", description="test", transport=TransportType.STDIO, command=CommandType.NPX, args=["server"], env={"KEY": "val"})`. Note: `command` is `CommandType` enum, `transport` is `TransportType` enum — use enum constructors, not strings.
  - Assert output is a typed `McpStdioServerConfig` (imported from `claude_agent_sdk.types`), keyed by `tool.name`.
  - Assert output dict value has `type="stdio"`, `command="npx"` (string from `.value`), `args=["server"]`, `env={"KEY": "val"}`.

- [ ] T002 [P] [US3] Write unit tests for `build_claude_mcp_configs()` with multiple MCP tools in `tests/unit/lib/backends/test_mcp_bridge.py`
  - **Ref**: `plan.md:372`, `spec.md:61` (US3 — multiple MCP tools)
  - Test with 2+ stdio MCP tools → output dict has one entry per tool keyed by `tool.name`.
  - Verify no key collisions when tools have unique names.

- [ ] T003 [P] [US3] Write unit tests for env var resolution in MCP bridge in `tests/unit/lib/backends/test_mcp_bridge.py`
  - **Ref**: `plan.md:369` ("Resolve env vars using existing `_resolve_mcp_env()` logic"), `research.md:188`
  - Test that `${VAR}` patterns in `tool.env` values are resolved from the process environment.
  - Test that `env_file` contents are loaded and merged (lower precedence than explicit `env`).
  - Test that `config` dict is passed through as `MCP_CONFIG` JSON env var.

- [ ] T004 [P] [US3] Write unit tests for non-stdio transport warning in `tests/unit/lib/backends/test_mcp_bridge.py`
  - **Ref**: `plan.md:366` ("non-stdio tools emit a warning and are skipped"), `spec.md:373` (Out of Scope — SSE/WebSocket/HTTP)
  - Test that an MCP tool with `transport=TransportType.SSE` emits a warning log and is excluded from the output dict.
  - Test that an MCP tool with `transport=TransportType.WEBSOCKET` is similarly skipped.
  - Test that a mix of stdio + non-stdio tools returns only the stdio entries.

- [ ] T005 [P] [US3] Write unit test for empty MCP tool list and `command=None` edge case in `tests/unit/lib/backends/test_mcp_bridge.py`
  - **Ref**: `plan.md:362` (function signature accepts `list[MCPTool]`), `factory.py:128` (default to "npx")
  - Test that passing an empty list returns an empty dict (no error).
  - Test that an `MCPTool` with `command=None` produces `"command": "npx"` in the output (matching existing `factory.py` default behavior).

### Implementation for Phase 6

- [ ] T006 [US3] Create `src/holodeck/lib/backends/mcp_bridge.py` with `build_claude_mcp_configs()` function
  - **Ref**: `plan.md:359–373`, `research.md:177–196`, `quickstart.md:326–358`
  - Import `MCPTool`, `TransportType`, `CommandType` from `holodeck.models.tool`.
  - Import `McpStdioServerConfig` from `claude_agent_sdk.types` for typed output.
  - **Env resolution** (I1 resolution): Do NOT import private `_resolve_env_vars()` from `factory.py`. Instead, import `substitute_env_vars` and `load_env_file` from `holodeck.config.env_loader` and write a local `_resolve_mcp_env(tool: MCPTool) -> dict[str, str]` function that:
    1. Loads `tool.env_file` if present (lower precedence).
    2. Applies `tool.env` dict with `substitute_env_vars()` per value (higher precedence).
    3. Handles `tool.config` dict → `MCP_CONFIG` JSON env var passthrough (matching `factory.py:71–72`).
  - Function signature: `def build_claude_mcp_configs(mcp_tools: list[MCPTool]) -> dict[str, McpStdioServerConfig]`
  - For each tool with `transport == TransportType.STDIO`: produce a `McpStdioServerConfig` with `type="stdio"`, `command=tool.command.value` (string extracted from `CommandType` enum), `args=tool.args or []`, `env=_resolve_mcp_env(tool)`.
  - **Edge case**: If `tool.command is None`, default to `"npx"` (matching existing `factory.py:128` behavior).
  - For non-stdio tools: emit `logger.warning(...)` with transport type and tool name, then skip.
  - Return the aggregated dict keyed by `tool.name`.

- [ ] T007 [US3] Export `build_claude_mcp_configs` from `src/holodeck/lib/backends/__init__.py`
  - **Ref**: `plan.md:97` (mcp_bridge.py in backends package)
  - Add import to `__init__.py` so ClaudeBackend (Phase 8) can import it cleanly.

- [ ] T008 [US3] Run Phase 6 tests and verify all pass in `tests/unit/lib/backends/test_mcp_bridge.py`
  - **Ref**: `plan.md:372` ("Test: Unit tests for config translation, env var resolution, and non-stdio warning")
  - Run: `pytest tests/unit/lib/backends/test_mcp_bridge.py -n auto -v`
  - All T001–T005 tests must pass.

- [ ] T009 [US3] Run code quality checks for Phase 6
  - **Ref**: `plan.md:553–561` (Phase 12 — run after each phase)
  - Run: `make format && make lint-fix && make type-check`
  - Fix any issues in `mcp_bridge.py` and `test_mcp_bridge.py`.

**Checkpoint**: `build_claude_mcp_configs()` is complete, tested, and passing code quality. MCP bridge is ready for integration into ClaudeBackend (Phase 8).

---

## Phase 7: OTel Bridge (Cross-cutting — Observability for Claude-Native Agents)

**Goal**: Translate HoloDeck's `ObservabilityConfig` YAML section into Claude Code subprocess environment variables, so OTel telemetry is enabled when the Claude subprocess starts. Emit named warnings for fields that have no Claude Code equivalent.

**Plan Reference**: `plan.md` lines 376–407 (Phase 7: OTel Bridge)
**Spec Reference**: `spec.md` lines 286–289 (FR-036, FR-037, FR-038)
**Research Reference**: `research.md` lines 330–351 (§15 — OTel Bridge Unsupported Fields)
**Data Model Reference**: `data-model.md` lines 296–330 (ObservabilityConfig model)
**Quickstart Reference**: `quickstart.md` lines 139–188 (§2 — build_options OTel integration)

**Independent Test**: Create an `ObservabilityConfig` with OTLP enabled and privacy controls. Verify `translate_observability()` returns the correct env var dict. Verify unsupported fields emit a consolidated warning.

### Tests for Phase 7 (TDD — write first, verify they FAIL)

- [ ] T010 [P] Write unit test for `translate_observability()` with OTLP exporter enabled in `tests/unit/lib/backends/test_otel_bridge.py`
  - **Ref**: `plan.md:383–391` (env var mapping list), `spec.md:287` (FR-036)
  - Test with `ObservabilityConfig(enabled=True, exporters=ExportersConfig(otlp=OTLPExporterConfig(enabled=True, endpoint="http://collector:4317", protocol=OTLPProtocol.GRPC)))`. Note: `protocol` is `OTLPProtocol` enum, not a string.
  - Assert output contains: `CLAUDE_CODE_ENABLE_TELEMETRY=1`, `OTEL_METRICS_EXPORTER=otlp`, `OTEL_LOGS_EXPORTER=otlp`, `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`, `OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317`.

- [ ] T011 [P] Write unit test for `translate_observability()` with custom metrics export interval in `tests/unit/lib/backends/test_otel_bridge.py`
  - **Ref**: `plan.md:390` (`OTEL_METRIC_EXPORT_INTERVAL`, `OTEL_LOGS_EXPORT_INTERVAL`)
  - Test with `ObservabilityConfig(enabled=True, metrics=MetricsConfig(export_interval_ms=10000), exporters=ExportersConfig(otlp=OTLPExporterConfig(enabled=True)))`.
  - Assert `OTEL_METRIC_EXPORT_INTERVAL=10000` is in output.
  - Assert `OTEL_LOGS_EXPORT_INTERVAL=10000` is also in output (C2 — both share the same interval since `LogsConfig` has no separate `export_interval_ms`).

- [ ] T012 [P] Write unit test for `translate_observability()` privacy controls (default off) in `tests/unit/lib/backends/test_otel_bridge.py`
  - **Ref**: `plan.md:391` (`OTEL_LOG_USER_PROMPTS`, `OTEL_LOG_TOOL_DETAILS`), `spec.md:289` (FR-038 — both default to off)
  - Test with default `ObservabilityConfig(enabled=True)` → assert `OTEL_LOG_USER_PROMPTS` is NOT in output (or is `"false"`).
  - Test with `traces.capture_content=True` → assert `OTEL_LOG_USER_PROMPTS=true`.

- [ ] T013 [P] Write unit test for `translate_observability()` with observability disabled in `tests/unit/lib/backends/test_otel_bridge.py`
  - **Ref**: `plan.md:383` (env var mapping only when enabled)
  - Test with `ObservabilityConfig(enabled=False)` → assert output is an empty dict.

- [ ] T014 [P] Write unit test for unmapped field warnings in `tests/unit/lib/backends/test_otel_bridge.py`
  - **Ref**: `plan.md:393–403` (unsupported fields list), `research.md:330–351` (§15)
  - Test with `ObservabilityConfig(enabled=True, exporters=ExportersConfig(azure_monitor=AzureMonitorExporterConfig(enabled=True, connection_string="InstrumentationKey=...")))`.
  - Assert a single consolidated warning is emitted listing `azure_monitor`.
  - Test with `prometheus` exporter enabled → assert warning includes `prometheus`.
  - Test with `traces.redaction_patterns=["secret.*"]` → assert warning includes `redaction_patterns`.
  - Test with `traces.sample_rate=0.5` → assert warning includes `sample_rate`.
  - Test with `logs.filter_namespaces=["my_ns"]` → assert warning includes `filter_namespaces`.

- [ ] T015 [P] Write unit test for `translate_observability()` with HTTP protocol in `tests/unit/lib/backends/test_otel_bridge.py`
  - **Ref**: `plan.md:389` (`OTEL_EXPORTER_OTLP_PROTOCOL`)
  - Test with `OTLPExporterConfig(enabled=True, protocol=OTLPProtocol.HTTP)` — use `OTLPProtocol` enum, not string.
  - Assert `OTEL_EXPORTER_OTLP_PROTOCOL == "http/protobuf"` (Claude Code expects `"http/protobuf"`, not bare `"http"`).

- [ ] T016 [P] Write unit test for `translate_observability()` logs export interval in `tests/unit/lib/backends/test_otel_bridge.py`
  - **Ref**: `plan.md:390` (`OTEL_LOGS_EXPORT_INTERVAL`)
  - Test that when `logs.enabled=True` the `OTEL_LOGS_EXPORTER=otlp` is set when OTLP is configured.

### Implementation for Phase 7

- [ ] T017 Create `src/holodeck/lib/backends/otel_bridge.py` with `translate_observability()` function
  - **Ref**: `plan.md:376–407`, `research.md:330–351`, `quickstart.md:168–172`
  - Import `ObservabilityConfig` from `holodeck.models.observability`.
  - Function signature: `def translate_observability(config: ObservabilityConfig) -> dict[str, str]`
  - If `config.enabled is False` → return empty dict.
  - Core mapping:
    - `CLAUDE_CODE_ENABLE_TELEMETRY` = `"1"`
    - `OTEL_METRICS_EXPORTER` = `"otlp"` if OTLP exporter is enabled, else `"none"`
    - `OTEL_LOGS_EXPORTER` = `"otlp"` if OTLP exporter is enabled and `logs.enabled`, else `"none"`
    - `OTEL_EXPORTER_OTLP_PROTOCOL` = `"grpc"` or `"http/protobuf"` based on `exporters.otlp.protocol`
    - `OTEL_EXPORTER_OTLP_ENDPOINT` = `exporters.otlp.endpoint`
    - `OTEL_METRIC_EXPORT_INTERVAL` = `str(metrics.export_interval_ms)`
    - `OTEL_LOGS_EXPORT_INTERVAL` = `str(metrics.export_interval_ms)` (reuse same interval)
  - Privacy controls (FR-038, default off):
    - `OTEL_LOG_USER_PROMPTS` = `"true"` only if `traces.capture_content is True`
    - `OTEL_LOG_TOOL_DETAILS` = `"true"` only if `traces.capture_content is True`
    - **Note (I7)**: Both env vars map to the same `traces.capture_content` field because `ObservabilityConfig` has no separate tool-detail toggle. This is a deliberate simplification — the Claude subprocess treats them as independent controls, but HoloDeck exposes a single "capture content" toggle.
  - **Note (I8)**: `OTEL_LOGS_EXPORT_INTERVAL` reuses `metrics.export_interval_ms` because `LogsConfig` has no separate export interval field. This is intentional per `plan.md:390`.

- [ ] T018 Implement unmapped field warning logic in `src/holodeck/lib/backends/otel_bridge.py`
  - **Ref**: `plan.md:393–403` (unmapped fields), `research.md:334–349`
  - Collect a list of unsupported field names that have non-default values:
    - `exporters.azure_monitor` (enabled) → `"exporters.azure_monitor"`
    - `exporters.prometheus` (enabled) → `"exporters.prometheus"`
    - `traces.redaction_patterns` (non-empty) → `"traces.redaction_patterns"`
    - `traces.sample_rate` (not 1.0) → `"traces.sample_rate"`
    - `logs.filter_namespaces` (non-default, i.e. `!= ["semantic_kernel"]`) → `"logs.filter_namespaces"`
  - If any unsupported fields are present, emit a single consolidated `logger.warning()`:
    `"The following observability settings are not supported by the Claude-native backend and will be ignored: {field_list}. Use a non-Anthropic provider or remove these fields."`

- [ ] T019 Export `translate_observability` from `src/holodeck/lib/backends/__init__.py`
  - **Ref**: `plan.md:98` (otel_bridge.py in backends package)
  - Add import to `__init__.py` so ClaudeBackend (Phase 8) can import it cleanly.

- [ ] T020 Run Phase 7 tests and verify all pass in `tests/unit/lib/backends/test_otel_bridge.py`
  - **Ref**: `plan.md:406` ("Test: Unit tests for each mapping, default-off privacy controls, and unmapped-field warning emission")
  - Run: `pytest tests/unit/lib/backends/test_otel_bridge.py -n auto -v`
  - All T010–T016 tests must pass.

- [ ] T021 Run code quality checks for Phase 7
  - **Ref**: `plan.md:553–561` (Phase 12 — run after each phase)
  - Run: `make format && make lint-fix && make type-check`
  - Fix any issues in `otel_bridge.py` and `test_otel_bridge.py`.

**Checkpoint**: `translate_observability()` is complete, tested, and passing code quality. OTel bridge is ready for integration into ClaudeBackend (Phase 8).

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 4 (SK backend refactor — COMPLETE)
    │
    ├── Phase 5 (tool adapters — COMPLETE)
    ├── Phase 6 (MCP bridge — THIS FILE, T001–T009)     ─┐
    └── Phase 7 (OTel bridge — THIS FILE, T010–T021)    ─┤── all parallel
                                                          │
                                                      Phase 8 (Claude backend — next)
```

- **Phase 6** and **Phase 7** have NO dependency on each other — they can be implemented in parallel.
- Both depend only on Phase 4 being complete (confirmed green via integration gate).
- Phase 8 (ClaudeBackend) depends on both Phase 6 and Phase 7 being complete.

### Within Phase 6 (MCP Bridge)

- T001–T005 (tests): All parallelizable — different test functions, same file.
- T006 (implementation): Depends on T001–T005 existing (TDD — tests written first).
- T007 (export): Depends on T006.
- T008 (test run): Depends on T006, T007.
- T009 (quality): Depends on T008.

### Within Phase 7 (OTel Bridge)

- T010–T016 (tests): All parallelizable — different test functions, same file.
- T017 (core implementation): Depends on T010–T016 existing (TDD — tests written first).
- T018 (warning logic): Depends on T017 (same file, extends the function).
- T019 (export): Depends on T017, T018.
- T020 (test run): Depends on T017, T018, T019.
- T021 (quality): Depends on T020.

### Parallel Opportunities

```bash
# Phase 6 and Phase 7 can run entirely in parallel:
# Stream A: T001–T009 (MCP Bridge)
# Stream B: T010–T021 (OTel Bridge)

# Within each phase, all test tasks are parallelizable:
# Phase 6 tests: T001, T002, T003, T004, T005 (all [P])
# Phase 7 tests: T010, T011, T012, T013, T014, T015, T016 (all [P])
```

---

## Implementation Strategy

### TDD Flow (per phase)

1. Write all test files (T001–T005 or T010–T016) — verify they FAIL (no implementation yet)
2. Implement the module (T006 or T017+T018)
3. Export from `__init__.py` (T007 or T019)
4. Run tests — verify all PASS (T008 or T020)
5. Run code quality — fix any issues (T009 or T021)

### Parallel Execution

Both phases can be worked on simultaneously by the same or different developers:

1. **Stream A**: Phase 6 MCP Bridge (T001–T009) — maps to US3 (MCP Tools)
2. **Stream B**: Phase 7 OTel Bridge (T010–T021) — cross-cutting observability

### Integration Point

After both phases complete, Phase 8 (ClaudeBackend) will:
- Call `build_claude_mcp_configs()` to translate MCP tool configs
- Call `translate_observability()` to build OTel env vars
- Merge both into `ClaudeAgentOptions` construction

---

## Summary

| Phase | Tasks | Test Tasks | Impl Tasks | User Story |
|-------|-------|-----------|------------|------------|
| 6 (MCP Bridge) | T001–T009 | 5 (T001–T005) | 4 (T006–T009) | US3 |
| 7 (OTel Bridge) | T010–T021 | 7 (T010–T016) | 5 (T017–T021) | Cross-cutting |
| **Total** | **21** | **12** | **9** | — |

**Parallel opportunities**: 12 test tasks can run in parallel (5 + 7). Both phases are fully independent.
**MVP scope**: Phase 6 alone delivers MCP bridge capability for US3.
**Format validation**: All 21 tasks follow checklist format (checkbox, ID, labels, file paths).

---

## Notes

- [P] tasks = different files or different functions in same file, no dependencies
- [US3] label = MCP Tools user story (Phase 6 only)
- Phase 7 tasks have no story label (cross-cutting observability concern)
- Env resolution in Phase 6 imports `substitute_env_vars` and `load_env_file` from `holodeck.config.env_loader` directly (NOT the private `_resolve_env_vars()` from `factory.py`)
- `MCPTool.command` is `CommandType` enum (use `.value` for string); `MCPTool.transport` is `TransportType` enum (compare directly)
- `OTLPExporterConfig.protocol` is `OTLPProtocol` enum; `OTLPProtocol.HTTP` maps to `"http/protobuf"` for Claude Code
- `LogsConfig` has no `export_interval_ms` — reuse `MetricsConfig.export_interval_ms` for both `OTEL_METRIC_EXPORT_INTERVAL` and `OTEL_LOGS_EXPORT_INTERVAL`
- `logs.filter_namespaces` default is `["semantic_kernel"]` (not `[]`) — non-default check must compare against this
- Observability model lives at `src/holodeck/models/observability.py:296–330`
- MCPTool model lives at `src/holodeck/models/tool.py:428–553`
