# Tasks: OpenTelemetry Observability

**Input**: Design documents from `/specs/018-otel-observability/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/observability-api.md, contracts/observability-config.schema.json

**Tests**: TDD approach - write tests FIRST, ensure they FAIL, then implement.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure) ✅

**Purpose**: Add dependencies and create module structure

- [x] T001 Add OpenTelemetry dependencies to pyproject.toml: `opentelemetry-sdk>=1.20.0`, `opentelemetry-exporter-otlp-proto-grpc>=1.20.0`, `opentelemetry-exporter-otlp-proto-http>=1.20.0`
- [x] T002 [P] Add Prometheus dependencies to pyproject.toml: `opentelemetry-exporter-prometheus>=0.60b1`, `prometheus-client>=0.17.0`
- [x] T003 [P] Add Azure Monitor dependency to pyproject.toml: `azure-monitor-opentelemetry-exporter>=1.0.0b24`
- [x] T004 Create observability module structure: src/holodeck/lib/observability/__init__.py
- [x] T005 [P] Create exporters submodule: src/holodeck/lib/observability/exporters/__init__.py
- [x] T006 [P] Create test directory structure: tests/unit/lib/observability/, tests/integration/observability/

---

## Phase 2: Foundational (Pydantic Models - Blocking Prerequisites) ✅

**Purpose**: Core data models that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

### Tests for Foundational (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T007 [P] Unit tests for LogLevel enum in tests/unit/models/test_observability.py
- [x] T008 [P] Unit tests for TracingConfig model (sample_rate validation, redaction_patterns regex validation) in tests/unit/models/test_observability.py
- [x] T009 [P] Unit tests for MetricsConfig model (export_interval_ms minimum validation) in tests/unit/models/test_observability.py
- [x] T010 [P] Unit tests for LogsConfig model (level enum validation) in tests/unit/models/test_observability.py
- [x] T011 [P] Unit tests for ConsoleExporterConfig model in tests/unit/models/test_observability.py
- [x] T012 [P] Unit tests for OTLPExporterConfig model (endpoint URL validation, protocol enum) in tests/unit/models/test_observability.py
- [x] T013 [P] Unit tests for PrometheusExporterConfig model (port range validation) in tests/unit/models/test_observability.py
- [x] T014 [P] Unit tests for AzureMonitorExporterConfig model (connection_string required when enabled) in tests/unit/models/test_observability.py
- [x] T015 [P] Unit tests for ExportersConfig model (get_enabled_exporters, uses_console_as_default) in tests/unit/models/test_observability.py
- [x] T016 Unit tests for ObservabilityConfig model (optional service_name, None default) in tests/unit/models/test_observability.py

### Implementation for Foundational

- [x] T017 Implement LogLevel enum (DEBUG, INFO, WARNING, ERROR, CRITICAL) in src/holodeck/models/observability.py
- [x] T018 Implement OTLPProtocol enum (GRPC, HTTP) in src/holodeck/models/observability.py
- [x] T019 [P] Implement TracingConfig model with sample_rate, capture_content, redaction_patterns, max_queue_size, max_export_batch_size in src/holodeck/models/observability.py
- [x] T020 [P] Implement MetricsConfig model with export_interval_ms, include_semantic_kernel_metrics in src/holodeck/models/observability.py
- [x] T021 [P] Implement LogsConfig model with level, include_trace_context, filter_namespaces in src/holodeck/models/observability.py
- [x] T022 [P] Implement ConsoleExporterConfig model with pretty_print, include_timestamps in src/holodeck/models/observability.py
- [x] T023 [P] Implement OTLPExporterConfig model with endpoint, protocol, headers, timeout_ms, compression, insecure in src/holodeck/models/observability.py
- [x] T024 [P] Implement PrometheusExporterConfig model with port, host, path in src/holodeck/models/observability.py
- [x] T025 [P] Implement AzureMonitorExporterConfig model with connection_string, disable_offline_storage, storage_directory in src/holodeck/models/observability.py
- [x] T026 Implement ExportersConfig model with get_enabled_exporters() and uses_console_as_default() methods in src/holodeck/models/observability.py
- [x] T027 Implement ObservabilityConfig model with enabled, optional service_name (str | None = None), traces, metrics, logs, exporters, resource_attributes in src/holodeck/models/observability.py
- [x] T028 Add ObservabilityConfig to AgentConfig model (optional field) in src/holodeck/models/agent.py
- [x] T029 Export all models from src/holodeck/models/observability.py in src/holodeck/models/__init__.py

**Checkpoint**: Foundation ready - all foundational tests pass - user story implementation can now begin ✅

---

## Phase 3: User Story 1 - Basic Observability Setup (Priority: P1) MVP ✅

**Goal**: User adds `observability: enabled: true` to agent.yaml and telemetry is output to console (default exporter)

**Independent Test**: Add observability section to agent.yaml, run `holodeck chat`, verify telemetry output to console

### Tests for User Story 1 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T030 [P] [US1] Unit tests for ObservabilityError, ObservabilityConfigError custom exceptions in tests/unit/lib/observability/test_errors.py
- [x] T031 [P] [US1] Unit tests for create_resource() function (agent_name parameter, service_name override, "holodeck-{agent_name}" default format, resource_attributes) in tests/unit/lib/observability/test_providers.py
- [x] T032 [P] [US1] Unit tests for ObservabilityContext dataclass (is_enabled, get_resource) in tests/unit/lib/observability/test_providers.py
- [x] T033 [P] [US1] Unit tests for initialize_observability() function in tests/unit/lib/observability/test_providers.py
- [x] T034 [P] [US1] Unit tests for shutdown_observability() function in tests/unit/lib/observability/test_providers.py
- [x] T035 [P] [US1] Unit tests for get_tracer() function in tests/unit/lib/observability/test_providers.py
- [x] T036 [P] [US1] Unit tests for get_meter() function in tests/unit/lib/observability/test_providers.py
- [x] T037 [P] [US1] Unit tests for console exporter creation (traces, metrics, logs) in tests/unit/lib/observability/test_exporters_console.py
- [x] T038 [US1] Unit tests for console exporter as default when no exporters configured in tests/unit/lib/observability/test_console_default.py

### Implementation for User Story 1

- [x] T039 [US1] Implement ObservabilityError, ObservabilityConfigError exceptions in src/holodeck/lib/observability/errors.py
- [x] T040 [US1] Implement create_resource(config, agent_name) function with service_name resolution (config.service_name or f"holodeck-{agent_name}") and resource_attributes in src/holodeck/lib/observability/providers.py
- [x] T041 [US1] Implement ObservabilityContext dataclass with tracer_provider, meter_provider, logger_provider, exporters in src/holodeck/lib/observability/providers.py
- [x] T042 [US1] Implement set_up_logging() function for LoggerProvider initialization in src/holodeck/lib/observability/providers.py
- [x] T043 [US1] Implement set_up_tracing() function for TracerProvider initialization in src/holodeck/lib/observability/providers.py
- [x] T044 [US1] Implement set_up_metrics() function for MeterProvider initialization in src/holodeck/lib/observability/providers.py
- [x] T045 [US1] Implement initialize_observability(config, agent_name) function with correct order (logging, tracing, metrics) in src/holodeck/lib/observability/providers.py
- [x] T046 [US1] Implement shutdown_observability() function with flush and close in src/holodeck/lib/observability/providers.py
- [x] T047 [US1] Implement get_tracer() and get_meter() helper functions in src/holodeck/lib/observability/providers.py
- [x] T048 [P] [US1] Implement ConsoleSpanExporter wrapper in src/holodeck/lib/observability/exporters/console.py
- [x] T049 [P] [US1] Implement ConsoleMetricExporter wrapper in src/holodeck/lib/observability/exporters/console.py
- [x] T050 [P] [US1] Implement ConsoleLogExporter wrapper in src/holodeck/lib/observability/exporters/console.py
- [x] T051 [US1] Implement create_console_exporters() factory function in src/holodeck/lib/observability/exporters/console.py
- [x] T052 [US1] Implement default-to-console logic in configure_exporters() in src/holodeck/lib/observability/config.py
- [x] T053 [US1] Export public API (initialize_observability, shutdown_observability, get_tracer, get_meter) in src/holodeck/lib/observability/__init__.py

**Checkpoint**: At this point, basic observability with console exporter works - all US1 tests pass ✅

---

## Phase 4: User Story 5 - GenAI Semantic Conventions (Priority: P1) ✅

**Goal**: Telemetry follows OpenTelemetry GenAI semantic conventions with `gen_ai.*` attributes

**Independent Test**: Invoke agent, examine exported spans, verify `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*` attributes present

**Implementation Note**: Semantic Kernel provides native OTel instrumentation with GenAI semantic conventions. We only need to enable it via environment variables. Tasks T054-T060 were skipped as SK handles attribute capture natively.

### Tests for User Story 5 (TDD)

> **NOTE: T054-T056 skipped - SK provides native instrumentation. Tests added for enable_semantic_kernel_telemetry() in test_instrumentation.py**

- [~] T054 [P] [US5] ~~Unit tests for GenAI attribute constants~~ - SKIPPED: SK provides native semantic conventions
- [~] T055 [P] [US5] ~~Unit tests for span naming convention~~ - SKIPPED: SK handles span naming natively
- [~] T056 [US5] ~~Integration test for SK instrumentation hooks~~ - SKIPPED: Covered by enable_semantic_kernel_telemetry tests

### Implementation for User Story 5

- [~] T057 [US5] ~~Implement GenAI operation name constants~~ - SKIPPED: SK provides natively
- [~] T058 [US5] ~~Implement GenAI attribute key constants~~ - SKIPPED: SK provides natively
- [~] T059 [US5] ~~Implement tool execution attribute constants~~ - SKIPPED: SK provides natively
- [~] T060 [US5] ~~Implement sensitive attribute constants~~ - SKIPPED: SK provides natively
- [x] T061 [US5] Implement enable_semantic_kernel_telemetry() to set environment variables in src/holodeck/lib/observability/instrumentation.py
- [x] T062 [US5] Call enable_semantic_kernel_telemetry() in initialize_observability() in src/holodeck/lib/observability/providers.py
- [x] T063 [US5] Export instrumentation module in src/holodeck/lib/observability/__init__.py

**Checkpoint**: Telemetry now includes GenAI semantic convention attributes via SK native instrumentation - US5 complete ✅

---

## Phase 5: User Story 2 - OTLP Exporter Configuration (Priority: P1)

**Goal**: User configures OTLP endpoint and headers to send telemetry to any OTLP collector (Jaeger, Honeycomb, Datadog, etc.)

**Independent Test**: Configure OTLP endpoint in YAML, start OTLP collector, verify data arrives

### Tests for User Story 2 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T064 [P] [US2] Unit tests for OTLP gRPC span exporter creation in tests/unit/lib/observability/test_exporters_otlp.py
- [ ] T065 [P] [US2] Unit tests for OTLP HTTP span exporter creation in tests/unit/lib/observability/test_exporters_otlp.py
- [ ] T066 [P] [US2] Unit tests for OTLP metric exporter creation (gRPC and HTTP) in tests/unit/lib/observability/test_exporters_otlp.py
- [ ] T067 [P] [US2] Unit tests for OTLP log exporter creation (gRPC and HTTP) in tests/unit/lib/observability/test_exporters_otlp.py
- [ ] T068 [P] [US2] Unit tests for custom headers injection (env var substitution) in tests/unit/lib/observability/test_exporters_otlp.py
- [ ] T069 [P] [US2] Unit tests for endpoint port adjustment based on protocol in tests/unit/lib/observability/test_exporters_otlp.py
- [ ] T070 [US2] Integration test for OTLP exporter with mock collector in tests/integration/observability/test_otlp_export.py

### Implementation for User Story 2

- [ ] T071 [US2] Implement create_otlp_span_exporter() for gRPC protocol in src/holodeck/lib/observability/exporters/otlp.py
- [ ] T072 [US2] Implement create_otlp_span_exporter() for HTTP protocol in src/holodeck/lib/observability/exporters/otlp.py
- [ ] T073 [P] [US2] Implement create_otlp_metric_exporter() for both protocols in src/holodeck/lib/observability/exporters/otlp.py
- [ ] T074 [P] [US2] Implement create_otlp_log_exporter() for both protocols in src/holodeck/lib/observability/exporters/otlp.py
- [ ] T075 [US2] Implement create_otlp_exporter() factory function (dispatches by signal type) in src/holodeck/lib/observability/exporters/otlp.py
- [ ] T076 [US2] Implement header resolution with env var substitution in src/holodeck/lib/observability/exporters/otlp.py
- [ ] T077 [US2] Integrate OTLP exporter into configure_exporters() in src/holodeck/lib/observability/config.py
- [ ] T078 [US2] Export OTLP exporter factory in src/holodeck/lib/observability/exporters/__init__.py

**Checkpoint**: OTLP exporter works with gRPC and HTTP protocols - all US2 tests pass

---

## Phase 6: User Story 3 - Prometheus Metrics Exporter (Priority: P2)

**Goal**: User enables Prometheus endpoint to expose metrics for scraping at configurable port

**Independent Test**: Enable Prometheus exporter, scrape `http://localhost:8889/metrics`, verify metrics in Prometheus format

### Tests for User Story 3 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T079 [P] [US3] Unit tests for PrometheusMetricReader creation in tests/unit/lib/observability/test_exporters_prometheus.py
- [ ] T080 [P] [US3] Unit tests for HTTP server startup on configured port in tests/unit/lib/observability/test_exporters_prometheus.py
- [ ] T081 [P] [US3] Unit tests for PortInUseError when port is already bound in tests/unit/lib/observability/test_exporters_prometheus.py
- [ ] T082 [US3] Integration test for Prometheus endpoint scraping in tests/integration/observability/test_prometheus_endpoint.py

### Implementation for User Story 3

- [ ] T083 [US3] Implement PortInUseError exception in src/holodeck/lib/observability/errors.py
- [ ] T084 [US3] Implement create_prometheus_exporter() factory function in src/holodeck/lib/observability/exporters/prometheus.py
- [ ] T085 [US3] Implement Prometheus HTTP server startup with port conflict detection in src/holodeck/lib/observability/exporters/prometheus.py
- [ ] T086 [US3] Integrate Prometheus exporter into configure_exporters() in src/holodeck/lib/observability/config.py
- [ ] T087 [US3] Export Prometheus exporter factory in src/holodeck/lib/observability/exporters/__init__.py

**Checkpoint**: Prometheus metrics endpoint works - all US3 tests pass

---

## Phase 7: User Story 4 - Azure Monitor Exporter (Priority: P2)

**Goal**: User exports telemetry directly to Azure Monitor/Application Insights

**Independent Test**: Configure Azure Monitor connection string, verify data appears in Application Insights

### Tests for User Story 4 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T088 [P] [US4] Unit tests for AzureMonitorTraceExporter creation in tests/unit/lib/observability/test_exporters_azure.py
- [ ] T089 [P] [US4] Unit tests for AzureMonitorMetricExporter creation in tests/unit/lib/observability/test_exporters_azure.py
- [ ] T090 [P] [US4] Unit tests for AzureMonitorLogExporter creation in tests/unit/lib/observability/test_exporters_azure.py
- [ ] T091 [P] [US4] Unit tests for ConnectionStringError when connection string invalid in tests/unit/lib/observability/test_exporters_azure.py
- [ ] T092 [P] [US4] Unit tests for connection_string resolution from environment variable in tests/unit/lib/observability/test_exporters_azure.py

### Implementation for User Story 4

- [ ] T093 [US4] Implement ConnectionStringError exception in src/holodeck/lib/observability/errors.py
- [ ] T094 [US4] Implement create_azure_monitor_trace_exporter() in src/holodeck/lib/observability/exporters/azure_monitor.py
- [ ] T095 [P] [US4] Implement create_azure_monitor_metric_exporter() in src/holodeck/lib/observability/exporters/azure_monitor.py
- [ ] T096 [P] [US4] Implement create_azure_monitor_log_exporter() in src/holodeck/lib/observability/exporters/azure_monitor.py
- [ ] T097 [US4] Implement create_azure_monitor_exporter() factory function (dispatches by signal type) in src/holodeck/lib/observability/exporters/azure_monitor.py
- [ ] T098 [US4] Integrate Azure Monitor exporter into configure_exporters() in src/holodeck/lib/observability/config.py
- [ ] T099 [US4] Export Azure Monitor exporter factory in src/holodeck/lib/observability/exporters/__init__.py

**Checkpoint**: Azure Monitor exporter works - all US4 tests pass

---

## Phase 8: User Story 6 - Sensitive Data Control (Priority: P2)

**Goal**: User controls whether prompts/completions are captured, with optional redaction patterns

**Independent Test**: Enable capture_content, verify prompts appear in spans; apply redaction pattern, verify data is redacted

### Tests for User Story 6 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T100 [P] [US6] Unit tests for capture_content enabling Semantic Kernel sensitive diagnostics in tests/unit/lib/observability/test_sensitive_data.py
- [ ] T101 [P] [US6] Unit tests for redaction pattern compilation and validation in tests/unit/lib/observability/test_sensitive_data.py
- [ ] T102 [P] [US6] Unit tests for RedactingSpanProcessor (SSN, email patterns) in tests/unit/lib/observability/test_sensitive_data.py
- [ ] T103 [US6] Integration test for sensitive content capture and redaction in tests/integration/observability/test_sensitive_data_flow.py

### Implementation for User Story 6

- [ ] T104 [US6] Implement enable_content_capture() to set SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE in src/holodeck/lib/observability/instrumentation.py
- [ ] T105 [US6] Implement validate_redaction_patterns() for regex validation in src/holodeck/lib/observability/config.py
- [ ] T106 [US6] Implement RedactingSpanProcessor class in src/holodeck/lib/observability/processors.py
- [ ] T107 [US6] Add RedactingSpanProcessor to tracer provider when patterns configured in src/holodeck/lib/observability/providers.py
- [ ] T108 [US6] Call enable_content_capture() when capture_content=True in initialize_observability() in src/holodeck/lib/observability/providers.py

**Checkpoint**: Sensitive data control works with redaction - all US6 tests pass

---

## Phase 9: User Story 7 - Multiple Exporters (Priority: P3)

**Goal**: User configures multiple exporters simultaneously, all receive same telemetry

**Independent Test**: Configure OTLP + Prometheus, verify data arrives at both destinations

### Tests for User Story 7 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T109 [P] [US7] Unit tests for multiple span exporters registration in tests/unit/lib/observability/test_multiple_exporters.py
- [ ] T110 [P] [US7] Unit tests for multiple metric readers registration in tests/unit/lib/observability/test_multiple_exporters.py
- [ ] T111 [P] [US7] Unit tests for independent exporter failure handling in tests/unit/lib/observability/test_multiple_exporters.py
- [ ] T112 [US7] Integration test for OTLP + Prometheus simultaneous export in tests/integration/observability/test_multiple_exporters.py

### Implementation for User Story 7

- [ ] T113 [US7] Refactor configure_exporters() to return lists of exporters/readers in src/holodeck/lib/observability/config.py
- [ ] T114 [US7] Update set_up_tracing() to add multiple BatchSpanProcessors in src/holodeck/lib/observability/providers.py
- [ ] T115 [US7] Update set_up_metrics() to register multiple MetricReaders in src/holodeck/lib/observability/providers.py
- [ ] T116 [US7] Implement error isolation for exporter failures in src/holodeck/lib/observability/providers.py

**Checkpoint**: Multiple exporters work simultaneously - all US7 tests pass

---

## Phase 10: CLI Integration ✅

**Purpose**: Integrate observability with chat, test, and serve commands

### Tests for CLI Integration (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T117 [P] Unit tests for chat command observability initialization in tests/unit/cli/commands/test_chat_observability.py
- [x] T118 [P] Unit tests for test command observability initialization in tests/unit/cli/commands/test_test_observability.py
- [x] T119 [P] Unit tests for serve command observability initialization in tests/unit/serve/test_server_observability.py
- [x] T120 [P] Unit tests for double logging prevention when console exporter active in tests/unit/lib/observability/test_logging_coordination.py

### Implementation for CLI Integration

- [x] T121 Implement configure_logging() for double logging prevention in src/holodeck/lib/observability/config.py
- [x] T122 Implement is_console_exporter_active() helper in src/holodeck/lib/observability/config.py
- [x] T123 Integrate observability initialization/shutdown into chat command in src/holodeck/cli/commands/chat.py
- [x] T124 Integrate observability initialization/shutdown into test command in src/holodeck/cli/commands/test.py
- [x] T125 Integrate observability into serve command FastAPI lifespan in src/holodeck/serve/server.py

**Checkpoint**: All CLI commands support observability - all CLI integration tests pass ✅

---

## Phase 11: Edge Cases and Error Handling

**Purpose**: Handle edge cases from spec.md

### Tests for Edge Cases (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T126 [P] Unit tests for bounded buffer (2048 spans/5MB) with oldest-first drop in tests/unit/lib/observability/test_buffer_limits.py
- [ ] T127 [P] Unit tests for configuration validation on startup in tests/unit/lib/observability/test_config_validation.py
- [ ] T128 [P] Unit tests for trace sampling rate application in tests/unit/lib/observability/test_sampling.py
- [ ] T129 [P] Integration test for exporter retry with bounded buffer in tests/integration/observability/test_resilience.py

### Implementation for Edge Cases

- [ ] T130 Implement bounded buffer configuration in BatchSpanProcessor setup in src/holodeck/lib/observability/providers.py
- [ ] T131 Implement validate_observability_config() for startup validation in src/holodeck/lib/observability/config.py
- [ ] T132 Implement TraceIdRatioBasedSampler configuration in src/holodeck/lib/observability/providers.py
- [ ] T133 Implement ExporterConnectionError with retry guidance in src/holodeck/lib/observability/errors.py

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Code quality and documentation

- [ ] T134 [P] Run `make format && make lint-fix` to ensure code quality
- [ ] T135 [P] Run `make type-check` to verify MyPy strict mode compliance
- [ ] T136 [P] Run `make test` to verify all unit tests pass
- [ ] T137 [P] Run `make security` to verify Bandit/Safety compliance
- [ ] T138 Validate quickstart.md examples work with implemented code
- [ ] T139 [P] Create docs/guides/observability.md user guide
- [ ] T140 [P] Add observability to docs/index.md navigation
- [ ] T141 Update README.md to mention observability capability

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational - Basic setup (MVP)
- **User Story 5 (Phase 4)**: Depends on US1 - GenAI conventions (pairs with MVP)
- **User Story 2 (Phase 5)**: Depends on US1 - OTLP exporter (can run parallel to US5)
- **User Story 3 (Phase 6)**: Depends on US1 - Prometheus (can run parallel to US2/US5)
- **User Story 4 (Phase 7)**: Depends on US1 - Azure Monitor (can run parallel to US2/US3/US5)
- **User Story 6 (Phase 8)**: Depends on US1 + US5 - Sensitive data control
- **User Story 7 (Phase 9)**: Depends on US2 + US3 + US4 - Multiple exporters
- **CLI Integration (Phase 10)**: Depends on US1 + any exporter (US2/US3/US4)
- **Edge Cases (Phase 11)**: Depends on all user stories complete
- **Polish (Phase 12)**: Depends on all phases complete

### TDD Workflow Within Each Phase

1. Write all tests for the phase FIRST
2. Run tests - verify they FAIL
3. Implement code to make tests pass
4. Refactor if needed (tests still pass)
5. Move to next phase

### User Story Dependencies

- **User Story 1 (P1)**: Basic Observability - No dependencies on other stories, MVP
- **User Story 5 (P1)**: GenAI Conventions - Depends on US1 foundation
- **User Story 2 (P1)**: OTLP Exporter - Can run parallel to US5
- **User Story 3 (P2)**: Prometheus - Can run parallel to US2/US5
- **User Story 4 (P2)**: Azure Monitor - Can run parallel to US2/US3/US5
- **User Story 6 (P2)**: Sensitive Data - Depends on US1 + US5
- **User Story 7 (P3)**: Multiple Exporters - Depends on US2/US3/US4

### Within Each User Story (TDD)

1. Write tests FIRST - ensure they FAIL
2. Models before services
3. Services before integration
4. Core implementation before exports
5. All tests pass before story complete

### Parallel Opportunities

- T002, T003, T005, T006 can run in parallel (Setup phase)
- T007-T015 can run in parallel (Foundational tests)
- T019-T025 can run in parallel (Foundational model implementations)
- T030-T038 can run in parallel (US1 tests)
- T048-T050 can run in parallel (Console exporter implementations)
- T064-T069 can run in parallel (US2 OTLP tests)
- T079-T081 can run in parallel (US3 Prometheus tests)
- T088-T092 can run in parallel (US4 Azure tests)
- US2, US3, US4 can start in parallel after US1 completes
- T117-T120 can run in parallel (CLI integration tests)
- T134-T137, T139-T140 can run in parallel (Polish tasks)

---

## Implementation Strategy

### MVP First (User Stories 1 + 5 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (tests first, then implementation)
3. Complete Phase 3: User Story 1 (tests first, then implementation)
4. Complete Phase 4: User Story 5 (tests first, then implementation)
5. **STOP and VALIDATE**: Console exporter with GenAI conventions works
6. Deploy/demo if ready - this is a usable MVP

### Incremental Delivery (TDD)

1. Setup + Foundational (tests → implementation) → Foundation ready
2. Add User Story 1 (tests → implementation) → Console exporter works
3. Add User Story 5 (tests → implementation) → GenAI conventions → **MVP!**
4. Add User Story 2 (tests → implementation) → OTLP exporter works
5. Add User Story 3 (tests → implementation) → Prometheus works
6. Add User Story 4 (tests → implementation) → Azure Monitor works
7. Add User Story 6 (tests → implementation) → Sensitive data control
8. Add User Story 7 (tests → implementation) → Multiple exporters
9. CLI Integration → Full integration with chat, test, serve
10. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (TDD)
2. Once Foundational is done:
   - Developer A: User Story 1 + User Story 5 (MVP path)
   - Developer B: User Story 2 (OTLP) + User Story 3 (Prometheus)
   - Developer C: User Story 4 (Azure Monitor)
3. Once exporters ready:
   - Developer A: User Story 6 (Sensitive data) + User Story 7 (Multiple exporters)
   - Developer B: CLI Integration
   - Developer C: Edge Cases + Polish
4. Stories complete and integrate independently - all tests pass

---

## Test Summary

| Phase | Test Tasks | Implementation Tasks | Status |
|-------|------------|---------------------|--------|
| Setup | 0 | 6 | ✅ Complete |
| Foundational | 10 | 13 | ✅ Complete |
| User Story 1 (MVP) | 9 | 15 | ✅ Complete |
| User Story 5 (GenAI) | 1* | 3* | ✅ Complete |
| User Story 2 (OTLP) | 7 | 8 | ⏳ Pending |
| User Story 3 (Prometheus) | 4 | 5 | ⏳ Pending |
| User Story 4 (Azure) | 5 | 7 | ⏳ Pending |
| User Story 6 (Sensitive) | 4 | 5 | ⏳ Pending |
| User Story 7 (Multiple) | 4 | 4 | ⏳ Pending |
| CLI Integration | 4 | 5 | ✅ Complete |
| Edge Cases | 4 | 4 | ⏳ Pending |
| Polish | 0 | 8 | ⏳ Pending |
| **Total** | **54** | **87** | |

*\* US5 reduced: SK provides native GenAI semantic conventions - only enable_semantic_kernel_telemetry() implementation needed (T061-T063)*

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- **TDD**: Write tests FIRST, verify they FAIL, then implement
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All tests must pass before moving to next phase
- File paths follow plan.md structure: `src/holodeck/lib/observability/`, `tests/*/lib/observability/`
- Console exporter is default when no exporters configured (FR-021)
- Double logging prevention when console exporter active (FR-022)
- Semantic Kernel provides native OTel instrumentation - leverage it (research.md)
