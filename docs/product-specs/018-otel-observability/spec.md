# Feature Specification: OpenTelemetry Observability with Semantic Conventions

**Feature Branch**: `018-otel-observability`
**Created**: 2026-01-04
**Status**: Draft
**Input**: User description: "Create a spec for enabling semantic conventions + OpenTelemetry with OTLP, Prometheus, and Azure Monitor exporters for HoloDeck agents using Semantic Kernel"

## Clarifications

### Session 2026-01-04

- Q: What buffer limits should apply when exporters fail? → A: Bounded buffer (2048 spans/5MB max) with oldest-first drop policy
- Q: What telemetry volume/scale should the system support? → A: Medium scale (~100 requests/min, ~10K spans/hour)
- Q: What authentication methods should exporters support? → A: Header-based auth only (API keys, bearer tokens via environment variables)
- Q: What should the default trace sampling rate be? → A: 100% (sample all traces by default)
- Q: What is explicitly out of scope for this feature? → A: Dashboards, alerting, and cost tracking (telemetry emission only)
- Q: Which CLI commands need observability integration? → A: chat, test, and serve commands
- Q: What is the default exporter when none are configured? → A: Console exporter (for development/debugging)
- Q: How to prevent double logging? → A: Suppress default HoloDeck logger when OTel console exporter is active

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic Observability Setup (Priority: P1)

As a HoloDeck user, I want to enable observability for my agent by adding a simple YAML configuration block, so that I can monitor my agent's performance without writing code.

**Why this priority**: Core value proposition - users need a no-code way to enable observability. This is the foundational capability all other features depend on.

**Independent Test**: Can be fully tested by adding an `observability` section to agent.yaml and verifying telemetry data is output to the console (default) or exported to a configured backend.

**Acceptance Scenarios**:

1. **Given** an agent.yaml without observability config, **When** I add `observability: enabled: true`, **Then** the agent exports traces, metrics, and logs to the console (default exporter)
2. **Given** an agent with observability enabled but no exporters configured, **When** I run `holodeck chat agent.yaml`, **Then** telemetry is output to the console for immediate feedback
3. **Given** an agent with observability enabled, **When** I run `holodeck chat agent.yaml`, **Then** telemetry spans are created for each LLM invocation
4. **Given** an agent with observability enabled, **When** I run `holodeck test agent.yaml`, **Then** telemetry captures test execution metrics
5. **Given** an agent with observability enabled, **When** I run `holodeck serve agent.yaml`, **Then** the REST API server exports telemetry for each request

---

### User Story 2 - OTLP Exporter Configuration (Priority: P1)

As a HoloDeck user, I want to configure OTLP export settings including custom endpoints and headers, so that I can send telemetry to my preferred observability backend (Jaeger, Honeycomb, Datadog, etc.).

**Why this priority**: OTLP is the universal standard for telemetry export and required for integration with most observability platforms.

**Independent Test**: Can be tested by configuring OTLP endpoint in YAML and verifying data arrives at a running collector.

**Acceptance Scenarios**:

1. **Given** observability enabled with OTLP exporter, **When** I specify `endpoint: http://collector:4317`, **Then** all telemetry is sent to that endpoint
2. **Given** OTLP config with custom headers, **When** I set `headers: {api-key: "${API_KEY}"}`, **Then** headers are included in export requests
3. **Given** OTLP config with gRPC protocol, **When** agent runs, **Then** telemetry uses gRPC transport
4. **Given** OTLP config with HTTP/protobuf protocol, **When** agent runs, **Then** telemetry uses HTTP transport

---

### User Story 3 - Prometheus Metrics Exporter (Priority: P2)

As a HoloDeck user, I want to expose metrics via a Prometheus endpoint, so that I can integrate with my existing Prometheus/Grafana monitoring stack.

**Why this priority**: Prometheus is widely adopted for metrics collection. This enables integration with existing infrastructure without requiring OTLP collectors.

**Independent Test**: Can be tested by enabling Prometheus exporter and scraping metrics from the exposed endpoint.

**Acceptance Scenarios**:

1. **Given** Prometheus exporter enabled, **When** agent starts, **Then** a metrics endpoint is exposed on the configured port
2. **Given** Prometheus endpoint at `http://localhost:8889/metrics`, **When** I scrape it, **Then** I receive metrics in Prometheus exposition format
3. **Given** agent processing requests, **When** I query Prometheus endpoint, **Then** I see token usage, request duration, and error count metrics

---

### User Story 4 - Azure Monitor Exporter (Priority: P2)

As a HoloDeck user deploying to Azure, I want to export telemetry directly to Azure Monitor/Application Insights, so that I can use Azure's native observability tools.

**Why this priority**: Azure Monitor is essential for users in Azure environments. Direct export simplifies deployment without requiring OTLP collectors.

**Independent Test**: Can be tested by configuring Azure Monitor connection string and verifying data appears in Application Insights.

**Acceptance Scenarios**:

1. **Given** Azure Monitor exporter configured with connection string, **When** agent runs, **Then** telemetry appears in Application Insights
2. **Given** Azure Monitor with `instrumentation_key` environment variable, **When** agent starts, **Then** it connects using the key
3. **Given** agent with Azure Monitor enabled, **When** I view traces in Azure portal, **Then** I see correlated traces, metrics, and logs

---

### User Story 5 - GenAI Semantic Conventions (Priority: P1)

As a HoloDeck user, I want my telemetry to follow OpenTelemetry GenAI semantic conventions, so that observability tools can automatically understand and visualize my AI agent data.

**Why this priority**: Standard semantic conventions enable interoperability with observability tools and consistent data analysis across the industry.

**Independent Test**: Can be tested by examining exported spans and verifying they contain standard `gen_ai.*` attributes.

**Acceptance Scenarios**:

1. **Given** agent invoking LLM, **When** span is created, **Then** it includes `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.prompt_tokens` attributes
2. **Given** agent completing LLM request, **When** span ends, **Then** it includes `gen_ai.response.finish_reasons` and token counts
3. **Given** agent using tools, **When** tool invoked, **Then** span includes standard tool invocation attributes

---

### User Story 6 - Sensitive Data Control (Priority: P2)

As a HoloDeck user, I want to control whether prompts and completions are captured in telemetry, so that I can protect sensitive data while still getting useful observability.

**Why this priority**: Data privacy is critical for enterprise adoption. Users must be able to control what data is captured.

**Independent Test**: Can be tested by configuring content capture settings and verifying presence/absence of prompt content in spans.

**Acceptance Scenarios**:

1. **Given** `capture_content: false` (default), **When** agent processes request, **Then** prompts and completions are NOT included in spans
2. **Given** `capture_content: true`, **When** agent processes request, **Then** prompts and completions ARE included in spans
3. **Given** redaction patterns configured, **When** content capture is enabled, **Then** matching patterns are replaced with redaction text

---

### User Story 7 - Multiple Exporters (Priority: P3)

As a HoloDeck user, I want to configure multiple telemetry exporters simultaneously, so that I can send data to different backends for different purposes.

**Why this priority**: Advanced use case for users with complex observability requirements (e.g., Prometheus for ops, Azure Monitor for business).

**Independent Test**: Can be tested by configuring two exporters and verifying data arrives at both destinations.

**Acceptance Scenarios**:

1. **Given** both OTLP and Prometheus exporters configured, **When** agent runs, **Then** traces go to OTLP and metrics exposed on Prometheus endpoint
2. **Given** OTLP and Azure Monitor configured, **When** agent runs, **Then** all telemetry is exported to both backends
3. **Given** multiple exporters fail independently, **When** one exporter errors, **Then** other exporters continue functioning

---

### Edge Cases

- What happens when the OTLP endpoint is unreachable? Telemetry is buffered (max 2048 spans or 5MB) with oldest-first drop policy and retried; agent continues functioning.
- What happens when Prometheus port is already in use? Clear error message on startup indicating port conflict.
- What happens when Azure Monitor connection string is invalid? Validation error on configuration load with helpful message.
- What happens when no exporters are configured but observability is enabled? Console exporter is used as the default fallback for development/debugging.
- How does system prevent double logging with console exporter? When console exporter is enabled, the default HoloDeck logger output is suppressed to avoid duplicate log entries.
- How does system handle high telemetry volume? Batch processors and sampling to prevent performance impact.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow enabling observability via YAML configuration without code changes
- **FR-002**: System MUST support OTLP exporter for traces, metrics, and logs
- **FR-003**: System MUST support Prometheus exporter for metrics
- **FR-004**: System MUST support Azure Monitor exporter for all telemetry signals
- **FR-005**: System MUST follow OpenTelemetry GenAI semantic conventions for all LLM-related spans
- **FR-006**: System MUST capture token usage metrics (prompt tokens, completion tokens, total tokens)
- **FR-007**: System MUST capture request duration metrics for LLM calls and tool invocations
- **FR-008**: System MUST use `"holodeck-{agent.name}"` as the default service name for OpenTelemetry resource attributes, with optional override via `observability.service_name`
- **FR-009**: System MUST support environment variable substitution in observability configuration (e.g., `${API_KEY}`)
- **FR-010**: System MUST allow users to control capture of sensitive content (prompts/completions)
- **FR-011**: System MUST support configurable redaction patterns for sensitive data
- **FR-012**: System MUST create correlated spans for parent-child operations (agent -> LLM -> tool)
- **FR-013**: System MUST support batch processing for efficient telemetry export
- **FR-014**: System MUST allow multiple exporters to be configured simultaneously
- **FR-015**: System MUST log structured output with trace context when enabled
- **FR-016**: System MUST validate observability configuration on startup and provide clear error messages
- **FR-017**: System MUST support configurable sampling rate for traces
- **FR-018**: System MUST expose Semantic Kernel's native telemetry through the observability pipeline
- **FR-019**: System MUST buffer telemetry when exporters are unavailable (max 2048 spans or 5MB) and drop oldest entries first when buffer is full
- **FR-020**: System MUST support header-based authentication for exporters (API keys, bearer tokens) via environment variable substitution
- **FR-021**: System MUST use console exporter as the default when observability is enabled but no exporters are explicitly configured
- **FR-022**: System MUST suppress default HoloDeck logger console output when OpenTelemetry console exporter is active to prevent duplicate log entries

### Key Entities

- **TelemetryConfig**: Top-level observability configuration containing enabled flag, optional service name override, and exporter settings
- **ExporterConfig**: Configuration for a single exporter (type, endpoint, headers, protocol)
- **TracingConfig**: Trace-specific settings (sample rate, content capture, redaction patterns)
- **MetricsConfig**: Metrics-specific settings (export interval, histogram buckets)
- **LogsConfig**: Logging-specific settings (level, format, trace context inclusion)
- **ResourceAttributes**: Standard OpenTelemetry resource attributes identifying the service (service name defaults to `"holodeck-{agent.name}"`)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can enable observability by adding 3 or fewer lines of YAML configuration
- **SC-002**: 100% of LLM invocations generate spans with GenAI semantic convention attributes
- **SC-003**: Telemetry export adds less than 5% overhead to agent response time under normal load
- **SC-004**: All three exporters (OTLP, Prometheus, Azure Monitor) can be configured and function independently
- **SC-005**: Users can view end-to-end request traces from agent invocation through LLM calls to tool executions
- **SC-006**: Sensitive data (prompts/completions) is NOT captured by default; users must explicitly enable
- **SC-007**: Configuration validation catches 100% of invalid exporter settings before agent startup
- **SC-008**: Multiple exporters can operate simultaneously without data loss or significant performance degradation
- **SC-009**: System handles ~100 requests/min and ~10K spans/hour without degradation

## Assumptions

- Users have access to an observability backend (OTLP collector, Prometheus, or Azure Monitor) for receiving telemetry
- Semantic Kernel provides native OpenTelemetry instrumentation that can be leveraged
- Default OTLP endpoint follows OpenTelemetry standard: `http://localhost:4317` for gRPC, `http://localhost:4318` for HTTP
- Prometheus metrics will be exposed on a configurable port, defaulting to 8889
- Azure Monitor requires a connection string or instrumentation key from Application Insights
- Batch processing and retry logic follow OpenTelemetry SDK defaults unless overridden
- Default trace sampling rate is 100% (all traces captured); users can configure lower rates for cost optimization

## Out of Scope

The following capabilities are explicitly excluded from this feature (may be addressed in future iterations):

- **Dashboards**: No built-in visualization UI; users rely on external tools (Grafana, Azure Portal, Jaeger UI)
- **Alerting**: No alert rules or notification configuration; users configure alerts in their observability backend
- **Cost Tracking**: No dedicated cost calculation or budget features; token metrics can be used externally for cost derivation
