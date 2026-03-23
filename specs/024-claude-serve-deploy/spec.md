# Feature Specification: Claude Backend Serve & Deploy Parity

**Feature Branch**: `024-claude-serve-deploy`
**Created**: 2026-03-20
**Status**: Draft
**Input**: User description: "Bring Claude backend to parity with SK backend for serve and deploy commands. Currently SK backend is supported in test, chat, serve, and deploy. Claude backend needs serve and deploy support."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Serve a Claude-Backed Agent via REST/AG-UI (Priority: P1)

A platform operator configures an agent with `provider: anthropic` and runs `holodeck serve agent.yaml`. The server starts, performs pre-flight validation (Node.js availability, API credentials), and begins accepting requests. Clients interact with the agent through REST or AG-UI protocol endpoints identically to how they would with an OpenAI-backed agent.

**Why this priority**: Serve is the primary production-facing capability. Without it, Claude agents cannot be exposed as APIs, blocking all downstream use cases (deploy, integrations, multi-agent orchestration).

**Independent Test**: Can be fully tested by running `holodeck serve` with a Claude agent config and sending requests via the REST endpoint. Delivers a working Claude-powered API server.

**Acceptance Scenarios**:

1. **Given** an agent config with `provider: anthropic` and Node.js installed, **When** the operator runs `holodeck serve agent.yaml`, **Then** the server starts successfully and the health endpoint reports ready status.
2. **Given** a running Claude-backed server, **When** a client sends a message to the REST endpoint, **Then** the server returns an `ExecutionResult`-compatible response with text, tool calls, and token usage.
3. **Given** a running Claude-backed server, **When** a client opens a streaming session, **Then** text chunks are delivered progressively as they arrive from the Claude Agent SDK.
4. **Given** an agent config with `provider: anthropic` but Node.js is not installed, **When** the operator runs `holodeck serve agent.yaml`, **Then** the server fails fast at startup with a clear error message indicating the Node.js requirement.
5. **Given** an agent config with `provider: anthropic` but no valid Anthropic credentials, **When** the operator runs `holodeck serve agent.yaml`, **Then** the server fails fast at startup with a clear error message about missing credentials.

---

### User Story 2 - Deploy a Claude-Backed Agent as a Container (Priority: P1)

A platform operator runs `holodeck deploy build agent.yaml` where the agent uses `provider: anthropic`. The build process detects the Claude backend requirement, generates a Dockerfile that includes Node.js, and produces a container image that can run the agent. The deployed container starts, validates its environment, and serves requests.

**Why this priority**: Deploy is the production delivery mechanism. Without Claude-aware container builds, operators cannot ship Claude agents to cloud environments, making the platform unusable for Anthropic-powered production workloads.

**Independent Test**: Can be fully tested by running `holodeck deploy build` with a Claude agent config and verifying the generated Dockerfile includes Node.js and the built image runs successfully.

**Acceptance Scenarios**:

1. **Given** an agent config with `provider: anthropic`, **When** the operator runs `holodeck deploy build agent.yaml`, **Then** the generated Dockerfile includes Node.js installation alongside the Python runtime.
2. **Given** a built container for a Claude agent, **When** the container starts, **Then** the entrypoint validates that Node.js is available and Anthropic credentials are configured before starting the serve process.
3. **Given** a built container for a Claude agent with valid credentials, **When** the container receives its first request, **Then** it successfully invokes the Claude Agent SDK and returns a response.
4. **Given** a built container for a Claude agent without credentials, **When** the container starts, **Then** it fails with a clear error message and a non-zero exit code rather than appearing healthy but failing on requests.
5. **Given** a `--dry-run` build for a Claude agent, **When** the operator inspects the generated Dockerfile, **Then** Node.js installation steps and security hardening (non-root user, capability drops) are visible in the output.

---

### User Story 3 - Secure Container Deployment with Credential Isolation (Priority: P2)

A security-conscious operator deploys a Claude agent container following the Anthropic secure deployment guidelines. The container supports credential injection via proxy pattern (`ANTHROPIC_BASE_URL`), runs as a non-root user, drops unnecessary Linux capabilities, and supports read-only root filesystems with tmpfs for working directories.

**Why this priority**: Production Claude deployments require defense-in-depth security. While basic serve/deploy works without this, enterprise adoption requires following Anthropic's published secure deployment patterns.

**Independent Test**: Can be tested by building a container and running it with `--cap-drop ALL --read-only --network none` flags alongside a credential proxy, verifying the agent still functions correctly.

**Acceptance Scenarios**:

1. **Given** a Claude agent container, **When** the operator sets `ANTHROPIC_BASE_URL` to a local proxy address, **Then** the Claude Agent SDK routes all sampling requests through that proxy instead of directly to the Anthropic API.
2. **Given** a Claude agent container built with the secure profile, **When** run with `--cap-drop ALL --security-opt no-new-privileges`, **Then** the agent functions normally without requiring elevated capabilities.
3. **Given** a Claude agent container, **When** run with `--read-only --tmpfs /tmp:rw,noexec,nosuid`, **Then** the Claude Agent SDK subprocess can use the tmpfs for working state while the root filesystem remains immutable.
4. **Given** a Claude agent container, **When** run with `--user 1000:1000`, **Then** the Node.js subprocess and Python process both run as the non-root user without permission errors.

---

### User Story 4 - Backend-Aware Health Checks (Priority: P2)

An infrastructure operator monitors deployed Claude agents. The health check endpoint validates not just that the server process is running, but that the Claude backend prerequisites are satisfied: Node.js is available, credentials are accessible, and the SDK subprocess can be spawned. This enables load balancers and orchestrators to route traffic only to genuinely ready instances.

**Why this priority**: Without backend-aware health checks, containers appear healthy to orchestrators but fail on actual requests, causing user-facing errors and complicating incident diagnosis.

**Independent Test**: Can be tested by starting a serve instance and calling the health endpoint, then removing Node.js or credentials and verifying the health status changes.

**Acceptance Scenarios**:

1. **Given** a Claude-backed serve instance with all prerequisites met, **When** the health endpoint is called, **Then** it returns a healthy status indicating the backend is ready.
2. **Given** a Claude-backed serve instance where Node.js becomes unavailable, **When** the health endpoint is called, **Then** it returns an unhealthy status with a diagnostic message about the missing dependency.
3. **Given** a Claude-backed serve instance where Anthropic credentials are invalid or expired, **When** the health endpoint is called, **Then** it returns a degraded status with a diagnostic message about credential issues.

---

### User Story 5 - Integration Test Coverage for Claude Serve/Deploy (Priority: P3)

A developer contributing to HoloDeck runs the test suite and gets confidence that Claude backend support for serve and deploy works correctly. Integration tests exercise the full path from serve command startup through request handling, and deploy build through Dockerfile generation, using the Claude backend.

**Why this priority**: Without test coverage, regressions in Claude serve/deploy support go undetected until users report them in production.

**Independent Test**: Can be tested by running `make test-integration` and verifying Claude-specific serve/deploy tests pass.

**Acceptance Scenarios**:

1. **Given** the integration test suite, **When** a developer runs serve tests, **Then** tests verify that a Claude-configured agent can start a server, handle requests, and shut down cleanly.
2. **Given** the integration test suite, **When** a developer runs deploy tests, **Then** tests verify that `deploy build` with a Claude agent produces a Dockerfile containing Node.js installation.
3. **Given** the integration test suite, **When** a developer runs deploy tests, **Then** tests verify that the container entrypoint validates Claude-specific prerequisites.

---

### Edge Cases

- What happens when Node.js is installed but at an incompatible version for the Claude Agent SDK?
- How does the system handle Claude Agent SDK subprocess crashes during a long-running serve session? → The server MUST return an error to the client with a retriable status code. The crashed session is terminated; the client decides whether to start a new session. No automatic retry or silent reconnection (conversation context is unrecoverable).
- What happens when the container's tmpfs fills up during heavy Claude Agent SDK usage?
- How does the system behave when `ANTHROPIC_BASE_URL` points to an unreachable proxy?
- What happens when multiple concurrent requests arrive and each needs a separate Claude SDK subprocess?

## Scope Boundary

**In scope:**
- Dockerfile generation with Node.js for Claude agents
- Container entrypoint fixes and Claude-specific prerequisite validation
- Serve pre-flight validation (Node.js, credentials) at startup
- Backend-aware health checks
- Subprocess lifecycle management with configurable session cap (new YAML field)
- OTel instrumentation preservation in serve mode
- Integration tests for Claude serve/deploy paths
- Environment variable pass-through for proxy credential injection

**Out of scope:**
- Kubernetes manifests, Helm charts, or cloud-specific deploy templates (ECS, Cloud Run, etc.)
- Multi-container architectures or sidecar proxy configurations
- New deployment targets beyond the existing `holodeck deploy` infrastructure
- Changes to the Claude Agent SDK itself
- Chat or test command modifications (already working)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The serve command MUST perform pre-flight validation of Claude backend prerequisites (Node.js, credentials) at startup, before accepting requests.
- **FR-002**: The deploy build command MUST detect when an agent uses `provider: anthropic` and generate a Dockerfile that includes Node.js installation.
- **FR-003**: The generated Dockerfile for Claude agents MUST follow secure deployment practices: non-root user, capability-aware, compatible with read-only root filesystem.
- **FR-004**: The container entrypoint MUST validate Claude-specific prerequisites (Node.js, credential environment variables) before starting the serve process.
- **FR-005**: The health check endpoint MUST report backend readiness status, not just process liveness, for Claude-backed agents.
- **FR-006**: The container configuration MUST pass through environment variables (`ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `HTTP_PROXY`, `HTTPS_PROXY`) to the Claude Agent SDK subprocess without filtering or overriding them, enabling proxy credential injection patterns.
- **FR-007**: The existing deploy build `--dry-run` output MUST show Claude-specific Dockerfile additions (Node.js installation, security hardening) when the agent uses `provider: anthropic`.
- **FR-008**: The serve command MUST surface clear, actionable error messages when Claude backend prerequisites are not met.
- **FR-009**: The system MUST pass all existing serve and deploy integration tests unchanged (no regressions to SK backend).
- **FR-010**: The system MUST include integration tests that exercise Claude backend through serve and deploy code paths.
- **FR-011**: The container entrypoint MUST correctly invoke `holodeck serve` with the agent config as a positional argument (fixing the existing `--config` flag bug in entrypoint.sh).
- **FR-012**: Pre-flight validation MUST verify that the installed Node.js version meets the Claude Agent SDK's minimum version requirement, not just that the `node` binary is present on PATH.
- **FR-013**: Serve mode MUST preserve existing GenAI semantic convention instrumentation (OTel traces, metrics) so that Claude agent invocations emit spans to the configured OpenTelemetry collector.
- **FR-014**: Serve mode MUST manage Claude SDK subprocess lifecycle with a default cap of 10 concurrent sessions per serve instance (configurable via agent YAML). Requests exceeding the cap MUST be queued or rejected with a clear capacity error. All active subprocesses MUST be gracefully shut down on server stop.

### Key Entities

- **Backend Prerequisites**: A set of system-level requirements (runtimes, credentials, network access) that a backend needs to function. Each backend declares its own prerequisites.
- **Container Profile**: A Dockerfile generation strategy that adapts based on the detected backend. Includes base image selection, system package installation, and security hardening steps.
- **Health Status**: An enriched health check response that includes backend-specific readiness information beyond process liveness.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Operators can serve Claude-backed agents with the same `holodeck serve` command used for other providers, with no additional manual setup beyond having Node.js installed.
- **SC-002**: Operators can build and deploy Claude-backed agent containers with `holodeck deploy build` and have the container run successfully on first attempt when credentials are provided.
- **SC-003**: When Claude prerequisites are missing, operators receive a clear error within 5 seconds of startup rather than after the first client request.
- **SC-004**: Deployed Claude agent containers pass Anthropic's secure deployment checklist: non-root user, dropped capabilities, proxy-compatible credential handling.
- **SC-005**: Integration test coverage exists for all Claude serve/deploy code paths, with tests running as part of the standard CI pipeline.
- **SC-006**: Zero regressions in existing SK backend serve/deploy functionality after changes are made.

## Clarifications

### Session 2026-03-20

- Q: What is the concurrent Claude session limit for serve mode? → A: Default cap of 10 concurrent sessions per serve instance, configurable via agent YAML. Requests exceeding cap are queued or rejected with a capacity error.
- Q: What happens when a Claude SDK subprocess crashes mid-session? → A: Return error to client with a retriable status code. Session is terminated; client decides whether to start a new session. No auto-retry (conversation context is unrecoverable).
- Q: What is the scope boundary for this feature? → A: Moderate scope — includes Dockerfile, entrypoint, pre-flight validation, health checks, session cap YAML field. Excludes K8s/Helm, cloud-specific templates, multi-container patterns.

## Assumptions

- Node.js >= 18 is the assumed minimum version required by the Claude Agent SDK. The exact version requirement should be verified against Claude Agent SDK documentation during the planning phase.
- The existing `BackendSelector` routing logic correctly dispatches to `ClaudeBackend` for `provider: anthropic` — the protocol-level plumbing is already in place.
- The serve infrastructure (AgentServer, protocol handlers, session store) is already backend-agnostic — changes are needed only at the validation, Dockerfile generation, and health check layers.
- Serve mode reuses the `chat` mode path in `BackendSelector` (there is no dedicated `serve` mode). Permission mode behavior in serve context should match chat: `manual`/`acceptEdits`/`acceptAll` as configured by the agent, with no test-mode overrides to `bypassPermissions`.
- Container deployments will use the existing `holodeck deploy` infrastructure; no new deployment targets are being added.
- The Anthropic secure deployment guidelines (proxy pattern, capability drops, read-only filesystem) represent best practices that should be supported but not mandated by default.
- The existing `docker/entrypoint.sh` has a pre-existing bug (uses `--config` flag instead of positional argument for `holodeck serve`) that must be fixed as a prerequisite for any container deployment to work.
