# Feature Specification: Simple SDK Config Additions

**Feature Branch**: `feature/007-claude-agent-features`
**Spec ID**: 026-sdk-config-additions
**Created**: 2026-03-28
**Status**: Draft
**Input**: Expose four missing Claude Agent SDK configuration parameters in HoloDeck YAML: fallback_model, effort, disallowed_tools, and max_budget_usd.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Control Thinking Effort Level (Priority: P1)

A user configuring a Claude agent wants to control how deeply the agent reasons without manually tuning `budget_tokens`. They set `effort: high` in their agent YAML and the agent automatically uses deeper thinking for complex tasks.

**Why this priority**: Effort level is the simplest, most impactful thinking control -- it replaces manual budget_tokens tuning with a single intuitive setting that the SDK natively supports.

**Independent Test**: Set `effort: high` in agent.yaml under `claude:`, run a test case, and verify the SDK receives the effort parameter.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `claude.effort: high`, **When** the agent is initialized, **Then** the SDK options include `effort="high"`.
2. **Given** an agent.yaml with `claude.effort: low`, **When** the agent is initialized, **Then** the SDK options include `effort="low"`.
3. **Given** an agent.yaml with `claude.effort: invalid`, **When** the config is loaded, **Then** a validation error is raised listing valid values (low, medium, high, max).
4. **Given** an agent.yaml with no `effort` field, **When** the agent is initialized, **Then** the SDK receives no effort parameter (SDK default behavior).

---

### User Story 2 - Cap Session Spending with Budget Controls (Priority: P1)

A team deploying agents in production wants to prevent runaway costs. They set `max_budget_usd: 5.0` in their YAML and the agent stops execution when the token budget is exhausted.

**Why this priority**: Budget controls are essential for production deployments where cost overruns are a real risk. Without this, users have no spending guardrails.

**Independent Test**: Set `max_budget_usd: 5.0` in agent.yaml under `claude:`, initialize the agent, and verify the SDK options include the budget limit.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `claude.max_budget_usd: 5.0`, **When** the agent is initialized, **Then** the SDK options include `max_budget_usd=5.0`.
2. **Given** an agent.yaml with `claude.max_budget_usd: 0`, **When** the config is loaded, **Then** a validation error is raised (budget must be positive).
3. **Given** an agent.yaml with no `max_budget_usd` field, **When** the agent is initialized, **Then** no budget limit is passed to the SDK (unlimited).

---

### User Story 3 - Configure Fallback Model (Priority: P2)

A user wants their agent to degrade gracefully if the primary model is unavailable or rate-limited. They set `fallback_model` in YAML so the SDK automatically retries with the fallback.

**Why this priority**: Fallback is a resilience feature -- important for production but not required for basic agent functionality.

**Independent Test**: Set `fallback_model: haiku` in agent.yaml under `claude:`, initialize the agent, and verify the SDK options include the fallback model.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `claude.fallback_model: haiku`, **When** the agent is initialized, **Then** the SDK options include `fallback_model="haiku"`.
2. **Given** an agent.yaml with no `fallback_model`, **When** the agent is initialized, **Then** no fallback model is passed to the SDK.

---

### User Story 4 - Block Specific Tools Unconditionally (Priority: P2)

A user wants to prevent the agent from ever using certain tools, regardless of other permission settings. They set `disallowed_tools` in YAML to create a deny list that takes precedence over allow lists.

**Why this priority**: Complements the existing `allowed_tools` allowlist. Deny lists are a safety net -- useful but secondary to the allowlist which is already implemented.

**Independent Test**: Set `disallowed_tools: [Bash, Write]` in agent.yaml under `claude:`, initialize the agent, and verify the SDK options include the disallowed tools list.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `claude.disallowed_tools: [Bash, Write]`, **When** the agent is initialized, **Then** the SDK options include `disallowed_tools=["Bash", "Write"]`.
2. **Given** both `allowed_tools` and `disallowed_tools` are set, **When** the config is loaded, **Then** validation succeeds (the SDK handles precedence -- disallowed wins).
3. **Given** an agent.yaml with no `disallowed_tools`, **When** the agent is initialized, **Then** no disallowed tools list is passed to the SDK.

---

### Edge Cases

- What happens when `effort` is set alongside `extended_thinking.budget_tokens`? Both are passed to the SDK; the SDK determines precedence. Document this interaction.
- What happens when `max_budget_usd` is set to a very small value (e.g., 0.01)? The agent may fail on the first turn. The system should surface the SDK's budget-exhausted error clearly.
- What happens when `fallback_model` is set to the same value as the primary model? Passed through as-is; the SDK handles this case.
- What happens when a tool appears in both `allowed_tools` and `disallowed_tools`? Passed through as-is; the SDK's precedence rules apply (disallowed wins).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept an `effort` field on `ClaudeConfig` with valid values: `low`, `medium`, `high`, `max`.
- **FR-002**: System MUST accept a `max_budget_usd` field on `ClaudeConfig` as a positive float.
- **FR-003**: System MUST accept a `fallback_model` field on `ClaudeConfig` as a string.
- **FR-004**: System MUST accept a `disallowed_tools` field on `ClaudeConfig` as a list of strings.
- **FR-005**: System MUST pass all four fields through to the Claude SDK options when present.
- **FR-006**: System MUST NOT pass any of the four fields when they are not set (preserve SDK defaults).
- **FR-007**: System MUST reject invalid `effort` values at config validation time with a clear error message.
- **FR-008**: System MUST reject non-positive `max_budget_usd` values at config validation time.

### Key Entities

- **ClaudeConfig**: Extended with four new optional fields (effort, max_budget_usd, fallback_model, disallowed_tools).
- **EffortLevel**: Enumeration of valid effort values (low, medium, high, max).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All four new configuration fields are accepted in agent YAML without errors when valid values are provided.
- **SC-002**: Invalid values for any of the four fields produce clear validation errors at config load time, before agent execution begins.
- **SC-003**: All four fields are correctly passed to the Claude SDK when set, verified by unit tests asserting on the built options dictionary.
- **SC-004**: Omitting any of the four fields results in SDK default behavior (no field sent), verified by unit tests.

## Assumptions

- The Claude Agent SDK's Python client (`ClaudeAgentOptions`) already supports `fallback_model`, `effort`, `disallowed_tools`, and `max_budget_usd` as parameters. If any are not yet available in the installed SDK version, the implementation should be gated behind a version check or documented as requiring a minimum SDK version.
- The `effort` field in the SDK accepts string literals ("low", "medium", "high", "max"), not numeric values.
- The `disallowed_tools` list uses the same tool naming convention as `allowed_tools` (e.g., "Bash", "Write", "mcp__server__tool").
