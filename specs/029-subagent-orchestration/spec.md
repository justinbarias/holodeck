# Feature Specification: Subagent Definitions & Multi-Agent Orchestration

**Feature Branch**: `feature/007-claude-agent-features`
**Spec ID**: 029-subagent-orchestration
**Created**: 2026-03-28
**Status**: Draft
**Input**: Enable users to define full subagent specifications in YAML and wire them into the Claude SDK's agent orchestration system, allowing multi-agent teams to be configured without code.
**Dependencies**: 027-mcp-http-sse-transport (parent agent may use HTTP/SSE MCP servers; subagents inherit access to those servers via their `tools` allowlist)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define a Multi-Agent Research Team (Priority: P1)

A user wants to create a research agent that delegates to specialized subagents: a Researcher that searches the web, a Data Analyst that processes findings, and a Report Writer that produces the final output. Each subagent has its own system prompt, model, and tool access. The user defines all of this in a single agent YAML file.

**Why this priority**: Multi-agent orchestration is the highest-value Claude SDK feature not yet exposed in HoloDeck. The demos repo shows this as the flagship pattern (Research Agent demo).

**Independent Test**: Define an agent.yaml with three subagent definitions under `claude.agents`, run a test case that requires delegation, and verify the SDK receives all three `AgentDefinition` objects.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `claude.agents` containing three named subagent definitions (researcher, analyst, writer), **When** the agent is initialized, **Then** the SDK options include all three as `AgentDefinition` objects with their respective prompts, models, and tool lists.
2. **Given** a subagent definition with `model: haiku`, **When** the agent is initialized, **Then** the subagent's model override is set to "haiku" in the SDK config.
3. **Given** a subagent definition with `tools: [WebSearch, WebFetch]`, **When** the agent is initialized, **Then** only those tools are available to that subagent.
4. **Given** a subagent definition with no `tools` field, **When** the agent is initialized, **Then** the subagent inherits all tools from the parent agent.

---

### User Story 2 - Restrict Subagent Tool Access (Priority: P2)

A user wants a subagent to only see a subset of the parent agent's tools, including a subset of MCP-provided tools. For example, a "Database Analyst" subagent receives the database MCP server's tools (e.g., `mcp__db__query`) while a "Researcher" subagent does not — even though both subagents share the same MCP server registration on the parent.

**Why this priority**: Tool-level isolation is the core security and capability boundary the SDK provides for subagents. The SDK does **not** support per-subagent MCP server registration — all MCP servers register at the parent level, and isolation is enforced via each subagent's `tools` allowlist.

**Independent Test**: Define two subagents whose `tools` lists name different MCP tool identifiers (`mcp__<server>__<tool>`); initialize the agent and verify each subagent's `AgentDefinition.tools` contains only the named entries.

**Acceptance Scenarios**:

1. **Given** a parent agent with an MCP server registered and a subagent definition whose `tools` list includes `mcp__db__query`, **When** the agent is initialized, **Then** the subagent's `AgentDefinition.tools` contains `mcp__db__query` and the SDK enforces that only those tools are usable from the subagent.
2. **Given** two subagents sharing one parent-level MCP server but with different `tools` lists, **When** both are initialized, **Then** each subagent only has access to the tool names it explicitly enumerates.

---

### User Story 3 - Limit HoloDeck-Side Subagent Concurrency (Priority: P3)

A user wants to cap how many HoloDeck-driven invocations of a multi-agent configuration run in parallel (e.g., during evaluation runs or batch test execution). They set `max_parallel` on the existing `subagents` config so the HoloDeck test runner throttles its own dispatch.

**Why this priority**: Resource management still matters for batch/test workflows. **However**, `max_parallel` does **not** control the SDK's internal subagent dispatch — once the parent agent starts orchestrating subagents via the Task tool, the SDK manages concurrency itself, and HoloDeck cannot intercede. This story therefore scopes `max_parallel` to HoloDeck-side concurrency only (e.g., the test runner's semaphore), not to the SDK.

**Independent Test**: Set `claude.subagents.max_parallel: 2` alongside subagent definitions, dispatch three test cases that each invoke the agent, and verify HoloDeck holds at most two concurrent agent sessions open at any time.

**Acceptance Scenarios**:

1. **Given** `claude.subagents.max_parallel: 2` and three concurrent test cases targeting an agent with subagent definitions, **When** the test runner executes them, **Then** at most two test cases hold an agent session simultaneously.
2. **Given** `claude.subagents.enabled: false` and subagent definitions present, **When** the config is loaded, **Then** a validation warning is raised that subagent definitions exist but subagents are disabled.

---

### User Story 4 - Define Subagent with Custom System Prompt (Priority: P1)

A user wants each subagent to have specialized instructions. They provide a `prompt` field (inline text or file path) for each subagent definition, giving each subagent a distinct personality and expertise area.

**Why this priority**: System prompts are what make subagents specialized. Without distinct prompts, subagents are just duplicates of the parent.

**Independent Test**: Define a subagent with `prompt: "You are a data analyst. Only analyze data, never write code."`, initialize the agent, and verify the SDK definition includes this prompt.

**Acceptance Scenarios**:

1. **Given** a subagent with `prompt: "You are a financial analyst."`, **When** the agent is initialized, **Then** the SDK `AgentDefinition` includes this prompt.
2. **Given** a subagent with `prompt_file: ./prompts/analyst.md`, **When** the agent is initialized, **Then** the prompt is loaded from the file and set on the SDK definition.
3. **Given** a subagent with neither `prompt` nor `prompt_file`, **When** the config is loaded, **Then** the subagent uses no custom prompt (inherits parent behavior).

---

### Edge Cases

- What happens when a subagent definition references a model not available to the user's API key? The SDK surfaces the error at runtime; HoloDeck should propagate it clearly.
- What happens when `claude.agents` is defined but `claude.subagents.enabled` is not set? Default to enabled if agent definitions are present.
- What happens when a subagent's tool list references tools not defined in the parent's tool configuration? Validation warning at config load time -- the tools may be built-in SDK tools (Read, Write, etc.), MCP-provided tools (`mcp__<server>__<tool>`), or the user may have made a mistake.
- What happens when subagent definitions are empty (no agents listed)? Treated the same as not having the `agents` section at all.
- What happens when a subagent's `prompt_file` path doesn't exist? A validation error is raised at config load time.
- What happens when neither `prompt` nor `prompt_file` is provided? A validation error is raised at config load time -- the SDK requires a non-empty prompt on every `AgentDefinition`.
- What happens when a subagent's `model` value is outside `{sonnet, opus, haiku, inherit}`? A validation error is raised at config load time -- the SDK's `AgentDefinition.model` only accepts those four literal values.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept an `agents` section under `claude` in agent YAML containing named subagent definitions.
- **FR-002**: Each subagent definition MUST support: `description` (string, required), `prompt` (inline string, required-if-no-`prompt_file`), `prompt_file` (file path, required-if-no-`prompt`), `tools` (list of tool names, optional), and `model` (string, optional, restricted to `sonnet | opus | haiku | inherit`).
- **FR-003**: System MUST translate each subagent definition into the Claude SDK's `AgentDefinition` dataclass with exactly its four supported fields: `description`, `prompt`, `tools`, `model`.
- **FR-004**: System MUST validate that each subagent has a `description` field (required by the SDK for routing decisions).
- **FR-005**: System MUST resolve `prompt_file` paths relative to the agent YAML directory and load file contents as the prompt.
- **FR-006**: System MUST validate that `prompt` and `prompt_file` are not both specified on the same subagent (mutually exclusive) and that at least one of them is set (the SDK requires a non-empty prompt on every `AgentDefinition`).
- **FR-007**: When `tools` is omitted from a subagent, the subagent MUST inherit all tools from the parent agent (achieved by passing `tools=None` to the SDK).
- **FR-008**: System MUST validate `model` against the SDK-allowed set `{sonnet, opus, haiku, inherit}` and surface a clear error for any other value.
- **FR-009**: System MUST allow subagent `tools` lists to reference parent-registered MCP tools by their fully qualified names (`mcp__<server>__<tool>`), so subagents share the parent's MCP server registrations but only see the tools enumerated in their `tools` list. Per-subagent MCP server registration is **not supported** by the SDK and MUST NOT be modeled in YAML.
- **FR-010**: The existing `subagents.max_parallel` field, when set, MUST cap HoloDeck-side concurrent agent-session dispatch (e.g., test runner / batch execution). It MUST NOT be claimed to control SDK-internal subagent dispatch — the SDK manages that concurrency itself.
- **FR-011**: System MUST default `subagents.enabled` to `true` when `claude.agents` definitions are present, unless explicitly set to `false`.
- **FR-012**: System MUST raise a validation warning when `subagents.enabled: false` but agent definitions are present.

### Key Entities

- **SubagentDefinition**: A named subagent specification with `description`, `prompt`/`prompt_file`, `tools`, and `model`. (Note: no per-subagent MCP servers — the SDK does not support that.)
- **ClaudeConfig.agents**: A dictionary mapping subagent names to their definitions; translated 1-to-1 to `ClaudeAgentOptions.agents` on the SDK side.
- **SubagentConfig**: Existing config (`enabled`, `max_parallel`); `enabled` gates whether YAML-defined `agents` are forwarded to the SDK, and `max_parallel` is a HoloDeck-side concurrency cap (not an SDK concurrency control).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can define multi-agent teams entirely in YAML with distinct prompts, models, and tool access per subagent.
- **SC-002**: Subagent definitions are correctly translated to SDK `AgentDefinition` objects, verified by unit tests asserting on the four fields (`description`, `prompt`, `tools`, `model`) of the built options.
- **SC-003**: Subagents with restricted `tools` lists (including MCP tool names) only see those tools, verified by unit tests asserting on each `AgentDefinition.tools` value.
- **SC-004**: Invalid subagent configurations (missing description, both prompt and prompt_file set, neither prompt nor prompt_file set, nonexistent prompt_file, model outside the allowed literal set) produce clear validation errors at config load time.
- **SC-005**: HoloDeck-side test runner / batch execution honors `max_parallel` as a session-dispatch cap, verified by an integration test that asserts on the maximum number of simultaneously open agent sessions.

## Assumptions

- The Claude Agent SDK's `AgentDefinition` dataclass exposes exactly four fields: `description: str` (required), `prompt: str` (required), `tools: list[str] | None`, `model: Literal["sonnet", "opus", "haiku", "inherit"] | None`. There are **no** `skills`, `memory`, `mcp_servers`, or `fallback_model` fields on `AgentDefinition`; these are top-level `ClaudeAgentOptions` concerns or are not exposed by the SDK at all.
- The parent agent automatically delegates to subagents based on their `description` fields -- no explicit routing logic is needed in HoloDeck.
- Subagent definitions are static (defined at config load time) -- dynamic subagent creation at runtime is out of scope.
- The `model` field on subagent definitions is restricted to the SDK's literal set `{sonnet, opus, haiku, inherit}`; full model IDs are not accepted by the SDK type.
- All MCP servers are registered at the parent (top-level `ClaudeAgentOptions.mcp_servers`); subagents share that registration and use their `tools` allowlist to scope which MCP tools they can call (`mcp__<server>__<tool>`).
- The SDK manages subagent-dispatch concurrency internally; HoloDeck does not control how many subagents the SDK runs in parallel during a single agent session. `max_parallel` only constrains HoloDeck-driven concurrent agent sessions (e.g., test runner).
