# Feature Specification: Subagent Definitions & Multi-Agent Orchestration

**Feature Branch**: `feature/007-claude-agent-features`
**Spec ID**: 029-subagent-orchestration
**Created**: 2026-03-28
**Status**: Draft
**Input**: Enable users to define full subagent specifications in YAML and wire them into the Claude SDK's agent orchestration system, allowing multi-agent teams to be configured without code.
**Dependencies**: 026-sdk-config-additions (fallback_model per subagent), 027-mcp-http-sse-transport (subagents may use HTTP/SSE MCP servers)

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

### User Story 2 - Configure Subagent with Custom MCP Servers (Priority: P2)

A user wants a subagent to have access to MCP servers that the parent agent doesn't use. For example, a "Database Analyst" subagent connects to a database MCP server while the parent agent has no database access.

**Why this priority**: MCP isolation per subagent is a key security and capability pattern -- subagents should only access the resources they need.

**Independent Test**: Define a subagent with its own `mcp_servers` section, initialize the agent, and verify the subagent's SDK definition includes the MCP server config.

**Acceptance Scenarios**:

1. **Given** a subagent definition with `mcp_servers` containing a database MCP server (STDIO or HTTP/SSE), **When** the agent is initialized, **Then** the subagent's SDK definition includes the MCP server configuration.
2. **Given** a subagent with MCP servers and the parent agent with different MCP servers, **When** both are initialized, **Then** each has only its own configured MCP servers.

---

### User Story 3 - Control Parallel Subagent Execution (Priority: P2)

A user wants to limit how many subagents run concurrently to control resource usage. They set `max_parallel` on the existing `subagents` config to cap concurrency.

**Why this priority**: Resource management is important for production but the existing `SubagentConfig.max_parallel` field already exists -- this story ensures it works with the new subagent definitions.

**Independent Test**: Set `claude.subagents.max_parallel: 2` alongside subagent definitions, initialize the agent, and verify the SDK options reflect the concurrency limit.

**Acceptance Scenarios**:

1. **Given** `claude.subagents.max_parallel: 2` and three subagent definitions, **When** the agent is initialized, **Then** the SDK is configured to run at most 2 subagents concurrently.
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
- What happens when a subagent's tool list references tools not defined in the parent's tool configuration? Validation warning at config load time -- the tools may be built-in SDK tools (Read, Write, etc.) or the user may have made a mistake.
- What happens when subagent definitions are empty (no agents listed)? Treated the same as not having the `agents` section at all.
- What happens when a subagent's `prompt_file` path doesn't exist? A validation error is raised at config load time.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept an `agents` section under `claude` in agent YAML containing named subagent definitions.
- **FR-002**: Each subagent definition MUST support: `description` (string, required), `prompt` (inline string, optional), `prompt_file` (file path, optional), `tools` (list of tool names, optional), `model` (string, optional), and `mcp_servers` (list of MCP configs, optional).
- **FR-003**: System MUST translate each subagent definition into the Claude SDK's `AgentDefinition` format.
- **FR-004**: System MUST validate that each subagent has a `description` field (required by the SDK for routing decisions).
- **FR-005**: System MUST resolve `prompt_file` paths relative to the agent YAML directory and load file contents as the prompt.
- **FR-006**: System MUST validate that `prompt` and `prompt_file` are not both specified on the same subagent (mutually exclusive).
- **FR-007**: When `tools` is omitted from a subagent, the subagent MUST inherit all tools from the parent agent.
- **FR-008**: System MUST translate subagent `mcp_servers` using the same MCP bridge logic as the parent (supporting STDIO, SSE, and HTTP transports).
- **FR-009**: The existing `subagents.max_parallel` field MUST be passed to the SDK to control concurrency.
- **FR-010**: System MUST default `subagents.enabled` to `true` when `claude.agents` definitions are present, unless explicitly set to `false`.
- **FR-011**: System MUST raise a validation warning when `subagents.enabled: false` but agent definitions are present.

### Key Entities

- **SubagentDefinition**: A named subagent specification with description, prompt, tools, model, and MCP servers.
- **ClaudeConfig.agents**: A dictionary mapping subagent names to their definitions.
- **SubagentConfig**: Existing config extended to interact with the new agent definitions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can define multi-agent teams entirely in YAML with distinct prompts, models, and tool access per subagent.
- **SC-002**: Subagent definitions are correctly translated to SDK `AgentDefinition` objects, verified by unit tests asserting on the built options.
- **SC-003**: Subagents with custom MCP servers receive only their configured servers, not the parent's.
- **SC-004**: Invalid subagent configurations (missing description, both prompt and prompt_file set, nonexistent prompt_file) produce clear validation errors at config load time.
- **SC-005**: The concurrency limit (`max_parallel`) is respected by the SDK when multiple subagents are active.

## Assumptions

- The Claude Agent SDK's `AgentDefinition` type supports `description`, `prompt`, `tools`, `model`, `skills`, `memory`, and `mcpServers` fields.
- The parent agent automatically delegates to subagents based on their `description` fields -- no explicit routing logic is needed in HoloDeck.
- Subagent definitions are static (defined at config load time) -- dynamic subagent creation at runtime is out of scope.
- The `model` field on subagent definitions accepts the same values as the SDK (e.g., "sonnet", "opus", "haiku", "inherit").
