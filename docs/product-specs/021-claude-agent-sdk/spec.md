# Feature Specification: Native Claude Agent SDK Integration

**Feature Branch**: `021-claude-agent-sdk`
**Created**: 2026-02-19
**Status**: Draft
**Input**: Native integration of the Claude Agent SDK into HoloDeck, bypassing Semantic Kernel, while preserving compatibility with existing vectorstore, hierarchical document, and MCP tools.

## Overview

HoloDeck currently uses a third-party agent orchestration framework (Semantic Kernel) as a middleware layer between YAML-configured agents and the underlying AI providers. This feature replaces that middleware for Anthropic-powered agents by routing execution directly through Anthropic's own agent platform. The result is a native, first-class integration that eliminates the translation overhead, provides immediate access to the latest Claude capabilities, and uses Anthropic's built-in tool-calling, memory, and streaming mechanisms — while keeping all of HoloDeck's existing tool configurations (vectorstore search, document retrieval, MCP) working without change.

## Architecture Note

**The Claude-native backend operates via a managed subprocess**, not an in-process function call. The Claude Agent SDK spawns a Claude Code process that owns its own agent loop, tool execution, permission enforcement, and streaming output. HoloDeck acts as the orchestrator: it configures the subprocess at startup, forwards tool definitions and capability flags, receives streamed events, and maps outputs back to HoloDeck's evaluation and chat interfaces.

This subprocess boundary has critical implications:

- **Tool registration**: HoloDeck's vectorstore and hierarchical document tools are wrapped using the Claude Agent SDK's `@tool` decorator and registered at subprocess initialization. When the subprocess calls a tool, the SDK routes the invocation back to HoloDeck's parent process transparently — HoloDeck does not implement any custom IPC server. The tool's `search()` method executes in the parent process and the result is returned to the subprocess via the SDK's built-in communication layer.
- **MCP tools**: The Claude Agent SDK has its own native MCP client. MCP server processes are started by the Claude Code subprocess, not by HoloDeck's SK-based MCP plugin factory. HoloDeck translates MCP tool configuration from `agent.yaml` into the SDK's MCP server specification format.
- **Hooks and callbacks** (`can_use_tool`, pre/post hooks): These are communicated to the subprocess as configuration at startup, not as live Python callables invoked across the process boundary.
- **Permission modes**: Enforced inside the Claude Code subprocess, not in HoloDeck's Python process. HoloDeck passes the configured mode at startup.
- **Streaming**: The subprocess emits an async event stream. HoloDeck consumes and forwards this to the terminal interface.
- **Embeddings**: The subprocess does not generate embeddings. HoloDeck initializes vectorstore and hierarchical document tool indexes in-process (using the existing SK embedding services) before the subprocess starts, so tools are ready to serve search results when called.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a Claude-Native Agent (Priority: P1)

A HoloDeck user has an `agent.yaml` with `provider: anthropic`. They run `holodeck test` or `holodeck chat` and the agent executes through the Claude Agent SDK rather than through the Semantic Kernel abstraction. The experience — YAML syntax, CLI commands, output format — is identical to today.

**Why this priority**: This is the foundational capability. Without a working end-to-end execution path for Claude-native agents, nothing else in this feature can be delivered or tested.

**Independent Test**: Run `holodeck chat agent.yaml` where `agent.yaml` specifies `provider: anthropic`. The chat session starts, accepts input, and returns a coherent response. This alone is a viable, demonstrable MVP.

**Acceptance Scenarios**:

1. **Given** a valid `agent.yaml` with `provider: anthropic` and an Anthropic API key configured, **When** the user runs `holodeck chat`, **Then** the agent starts a session, accepts user messages, and returns responses without errors.
2. **Given** the same agent configuration, **When** the user runs `holodeck test`, **Then** all defined test cases are submitted to the agent and responses are captured for evaluation.
3. **Given** a user who previously ran the same agent via the SK backend, **When** they switch to the Claude-native backend without changing their YAML, **Then** the agent produces functionally equivalent responses.
4. **Given** an invalid or missing API key, **When** the user attempts to start the agent, **Then** the system displays a clear, actionable error message indicating the credential issue.

---

### User Story 2 - Vectorstore and Document Tools Work with Claude-Native Agents (Priority: P2)

A user has an agent configured with a vectorstore or hierarchical document tool. When running via the Claude-native backend, the agent can invoke these tools during a conversation or test run, and the results are used in the agent's response — exactly as they would with the SK-based backend.

**Why this priority**: Tools are the primary way HoloDeck agents add domain knowledge. An agent without working tools is a plain chat interface, which defeats HoloDeck's value proposition.

**Independent Test**: Configure an agent with a `vectorstore` or `hierarchical_document` tool and run a test case whose answer requires searching that tool. The agent invokes the tool, retrieves relevant results, and incorporates them into its response.

**Acceptance Scenarios**:

1. **Given** an agent with a vectorstore tool and a test case requiring knowledge retrieval, **When** the test runs via the Claude-native backend, **Then** the agent invokes the search tool at least once and returns a response that references retrieved content.
2. **Given** an agent with a hierarchical document tool (semantic, keyword, or hybrid search mode), **When** a question is asked that requires document lookup, **Then** the agent retrieves relevant chunks and incorporates them correctly.
3. **Given** a tool invocation that returns no results (empty search), **When** the agent processes the empty result, **Then** the agent communicates the lack of results gracefully rather than hallucinating.
4. **Given** a tool that fails during execution (e.g., corrupted index), **When** the agent attempts to use it, **Then** the agent receives an error notification and responds accordingly without crashing the session.

---

### User Story 3 - MCP Tools Work with Claude-Native Agents (Priority: P3)

A user has an agent with one or more MCP tool entries in their `agent.yaml`. When running via the Claude-native backend, these MCP-connected tools are available to the agent and can be invoked during conversations or test cases.

**Why this priority**: MCP is the designated integration path for external APIs and services in HoloDeck. It must work with the Claude-native backend to preserve the "no custom API types" design principle.

**Independent Test**: Configure an agent with an MCP tool (e.g., filesystem or a local MCP server). Run a test case that requires that tool. The agent discovers and calls the tool successfully.

**Acceptance Scenarios**:

1. **Given** an agent with an MCP (stdio transport) tool configured, **When** the agent is started via the Claude-native backend, **Then** the MCP server process starts and the tool is available for invocation.
2. **Given** the agent receives a query requiring an MCP tool, **When** it invokes the tool, **Then** the tool returns results and the agent incorporates them into its response.
3. **Given** an MCP server that is unavailable or fails to start, **When** the agent is initialized, **Then** the user receives a clear error identifying which MCP tool failed and why.

---

### User Story 4 - Run Evaluations Against Claude-Native Agents (Priority: P4)

A user runs `holodeck test` with evaluation metrics configured (NLP, G-Eval, RAG). The test runner executes all test cases against the Claude-native agent and applies all configured metrics to the responses.

**Why this priority**: Evaluation is HoloDeck's core quality gate. If evaluations don't work, users lose the ability to measure and improve agent quality.

**Independent Test**: Define test cases with `ground_truth` and configure at least one evaluation metric (e.g., BLEU or G-Eval). Run `holodeck test` against a Claude-native agent. Each test case receives a score for each configured metric, and the report is generated normally.

**Acceptance Scenarios**:

1. **Given** test cases with `ground_truth` values and NLP metrics configured, **When** `holodeck test` is run against a Claude-native agent, **Then** each test case is scored and the final report is generated in the requested format.
2. **Given** G-Eval or RAG metrics configured, **When** the test runner evaluates Claude-native agent responses, **Then** the LLM-based evaluation executes and produces scores.
3. **Given** a test case that the agent fails to answer (e.g., timeout or error), **When** evaluation runs, **Then** the test is marked as failed with a clear reason, without blocking the rest of the test suite.

---

### User Story 5 - Streaming Chat with Claude-Native Agents (Priority: P5)

A user runs `holodeck chat` with a Claude-native agent. The agent's response streams to the terminal in real time — words appearing progressively rather than all at once after a delay.

**Why this priority**: Streaming is a critical quality-of-life feature for interactive chat. Users find non-streaming responses noticeably worse, especially for long answers.

**Note**: The current `holodeck chat` executor is non-streaming (it returns a complete response after the full turn). Streaming is a net-new capability for HoloDeck's chat interface — it is not currently implemented for any backend. The Claude Agent SDK's subprocess model emits an async event stream natively, making it the natural place to implement streaming for the first time. A prerequisite of this user story is restructuring the chat executor to consume an async event stream rather than a blocking response.

**Independent Test**: Run `holodeck chat` with a Claude-native agent. Ask a question that produces a multi-sentence response. Observe that text appears token-by-token, not all at once.

**Acceptance Scenarios**:

1. **Given** a running chat session with a Claude-native agent, **When** the user submits a prompt, **Then** the response begins appearing within 3 seconds and streams progressively to the terminal.
2. **Given** a streaming response in progress, **When** the response completes, **Then** the chat prompt returns to accept the next input without any delay or error.
3. **Given** a streaming response that is interrupted (e.g., network issue), **When** the stream terminates early, **Then** whatever was received is displayed and the user is informed the response was incomplete.

---

### User Story 6 - Parallel Subagent Execution (Priority: P3)

A user configures an agent that breaks large tasks into parallel subtasks — each handled by an isolated sub-agent with its own context window. Results are consolidated into a single response. This enables HoloDeck agents to tackle complex, multi-step workflows without exhausting a single context window.

**Why this priority**: Subagents unlock a qualitatively different class of task complexity. Without parallelism, Claude-native agents are bounded by single-context limitations that SK agents face too.

**Independent Test**: Configure an agent with a task that explicitly requests parallel research (e.g., "compare three products"). Verify that the agent spawns sub-agents and returns a consolidated answer.

**Acceptance Scenarios**:

1. **Given** an agent configured with subagent support, **When** the task requires processing multiple independent workstreams, **Then** the agent spawns isolated sub-agents and consolidates their outputs into the final response.
2. **Given** a sub-agent that fails mid-task, **When** the parent agent receives the failure, **Then** the parent reports which sub-task failed and continues with the available results.

---

### User Story 7 - File System and Execution Access (Priority: P4)

A user configures a Claude-native agent with a working directory and grants it file system and command execution capabilities. The agent can read existing files, write or edit outputs, discover project structure, and run shell commands — all scoped to the configured directory.

**Why this priority**: File system and execution access are what differentiate a Claude Code-powered agent from a plain chat API call. These capabilities enable autonomous code generation, test running, and document authoring workflows.

**Independent Test**: Configure an agent with a working directory and run a task that requires reading a file and writing a modified version. Verify the output file exists with correct content.

**Acceptance Scenarios**:

1. **Given** an agent with a working directory configured, **When** the agent reads a file within that directory, **Then** the file contents are returned and available for use in the response.
2. **Given** an agent with write access, **When** the agent creates or overwrites a file, **Then** the file exists at the expected path with the correct content.
3. **Given** an agent with bash execution enabled, **When** the agent runs a shell command, **Then** the command executes within the sandbox constraints and its output is available to the agent.
4. **Given** an agent configured with excluded commands, **When** the agent attempts to run a disallowed command, **Then** execution is blocked and the agent receives a clear permission denial.
5. **Given** an agent without write access configured, **When** the agent attempts to write a file outside its permitted scope, **Then** the operation is denied without crashing the session.

---

### User Story 8 - Permission and Safety Governance (Priority: P3)

A user or team configures the autonomy level of their Claude-native agent via `agent.yaml`. In `manual` mode, every agent action requires user approval before execution. In `acceptEdits` mode, file edits are auto-approved but other actions require approval. In `acceptAll` mode, the agent operates fully autonomously.

**Why this priority**: Enterprise and regulated environments require human-in-the-loop controls. Permission modes make Claude-native agents safe to deploy in sensitive contexts.

**Independent Test**: Configure an agent in `manual` mode. Run a task requiring a file write. Verify the agent pauses and prompts for approval before writing.

**Acceptance Scenarios**:

1. **Given** an agent configured with `permission_mode: manual`, **When** the agent attempts any action (file write, bash command, tool call), **Then** the user is prompted to approve or deny before execution proceeds.
2. **Given** an agent configured with `permission_mode: acceptEdits`, **When** the agent performs a file edit, **Then** it proceeds without prompting; when it attempts a bash command, **Then** it pauses for approval.
3. **Given** an agent configured with `permission_mode: acceptAll`, **When** the agent performs any permitted action, **Then** it proceeds without prompting.

---

### User Story 9 - Flexible Authentication (Priority: P5)

A user configures their Claude-native agent's authentication method via `agent.yaml`. Individual users can authenticate via an Anthropic API key or a Claude Code OAuth token; enterprise teams can route through AWS Bedrock, Google Vertex AI, or Azure AI Foundry using existing cloud credentials. No code changes required — the auth method is selected entirely through YAML.

**Why this priority**: Different users and organisations have different credential models. Enterprise teams often cannot use direct API keys due to security policies. Supporting all common auth paths unlocks HoloDeck adoption across individual developers, teams, and regulated industries.

**Independent Test**: Configure an agent with `auth_provider: oauth_token` and a valid `CLAUDE_CODE_OAUTH_TOKEN` in the environment. Run a test case. Verify the agent executes successfully without an Anthropic API key.

**Acceptance Scenarios**:

1. **Given** an agent with no `auth_provider` set (default) and a valid `ANTHROPIC_API_KEY` in the environment, **When** `holodeck chat` or `holodeck test` is run, **Then** the agent authenticates via the API key and responds correctly.
2. **Given** an agent with `auth_provider: api_key` and a valid `ANTHROPIC_API_KEY`, **When** the agent runs, **Then** it authenticates via the API key and responds correctly.
3. **Given** an agent with `auth_provider: oauth_token` and a valid `CLAUDE_CODE_OAUTH_TOKEN` in the environment, **When** `holodeck chat` or `holodeck test` is run, **Then** the agent authenticates via the OAuth token without requiring an Anthropic API key.
4. **Given** an agent with `auth_provider: oauth_token` but `CLAUDE_CODE_OAUTH_TOKEN` is absent, **When** the agent starts, **Then** the system displays a clear error message identifying the missing token and directing the user to run `claude setup-token`.
5. **Given** an agent with `auth_provider: bedrock` and valid AWS credentials in the environment, **When** `holodeck chat` or `holodeck test` is run, **Then** the agent authenticates via Bedrock and responds correctly.
6. **Given** an agent with `auth_provider: vertex` and valid GCP credentials, **When** the agent runs, **Then** it authenticates via Vertex AI and responds correctly.
7. **Given** an agent with `auth_provider: foundry` and valid Azure credentials, **When** the agent runs, **Then** it authenticates via Azure AI Foundry and responds correctly.
8. **Given** any auth provider configured but the required credentials missing or invalid, **When** the agent starts, **Then** the system displays a specific error identifying the missing credential and how to provide it.

### User Story 10 - Structured Output from Claude-Native Agents (Priority: P4)

A user configures a Claude-native agent with a `response_format` schema in `agent.yaml`. The agent's response is returned as a validated, typed JSON object matching the schema — not free-form text. This enables downstream code (evaluations, pipelines, integrations) to consume agent output programmatically without parsing.

**Why this priority**: Structured output is critical for agents embedded in automated pipelines where downstream code must consume the response. Free-form text is unusable in those contexts. The Claude Agent SDK supports this natively, so there is no reason to leave it out of scope.

**Independent Test**: Configure an agent with a `response_format` defining a schema with at least two typed fields. Run a test case. Verify the response is a validated object matching the schema, not a plain text string.

**Acceptance Scenarios**:

1. **Given** an agent with `response_format` defining a JSON schema, **When** the agent completes a turn, **Then** the response is returned as a structured object validated against that schema.
2. **Given** a structured response is produced, **When** the test runner evaluates it, **Then** NLP and G-Eval metrics operate on the serialised text representation, and the structured object is also preserved in the result for downstream use.
3. **Given** an agent response that does not conform to the configured schema, **When** the result is returned, **Then** the system surfaces a clear validation error identifying the schema violation rather than silently returning malformed output.
4. **Given** a `response_format` schema is configured, **When** the agent runs in streaming mode, **Then** the structured output is emitted as a complete object once the turn finishes, not as partial streaming tokens.

---

### Edge Cases

- When the Anthropic API rate limit is hit or a transient error occurs mid-conversation, the system retries up to 3 times with exponential backoff, then surfaces a clear error to the user if all retries fail.
- How does the system behave when a tool invocation exceeds a reasonable time limit?
- What happens when the Claude-native agent produces a tool call for a tool that is not defined in `agent.yaml`?
- How are multi-turn conversations with tool results handled when the agent calls multiple tools in a single turn?
- What happens when the user configures a model that is not supported by the Claude Agent SDK?
- How does the system handle agents that have both Anthropic and non-Anthropic tool dependencies?
- **CLAUDE.md collision**: The Claude Agent SDK subprocess automatically loads `CLAUDE.md` from the configured working directory. If the user sets `working_directory` to the HoloDeck project root (or any directory containing a `CLAUDE.md` not intended for the agent), that file's contents become part of the agent's context. The system MUST warn the user at startup if the working directory's `CLAUDE.md` appears to be a project-level developer file rather than an agent instruction file. Guidance must direct users to set a more specific working directory or provide instructions via `agent.yaml`'s `instructions` field instead.
- **Embedding provider absent with vectorstore tools**: If `provider: anthropic` is set alongside vectorstore or hierarchical document tools but no `embedding_provider` is configured, the system MUST raise a clear validation error at startup — not a runtime failure mid-search.
- **Structured output schema incompatible with SDK**: If the schema provided in `response_format` cannot be serialised to a valid JSON Schema (e.g. uses unsupported Pydantic field types), the system MUST raise a clear validation error at startup identifying the incompatible field, rather than passing a malformed schema to the subprocess.
- **Tool filtering configured on Anthropic agent**: The `tool_filtering` semantic search capability depends on SK's kernel and is not supported in the Claude-native backend. If `tool_filtering` is configured alongside `provider: anthropic`, the system MUST emit a warning and skip tool filtering rather than crashing.
- **`max_turns` cap reached mid-task**: When the agent hits the `max_turns` limit before completing a task, the test case is marked as failed with a reason of "max_turns limit reached" rather than as an execution error. Partial results accumulated before the cap are preserved in the result output.
- **OTLP endpoint unreachable**: If the configured OTLP endpoint is unavailable, the agent session continues normally. Telemetry is silently dropped per standard OTel SDK behaviour. HoloDeck MUST NOT fail agent startup or abort a session due to an unreachable observability endpoint.
- **Subprocess terminates unexpectedly**: If the Claude Code subprocess exits mid-session due to an OS-level kill (OOM, SIGKILL), an unhandled internal error, or any other unexpected cause, HoloDeck MUST detect the termination, surface a clear "agent session terminated unexpectedly" error to the user, mark any in-progress test case as failed (not as an evaluation failure), and release all associated resources (tool indexes, open handles).
- **No credentials found for any auth method**: If `auth_provider: api_key` (or default) is used but `ANTHROPIC_API_KEY` is absent, and `auth_provider: oauth_token` is used but `CLAUDE_CODE_OAUTH_TOKEN` is absent, the system MUST detect the missing credential at startup and surface a specific, actionable error — not a cryptic subprocess failure after the agent loop starts.
- **Node.js not installed**: The Claude Agent SDK requires Node.js as a host runtime. If Node.js is not found on `PATH` at agent startup, the system MUST surface a clear error identifying Node.js as a missing prerequisite and directing the user to install it, rather than failing with a cryptic subprocess spawn error.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST execute agents configured with `provider: anthropic` through the Claude native execution path, not through the Semantic Kernel layer.
- **FR-001a**: All Claude-native capabilities (bash execution, file system access, extended thinking, web search, subagents, hooks, etc.) MUST be disabled by default and require explicit individual opt-in via `agent.yaml`. An agent with no capability flags enabled MUST behave as a plain conversational agent.
- **FR-002**: Existing `agent.yaml` files that do not include Claude-native capability fields MUST continue to work without modification when switching to `provider: anthropic`. All new capability fields introduced by this feature are optional and default to disabled. Users are NOT required to remove or change any existing fields to adopt the Claude-native backend — however, new fields are required to unlock new capabilities.
- **FR-002a**: The `Agent` configuration model MUST be extended to support new optional Claude-native fields (working directory, permission mode, max turns, extended thinking budget, capability flags, hooks, auth provider, etc.). Because the existing model uses strict field validation, any new field MUST be explicitly modelled — unrecognised fields MUST continue to produce a clear validation error rather than silently being ignored.
- **FR-003**: System MUST make vectorstore tools (including structured and unstructured modes) available as callable tools within Claude-native agent sessions.
- **FR-004**: System MUST make hierarchical document tools (semantic, keyword, and hybrid search modes) available as callable tools within Claude-native agent sessions.
- **FR-005**: System MUST support MCP tools (stdio transport) for Claude-native agents, launching and communicating with MCP server processes as configured.
- **FR-006**: System MUST preserve full conversation history across turns in Claude-native agent sessions, including tool invocation results. When the context window approaches its limit, the system MUST use the Claude Agent SDK's native compaction capability to compress history automatically, maintaining conversation continuity without data loss or manual truncation.
- **FR-007**: System MUST deliver streaming responses from Claude-native agents to the `holodeck chat` terminal interface.
- **FR-008**: System MUST execute all configured evaluation metrics (NLP, G-Eval, RAG) against Claude-native agent responses in the test runner.
- **FR-009**: System MUST provide specific, actionable error messages when Claude-native agent execution fails (credential issues, unavailable models, tool failures).
- **FR-009a**: System MUST verify Node.js is available on `PATH` before attempting to start any Claude-native agent session. If Node.js is absent, the system MUST raise a clear startup error identifying Node.js as a required runtime and providing installation guidance. The Claude Agent SDK bundles the `claude` CLI and does not require it to be separately installed — Node.js is the only host-level prerequisite.
- **FR-009b**: System MUST monitor the Claude Code subprocess for unexpected termination during a session. If the subprocess exits with a non-zero code or is killed by the OS, HoloDeck MUST detect the termination promptly, surface a clear user-facing error distinguishing subprocess crash from a normal agent error, mark any in-progress test case as failed with reason "subprocess terminated unexpectedly", and release all associated resources cleanly without leaving dangling processes or open file handles.
- **FR-013**: System MUST retry failed Anthropic API calls (rate limit, 5xx, timeout) with exponential backoff for up to 3 attempts before surfacing a clear error to the user. **Note**: If the Claude Agent SDK subprocess handles API-level retries internally, HoloDeck MUST NOT add a second retry layer on top — doing so would produce up to 9 effective attempts (3 × 3) and unpredictable wait times. During implementation, the SDK's built-in retry behaviour MUST be verified; if the SDK already retries, HoloDeck's retry logic for the Claude-native path MUST be limited to session-level failures (e.g., subprocess crash, spawn failure) only. **Note**: The current `AgentThreadRun._invoke_with_retry()` catches only `ConnectionError` and `TimeoutError`. `anthropic.RateLimitError` (HTTP 429) is not a subclass of either and currently falls into the non-retryable branch, raising immediately. The Claude-native retry implementation MUST explicitly include `RateLimitError` (and any SDK-equivalent rate limit exception) in its retryable exception set.
- **FR-010**: All existing `agent.yaml` configurations using non-Anthropic providers MUST continue to work without any modification.
- **FR-011**: When `provider: anthropic` is specified in `agent.yaml`, the system MUST route execution exclusively through the Claude-native backend. The existing Semantic Kernel execution layer MUST NOT be used for Anthropic-provider agents.
- **FR-012**: System MUST initialize vectorstore and hierarchical document tool indexes at agent startup, exactly as the existing backend does, ensuring the tools are ready before the first query.
- **FR-012a**: Because the Claude Agent SDK does not generate text embeddings, the `Agent` configuration model MUST be extended with a top-level optional `embedding_provider` field of type `LLMProvider` (the same type as the existing `model:` field). This field configures the provider used exclusively for embedding generation (e.g., OpenAI or Azure OpenAI) independently of the chat `provider`. When a Claude-native agent uses vectorstore or hierarchical document tools, `embedding_provider` MUST be configured. If vectorstore tools are present and no `embedding_provider` is configured, the system MUST raise a clear validation error at startup rather than attempting to use the Anthropic provider for embeddings. **Note**: The current `AgentFactory._register_embedding_service()` method explicitly raises `AgentFactoryError` for `provider: anthropic` — this is the immediate blocker for US-2. When the Claude-native backend is active, this code path MUST be bypassed entirely; embedding service registration MUST use `embedding_provider` credentials instead of the chat provider credentials.

#### Abstract Execution Layer

- **FR-012b**: HoloDeck MUST define a provider-agnostic execution result interface that both the SK backend and the Claude-native backend implement. This interface captures: the agent's final text response, tool calls made during the turn, tool results, and token usage. All downstream consumers (chat executor, test executor, evaluation pipeline) MUST depend on this interface rather than on SK-specific types (`ChatHistory`, `ChatMessageContent`, etc.). This is a prerequisite for the Claude-native backend to integrate with the existing test runner and chat layer without forking those components. **⚠ Prerequisite**: This FR MUST be implemented and verified before any other FR in this feature can be tested end-to-end. All other FRs depend on both backends producing a result that the existing `TestExecutor` and `AgentExecutor` can consume without modification.

#### Structured Output

- **FR-039**: System MUST support structured output for Claude-native agents via the existing `response_format` field in `agent.yaml`. When `response_format` is configured alongside `provider: anthropic`, HoloDeck MUST translate the schema into the Claude Agent SDK's `output_format` parameter (`{"type": "json_schema", "schema": <json_schema>}`) before spawning the subprocess. **Note**: The existing SK/OpenAI response format wrapper (`_wrap_response_format`) produces a structurally different format (`{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": true}}`) and MUST NOT be reused for the Claude-native path. A separate translation function is required.
- **FR-039a**: The structured output returned by the subprocess MUST be validated against the configured schema and surfaced as a typed object in both the chat interface and the test runner result. The serialised text representation of the object MUST also be available so that evaluation metrics (NLP, G-Eval) can operate on it normally.
- **FR-039b**: If the agent's response does not conform to the configured schema, the system MUST surface a clear validation error identifying the schema violation. It MUST NOT silently return malformed or unvalidated output.

#### Agent Loop Controls

- **FR-014**: System MUST support a configurable `max_turns` limit that caps the number of agent loop iterations, preventing runaway execution and controlling cost.
- **FR-015**: System MUST support spawning parallel sub-agents with isolated conversation contexts, consolidating their outputs back to the parent agent.

#### Reasoning

- **FR-016**: System MUST support extended thinking (deep reasoning) mode for Claude-native agents, configurable via `agent.yaml` with a token budget parameter.

#### File System Access

- **FR-017**: System MUST allow Claude-native agents to read file contents from within the configured working directory.
- **FR-018**: System MUST allow Claude-native agents to create and overwrite files within the configured working directory.
- **FR-019**: System MUST allow Claude-native agents to perform surgical find-and-replace edits on files without rewriting the entire file.
- **FR-020**: System MUST allow Claude-native agents to discover files by glob pattern and search file contents by regular expression.

#### Command Execution

- **FR-021**: System MUST allow Claude-native agents to execute shell commands within a configurable sandbox, supporting explicit lists of excluded commands and permitted unsafe commands.

#### Tool System

- **FR-023**: HoloDeck MUST internally wrap its existing tool implementations (vectorstore search, hierarchical document search) as Claude Agent SDK-compatible tool definitions using the SDK's `@tool` decorator/registration mechanism. The SDK manages all subprocess↔parent communication for tool invocation internally — HoloDeck does not build any custom IPC server. This is an internal implementation concern — users do NOT write Python decorators. Users continue to configure tools via `agent.yaml` exactly as today; HoloDeck generates the tool definitions from that configuration automatically.
- **FR-026**: System MUST support an `allowed_tools` list in `agent.yaml` specifying which tools the agent is permitted to access. This is communicated to the subprocess at startup.

#### Configuration

- **FR-028**: System MUST support setting a working directory for file system operations via `agent.yaml`, scoping the agent's file access to a specific project path.
- **FR-029**: System MUST load agent context and persistent instructions from `CLAUDE.md` and `.claude/` project configuration files when present in the working directory.
- **FR-030**: System MUST support enabling built-in web search as a configurable capability for Claude-native agents.

#### Permission Modes

- **FR-031**: System MUST support a `permission_mode` configuration with three levels:
  - `manual` — every agent action requires explicit user approval before execution.
  - `acceptEdits` — file edits are auto-approved; all other actions require approval.
  - `acceptAll` — all permitted actions proceed without prompting (fully autonomous).

#### Observability

- **FR-036**: System MUST support OpenTelemetry observability for Claude-native agents by translating the `observability` section of `agent.yaml` into the corresponding Claude Code environment variables (`CLAUDE_CODE_ENABLE_TELEMETRY`, `OTEL_METRICS_EXPORTER`, `OTEL_LOGS_EXPORTER`, `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_METRIC_EXPORT_INTERVAL`, `OTEL_LOGS_EXPORT_INTERVAL`) before the Claude Code subprocess is spawned. The existing SK-based OTel decorator approach is not used for Anthropic agents.
- **FR-037**: The following native telemetry signals MUST be available when observability is enabled: token usage metrics (input/output, by model), API request counts with success/failure status, API latency distributions, active usage time, user prompt events (length; content opt-in), tool result events (tool name, success/fail, execution time, accept/reject decision), bash command events, and MCP tool events (server name, tool name opt-in).
- **FR-038**: System MUST support opt-in privacy controls via `agent.yaml`: logging of full user prompt content (maps to `OTEL_LOG_USER_PROMPTS`) and logging of MCP server and tool names (maps to `OTEL_LOG_TOOL_DETAILS`). Both MUST default to off.

#### MCP Integration

- **FR-035**: HoloDeck MUST support two distinct MCP integration paths, selected automatically based on the active backend:
  - **SK backend path**: MCP tool configurations produce SK `MCPStdioPlugin` instances registered on the SK Kernel (existing behaviour, unchanged).
  - **Claude-native backend path**: MCP tool configurations are translated into the Claude Agent SDK's native MCP server specification format, which the Claude Code subprocess uses to launch and communicate with MCP server processes directly. The existing SK MCP plugin factory is NOT used for Anthropic agents.
  - Users configure MCP tools identically in `agent.yaml` regardless of backend. The routing is transparent.

#### Enterprise Authentication

- **FR-032**: The `LLMProvider` configuration model MUST be extended with an optional `auth_provider` field supporting five values:
  - `api_key` *(default)* — authenticates using `ANTHROPIC_API_KEY` from the environment or `.env` file. This is the default when `auth_provider` is not specified.
  - `oauth_token` — authenticates using `CLAUDE_CODE_OAUTH_TOKEN` from the environment or `.env` file. Intended for users who authenticate via `claude setup-token` rather than a direct API key.
  - `bedrock` — authenticates via AWS Bedrock using standard AWS environment credentials, without requiring `ANTHROPIC_API_KEY`.
  - `vertex` — authenticates via Google Vertex AI using standard GCP environment credentials, without requiring `ANTHROPIC_API_KEY`.
  - `foundry` — authenticates via Azure AI Foundry using standard Azure environment credentials, without requiring `ANTHROPIC_API_KEY`.
- **FR-032a**: When `auth_provider: api_key` is set (or when `auth_provider` is absent), the system MUST pass `ANTHROPIC_API_KEY` into the Claude Code subprocess environment. If the key is missing, the system MUST raise a clear startup error identifying the missing variable and how to set it.
- **FR-032b**: When `auth_provider: oauth_token` is set, the system MUST read `CLAUDE_CODE_OAUTH_TOKEN` from the environment or `.env` file and pass it into the Claude Code subprocess environment. If the token is absent, the system MUST raise a clear startup error identifying the missing variable and directing the user to run `claude setup-token`.
- **FR-033**: System MUST support authenticating Claude-native agents via AWS Bedrock using standard AWS environment credentials when `auth_provider: bedrock` is configured.
- **FR-034**: System MUST support authenticating Claude-native agents via Google Vertex AI using standard GCP environment credentials when `auth_provider: vertex` is configured.
- **FR-034a**: System MUST support authenticating Claude-native agents via Azure AI Foundry using standard Azure environment credentials when `auth_provider: foundry` is configured.

### Key Entities

- **Agent Execution Backend**: The runtime responsible for processing user messages, invoking tools, maintaining conversation state, and producing responses. This feature introduces a Claude-native backend as an alternative to the existing one.
- **Tool Adapter**: The bridge that exposes HoloDeck's existing tools (vectorstore, hierarchical document) to the Claude-native backend's tool-calling mechanism. It translates tool configurations and result formats between HoloDeck's internal model and what the Claude platform expects.
- **MCP Tool Bridge**: The component that connects MCP server processes to the Claude-native backend. It launches, manages, and communicates with MCP servers per the existing MCP tool configuration.
- **Execution Context**: The state maintained during an agent session — conversation history, tool results, active tool instances — that must be preserved and accessible across multiple turns.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of test cases submitted to a Claude-native agent are executed and produce a response (pass or fail), with no unhandled errors that abort the test run.
- **SC-002**: Vectorstore and hierarchical document search tools return results when queried by a Claude-native agent, with search accuracy within 10% of the same tool queried by the existing backend on identical inputs.
- **SC-003**: MCP tools configured for Claude-native agents are successfully invoked in 100% of valid test scenarios where tool usage is expected.
- **SC-004**: Streaming chat responses begin appearing within 3 seconds of a user submitting a message in the `holodeck chat` interface.
- **SC-005**: All evaluation metrics (NLP, G-Eval, RAG) produce a numeric score for 100% of test cases that have a valid response, with no metric silently skipped.
- **SC-006**: Zero existing agent YAML configurations (non-Anthropic providers) require modification after this feature is delivered.
- **SC-007**: A user familiar with HoloDeck's existing YAML syntax can configure and run a Claude-native agent without consulting documentation beyond the provider selection option.
- **SC-008**: An agent configured with `max_turns` never exceeds that iteration limit, regardless of task complexity.
- **SC-009**: File read, write, edit, glob, and grep operations configured in `agent.yaml` execute correctly in 100% of valid test cases where file access is within the working directory scope.
- **SC-010**: Shell commands executed by agents with bash access run within the configured sandbox; 100% of explicitly excluded commands are blocked without crashing the session.
- **SC-011**: In `manual` permission mode, 100% of agent actions trigger an approval prompt before execution — no action proceeds silently.
- **SC-012**: An agent configured with any supported auth method (`api_key`, `oauth_token`, `bedrock`, `vertex`, or `foundry`) runs a complete test case successfully using the corresponding credentials. For cloud providers (Bedrock, Vertex, Foundry), no Anthropic API key is required.
- **SC-013**: A tool configured in `agent.yaml` is callable by the Claude-native agent and returns correct results in 100% of valid invocations where the tool is correctly configured and the input is within the tool's expected parameters.
- **SC-014**: When an `allowed_tools` list is configured, 100% of tool calls to tools not on the list are blocked by the subprocess without crashing the session.
- **SC-015**: When observability is enabled for a Claude-native agent, token usage metrics, API request events, and tool result events are exported to the configured OTLP endpoint within 30 seconds of agent activity, verifiable via the OTLP backend.
- **SC-016**: When `response_format` is configured for a Claude-native agent, 100% of successful agent responses are returned as validated structured objects matching the schema. Evaluation metrics receive a serialised text representation and produce scores normally.

## Clarifications

### Session 2026-02-19

- Q: How should the system respond to Anthropic API failures (rate limit, 5xx errors, timeouts) mid-conversation? → A: Retry with exponential backoff (up to 3 attempts), then fail with a clear user-facing error message.
- Q: What is the scope of Semantic Kernel removal — agent loop only, or including embeddings and non-Anthropic providers? → A: SK stays installed. Only the agent conversation loop is replaced for Anthropic providers. SK continues to be used for embeddings (including in Anthropic agents) and for all non-Anthropic providers.
- Q: Which native Claude capabilities should be exposed through HoloDeck's YAML configuration? → A: Full capability set including: async streaming, multi-turn sessions, subagents, permission modes, max turns; extended thinking; file system (read/write/edit/glob/grep); bash execution with sandbox; notebook execution; @tool decorator and in-process MCP server; external MCP servers; tool interception (can_use_tool); allowed_tools allowlist; pre/post hooks; working directory; CLAUDE.md config loading; web search; and enterprise cloud auth (AWS Bedrock, Google Vertex AI, Azure AI Foundry).
- Q: How should long conversation history be managed when context limits are approached? → A: Use the Claude Agent SDK's native compaction capability, which automatically compresses conversation history when the context window fills, preserving continuity without manual truncation or summarization.
- Q: How do new capabilities (bash, file system, extended thinking, web search, etc.) surface in agent.yaml — individually or as bundles? → A: Individual opt-in flags per capability. Each feature must be explicitly enabled in agent.yaml. All capabilities are off by default; agents are minimal and least-privilege unless the user explicitly configures each capability.

## Assumptions

- The Claude Agent SDK is stable enough for the use cases in this spec (interactive chat, test execution, tool calling). Known alpha limitations are documented but do not block delivery.
- The Claude Agent SDK bundles the `claude` CLI as a package dependency — it does not need to be separately installed. However, **Node.js must be present on the host machine** as a runtime prerequisite for the SDK. HoloDeck MUST verify Node.js availability at agent startup and surface a clear error with installation guidance if it is absent.
- Vectorstore and hierarchical document tools continue to use SK-powered embedding services for index generation, even when the agent conversation loop uses the Claude-native backend. The Claude Agent SDK subprocess does not generate embeddings — HoloDeck initializes all tool indexes in-process before the subprocess starts. A top-level `embedding_provider: LLMProvider` field in `agent.yaml` configures which provider generates embeddings for Anthropic agents with vectorstore tools (e.g., `provider: openai`, `name: text-embedding-3-small`).
- MCP tools using stdio transport are the initial scope. SSE, WebSocket, and HTTP transports remain out of scope for this feature (as they are for the existing backend). For Anthropic agents, the Claude Code subprocess manages MCP server processes directly using its native MCP client — HoloDeck does not use SK's `MCPStdioPlugin` for these agents.
- Evaluation metrics operate on the agent's final text response, which both backends produce. The evaluation pipeline requires a provider-agnostic execution result interface (FR-012b) to extract responses without depending on SK's `ChatHistory` type.
- Users will have valid credentials for their chosen auth method configured via environment variable or `.env` file: `ANTHROPIC_API_KEY` for direct API access, `CLAUDE_CODE_OAUTH_TOKEN` for OAuth token auth (obtained via `claude setup-token`), or standard cloud provider environment credentials for Bedrock, Vertex, or Foundry. HoloDeck validates credential presence at startup, before the subprocess is spawned.
- The Claude Agent SDK's native context compaction handles conversation history management automatically; no custom truncation or summarization logic is required in HoloDeck.
- The existing `chat/executor.py` non-streaming architecture requires restructuring to consume an async event stream. This is treated as a prerequisite internal to this feature, not a separate feature. The streaming change is implemented as part of US-5 delivery.
- Tool filtering (`tool_filtering`) is gracefully disabled for `provider: anthropic` with a user-visible warning, not a silent failure.
- The Claude Code subprocess has built-in OTel support activated entirely via environment variables set before spawn. HoloDeck does not need to instrument the subprocess directly — it only needs to translate `agent.yaml` observability config into the correct env vars. The SK-based OTel decorators used for non-Anthropic backends do not apply here and are not used.

## Out of Scope

- SSE, WebSocket, and HTTP MCP transports (not yet supported in the existing backend either).
- Support for non-Anthropic providers (OpenAI, Azure, Ollama) through the Claude-native backend.
- Replacing the existing backend for non-Anthropic providers.
- The `holodeck deploy` command integration (planned separately in spec 019).
- Changes to evaluation metric logic or scoring algorithms.
- A GUI or web-based approval interface for `manual` permission mode (terminal prompt only for this spec).
- Removing Semantic Kernel from the project dependency tree (SK remains installed; only the agent conversation loop is replaced for Anthropic providers).
- **Semantic tool filtering (`tool_filtering`) for Claude-native agents**: The `tool_filtering` capability depends on SK's in-process kernel and is not compatible with the subprocess model. It is silently disabled with a warning for `provider: anthropic`. A follow-up spec will address tool filtering for the Claude-native backend.
- **Streaming for non-Anthropic backends**: Streaming is introduced in this feature via the Claude Agent SDK's event stream. Backporting streaming to the SK-based backend (OpenAI, Azure, Ollama) is a separate effort not in scope here.

- **`can_use_tool` callback interception**: Python callables cannot be serialized across the subprocess boundary. Implementing `can_use_tool` requires an IPC round-trip (subprocess emits permission request → parent evaluates → subprocess receives approve/deny). This is deferred to a follow-up spec.
- **Pre/post execution hooks (Python callables)**: Same subprocess boundary constraint as `can_use_tool`. Live Python hook execution within the Claude Code agent loop is not achievable in this model without a defined IPC protocol. Deferred to a follow-up spec.
