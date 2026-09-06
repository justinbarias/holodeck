# Feature Specification: MCP Tool Operations

**Feature Branch**: `010-mcp-tool-operations`
**Created**: 2025-11-28
**Status**: Draft
**Input**: User Story US 6.3 - MCP Tool Operations (Priority: P1)

## Clarifications

### Session 2025-11-28

- Q: What should be the default `request_timeout` value for MCP operations when not explicitly configured? → A: 60 seconds (generous - complex operations)
- Q: When an environment variable reference cannot be resolved, what should the system do? → A: Fail at config load time with clear error message
- Q: How should the system handle concurrent/parallel calls to the same MCP server? → A: Allow parallel calls (server handles concurrency)

## Reference Implementations

This specification is informed by:
- **VS Code MCP Configuration**: [mcp.json specification](https://code.visualstudio.com/docs/copilot/customization/mcp-servers)
- **Semantic Kernel MCP Module**: [mcp.py](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/connectors/mcp.py)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Standard MCP Server Integration (Priority: P1)

A developer defines an MCP (Model Context Protocol) tool in their agent.yaml that references a standardized MCP server package (e.g., `@modelcontextprotocol/server-filesystem`). When the agent executes and needs to use this tool, the HoloDeck agent engine establishes a connection to the MCP server via stdio transport, invokes the tool using the standard Model Context Protocol, and receives the response.

**Why this priority**: This is the core MCP functionality. Without the ability to invoke standard MCP servers, no other MCP features can work. This enables developers to leverage the entire ecosystem of existing MCP servers.

**Independent Test**: Can be tested by configuring a simple MCP tool (e.g., filesystem server), calling it through the agent, and verifying the tool executes and returns results.

**Acceptance Scenarios**:

1. **Given** a tool defined as `type: mcp` with `server: @modelcontextprotocol/server-filesystem` in agent.yaml, **When** the agent calls this tool, **Then** the MCP server is invoked using the standard Model Context Protocol and returns the expected result
2. **Given** an MCP tool with `command: npx` and `args: ["-y", "@modelcontextprotocol/server-memory"]`, **When** the agent starts, **Then** the MCP server process is spawned with the correct command and arguments
3. **Given** an MCP server returns a structured response, **When** the agent receives the response, **Then** the response is properly parsed according to MCP protocol specifications

---

### User Story 2 - MCP Server Configuration (Priority: P1)

A developer configures an MCP tool with custom settings such as environment variables, allowed directories, API tokens, or other server-specific configuration. When the agent executes, the configuration is properly passed to the MCP server, allowing it to operate within the specified constraints.

**Why this priority**: Most real-world MCP integrations require configuration (API keys, file paths, permissions). Without configuration support, MCP tools cannot be practically used.

**Independent Test**: Can be tested by configuring an MCP tool with specific environment variables or configuration options and verifying the MCP server receives and uses them.

**Acceptance Scenarios**:

1. **Given** an MCP tool with `env: {API_KEY: "${MY_API_KEY}"}` configuration, **When** the agent executes, **Then** the environment variable is properly resolved and passed to the MCP server process
2. **Given** an MCP tool with `config: {allowed_directories: ["/workspace/data"]}`, **When** the MCP server initializes, **Then** the configuration is passed as server initialization parameters
3. **Given** an MCP tool with `envFile: ".env"` specified, **When** the agent starts, **Then** environment variables from the file are loaded and passed to the MCP server
4. **Given** an MCP tool with `encoding: "utf-8"` specified for stdio transport, **When** the agent communicates with the server, **Then** the specified encoding is used for stdin/stdout streams

---

### User Story 3 - MCP Response Processing (Priority: P1)

When an MCP server returns structured data (tool results, resources, or prompts), the agent engine parses the response according to MCP protocol specifications and makes it available for the agent's reasoning process. The response data is formatted appropriately for the LLM context, following Semantic Kernel's content type conversion patterns.

**Why this priority**: The value of MCP tools comes from their responses being usable in agent reasoning. Without proper response processing, MCP tool calls would be ineffective.

**Independent Test**: Can be tested by calling an MCP tool that returns structured data and verifying the response is correctly parsed and integrated into the agent's conversation context.

**Acceptance Scenarios**:

1. **Given** an MCP server that returns a text content block, **When** the agent receives the response, **Then** the text is extracted and converted to TextContent for LLM consumption
2. **Given** an MCP server that returns an image content block, **When** the agent receives the response, **Then** the image data is converted to ImageContent (base64 encoded or URL referenced)
3. **Given** an MCP server that returns an audio content block, **When** the agent receives the response, **Then** the audio data is converted to AudioContent
4. **Given** an MCP server that returns multiple content blocks, **When** the agent receives the response, **Then** all content blocks are processed and concatenated appropriately
5. **Given** an MCP server that returns an EmbeddedResource or ResourceLink, **When** the agent receives the response, **Then** the resource is converted to BinaryContent

---

### User Story 4 - MCP Tool Discovery (Priority: P1)

When an MCP server is connected, the agent engine automatically discovers available tools and prompts from the server. The discovered tools are registered with the agent and can be invoked through the standard tool calling mechanism. Tool names are normalized to be compatible with the agent's naming conventions.

**Why this priority**: Automatic tool discovery eliminates the need to manually specify each tool, making MCP integration seamless and reducing configuration overhead.

**Independent Test**: Can be tested by connecting to an MCP server and verifying all server-provided tools are discovered and callable.

**Acceptance Scenarios**:

1. **Given** an MCP server that exposes multiple tools, **When** the agent connects and `load_tools: true` (default), **Then** all server tools are discovered and registered with the agent
2. **Given** an MCP tool with `load_prompts: true`, **When** the agent connects, **Then** server-provided prompts are discovered and available for use
3. **Given** an MCP server tool with special characters in its name, **When** the tool is registered, **Then** the name is normalized (invalid characters replaced with "-")
4. **Given** an MCP server that sends a `notifications/tools/list_changed` notification, **When** the agent receives it, **Then** the tool list is automatically refreshed

---

### User Story 5 - MCP Error Handling (Priority: P2)

When an MCP server connection fails, the server process crashes, or the server returns an error response, the agent engine catches the error and provides a clear, actionable message about the MCP server unavailability. The agent can gracefully handle the failure and potentially continue with alternative approaches.

**Why this priority**: Robust error handling is essential for production-ready agents, but the core happy path functionality takes precedence.

**Independent Test**: Can be tested by intentionally causing MCP server failures (invalid command, network timeout, malformed response) and verifying appropriate error messages are returned.

**Acceptance Scenarios**:

1. **Given** an MCP server command that does not exist, **When** the agent attempts to start the server, **Then** a clear error message indicates the server could not be started with the command path
2. **Given** an MCP server that crashes during operation, **When** the agent attempts to use the tool, **Then** an error is returned indicating the server is no longer available
3. **Given** an MCP server that returns an error response, **When** the agent processes the response, **Then** the error details are extracted and presented as a tool error (FunctionExecutionException pattern)
4. **Given** an MCP server connection that times out, **When** the agent waits for a response, **Then** a timeout error is returned with configurable `request_timeout` duration

---

### User Story 6 - HTTP/SSE MCP Servers (Priority: P2)

A developer configures an MCP tool to connect to a remote MCP server via HTTP or Server-Sent Events (SSE) transport. The agent engine establishes the HTTP connection, handles authentication headers, and communicates using the MCP protocol over HTTP.

**Why this priority**: HTTP/SSE support enables cloud-hosted MCP servers and enterprise integrations, but stdio is the more common pattern for local development.

**Independent Test**: Can be tested by configuring an MCP tool with HTTP transport and verifying connection, authentication, and request/response flow.

**Acceptance Scenarios**:

1. **Given** an MCP tool with `transport: sse` and `url: "https://mcp.example.com"`, **When** the agent calls this tool, **Then** the connection is established using Server-Sent Events
2. **Given** an MCP tool with `headers: {Authorization: "Bearer ${TOKEN}"}` configured, **When** the agent sends a request, **Then** the headers are included in the HTTP request
3. **Given** an MCP tool with `timeout` and `sse_read_timeout` configured, **When** the agent establishes a connection, **Then** the specified timeouts are applied

---

### User Story 7 - WebSocket MCP Servers (Priority: P3)

A developer configures an MCP tool to connect to an MCP server via WebSocket transport for bidirectional communication. This is useful for servers that require persistent connections or real-time updates.

**Why this priority**: WebSocket support is less common but needed for certain advanced MCP server implementations.

**Independent Test**: Can be tested by configuring an MCP tool with WebSocket transport and verifying bidirectional communication.

**Acceptance Scenarios**:

1. **Given** an MCP tool with `transport: websocket` and `url: "wss://mcp.example.com"`, **When** the agent connects, **Then** a WebSocket connection is established
2. **Given** an active WebSocket connection, **When** the agent calls a tool, **Then** the request is sent over the existing WebSocket connection

---

### User Story 8 - Streamable HTTP Transport (Priority: P3)

A developer configures an MCP tool to use HTTP with streaming response support. The agent engine handles streaming responses and can optionally terminate the connection on close.

**Why this priority**: Streamable HTTP is an advanced transport option for specific use cases.

**Independent Test**: Can be tested by configuring streamable HTTP transport and verifying streaming responses are handled correctly.

**Acceptance Scenarios**:

1. **Given** an MCP tool with `transport: http` and `url` configured, **When** the agent calls this tool, **Then** HTTP transport with streaming support is used
2. **Given** an MCP tool with `terminate_on_close: true`, **When** the agent finishes using the tool, **Then** the HTTP connection is terminated

---

### Edge Cases

- What happens when an MCP server returns an empty response?
- How does the system handle concurrent calls to the same MCP server? (Answer: Allow parallel calls; server handles concurrency)
- What happens when the MCP server returns content types the agent doesn't support?
- How are large responses (exceeding token limits) handled?
- What happens when environment variable references cannot be resolved? (Answer: Fail at config load time with clear error)
- How does the system behave when an MCP server requires initialization but initialization fails?
- What happens when the same MCP server is configured multiple times with different names?
- How does the system handle MCP server logging callbacks at different log levels?
- What happens when an MCP server sends a sampling request requiring LLM completion?

## Requirements *(mandatory)*

### Functional Requirements

#### Core MCP Support
- **FR-001**: System MUST support defining MCP tools with `type: mcp` in agent.yaml configuration
- **FR-002**: System MUST implement MCP protocol lifecycle (initialize session, call_tool, close)
- **FR-003**: System MUST parse MCP server responses according to protocol specification (content blocks, tool results)

#### Transport Types
- **FR-004**: System MUST support stdio transport for MCP servers, spawning server processes with configurable `command`, `args`, `env`, and `encoding`
- **FR-005**: System MUST support SSE (Server-Sent Events) transport with configurable `url`, `headers`, `timeout`, and `sse_read_timeout`
- **FR-006**: System MUST support WebSocket transport with configurable `url`
- **FR-007**: System MUST support Streamable HTTP transport with configurable `url`, `headers`, `timeout`, `sse_read_timeout`, and `terminate_on_close`

#### Configuration
- **FR-008**: System MUST support environment variable configuration for MCP servers via `env` mapping
- **FR-009**: System MUST support environment variable file loading via `env_file` configuration
- **FR-010**: System MUST resolve environment variable references using `${VAR_NAME}` syntax in configuration; unresolvable references MUST cause configuration load to fail with a clear error message identifying the missing variable
- **FR-011**: System MUST support configuration passthrough to MCP servers via `config` object
- **FR-012**: System MUST support `request_timeout` configuration for MCP operations (default: 60 seconds)

#### Tool & Prompt Discovery
- **FR-013**: System MUST automatically discover and register tools from MCP servers when `load_tools: true` (default)
- **FR-014**: System MUST automatically discover prompts from MCP servers when `load_prompts: true` (default)
- **FR-015**: System MUST normalize tool names by replacing invalid characters with "-" (per Semantic Kernel pattern)
- **FR-016**: System MUST handle `notifications/tools/list_changed` and `notifications/prompts/list_changed` notifications to refresh tool/prompt lists

#### Content Type Conversion
- **FR-017**: System MUST convert MCP TextContent to internal TextContent representation
- **FR-018**: System MUST convert MCP ImageContent to internal ImageContent representation
- **FR-019**: System MUST convert MCP AudioContent to internal AudioContent representation
- **FR-020**: System MUST convert MCP EmbeddedResource and ResourceLink to internal BinaryContent representation

#### Error Handling
- **FR-021**: System MUST provide clear error messages when MCP server connection fails, including the command that failed
- **FR-022**: System MUST provide clear error messages when MCP server returns an error response
- **FR-023**: System MUST handle server process crashes gracefully and report server unavailability

#### Lifecycle Management
- **FR-024**: System MUST manage MCP server process lifecycle (start on first use, terminate on agent shutdown)
- **FR-025**: System MUST support async context management for proper resource cleanup
- **FR-026**: System MUST validate MCP tool configuration at agent load time and provide helpful validation errors

#### Logging & Callbacks
- **FR-027**: System MUST support MCP logging callbacks and map MCP log levels to standard logging levels
- **FR-028**: System MUST support message handler callbacks for server exceptions and notifications

### Key Entities

- **MCPPlugin**: Base representation of an MCP server plugin with name, description, load_tools/load_prompts flags, session management, request_timeout, and kernel reference (following Semantic Kernel MCPPluginBase pattern)
- **MCPStdioPlugin**: Stdio transport plugin with command, args, env, and encoding configuration
- **MCPSsePlugin**: SSE transport plugin with url, headers, timeout, and sse_read_timeout configuration
- **MCPWebsocketPlugin**: WebSocket transport plugin with url configuration
- **MCPStreamableHttpPlugin**: Streamable HTTP transport plugin with url, headers, timeouts, and terminate_on_close configuration
- **MCPSession**: Manages the MCP protocol session with read/write streams and message handling
- **MCPContentBlock**: Represents content in an MCP response (TextContent, ImageContent, AudioContent, EmbeddedResource, ResourceLink)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Developers can configure and use any standard MCP server package in their agent within 5 minutes following documentation
- **SC-002**: MCP tool invocations complete within 10 seconds for typical operations (file access, simple queries)
- **SC-003**: 95% of MCP tool errors produce actionable error messages that indicate the root cause
- **SC-004**: Configuration validation catches 100% of malformed MCP tool definitions before agent execution
- **SC-005**: Developers can successfully use MCP servers requiring authentication (API keys, tokens) without exposing secrets in configuration files
- **SC-006**: Agent can recover gracefully from MCP server failures and continue operating with remaining tools
- **SC-007**: All four transport types (stdio, SSE, WebSocket, streamable HTTP) function correctly when configured
- **SC-008**: Tool discovery automatically registers 100% of tools exposed by connected MCP servers
