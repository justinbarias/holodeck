# Feature Specification: MCP HTTP/SSE Transport Support

**Feature Branch**: `feature/007-claude-agent-features`
**Spec ID**: 027-mcp-http-sse-transport
**Created**: 2026-03-28
**Status**: Draft
**Input**: Extend MCP tool bridge to support HTTP and SSE transports in addition to the existing STDIO transport, enabling connection to remote MCP servers.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Connect to a Remote MCP Server via SSE (Priority: P1)

A user wants to connect their Claude agent to a cloud-hosted MCP server that uses Server-Sent Events (SSE) for communication. They specify `transport: sse` with a URL in their tool configuration and the agent discovers and uses the remote tools.

**Why this priority**: SSE is the most common transport for remote MCP servers. Many MCP server implementations (including popular community servers) default to SSE. Without this, users are limited to locally-spawned STDIO processes only.

**Independent Test**: Configure an MCP tool with `transport: sse` and `url: http://localhost:3000/sse` in agent.yaml, initialize the agent, and verify the SDK receives an SSE server configuration.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with an MCP tool configured as `transport: sse` and a valid URL, **When** the agent is initialized, **Then** the SDK receives an SSE server config with the URL.
2. **Given** an SSE MCP tool with custom `headers` specified, **When** the agent is initialized, **Then** the headers are included in the SSE server config.
3. **Given** an SSE MCP tool with no `url` field, **When** the config is loaded, **Then** a validation error is raised requiring a URL for SSE transport.
4. **Given** an SSE MCP tool with environment variable references in the URL (e.g., `${MCP_SERVER_URL}`), **When** the agent is initialized, **Then** the URL is resolved from environment variables.

---

### User Story 2 - Connect to an HTTP-Based MCP Server (Priority: P1)

A user wants to connect to an MCP server exposed over HTTP (streamable HTTP transport). They specify `transport: http` with a URL and optional authentication headers.

**Why this priority**: HTTP transport is the newer, recommended transport for remote MCP servers. It supports bidirectional communication and is increasingly adopted.

**Independent Test**: Configure an MCP tool with `transport: http` and a URL in agent.yaml, initialize the agent, and verify the SDK receives an HTTP server configuration.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with an MCP tool configured as `transport: http` and a valid URL, **When** the agent is initialized, **Then** the SDK receives an HTTP server config with the URL.
2. **Given** an HTTP MCP tool with `headers` including an Authorization token, **When** the agent is initialized, **Then** the headers are passed through to the SDK config.
3. **Given** an HTTP MCP tool with no `url` field, **When** the config is loaded, **Then** a validation error is raised.

---

### User Story 3 - Graceful Handling of Unsupported Transports (Priority: P3)

A user configures an MCP tool with a transport type not supported by the Claude SDK (e.g., WebSocket). The system provides a clear warning explaining which transports are supported.

**Why this priority**: Error handling for unsupported cases -- important for user experience but not core functionality.

**Independent Test**: Configure an MCP tool with `transport: websocket`, load the config, and verify a clear warning message is produced.

**Acceptance Scenarios**:

1. **Given** an MCP tool with `transport: websocket`, **When** the Claude backend builds MCP configs, **Then** the tool is skipped with a warning naming the unsupported transport and listing supported alternatives (stdio, sse, http).

---

### Edge Cases

- What happens when the remote SSE/HTTP server is unreachable at agent initialization time? The SDK handles connection failures; the system should surface the SDK's error clearly.
- What happens when headers contain sensitive values (API keys)? Headers should support environment variable substitution (e.g., `Authorization: Bearer ${API_KEY}`).
- What happens when an MCP tool has `transport: stdio` (the existing case)? Existing behavior is preserved with no changes.
- What happens when multiple MCP tools use different transport types? Each tool is translated to its appropriate SDK config type independently.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST translate MCP tools with `transport: sse` into the Claude SDK's SSE server config format, including URL and optional headers.
- **FR-002**: System MUST translate MCP tools with `transport: http` into the Claude SDK's HTTP server config format, including URL and optional headers.
- **FR-003**: System MUST continue to support `transport: stdio` with no behavior changes.
- **FR-004**: System MUST validate that SSE and HTTP MCP tools include a `url` field, raising a clear error if missing.
- **FR-005**: System MUST resolve environment variable references in URLs and header values (e.g., `${API_KEY}`).
- **FR-006**: System MUST skip MCP tools with unsupported transports (e.g., websocket) with a warning listing supported alternatives.
- **FR-007**: System MUST support mixing different transport types across multiple MCP tools in the same agent configuration.

### Key Entities

- **MCPTool**: Extended with `url` (string, required for SSE/HTTP) and `headers` (dict, optional) fields.
- **TransportType**: Existing enum already includes `sse`, `http`, `websocket` -- the bridge now acts on `sse` and `http` instead of skipping them.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can connect to remote MCP servers via SSE transport by specifying `transport: sse` and a URL in their agent YAML.
- **SC-002**: Users can connect to remote MCP servers via HTTP transport by specifying `transport: http` and a URL in their agent YAML.
- **SC-003**: Existing STDIO MCP tool configurations continue to work without any changes.
- **SC-004**: Missing required fields (URL for SSE/HTTP) produce clear validation errors before agent execution.
- **SC-005**: Environment variable substitution works in URLs and headers for all transport types.

## Assumptions

- The Claude Agent SDK supports `McpSseServerConfig` and `McpHttpServerConfig` (or equivalent) types for SSE and HTTP transports respectively.
- The HoloDeck `TransportType` enum in `tool.py` already includes `sse` and `http` variants -- they just need to be wired through the MCP bridge.
- Header values may contain secrets and should be resolved from environment variables rather than hardcoded in YAML.
