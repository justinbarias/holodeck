# Feature Specification: MCP CLI Command Group

**Feature Branch**: `013-mcp-cli`
**Created**: 2025-12-13
**Status**: Draft
**Input**: User description: "Create a spec for a new cli command group 'holodeck mcp' with subcommands: search, list, add, remove. Search uses the official MCP registry API. List shows installed MCP servers. Add/Remove work with agent files or global config."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Search MCP Registry (Priority: P1)

As a developer, I want to search the official MCP registry for available MCP servers so that I can discover tools to extend my agent's capabilities.

**Why this priority**: This is the discovery entry point - users must find servers before they can add them. Without search, users would need to know server names beforehand.

**Independent Test**: Can be fully tested by running `holodeck mcp search <query>` and verifying results are returned from the registry. Delivers value by enabling discovery of available integrations.

**Acceptance Scenarios**:

1. **Given** the MCP registry is accessible, **When** user runs `holodeck mcp search filesystem`, **Then** display matching servers with name, description, and available transports
2. **Given** the MCP registry is accessible, **When** user runs `holodeck mcp search` with no query, **Then** display a paginated list of all available servers
3. **Given** the MCP registry is inaccessible, **When** user runs `holodeck mcp search <query>`, **Then** display a clear error message about network connectivity
4. **Given** no servers match the query, **When** user runs `holodeck mcp search xyz123nonexistent`, **Then** display "No servers found matching 'xyz123nonexistent'"

---

### User Story 2 - Add MCP Server to Agent (Priority: P1)

As a developer, I want to add an MCP server to my agent configuration so that my agent can use the server's tools and capabilities.

**Why this priority**: Core functionality - adding servers is the primary action users take after discovery. Enables immediate productivity.

**Independent Test**: Can be fully tested by running `holodeck mcp add <server-name>` and verifying the agent.yaml file is updated with the correct MCP tool configuration.

**Acceptance Scenarios**:

1. **Given** an agent.yaml exists in the current directory, **When** user runs `holodeck mcp add io.github.user/server`, **Then** add the MCP server configuration to the tools section of agent.yaml
2. **Given** user specifies `--agent custom-agent.yaml`, **When** user runs `holodeck mcp add io.github.user/server --agent custom-agent.yaml`, **Then** add the server to the specified agent file
3. **Given** user specifies `-g` flag, **When** user runs `holodeck mcp add io.github.user/server -g`, **Then** add the server to `~/.holodeck/config.yaml` in a global mcp_servers section
4. **Given** the server is already installed, **When** user runs `holodeck mcp add io.github.user/server`, **Then** display "Server 'io.github.user/server' is already configured" and do not duplicate
5. **Given** no agent.yaml exists and no `-g` flag, **When** user runs `holodeck mcp add io.github.user/server`, **Then** display error "No agent.yaml found in current directory. Use --agent to specify a file or -g for global install."

---

### User Story 3 - List Installed MCP Servers (Priority: P2)

As a developer, I want to see all MCP servers installed in my agent and global configuration so that I can understand what integrations are currently available.

**Why this priority**: Important for visibility but secondary to add/search. Users need to see what's installed, but this is typically done less frequently than adding servers.

**Independent Test**: Can be fully tested by running `holodeck mcp list` after adding servers and verifying the output shows all configured MCP servers.

**Acceptance Scenarios**:

1. **Given** an agent.yaml exists with MCP tools, **When** user runs `holodeck mcp list`, **Then** display all MCP servers from agent.yaml with their names and transport types
2. **Given** global config has MCP servers, **When** user runs `holodeck mcp list -g`, **Then** display MCP servers from `~/.holodeck/config.yaml`
3. **Given** both agent and global config have MCP servers, **When** user runs `holodeck mcp list --all`, **Then** display servers from both sources with clear labels indicating source
4. **Given** no MCP servers are configured, **When** user runs `holodeck mcp list`, **Then** display "No MCP servers configured. Use 'holodeck mcp search' to find available servers."

---

### User Story 4 - Remove MCP Server (Priority: P2)

As a developer, I want to remove an MCP server from my configuration so that I can clean up unused integrations.

**Why this priority**: Cleanup functionality is important but used less frequently than adding servers.

**Independent Test**: Can be fully tested by running `holodeck mcp remove <server-name>` and verifying the server is removed from the configuration file.

**Acceptance Scenarios**:

1. **Given** an MCP server is configured in agent.yaml, **When** user runs `holodeck mcp remove io.github.user/server`, **Then** remove the server from agent.yaml tools section
2. **Given** user specifies `--agent custom-agent.yaml`, **When** user runs `holodeck mcp remove io.github.user/server --agent custom-agent.yaml`, **Then** remove from the specified file
3. **Given** user specifies `-g` flag, **When** user runs `holodeck mcp remove io.github.user/server -g`, **Then** remove the server from `~/.holodeck/config.yaml`
4. **Given** the server is not installed, **When** user runs `holodeck mcp remove io.github.user/server`, **Then** display "Server 'io.github.user/server' is not configured"
5. **Given** server exists in both agent and global config, **When** user runs `holodeck mcp remove io.github.user/server` without flags, **Then** remove only from agent.yaml (local takes precedence)

---

### Edge Cases

- What happens when the MCP registry API is slow or fails? Fail fast with 5s timeout; display clear error message with suggestion to retry manually.
- How does the system handle servers with multiple transport types (stdio, sse, http)? Auto-select stdio as default; users can override with `--transport` flag.
- What happens when adding a server that requires specific environment variables? Display required env vars and prompt user to configure them.
- How does the system handle malformed agent.yaml files? Display validation error and do not modify the file.
- What happens when the global config directory (`~/.holodeck/`) doesn't exist? Create it automatically with appropriate permissions.
- How does merging work when the same server exists in both global and agent config? Agent-level configuration takes precedence; global servers provide defaults.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a `holodeck mcp` command group with `search`, `list`, `add`, and `remove` subcommands
- **FR-002**: System MUST query the official MCP registry API (https://registry.modelcontextprotocol.io) for the `search` subcommand
- **FR-003**: System MUST support searching servers by name substring match via the registry API's `search` parameter
- **FR-004**: System MUST display server information including name, description, and available transport types in search results
- **FR-005**: System MUST paginate search results when the registry returns multiple pages
- **FR-006**: System MUST add MCP server configurations to the `tools` section of agent YAML files
- **FR-007**: System MUST support the `--agent <filename>` option to specify a custom agent file
- **FR-008**: System MUST support the `-g` or `--global` flag to install servers to `~/.holodeck/config.yaml`
- **FR-009**: System MUST create `~/.holodeck/config.yaml` if it does not exist when using global install
- **FR-010**: System MUST prevent duplicate server installations by checking existing configurations
- **FR-011**: System MUST list MCP servers from agent configuration by default
- **FR-012**: System MUST support `--all` flag to list servers from both agent and global configurations
- **FR-013**: System MUST clearly indicate the source (agent/global) when listing servers from multiple sources
- **FR-014**: System MUST remove servers only from the specified configuration scope (agent or global)
- **FR-015**: System MUST validate YAML files before and after modifications
- **FR-016**: System MUST preserve existing YAML formatting and comments when modifying files
- **FR-017**: System MUST provide clear error messages for all failure scenarios
- **FR-018**: System MUST display required environment variables when adding servers that need them
- **FR-019**: System MUST support version selection via `--version` flag when adding servers (defaults to latest)
- **FR-020**: System MUST auto-select `stdio` transport when multiple transports are available, with `--transport` flag to override
- **FR-021**: System MUST use 5-second timeout for registry API calls with no automatic retries (fail fast behavior)

### Key Entities

- **MCP Server**: Represents a Model Context Protocol server from the registry. Key attributes: name (reverse-DNS format), description, version, transport configurations, required environment variables.
- **Global Config**: The user-level configuration at `~/.holodeck/config.yaml` containing default MCP servers that apply to all agents. New section: `mcp_servers` containing a list of MCPTool configurations.
- **Agent Config**: The project-level `agent.yaml` containing agent-specific tools. MCP servers appear in the `tools` section with `type: mcp`.
- **Server Package**: Distribution information for an MCP server including registry type (npm, pypi, docker), package identifier, and transport configuration.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can discover relevant MCP servers within 30 seconds using the search command
- **SC-002**: Users can add an MCP server to their agent configuration in under 1 minute from search to successful add
- **SC-003**: Users can view all installed MCP servers (local and global) with a single command
- **SC-004**: Configuration changes are atomic - partial failures leave files unchanged
- **SC-005**: All error messages provide actionable guidance for resolution
- **SC-006**: Global MCP servers are correctly merged with agent-level servers when agents execute (agent takes precedence)

## Clarifications

### Session 2025-12-13

- Q: When a server offers multiple transport types, what behavior should `holodeck mcp add` use? → A: Auto-select stdio as default, allow `--transport` flag override
- Q: What timeout and retry behavior for MCP registry API? → A: 5s timeout, no retries (fail fast)

## Assumptions

- The MCP registry API at https://registry.modelcontextprotocol.io follows the OpenAPI specification documented at https://github.com/modelcontextprotocol/registry
- The registry API is publicly accessible without authentication for read operations (search, list versions, get server details)
- Server names in the registry use reverse-DNS format (e.g., `io.github.user/server-name`)
- Most common transport type is `stdio` which will be the default when multiple transports are available
- The existing MCPTool Pydantic model in `src/holodeck/models/tool.py` is sufficient for representing servers from the registry
- Users have network connectivity to access the registry (offline mode is not required for initial implementation)
