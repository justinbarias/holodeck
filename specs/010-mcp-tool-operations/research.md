# Research: MCP Tool Operations

**Feature**: 010-mcp-tool-operations
**Date**: 2025-11-28

## Research Questions Resolved

### 1. Semantic Kernel MCP Module Integration

**Decision**: Use Semantic Kernel's MCP plugins directly via composition/wrapping

**Rationale**:
- Semantic Kernel (already a dependency at v1.37.1) provides complete MCP implementation
- Four plugin types available: MCPStdioPlugin, MCPSsePlugin, MCPWebsocketPlugin, MCPStreamableHttpPlugin
- Plugins handle protocol lifecycle, tool discovery, content conversion
- Avoids reimplementing MCP protocol from scratch

**Alternatives Considered**:
- Direct mcp-sdk usage: Rejected - would duplicate work already done in SK
- Custom MCP implementation: Rejected - unnecessary complexity, maintenance burden

### 2. MCP Protocol Version Compatibility

**Decision**: Target MCP protocol version supported by Semantic Kernel (2024-11-05 spec)

**Rationale**:
- SK maintains compatibility with official MCP specification
- Protocol includes: initialize, tools/list, tools/call, prompts/list, prompts/get, notifications
- Content types: TextContent, ImageContent, AudioContent, EmbeddedResource, ResourceLink

**Alternatives Considered**:
- Pin specific MCP version: Rejected - let SK handle version management

### 3. Transport Type Detection

**Decision**: Explicit `transport` field in configuration (not auto-detection)

**Rationale**:
- Clear user intent
- No ambiguity between http and sse transports
- Follows VS Code mcp.json pattern
- Default to `stdio` when not specified

**Transport Mapping**:
| transport value | SK Plugin Class |
|-----------------|-----------------|
| stdio (default) | MCPStdioPlugin |
| sse | MCPSsePlugin |
| websocket | MCPWebsocketPlugin |
| http | MCPStreamableHttpPlugin |

### 3.1 Stdio Command Restrictions (Security)

**Decision**: Only allow `npx`, `uvx`, or `docker` as stdio commands

**Rationale**:
- **Security**: Prevents command injection attacks via arbitrary shell commands
- **Consistency**: All MCP servers run through known package managers or containers
- **Auditability**: Easy to understand what's being executed
- **Ecosystem alignment**: Most MCP servers are distributed via npm (npx) or Python (uvx)

**Supported Commands**:
| Command | Runtime | Package Source |
|---------|---------|----------------|
| `npx` | Node.js | npm registry |
| `uvx` | Python | PyPI via uv |
| `docker` | Docker | Container registries |

**Alternatives Rejected**:
- Arbitrary command execution: Security risk (shell injection)
- Python/node direct: Less portable, harder to manage dependencies
- Allowlist expansion: Can be added later if needed with security review

### 4. Environment Variable Resolution Strategy

**Decision**: Fail-fast at configuration load time, reusing existing `env_loader.py`

**Rationale**:
- Per clarification session decision
- Prevents runtime surprises
- Security benefit: no empty credentials passed accidentally
- Clear error messages help debugging
- **DRY**: Existing `substitute_env_vars()` already implements fail-fast behavior

**Implementation** (REUSE existing code):
- **REUSE**: `holodeck.config.env_loader.substitute_env_vars()` for `${VAR}` pattern
- **REUSE**: `holodeck.config.env_loader.load_env_file()` for `env_file` loading
- **DO NOT**: Create new env resolution logic - use existing functions

**Existing Functions in `env_loader.py`**:
```python
substitute_env_vars(text: str) -> str  # Already fails-fast on missing vars
load_env_file(path: str) -> dict[str, str]  # Already loads .env files
get_env_var(key: str, default: Any) -> Any  # Get single var with default
```

### 5. Tool Name Normalization Pattern

**Decision**: Replace invalid characters with "-" (per Semantic Kernel pattern)

**Rationale**:
- Consistent with SK MCPPluginBase implementation
- Pattern: `re.sub(r"[^a-zA-Z0-9_]", "-", tool_name)`
- Ensures tool names are valid Python identifiers

### 6. Content Type Conversion Strategy

**Decision**: Map MCP content types to HoloDeck internal representation

**Rationale**:
- HoloDeck already has content models for LLM integration
- Conversion ensures consistent handling across tool types

**Mapping**:
| MCP Type | HoloDeck Type |
|----------|---------------|
| TextContent | str (plain text) |
| ImageContent | dict with base64/url |
| AudioContent | dict with base64/url |
| EmbeddedResource | bytes (binary) |
| ResourceLink | str (URI reference) |

### 7. Error Handling Strategy

**Decision**: Custom exception hierarchy extending HoloDeckError

**Rationale**:
- Consistent with existing error patterns in `lib/errors.py`
- Clear categorization: config errors vs runtime errors vs protocol errors

**Exception Types**:
- `MCPConfigError`: Invalid MCP configuration (extends ConfigError)
- `MCPConnectionError`: Server connection failures (extends ToolError)
- `MCPProtocolError`: Protocol-level errors (extends ToolError)
- `MCPTimeoutError`: Request timeout (extends MCPConnectionError)

### 8. Async Context Management

**Decision**: Use async context managers for MCP server lifecycle

**Rationale**:
- SK plugins already implement `__aenter__`/`__aexit__`
- Ensures proper cleanup on agent shutdown
- Supports multiple concurrent MCP connections

**Pattern**:
```python
async with mcp_plugin:
    result = await mcp_plugin.call_tool(tool_name, **kwargs)
```

### 9. Concurrent Call Handling

**Decision**: Allow parallel calls; server handles concurrency

**Rationale**:
- Per clarification session decision
- MCP servers designed to handle concurrent requests
- No client-side queuing needed
- Simplifies implementation

### 10. Default Timeout Configuration

**Decision**: 60 seconds default request_timeout

**Rationale**:
- Per clarification session decision
- Generous timeout for complex operations (file processing, database queries)
- Configurable per-tool via `request_timeout` field

## Technology Best Practices

### Semantic Kernel MCP Best Practices

1. **Plugin Lifecycle**: Always use async context manager
2. **Tool Discovery**: Set `load_tools=True` (default), call `connect()` to discover
3. **Prompt Discovery**: Set `load_prompts=True` if needed
4. **Kernel Reference**: Pass kernel for sampling callback support
5. **Logging**: Set log level matching Python logger configuration

### YAML Configuration Best Practices

1. **Transport Selection**: Always explicit, default to stdio
2. **Environment Variables**: Use `${VAR}` for process env vars
3. **Timeouts**: Configure realistic timeouts based on expected operation duration
4. **Server References**: Use npm package names for stdio (e.g., `@modelcontextprotocol/server-filesystem`)

## DRY Compliance: Reuse Existing Code

### Existing Code to Reuse

| Module | Function | Purpose | MCP Usage |
|--------|----------|---------|-----------|
| `holodeck.config.env_loader` | `substitute_env_vars(text)` | `${VAR}` substitution with fail-fast | Resolve env vars in MCP config |
| `holodeck.config.env_loader` | `load_env_file(path)` | Load .env file to dict | Handle `env_file` config option |
| `holodeck.config.env_loader` | `get_env_var(key, default)` | Get single env var | Optional fallback lookups |
| `holodeck.lib.errors` | `ConfigError` | Configuration errors | Base for MCPConfigError |
| `holodeck.lib.errors` | `HoloDeckError` | Base exception | Base for MCPError |
| `holodeck.lib.errors` | `ExecutionError` | Runtime execution errors | Pattern for MCP runtime errors |

### Code NOT to Duplicate

- ❌ Do NOT create new regex for `${VAR}` pattern - use `substitute_env_vars()`
- ❌ Do NOT create new .env parser - use `load_env_file()`
- ❌ Do NOT create new error base classes - extend existing `ConfigError`/`ToolError`

### New Code Required

| Module | Function/Class | Purpose |
|--------|----------------|---------|
| `holodeck.tools.mcp.errors` | `MCPConfigError(ConfigError)` | MCP-specific config errors |
| `holodeck.tools.mcp.errors` | `MCPError(HoloDeckError)` | MCP runtime errors base |
| `holodeck.tools.mcp.errors` | `MCPConnectionError(MCPError)` | Server connection failures |
| `holodeck.tools.mcp.errors` | `MCPTimeoutError(MCPConnectionError)` | Timeout errors |
| `holodeck.tools.mcp.errors` | `MCPProtocolError(MCPError)` | Protocol-level errors |
| `holodeck.tools.mcp.factory` | `create_mcp_plugin(config)` | Factory for SK plugins |

**Note**: Do NOT create `UnresolvedEnvVarError` - let `substitute_env_vars()` raise
its existing `ConfigError` and propagate it unchanged.

## Dependencies Analysis

### Required (Already Installed)
- `semantic-kernel>=1.37.1` - MCP plugin implementations
- `pydantic>=2.0.0` - Configuration models
- `python-dotenv>=1.0.0` - Environment file loading

### Transitive (via semantic-kernel)
- `mcp` - MCP protocol SDK
- `httpx` / `httpx-sse` - HTTP/SSE transports
- `websockets` - WebSocket transport

### No Additional Dependencies Required
All MCP functionality available through existing semantic-kernel dependency.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SK MCP API changes | Low | Medium | Pin SK version, test on upgrades |
| MCP server compatibility | Medium | Low | Test with official MCP servers |
| Async complexity | Medium | Medium | Comprehensive async tests |
| Environment variable leaks | Low | High | Fail-fast validation, no logging of values |

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Semantic Kernel MCP Module](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/connectors/mcp.py)
- [VS Code MCP Configuration](https://code.visualstudio.com/docs/copilot/customization/mcp-servers)
- [HoloDeck VISION.md - MCP Section](../../VISION.md)
