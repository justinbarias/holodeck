# Research: Interactive Init Wizard

**Feature Branch**: `011-interactive-init-wizard`
**Date**: 2025-11-29

## Research Tasks

### 1. MCP Registry API Integration

**Context**: Need to fetch list of available MCP servers from official registry for user selection.

**Decision**: Use official MCP Registry API at `https://registry.modelcontextprotocol.io/v0/servers`

**Rationale**:
- Official registry endorsed by Model Context Protocol maintainers
- RESTful API with JSON responses, easy to integrate
- No authentication required for read operations
- Supports search and pagination

**API Specification**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v0/servers` | GET | List all registered MCP servers |
| `/v0/servers?search={query}` | GET | Search servers by keyword |
| `/v0/servers?limit={n}` | GET | Limit results (max 100) |

**Response Structure**:
```json
{
  "servers": [
    {
      "server": {
        "name": "io.github.user/server-name",
        "description": "Server description",
        "version": "1.0.0",
        "packages": [
          {
            "registryType": "npm",
            "identifier": "@scope/package-name",
            "transport": {"type": "stdio"}
          }
        ]
      },
      "_meta": {
        "io.modelcontextprotocol.registry/official": {
          "status": "active",
          "publishedAt": "2025-09-16T16:43:44.243Z"
        }
      }
    }
  ],
  "metadata": {
    "count": 1,
    "nextCursor": "cursor_string"
  }
}
```

**Default MCP Servers** (from spec clarifications):
1. `@modelcontextprotocol/server-filesystem` - File system access
2. `@modelcontextprotocol/server-memory` - Key-value memory storage
3. `@modelcontextprotocol/server-sequential-thinking` - Structured reasoning

**Error Handling**: Per spec FR-014, if registry is unreachable, display clear error and exit.

**Alternatives Considered**:
- Hardcoded server list: Rejected because it would become stale and wouldn't reflect new servers
- GitHub API for server repos: Rejected because it's not the authoritative source

**Sources**:
- [MCP Registry GitHub](https://github.com/modelcontextprotocol/registry)
- [Nordic APIs - MCP Registry API Guide](https://nordicapis.com/getting-started-with-the-official-mcp-registry-api/)

---

### 2. Interactive CLI Library Selection

**Context**: Click's native `prompt` with `multiple=True` doesn't work well. Need library for multi-select prompts.

**Decision**: Use **InquirerPy** for interactive prompts

**Rationale**:
1. Built on prompt_toolkit (cross-platform, including Windows)
2. Native `checkbox` prompt for multi-select with pre-selection support
3. Native `select` prompt for single-selection
4. Active maintenance and good documentation
5. No Click conflicts - works alongside Click decorators

**Installation**: `inquirerpy` (already follows project dependency patterns)

**Key Features Required**:

| Feature | InquirerPy Support | Usage |
|---------|-------------------|-------|
| Single-select | `inquirer.select()` | LLM provider, Vector store |
| Multi-select | `inquirer.checkbox()` | MCP servers |
| Pre-selection | `Choice(value, enabled=True)` | Default MCP servers |
| Descriptions | `Choice(value, name="Display")` | Show server descriptions |
| Validation | `validate=lambda r: len(r) >= 1` | Ensure selections made |

**Example Integration**:
```python
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

# Single-select for LLM provider
provider = inquirer.select(
    message="Select LLM provider:",
    choices=[
        Choice("ollama", name="Ollama (default, local)"),
        Choice("openai", name="OpenAI"),
        Choice("azure_openai", name="Azure OpenAI"),
        Choice("anthropic", name="Anthropic Claude"),
    ],
    default="ollama",
).execute()

# Multi-select for MCP servers
servers = inquirer.checkbox(
    message="Select MCP servers (space to toggle, enter to confirm):",
    choices=[
        Choice("filesystem", name="Filesystem", enabled=True),
        Choice("memory", name="Memory", enabled=True),
        Choice("sequential-thinking", name="Sequential Thinking", enabled=True),
        Choice("github", name="GitHub"),
    ],
).execute()
```

**Alternatives Considered**:
- **click-prompt**: Less active, not as well documented
- **questionary**: Good alternative, but InquirerPy has more customization
- **python-inquirer**: Windows support is experimental
- **Raw Click prompts**: Doesn't support multi-select checkbox UI

**Sources**:
- [InquirerPy PyPI](https://pypi.org/project/inquirerpy/)
- [InquirerPy Checkbox Docs](https://inquirerpy.readthedocs.io/en/latest/pages/prompts/checkbox.html)
- [InquirerPy Select Docs](https://inquirerpy.readthedocs.io/en/latest/pages/prompts/list.html)
- [Click Prompts Documentation](https://click.palletsprojects.com/en/stable/prompts/)

---

### 3. Terminal Interactivity Detection

**Context**: Need to detect when terminal doesn't support interactive prompts (per FR-013).

**Decision**: Use `sys.stdin.isatty()` for detection, fall back to defaults

**Rationale**:
- Standard Python approach, no additional dependencies
- Click also uses this pattern internally
- Works across platforms

**Implementation Pattern**:
```python
import sys

def is_interactive() -> bool:
    """Check if terminal supports interactive prompts."""
    return sys.stdin.isatty() and sys.stdout.isatty()
```

**Fallback Behavior**:
- When non-interactive: Use all defaults without prompting
- Log info message explaining defaults were used
- Same behavior as `--non-interactive` flag

**Alternatives Considered**:
- Environment variable check only: Insufficient, doesn't detect piped input
- prompt_toolkit's detection: Overkill for this use case

---

### 4. Non-Interactive Mode CLI Design

**Context**: Need CLI flags for scripted/CI usage (FR-007).

**Decision**: Add flags `--llm`, `--vectorstore`, `--mcp`, and `--non-interactive`

**Rationale**:
- Follows existing CLI patterns in holodeck
- Clear, explicit flag names matching wizard prompts
- `--non-interactive` provides explicit opt-out from prompts

**Flag Specification**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--llm` | Choice | ollama | LLM provider selection |
| `--vectorstore` | Choice | chromadb | Vector store selection |
| `--mcp` | String (comma-sep) | filesystem,memory,sequential-thinking | MCP servers |
| `--non-interactive` | Flag | False | Skip all prompts, use defaults/flags |

**Validation**:
- Invalid `--llm` or `--vectorstore` values: Error with valid options listed
- Invalid `--mcp` server names: Warning + skip invalid, continue with valid

**Alternatives Considered**:
- JSON config file input: Over-engineering for this use case
- Environment variables only: Less discoverable than CLI flags

---

### 5. Clean Cancellation (Ctrl+C) Handling

**Context**: Per FR-010, no partial files should remain if user cancels mid-wizard.

**Decision**: Wrap wizard in try/except with cleanup, use existing `ProjectInitializer` cleanup pattern

**Rationale**:
- Existing `ProjectInitializer` already handles cleanup on failure
- InquirerPy raises `KeyboardInterrupt` on Ctrl+C
- Defer file creation until all prompts complete

**Implementation Pattern**:
```python
try:
    # Collect all wizard inputs first (no file I/O)
    wizard_result = run_wizard()

    # Only create files after all inputs collected
    initializer = ProjectInitializer()
    initializer.initialize(wizard_result)

except KeyboardInterrupt:
    click.echo("\nWizard cancelled.")
    # No cleanup needed - files weren't created yet
    raise click.Abort()
```

**Key Insight**: Separate prompt collection phase from file creation phase.

**Alternatives Considered**:
- Transactional file system: Overkill, complex
- Temp directory + atomic move: Unnecessary complexity

---

### 6. Existing Codebase Integration Points

**Context**: Understanding where wizard logic should integrate.

**Analysis**:

**Current `holodeck init` Flow**:
1. `init.py` parses CLI args → `ProjectInitInput`
2. `ProjectInitializer.initialize()` validates and creates files
3. Uses `TemplateRenderer` for Jinja2 template processing

**Integration Points**:
1. **`src/holodeck/cli/commands/init.py`**: Add wizard invocation before `ProjectInitializer`
2. **`src/holodeck/cli/utils/wizard.py`** (new): Wizard logic and prompt definitions
3. **`src/holodeck/lib/mcp_registry.py`** (new): MCP registry API client
4. **`src/holodeck/models/wizard_config.py`** (new): Wizard state and result models

**Existing Models to Extend**:
- `LLMProvider` / `ProviderEnum` in `models/llm.py` - already has all 4 providers
- `DatabaseConfig` in `models/tool.py` - has chromadb, redis, in-memory
- `MCPTool` in `models/tool.py` - MCP server configuration

**Template Updates Required**:
- Update `agent.yaml.j2` templates to accept wizard selections
- Add placeholder variables for LLM config, vectorstore, MCP tools

---

## Summary of Decisions

| Area | Decision | Key Dependency |
|------|----------|----------------|
| MCP Server List | Official Registry API | `requests` (existing) |
| Interactive Prompts | InquirerPy | New dependency |
| TTY Detection | `sys.stdin.isatty()` | Standard library |
| CLI Flags | `--llm`, `--vectorstore`, `--mcp`, `--non-interactive` | Click (existing) |
| Cancellation | Prompt-first, then file creation | Existing cleanup patterns |

## New Dependencies

```toml
# Add to pyproject.toml dependencies
"inquirerpy>=0.3.4,<0.4.0"  # Interactive CLI prompts
```
