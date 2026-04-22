# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project Pointers

- **Constitution (authoritative principles):** `.specify/memory/constitution.md`
- **Product & user documentation:** `docs/`
- **Feature specs, plans, tasks, status:** `specs/` (spec-kit artifacts per feature)
- **Agent YAML schema:** `schemas/agent.schema.json`
- **Vision & roadmap:** `VISION.md`
- **Comprehensive agent docs:** `AGENTS.md`

HoloDeck is an open-source, no-code platform for building, testing, and deploying AI agents via YAML. Stack: Python 3.10+, UV, Pydantic v2, Click, Semantic Kernel, Claude Agent SDK, pytest.

## Behavioral Guidelines

Bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions *your* changes made unused. Don't remove pre-existing dead code unless asked.

Test: every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
Define success criteria. Loop until verified.
- "Add validation" → write tests for invalid inputs, then make them pass.
- "Fix the bug" → write a test that reproduces it, then make it pass.
- "Refactor X" → ensure tests pass before and after.

For multi-step tasks, state a brief plan with per-step verification.

## High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Layer (holodeck)                      │
│  init · test · chat · config                                 │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Configuration Management                        │
│  ConfigLoader · Validator · Merge · EnvLoader                │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Pydantic Models (Schema)                     │
│  Agent · LLMProvider · ClaudeConfig · ToolUnion ·            │
│  EvaluationConfig · TestCaseModel                            │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           Backend Abstraction Layer (auto-routed)            │
│  BackendSelector · AgentBackend · AgentSession ·             │
│  ExecutionResult · ContextGenerator                          │
├─────────────────────────────┬───────────────────────────────┤
│   SK Backend                │   Claude Backend              │
│   (OpenAI, Azure, Ollama)   │   (Anthropic — first-class)   │
└─────────────────────────────┴───────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Evaluation Framework                            │
│  NLP · Azure AI · DeepEval · Test Runner                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Architectural Decisions

1. **Configuration-Driven**: All agent behavior defined via YAML with Pydantic validation. See `schemas/agent.schema.json`.
2. **Multi-Backend**: Protocol-driven, provider-agnostic. Consumers use `AgentBackend`/`AgentSession`/`ExecutionResult` only. `BackendSelector` auto-routes by `model.provider`.
3. **Plugin Tools**: 6 types — vectorstore, function, MCP, prompt, plugin, hierarchical_document. Claude adapter bridge wraps HoloDeck tools as SDK-compatible MCP tools.
4. **MCP for APIs**: External integrations must use MCP servers, not custom API tool types.
5. **Claude First-Class**: Native backend via Claude Agent SDK; prioritize Claude-native capabilities (hooks, tools, subagents).
6. **OpenTelemetry Native**: Observability follows GenAI semantic conventions.
7. **Evaluation Flexibility**: Model configuration at global, run, and metric levels.
8. **Multimodal Testing**: First-class images, PDFs, Office docs.
9. **Streaming**: async/await throughout.

### Backend Routing

- OpenAI / Azure OpenAI / Ollama → `SKBackend`
- Anthropic → `ClaudeBackend`

| Protocol           | Methods                                                           |
| ------------------ | ----------------------------------------------------------------- |
| `AgentBackend`     | `initialize()`, `invoke_once()`, `create_session()`, `teardown()` |
| `AgentSession`     | `send()`, `send_streaming()`, `close()`                           |
| `ContextGenerator` | `contextualize_batch()`                                           |

`ExecutionResult` fields: `response`, `tool_calls`, `tool_results`, `token_usage`, `structured_output`, `num_turns`, `is_error`, `error_reason`.

## Development Setup

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: brew install uv

make init                     # venv + deps + pre-commit
source .venv/bin/activate
holodeck --version
```

### Dependency Management

```bash
make install-dev              # All deps including dev
make install-prod             # Production only
uv add <package>              # Add dependency
uv add --dev <package>        # Add dev dependency
uv remove <package>           # Remove dependency
make update-deps              # Update all
```

### Environment Variables

Priority: shell env > `.env` (project) > `~/.holodeck/.env` (user).

### Testing

Always run pytest with `-n auto` (parallel).

```bash
make test                     # All tests (parallel)
make test-unit                # Unit only
make test-integration         # Integration only
make test-coverage            # With coverage report
make test-failed              # Re-run failed

pytest tests/unit/ -n auto -v
```

Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`. Use AAA (Arrange/Act/Assert).

### Code Quality

```bash
make format                   # Black + Ruff
make format-check             # CI-safe check
make lint                     # Ruff + Bandit
make lint-fix                 # Auto-fix
make type-check               # MyPy (strict)
make security                 # pip-audit + Bandit + detect-secrets
make ci                       # Full CI locally
```

Pre-commit: `make install-hooks`, `make pre-commit`.

Always `source .venv/bin/activate` before Python commands.

## Code Standards

- **Style:** Google Python Style Guide. Black (88 cols), Ruff, MyPy strict, Bandit. Target Python 3.10+.
- **Type hints:** everywhere, for parameters and return values.
- **Docstrings (PEP 257):** required on public functions — Args, Returns, Raises.
- **Errors:** use `holodeck.lib.errors` hierarchy (`HoloDeckError`, `ConfigError`, `ValidationError`, `ToolError`, `EvaluationError`). Never catch broad exceptions without re-raising.
- **Async:** async/await for all I/O. No sync I/O in async functions.
- **Config:** Pydantic models only. No hardcoded configuration — use env vars + YAML.
- **CLI output:** Click's `echo()` in CLI, `logging` elsewhere. Never `print()`.
- **Mutable defaults:** use `None` sentinel, never `[]`/`{}` as defaults.
- **Backend usage:** always via `BackendSelector`; use protocol types in signatures.
- **External APIs:** MCP servers.

## Code Navigation: LSP vs Grep

**LSP** for semantic questions — references, types, definitions, symbol hierarchy, call graphs, implementations of protocols.

**Grep** for textual questions — non-Python files (YAML, markdown, `.env`), regex/partial patterns, strings, TODOs, config keys, unparseable files.

## Claude-Specific Infrastructure

| Module             | Purpose                                                      |
| ------------------ | ------------------------------------------------------------ |
| `tool_adapters.py` | Wraps VectorStore/HierarchicalDoc tools as SDK MCP tools     |
| `mcp_bridge.py`    | Translates HoloDeck MCP configs to Claude SDK format         |
| `otel_bridge.py`   | Translates observability config to subprocess env vars       |
| `validators.py`    | Pre-flight checks (Node.js, credentials, embedding provider) |

## Workflow (spec-kit)

1. `/speckit.specify` — create spec
2. `/speckit.clarify` — resolve ambiguity
3. `/speckit.plan` — design artifacts
4. `/speckit.tasks` — dependency-ordered tasks
5. Read all files in `specs/<feature>/` before planning; always surface the plan path for review.
6. Implement with Claude tasks.
7. Run `make format`, `make lint`, `make type-check`, `make security` after each task.

## Git Commits

- Do NOT attribute Claude Code or include "Generated with Claude Code".
- Conventional, focused on the change.
