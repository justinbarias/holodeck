# CLAUDE.md

Guidance for Claude Code working in this repository.

HoloDeck is an open-source, no-code platform for building, testing, and deploying AI agents via YAML. Stack: Python 3.10+, UV, Pydantic v2, Click, Semantic Kernel, Claude Agent SDK (`claude-agent-sdk==0.2.82`), OpenAI Agents SDK (optional extra), pytest.

## Authoritative References

- **Constitution (project principles):** `.specify/memory/constitution.md`
- **Vision & roadmap:** `VISION.md`
- **Comprehensive agent docs:** `AGENTS.md`
- **Product & user documentation:** `docs/`
- **Feature specs and status:** `specs/<feature>/` — read the whole feature directory before working on one
- **YAML schemas:** `schemas/agent.schema.json`, `schemas/workflow.schema.json`, `schemas/optimize-progress.schema.json`

## Codebase Index

The map below is the fastest way to locate code. Every path is relative to `src/holodeck/` unless noted.

```
src/holodeck/
├── cli/                    Click entry point (holodeck)
│   └── commands/           One module per subcommand:
│                           init, test, chat, serve, deploy, workflow,
│                           optimize, mcp, config, test_view
├── config/                 ConfigLoader, validation, env/YAML merge
├── models/                 Pydantic v2 schema layer (no I/O, no business logic)
│   ├── agent.py            Agent — root model for agent.yaml
│   ├── workflow.py         Workflow — root model for workflow.yaml (036)
│   ├── decision_table.py   DecisionTable + loader (DMN-style tables)
│   ├── llm.py              LLMProvider union
│   ├── claude_config.py    Claude backend options
│   ├── openai_config.py    OpenAI backend options
│   ├── tool.py             ToolUnion — 6 tool types
│   ├── evaluation.py       EvaluationConfig, metrics
│   ├── test_case.py        TestCaseModel
│   ├── observability.py    ObservabilityConfig (OTel)
│   └── deployment.py       Deploy target config
├── lib/                    Business logic
│   ├── backends/           Backend abstraction layer
│   │   ├── base.py         AgentBackend / AgentSession / ExecutionResult protocols
│   │   ├── selector.py     BackendSelector — routes by model.provider
│   │   ├── claude_backend.py         Claude Agent SDK backend
│   │   ├── tool_adapters.py          HoloDeck tools → SDK MCP tools
│   │   ├── mcp_bridge.py             HoloDeck MCP configs → Claude SDK format
│   │   ├── otel_bridge.py            Observability config → subprocess env vars
│   │   ├── validators.py             Pre-flight checks (Node.js, credentials)
│   │   └── openai_agents_*.py        OpenAI Agents SDK backend + adapters
│   ├── workflow/           Deterministic spine (feature 036)
│   │   ├── runner.py       prepare_workflow (all validation, zero LLM) /
│   │   │                   execute_workflow (topological execution)
│   │   ├── edge.py         Edge-node executor + gate schema validation
│   │   ├── table_eval.py   Hit-policy evaluation (UNIQUE/FIRST/PRIORITY) → Verdict
│   │   ├── feel.py         FEEL expression subset (bkflow-feel) + static rejection
│   │   └── input_data.py   Fact-of-record validation against JSON Schema
│   ├── evaluators/         NLP, Azure AI, DeepEval metric implementations
│   ├── test_runner/        Test execution engine
│   ├── eval_run/           Evaluation run orchestration
│   ├── observability/      OTel setup, GenAI semantic conventions, exporters
│   ├── vector_store.py     Vector store integration (qdrant etc.)
│   ├── hybrid_search.py    Hybrid retrieval
│   ├── errors.py           Error hierarchy — always use these
│   └── runtime.py          Shared runtime helpers
├── tools/                  Tool implementations
│   ├── vectorstore_tool.py
│   ├── hierarchical_document_tool.py
│   └── mcp/                MCP tool integration
├── serve/                  HTTP server (AG-UI protocol, sessions, middleware)
├── deploy/                 Docker build (builder.py, dockerfile.py) +
│   └── deployers/          per-target deployers (Azure Container Apps, …)
├── optimizer/              Prompt/agent optimization loop (proposers, scorer)
├── chat/                   Interactive chat session logic
├── dashboard/              Test-results dashboard (views, components)
├── services/               Shared service layer
└── templates/              `holodeck init` project templates
                            (conversational, customer-support, research)

tests/
├── unit/                   Mirrors src/ layout (e.g. tests/unit/workflow/)
├── integration/            Cross-component, may need credentials
├── contract/               Schema/contract tests
└── fixtures/               Committed test fixtures

schemas/                    Published JSON Schemas — keep in sync with models/
                            (sync enforced by tests/unit/test_workflow_schema_sync.py)
docs/                       Docsite content (https://docs.useholodeck.ai/)
sample/                     Local-only sample agents (git-ignored, see .gitignore)
specs/                      Feature specs, plans, task lists per feature
```

Keep this index current: when you add, move, or remove a top-level module or package, update the tree above in the same change.

## Architecture Essentials

1. **Configuration-driven**: all agent behavior defined via YAML, validated by Pydantic models in `models/`. JSON Schemas in `schemas/` are the published contract.
2. **Multi-backend, protocol-driven**: consumers depend only on `AgentBackend` / `AgentSession` / `ExecutionResult`. Never construct a backend directly — go through `BackendSelector`.
   - OpenAI / Azure OpenAI → `OpenAIAgentsBackend`
   - Anthropic / Ollama → `ClaudeBackend`
3. **Plugin tools**: 6 types — vectorstore, function, MCP, prompt, plugin, hierarchical_document. External API integrations must be MCP servers, never custom API tool types.
4. **Claude first-class**: native backend via Claude Agent SDK; prefer Claude-native capabilities (hooks, tools, subagents).
5. **OpenTelemetry native**: observability follows GenAI semantic conventions.
6. **Streaming**: async/await throughout; no sync I/O in async functions.

| Protocol           | Methods                                                           |
| ------------------ | ----------------------------------------------------------------- |
| `AgentBackend`     | `initialize()`, `invoke_once()`, `create_session()`, `teardown()` |
| `AgentSession`     | `send()`, `send_streaming()`, `close()`                           |
| `ContextGenerator` | `contextualize_batch()`                                           |

`ExecutionResult` fields: `response`, `tool_calls`, `tool_results`, `token_usage`, `structured_output`, `num_turns`, `is_error`, `error_reason`.

## Working Norms

- Surgical changes: touch only what the task requires; match existing style; don't refactor or "improve" adjacent code. Remove only imports/variables *your* change made unused.
- No speculative features, abstractions, or configurability beyond what was asked.
- Surface assumptions and tradeoffs; if multiple interpretations exist, say so instead of picking silently.
- Define success criteria up front (usually a test), then verify against them before claiming done.
- After each task run: `make format`, `make lint`, `make type-check`, `make security`.

## Development Commands

```bash
make init                     # venv + deps + pre-commit
source .venv/bin/activate     # always, before any Python command
holodeck --version
```

Dependencies: `uv add <pkg>`, `uv add --dev <pkg>`, `uv remove <pkg>`, `make update-deps`. Env var priority: shell env > `.env` (project) > `~/.holodeck/.env` (user).

### Testing

Always run pytest with `-n auto` (parallel). AAA (Arrange/Act/Assert). Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`.

```bash
make test                     # All tests (parallel)
make test-unit                # Unit only
make test-integration         # Integration only
make test-coverage            # Coverage report
make test-failed              # Re-run failed
pytest tests/unit/workflow/ -n auto -v   # Targeted
```

### Code Quality

```bash
make format                   # Black + Ruff
make lint                     # Ruff + Bandit
make type-check               # MyPy (strict)
make security                 # pip-audit + Bandit + detect-secrets
make ci                       # Full CI locally
```

## Code Standards

- Google Python Style Guide. Black (88 cols), Ruff, MyPy strict, Bandit. Target Python 3.10+.
- Type hints everywhere; PEP 257 docstrings on public functions (Args, Returns, Raises).
- Errors: use the `holodeck.lib.errors` hierarchy (`HoloDeckError`, `ConfigError`, `ValidationError`, `ToolError`, `EvaluationError`). Never catch broad exceptions without re-raising.
- CLI output via Click's `echo()`; `logging` everywhere else. Never `print()`.
- Mutable defaults: use `None` sentinel, never `[]`/`{}`.
- Config through Pydantic models + env vars + YAML only — no hardcoded configuration.

## Code Navigation: LSP vs Grep

**LSP** for semantic questions — references, definitions, call graphs, protocol implementations. **Grep** for textual questions — YAML/markdown/`.env`, regex patterns, strings, config keys.

## End-to-End Deploy Validation Loop

**Run only when the user explicitly asks** (e.g. "run the deploy validation loop", "do the local base + deploy build + deploy run validation"). **Do NOT run automatically** after every change — it builds a docker image, pushes to GHCR, and rolls a live Azure Container Apps revision. Unit tests are the default contract; this loop is reserved for verifying that a fix actually takes effect end-to-end (the `FROM ghcr.io/justinbarias/holodeck-base:latest` chain pins the published wheel by default, so local source changes are invisible until baked into the base).

The default sample for this loop is `sample/financial-assistant/claude` (qdrant cloud + Aspire OTEL + Azure Container Apps). Substitute the path if the user names a different agent.

### Sequence

```bash
# 1. Build local wheel (reflects working-tree source)
rm -rf dist && uv build --wheel

# 2. Build local base image with the wheel baked in.
#    docker/Dockerfile.local exists for this — it installs from dist/*.whl
#    instead of PyPI. --no-cache is required because docker buildx will
#    happily reuse the layer that did the PyPI install.
docker buildx build --platform linux/amd64 --no-cache \
    -f docker/Dockerfile.local \
    -t ghcr.io/justinbarias/holodeck-base:latest --load .

# Verify the base actually carries the local wheel:
docker run --rm --entrypoint python ghcr.io/justinbarias/holodeck-base:latest \
    -c "import holodeck; print(holodeck.__version__)"
# Expect a dev version (e.g. 0.6.35.dev1), NOT the published release.

# 3. Temporarily disable the pull=True in src/holodeck/deploy/builder.py
#    (search for `pull=True,  # Always pull base image`). Otherwise
#    `holodeck deploy build` re-pulls the registry base and clobbers
#    your locally tagged image. Revert this edit before committing.

# 4. Build + push the agent image.
cd sample/financial-assistant/claude
holodeck deploy build
docker push ghcr.io/justinbarias/holodeck-financial-assistant:<tag>
# Tag is `git_sha` by default — first 7 chars of HEAD. Confirm via
# the build output's "Image:" line.

# 5. Deploy to Azure Container Apps.
holodeck deploy run

# 6. Wait until /health is up, then exercise the AG-UI endpoint with
#    a known-good query and confirm a 200 + sensible content. For the
#    financial-assistant sample, a single ConvFinQA turn (e.g. ALXN/2007
#    rental payments) covers ingestion + hybrid search + tool-loop.
URL=https://financial-assistant.nicemoss-50caf9f5.eastus.azurecontainerapps.io
until curl -sf -o /dev/null --max-time 5 "$URL/health"; do sleep 3; done
curl -sS -X POST "$URL/awp" -H 'content-type: application/json' \
    -d '{"threadId":"v","runId":"r","state":{},"messages":[{"id":"m1","role":"user","content":"<your query>"}],"tools":[],"context":[],"forwardedProps":{}}' \
    --max-time 180

# 7. Revert the builder.py edit (re-enable pull=True) before committing.
```

### Caveats

- **`pull=True` clobbering**: `holodeck deploy build` calls Docker SDK with `pull=True` so it gets the right platform — but this overrides your local tag with whatever's on GHCR. Either flip the flag temporarily (step 3) or accept that the local base won't be used.
- **arm64 vs amd64**: Container Apps run amd64. Build the base + agent for `linux/amd64` even on Apple Silicon (use `--platform linux/amd64`). For local docker-network testing, arm64 is fine.
- **Tag re-use**: Image tag stays `<git_sha>` across iterations, but each rebuild produces a new digest. ACA picks up the new digest on `deploy run` (it inspects the image rather than caching by tag).
- **Cold-start latency**: First post-deploy query is 40–70s (image pull + tool init + first ingestion check). Subsequent queries are sub-10s.
- **Validation queries should be deterministic**: prefer questions with a single grounded answer from the corpus (e.g. a specific filing's table value) so a regression is unambiguous.

## Git Commits

- Conventional commits, focused on the change.
- Do NOT attribute Claude Code or include "Generated with Claude Code".
