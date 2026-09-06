# Plan Analysis: 021-claude-agent-sdk

**Date**: 2026-02-21
**Status**: Findings recorded — decisions pending

Cross-reference of `plan.md` against the actual codebase. Findings are ordered by severity.

---

## 🔴 Blockers — Must fix before implementation

### 1. `AgentExecutionResult` has no `response` field

**Problem**: `AgentThreadRun.invoke()` returns `AgentExecutionResult(tool_calls, tool_results, chat_history, token_usage)`. There is no `response: str`. The response text is extracted by consumers *after* the call:

```python
# executor.py:551
agent_response = extract_last_assistant_content(result.chat_history)

# chat/executor.py:124
content = extract_last_assistant_content(result.chat_history)
```

The plan's `ExecutionResult` correctly has `response: str`, but never says who populates it when wrapping the SK backend. Without explicitly moving the extraction step inside `SKBackend.invoke_once()`, every SK-backend invocation returns `ExecutionResult(response="", ...)` — a silent failure.

**Fix**: Phase 4 must explicitly state that `SKBackend.invoke_once()` calls `extract_last_assistant_content(result.chat_history)` internally before building `ExecutionResult`.

**Decision required**: No — plan correction only.

---

### 2. `models/chat.py`: `ChatSession.history: ChatHistory` not in plan scope

**Problem**: `models/chat.py` line 74:

```python
class ChatSession(BaseModel):
    history: ChatHistory   # ← Semantic Kernel type, in a data model
```

`ChatSessionManager.start()` creates it:
```python
history = ChatHistory()
self.session = ChatSession(agent_config=..., history=history, ...)
```

The plan lists `chat/session.py` and `chat/executor.py` as files to update, but never mentions `models/chat.py`. For Claude agents there is no `ChatHistory` object. The model cannot represent a Claude chat session as-is.

**Options**:

- **Option A** (minimal): Make `history` nullable — `history: ChatHistory | None = None`. Claude sessions don't populate it. Simple, can be revisited.
- **Option B** (clean): Replace with `history: list[dict[str, Any]] = []`. SK backend serialises `ChatHistory` to dicts internally. Cleaner long-term, more work, touches anything reading `session.history`.

**Decision required**: Choose Option A or B.

---

### 3. `chat_history_utils.py` — SK utility not in plan scope

**Problem**: `lib/chat_history_utils.py` imports `ChatHistory` at the module level:

```python
from semantic_kernel.contents import ChatHistory

def extract_last_assistant_content(history: ChatHistory) -> str: ...
def extract_tool_names(tool_calls: list[dict]) -> list[str]: ...
```

Both `executor.py` (test runner) and `chat/executor.py` import from it. The plan never mentions this file. After the refactor, importing this module anywhere in the codebase pulls in SK, contradicting the decoupling goal.

**Fix**: `extract_last_assistant_content` moves inside `sk_backend.py` as a private function. `extract_tool_names` is SK-free and can stay or be inlined. Add `lib/chat_history_utils.py` to Phase 4 scope.

**Decision required**: No — plan correction only.

---

### 4. `TestExecutor.__init__` is sync — `BackendSelector.create()` is async

**Problem**: `TestExecutor.__init__` synchronously constructs `AgentFactory`:

```python
self.agent_factory = agent_factory or self._create_agent_factory()
# _create_agent_factory() → AgentFactory(...) → __init__ runs kernel creation synchronously
```

The plan's `BackendSelector.create()` is `async def` with `await backend.initialize()`. Python `__init__` cannot `await`. The plan's eager-init lifecycle diagram cannot be wired into `TestExecutor.__init__`.

**Fix**: The Claude backend must adopt the same lazy-init pattern used by the existing `AgentFactory._ensure_tools_initialized()`. `ClaudeBackend.__init__` stores config only. `initialize()` is called internally on first `invoke_once()` or `create_session()`. The plan's lifecycle diagram needs updating.

**Decision required**: No — lazy init is clearly correct.

---

### 5. `VectorStoreTool` uses `model.provider` for embedding dimensions — must use `embedding_provider`

**Problem**: `agent_factory.py:847`:

```python
provider_type = self.agent_config.model.provider.value  # "anthropic" for Claude agents
await tool.initialize(force_ingest=self._force_ingest, provider_type=provider_type)
```

`provider_type` controls embedding vector dimensionality (e.g., OpenAI → 1536 dims). For Claude-native agents, `model.provider` is `"anthropic"`, but embeddings come from `embedding_provider` (OpenAI or Azure). Passing `provider_type="anthropic"` will fail or produce wrong dimensions.

**Fix**: The Claude backend must pass `agent.embedding_provider.provider.value` as `provider_type`. Explicitly call this out in Phase 5.

**Decision required**: No — plan correction only.

---

### 6. Closure bug in tool adapter loop

**Problem**: The `quickstart.md` and the implied `tool_adapters.py` implementation use a `@tool` decorator inside a `for` loop:

```python
for vs_tool in vectorstore_tools:
    @tool(tool_name, ...)
    async def search_fn(args: dict) -> dict:
        result = await vs_tool.search(args["query"])  # captures 'vs_tool' by reference
```

Python closures capture variables by reference. After the loop, all `search_fn` functions reference the last `vs_tool`. If there are three tools — `docs`, `faq`, `kb` — all three search functions silently call `kb.search()`. No exception, wrong results.

**Fix**: Use a factory function:

```python
def make_search_fn(t: VectorStoreTool, name: str) -> SdkMcpTool:
    @tool(name, ...)
    async def search_fn(args: dict) -> dict:
        return {"content": [{"type": "text", "text": await t.search(args["query"])}]}
    return search_fn
```

Must be corrected in both `quickstart.md` and the actual `tool_adapters.py` implementation.

**Decision required**: No — clear bug fix.

---

## 🟡 Inconsistencies — Plan vs Actual Code

### 7. `TokenUsage` vs `dict` — field names and types are incompatible

**Problem**: The codebase has a `TokenUsage` Pydantic model with fields `prompt_tokens`, `completion_tokens`, `total_tokens` (with cross-field validation). The plan proposes `ExecutionResult.token_usage: dict[str, int]` with Anthropic SDK field names: `input_tokens`, `output_tokens`, `total_tokens`. These are incompatible in both type and field names. `ChatSessionManager` accumulates token usage using `TokenUsage` arithmetic.

**Options**:

- **Option A** (recommended): Keep `TokenUsage` in `ExecutionResult`. Claude backend translates `input_tokens → prompt_tokens`, `output_tokens → completion_tokens` on the way out. No changes to `ChatSessionManager`. Three lines of translation in `claude_backend.py`.
- **Option B**: Replace everything with `dict[str, int]`. Update `ChatSessionManager`, all token accumulation code, and the test runner. More work, loses Pydantic validation benefit.

**Decision required**: Choose Option A or B.

---

### 8. OTel bridge silently drops fields users have configured

**Problem**: The existing `ObservabilityConfig` supports rich exporters that have no Claude Code env var equivalent:

| Config Field | Claude Code Env Var | Status |
|---|---|---|
| `exporters.otlp.endpoint` | `OTEL_EXPORTER_OTLP_ENDPOINT` | ✅ Mapped |
| `exporters.otlp.protocol` | `OTEL_EXPORTER_OTLP_PROTOCOL` | ✅ Mapped |
| `metrics.export_interval_ms` | `OTEL_METRIC_EXPORT_INTERVAL` | ✅ Mapped |
| `traces.capture_content` | `OTEL_LOG_USER_PROMPTS` | ✅ Mapped (approximate) |
| `exporters.azure_monitor` | — | ❌ No mapping |
| `exporters.prometheus` | — | ❌ No mapping |
| `traces.redaction_patterns` | — | ❌ No mapping |
| `traces.sample_rate` | — | ❌ No mapping |
| `logs.filter_namespaces` | — | ❌ No mapping |

An enterprise user who configured `azure_monitor` silently loses their observability setup with no feedback when switching to `provider: anthropic`.

**Options**:

- **Option A** (recommended): Emit named warnings at startup for each unsupported field. "The following observability settings are not supported by the Claude-native backend: `azure_monitor`, `prometheus`..." User can adapt.
- **Option B**: Error at startup if unsupported exporters are configured alongside `provider: anthropic`. Forces explicit acknowledgement.
- **Option C**: Silently ignore. Acceptable only with explicit changelog entry. Argued against.

**Decision required**: Choose warning policy (A or B).

---

### 9. `AgentFactory` constructor contract is self-contradictory in plan

**Problem**: The plan says `agent_factory.py` becomes "a thin backwards-compatible facade." But `AgentFactory.__init__` currently creates the SK kernel synchronously. If it delegates to `BackendSelector`, the constructor either still creates an SK kernel (wasteful for Claude agents) or does nothing (changes observable behaviour, breaking callers who expect a ready kernel).

**Fix**: Preserve the synchronous constructor contract. For SK agents, `__init__` still creates the kernel immediately. For Claude agents, `__init__` stores config and does nothing else — async init remains lazy. The plan needs precise language rather than "thin facade."

**Decision required**: No — plan wording correction.

---

### 10. `get_history() -> ChatHistory` on `AgentExecutor` not in plan scope

**Problem**: `chat/executor.py` exposes:

```python
def get_history(self) -> ChatHistory:
    if self._thread_run is not None:
        return self._thread_run.chat_history
    return ChatHistory()
```

This public method returns an SK type. If called for a Claude-backend session, it returns an empty `ChatHistory()` (misleading) or crashes. Not listed in the plan's files to update.

**Fix**: Add to Phase 10 scope. Method either becomes provider-specific with nullable return, or is replaced with a generic `get_message_count() -> int` that both backends can satisfy.

**Decision required**: No — plan scope correction.

---

## 🔵 Challenged Assumptions

### 11. `claude-agent-sdk` API not verified from an actual install

**Problem**: The entire backend is planned around class names, method signatures, and dataclass fields researched via web document fetching, not from installing the package. The package is Alpha, released 2026-02-19 — the same day as the spec.

Specific API details that may be wrong:
- Is the class `ClaudeAgentOptions` or something else?
- Is `PermissionMode` a string literal or an enum? Is `"bypassPermissions"` the correct casing?
- Does `@tool` work as described, or is registration different?
- Does `create_sdk_mcp_server()` exist by that name?
- Is `ResultMessage.structured_output` a field?
- Does `ClaudeSDKClient` exist, or is the stateful API different?

If any of these are wrong, Phase 8 fails on the first import. This is the highest single risk in the plan.

**Recommended action**: Before any implementation starts, install the package and write a 20-line smoke test against the actual API. This takes an hour and eliminates the largest source of risk.

**Decision required**: Should SDK verification be a mandatory Phase 0 of implementation?

---

### 12. `bypassPermissions` for test runs may allow destructive operations

**Problem**: The plan auto-overrides to `bypassPermissions` during `holodeck test`. If a user has configured `bash.enabled: true` and `file_system.write: true` for their agent, test runs will execute shell commands and write files in the working directory without any approval. A test input like "delete the old build artifacts" triggers real deletion in an automated, unattended context.

**Options**:

- **Option A**: Disable bash and file system in test mode regardless of agent config. Tests are pure conversation tests. Safest, but may block legitimate scenarios that verify file output.
- **Option B**: Keep `bypassPermissions` but require explicit opt-in via `--allow-side-effects` CLI flag. Makes risk visible.
- **Option C**: Keep current plan — `bypassPermissions` automatically. Accept the risk, document it.

**Decision required**: Choose safety policy for test runs.

---

### 13. Multi-turn state for `ClaudeSDKClient` underspecified

**Problem**: The plan says `ClaudeSDKClient` handles multi-turn state. But `ClaudeAgentOptions` has `continue_conversation: bool = False`. If this defaults to `False`, each `query()` call starts a fresh conversation — multi-turn chat is broken from the first test.

**Why it matters**: `holodeck chat` is explicitly multi-turn. If the session statefulness configuration is wrong, users ask follow-up questions and the agent has no context of previous messages.

**Fix**: Verify against the actual SDK (contingent on #11). The correct configuration is likely either `continue_conversation=True` in `ClaudeAgentOptions` or maintaining `session_id` across calls. Phase 8 `ClaudeSession` implementation must explicitly handle this.

**Decision required**: Contingent on #11 SDK verification.

---

### 14. SK refactor (1,359 lines) is presented as low-risk

**Problem**: Phase 4 moves essentially all of `agent_factory.py` into `sk_backend.py`. This code manages Kernel lifecycle, MCP plugin async context managers (with explicit ordering constraints), vectorstore tool initialization, hierarchical document tools, tool filtering, retry logic, and response format wrapping. The comment in `AgentFactory.shutdown()` explicitly notes: "Must be called from the same task context where the factory was used." This asyncio constraint doesn't move automatically.

**Mitigation**: Add an explicit integration test gate between Phase 4 and Phase 5. Run a full `holodeck test` workflow against a real non-Anthropic agent (with at least one MCP tool and one vectorstore tool) before proceeding. This catches lifecycle regressions before they get buried under Claude backend code.

**Decision required**: No — add integration gate to plan.

---

### 15. Phase parallelism is overstated

**Problem**: The plan says Phases 4, 5, 6, 7 can run in parallel after Phase 3. But:

- Phase 4 changes `agent_factory.py`, which Phase 9 (test runner) also modifies. Running them in parallel causes merge conflicts on the same file.
- If `ExecutionResult`'s shape changes during Phase 4 (e.g., TokenUsage vs dict — finding #7), Phases 5 and 6 built against it need rework.

**Corrected critical path**:

```
1 → 2 → 3 → 4 → [5 + 6 + 7 in parallel] → 8 → 9 → 10 → 11
```

Phase 4 is not parallelisable with 5 and 6. Phase 4 must be stable and tested before Phases 5 and 6 begin.

**Decision required**: No — plan correction only.

---

## Decisions Required Before Plan Update

| # | Finding | Decision |
|---|---|---|
| 2 | `ChatSession.history` type | Option A (nullable) or Option B (list[dict]) |
| 7 | TokenUsage vs dict in ExecutionResult | Option A (keep TokenUsage) or Option B (use dict) |
| 8 | OTel bridge unmapped fields | Warning (A) or Error (B) at startup |
| 11 | SDK API verification | Make Phase 0 mandatory? |
| 12 | Test run safety with bash/file access | Disable (A), opt-in flag (B), or accept risk (C) |
| 13 | Multi-turn ClaudeSDKClient config | Contingent on #11 |

All other findings (#1, 3, 4, 5, 6, 9, 10, 14, 15) are plan corrections that do not require your input.
