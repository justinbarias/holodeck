# Plan: Decouple Context Generator — Full ABC Refactor with Claude SDK Backend

## Context

`LLMContextGenerator` generates short context snippets for document chunks (Anthropic's contextual retrieval approach, improving retrieval by 35-49%). It's hardwired to Semantic Kernel's `ChatCompletionClientBase`. The Claude backend calls `initialize_tools()` **without** a `chat_service`, so contextual embeddings silently degrade to no-op for all Anthropic agents.

**The key motivation:** Claude backend users authenticate via `CLAUDE_CODE_OAUTH_TOKEN` — no API keys in YAML. We should leverage this for context generation by using the Claude Agent SDK directly with Haiku (cheap/fast), rather than requiring users to configure separate API keys for a `context_model`.

**Design:** Introduce a `ContextGenerator` protocol with two implementations:
- `LLMContextGenerator` (existing, for SK backend)
- `ClaudeSDKContextGenerator` (new, uses `query()` with batched chunks + concurrent sessions)

---

## Step 1 — Add `ContextGenerator` protocol

**File:** `src/holodeck/lib/backends/base.py`

Add after `AgentBackend` protocol (~line 133):

```python
@runtime_checkable
class ContextGenerator(Protocol):
    """Backend-agnostic contextual embedding generation."""

    async def contextualize_batch(
        self,
        chunks: list["DocumentChunk"],
        document_text: str,
        concurrency: int | None = None,
    ) -> list[str]: ...
```

Guard `DocumentChunk` import under `TYPE_CHECKING`. Export from `src/holodeck/lib/backends/__init__.py`.

---

## Step 2 — Verify `LLMContextGenerator` conforms (no structural changes)

**File:** `src/holodeck/lib/llm_context_generator.py`

The existing `contextualize_batch()` signature already matches the protocol exactly. Add a `TYPE_CHECKING` conformance assertion and update the class docstring. No method changes needed.

---

## Step 3 — Create `ClaudeSDKContextGenerator`

**New file:** `src/holodeck/lib/claude_context_generator.py`

Uses `query()` (stateless one-shot, one subprocess per call). For 100 chunks with batch_size=10, that's 10 subprocess spawns — manageable at ingestion time.

### Batch strategy
- Group chunks into batches (default 10 per prompt)
- Each batch prompt includes the document + N numbered chunks
- Ask for JSON array of N context strings
- If JSON parsing fails, fall back to individual per-chunk calls

### Concurrency
- `asyncio.Semaphore(concurrency)` limits concurrent `query()` calls (default 5)
- Each batch runs as an async task

### Key class outline

```python
@dataclass
class ClaudeContextConfig:
    model: str = "claude-haiku-4-5-20251001"
    batch_size: int = 10
    concurrency: int = 5
    max_retries: int = 3
    base_delay: float = 1.0
    max_document_tokens: int = 8000

class ClaudeSDKContextGenerator:
    def __init__(self, config=None, max_context_tokens=100)

    # Batch prompt: document + N chunks → JSON array of N context strings
    def _build_batch_prompt(chunks, document_text) -> str
    def _build_single_prompt(chunk_text, document_text) -> str  # fallback
    def _parse_batch_response(response, expected_count) -> list[str] | None

    # SDK interaction
    async def _query_claude(prompt) -> str  # wraps query() with minimal options

    # Processing
    async def _process_batch(chunks, document_text) -> list[str]  # try batch, fallback individual
    async def contextualize_batch(chunks, document_text, concurrency=None) -> list[str]
```

`_query_claude` uses:
```python
ClaudeAgentOptions(
    model=self._config.model,
    system_prompt="You are a context generation assistant...",
    permission_mode="bypassPermissions",
    max_turns=1,
)
```

Auth: inherited `CLAUDE_CODE_OAUTH_TOKEN` from environment — zero config.

---

## Step 4 — Update `HierarchicalDocumentTool`

**File:** `src/holodeck/tools/hierarchical_document_tool.py`

### 4a. Widen type of `_context_generator` (line 124)

```python
# Before
self._context_generator: LLMContextGenerator | None = None
# After
self._context_generator: ContextGenerator | None = None
```

Add `ContextGenerator` to `TYPE_CHECKING` imports.

### 4b. Add `set_context_generator()` method

```python
def set_context_generator(self, generator: Any) -> None:
    """Set the context generator for contextual embeddings.

    Accepts any ContextGenerator protocol implementation.
    """
    self._context_generator = generator
```

### 4c. Keep `set_chat_service()` as backward-compat shim

No deprecation warning (all callers are internal). Internally wraps the SK service in `LLMContextGenerator` and stores it as `_context_generator`, same as today.

### 4d. Simplify guard in `_ingest_documents()` (line 533-536)

Remove the `and self._chat_service is not None` check:

```python
# Before
if (self.config.contextual_embeddings
    and self._context_generator is not None
    and self._chat_service is not None):

# After
if (self.config.contextual_embeddings
    and self._context_generator is not None):
```

Also remove the `"no chat service available"` reason branch (line 554-555) since `_context_generator` is now the single source of truth.

---

## Step 5 — Update `tool_initializer.py`

**File:** `src/holodeck/lib/tool_initializer.py`

### 5a. Add `context_generator` parameter to `initialize_tools()` and `_initialize_hierarchical_doc_tools()`

### 5b. New priority chain in `_initialize_hierarchical_doc_tools()`

```
1. Explicit context_generator arg    → tool.set_context_generator(it)
2. chat_service arg (SK path)        → tool.set_context_generator(LLMContextGenerator(chat_service))
3. tool_config.context_model set     → create SK chat service → LLMContextGenerator
4. agent.model.provider == anthropic → auto-create ClaudeSDKContextGenerator
5. None of the above                 → no context generation (graceful degradation)
```

The LLMContextGenerator wrapping (cases 2-3) is constructed with `tool_config.context_max_tokens` and `tool_config.context_concurrency`.

Case 4 auto-creates `ClaudeSDKContextGenerator` with `ClaudeContextConfig(concurrency=tool_config.context_concurrency)`.

### 5c. Clean up existing `_create_chat_service_from_config()` and `_resolve_context_model_config()`

These stay for case 3 (explicit `context_model` with API key). The recently added code from the previous PR is preserved.

---

## Step 6 — Update `agent_factory.py` to delegate to `tool_initializer`

**File:** `src/holodeck/lib/test_runner/agent_factory.py`

### Problem

The current `_register_hierarchical_document_tools()` (line 950-1013) duplicates context generator wiring: it calls `tool.set_chat_service(self._llm_service)` directly, bypassing the priority chain and `context_model` resolution in `tool_initializer.py`. This means the same YAML config produces different behavior depending on which backend runs it:

| Feature | `tool_initializer.py` (Claude backend) | `agent_factory.py` (SK backend) |
|---------|----------------------------------------|----------------------------------|
| `context_model` override | Resolved via `_resolve_context_model_config()` | **Ignored** |
| Anthropic auto-detection | Auto-creates `ClaudeSDKContextGenerator` | **No** — always uses main agent model |
| Explicit `context_generator` | Supported (Step 5) | **No** |

### Solution: Delegate to `tool_initializer._initialize_hierarchical_doc_tools()`

Replace the inline tool initialization loop with a call to the shared function. This gives AgentFactory the same 5-tier priority chain as the Claude backend path.

### 6a. Replace `_register_hierarchical_document_tools()` body

```python
async def _register_hierarchical_document_tools(self) -> None:
    """Register hierarchical document tools from agent config.

    Delegates tool initialization (embedding, context generation, ingestion)
    to the shared tool_initializer, then registers search methods as
    KernelFunctions.
    """
    if not self.agent_config.tools:
        return

    if not self._has_hierarchical_document_tools():
        return

    from holodeck.lib.tool_initializer import _initialize_hierarchical_doc_tools

    provider_type = self.agent_config.model.provider.value

    instances = await _initialize_hierarchical_doc_tools(
        agent=self.agent_config,
        embedding_service=self._embedding_service,
        chat_service=self._llm_service,
        force_ingest=self._force_ingest,
        provider_type=provider_type,
    )

    for tool_name, tool in instances.items():
        tool_config = next(
            tc for tc in self.agent_config.tools
            if hasattr(tc, "name") and tc.name == tool_name
        )

        kernel_function = self._create_search_kernel_function(
            tool=tool,
            tool_name=tool_config.name,
            tool_description=tool_config.description,
        )

        self.kernel.add_function(
            plugin_name="hierarchical_document", function=kernel_function
        )
        self._hierarchical_document_tools.append(tool)

        logger.info(f"Registered hierarchical document tool: {tool_config.name}")
```

### 6b. What this achieves

1. **Single source of truth** — context generator resolution logic lives only in `tool_initializer.py`
2. **Config parity** — `context_model` YAML field works identically for both backends
3. **Anthropic auto-detection** — SK backend agents with `provider: anthropic` automatically get `ClaudeSDKContextGenerator` (same as Claude backend)
4. **Eliminates dead code** — the inline `tool.set_chat_service(self._llm_service)` / `tool.set_embedding_service()` / `tool.initialize()` sequence is replaced by one function call
5. **Future-proof** — any new context generator strategies added to `tool_initializer.py` automatically benefit both backends

### 6c. Impact on `_initialize_hierarchical_doc_tools()` (minor)

The `tool_initializer._initialize_hierarchical_doc_tools()` function already accepts `chat_service` as a parameter. For the SK path, `agent_factory` will pass `self._llm_service` (the main SK chat service). The priority chain in Step 5 then resolves:

```
1. Explicit context_generator arg       → (not used from AgentFactory)
2. chat_service arg (self._llm_service)  → LLMContextGenerator wrapping the main agent model
3. tool_config.context_model set         → create separate SK chat service → LLMContextGenerator
4. agent.model.provider == anthropic     → auto-create ClaudeSDKContextGenerator
5. None of the above                     → graceful degradation
```

For the typical SK case (OpenAI/Azure agent, no `context_model`), path 2 fires — same behavior as today, just routed through the shared function.

### 6d. What NOT to change

- `_create_search_kernel_function()` stays in `agent_factory.py` — it creates SK-specific `KernelFunction` objects
- `_has_hierarchical_document_tools()` stays — used for early-exit check
- `self._hierarchical_document_tools` list stays — used for cleanup

---

## Step 7 — Tests

### New: `tests/unit/lib/test_claude_context_generator.py`
- `ClaudeContextConfig` defaults
- Batch prompt construction (all chunks present, document truncation)
- JSON response parsing (valid, markdown-fenced, wrong count, invalid)
- `contextualize_batch()` — empty input, batch success, fallback to individual
- Protocol conformance: `isinstance(gen, ContextGenerator)`

### Update: `tests/unit/lib/test_tool_initializer.py`
- Explicit `context_generator` param takes priority
- Anthropic provider auto-creates `ClaudeSDKContextGenerator`
- SK chat_service wraps in `LLMContextGenerator`
- `context_model` override creates SK service → `LLMContextGenerator`

### Update: `tests/unit/lib/test_llm_context_generator.py`
- Protocol conformance: `isinstance(gen, ContextGenerator)`

### Update: `tests/unit/tools/test_hierarchical_document_tool.py`
- `set_context_generator()` stores generator
- `_ingest_documents()` uses generator without `_chat_service` check

### Update: `tests/unit/lib/test_runner/test_agent_factory.py`
- `_register_hierarchical_document_tools()` delegates to `_initialize_hierarchical_doc_tools()`
- Verify `_initialize_hierarchical_doc_tools()` called with correct args (agent, embedding_service, chat_service, force_ingest, provider_type)
- Verify returned tool instances are registered as KernelFunctions
- Verify `context_model` override works through the delegation path

---

## Files Modified

| File | Change |
|------|--------|
| `src/holodeck/lib/backends/base.py` | Add `ContextGenerator` protocol |
| `src/holodeck/lib/backends/__init__.py` | Export `ContextGenerator` |
| `src/holodeck/lib/llm_context_generator.py` | Conformance assertion, docstring |
| `src/holodeck/lib/claude_context_generator.py` | **New** — `ClaudeSDKContextGenerator` |
| `src/holodeck/tools/hierarchical_document_tool.py` | Add `set_context_generator()`, widen type, simplify guard |
| `src/holodeck/lib/tool_initializer.py` | Add `context_generator` param, auto-detection logic, unified priority chain |
| `src/holodeck/lib/test_runner/agent_factory.py` | Delegate to `tool_initializer._initialize_hierarchical_doc_tools()` |
| `tests/unit/lib/test_claude_context_generator.py` | **New** |
| `tests/unit/lib/test_tool_initializer.py` | New wiring tests |
| `tests/unit/lib/test_llm_context_generator.py` | Protocol conformance |
| `tests/unit/lib/test_runner/test_agent_factory.py` | Delegation tests |

## Verification

```bash
# Targeted tests
pytest tests/unit/lib/test_claude_context_generator.py tests/unit/lib/test_tool_initializer.py tests/unit/lib/test_llm_context_generator.py tests/unit/lib/test_runner/test_agent_factory.py -n auto -v

# Full suite
make test
make type-check
make lint
```
