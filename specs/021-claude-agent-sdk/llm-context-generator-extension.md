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

## Step 6 — Update `agent_factory.py` (SK backend)

**File:** `src/holodeck/lib/test_runner/agent_factory.py` (line 978-979)

Change:
```python
if self._llm_service:
    tool.set_chat_service(self._llm_service)
```
To:
```python
if self._llm_service:
    from holodeck.lib.llm_context_generator import LLMContextGenerator
    tool.set_context_generator(
        LLMContextGenerator(
            chat_service=self._llm_service,
            max_context_tokens=tool_config.context_max_tokens,
            concurrency=tool_config.context_concurrency,
        )
    )
```

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

---

## Files Modified

| File | Change |
|------|--------|
| `src/holodeck/lib/backends/base.py` | Add `ContextGenerator` protocol |
| `src/holodeck/lib/backends/__init__.py` | Export `ContextGenerator` |
| `src/holodeck/lib/llm_context_generator.py` | Conformance assertion, docstring |
| `src/holodeck/lib/claude_context_generator.py` | **New** — `ClaudeSDKContextGenerator` |
| `src/holodeck/tools/hierarchical_document_tool.py` | Add `set_context_generator()`, widen type, simplify guard |
| `src/holodeck/lib/tool_initializer.py` | Add `context_generator` param, auto-detection logic |
| `src/holodeck/lib/test_runner/agent_factory.py` | Use `set_context_generator()` |
| `tests/unit/lib/test_claude_context_generator.py` | **New** |
| `tests/unit/lib/test_tool_initializer.py` | New wiring tests |
| `tests/unit/lib/test_llm_context_generator.py` | Protocol conformance |

## Verification

```bash
# Targeted tests
pytest tests/unit/lib/test_claude_context_generator.py tests/unit/lib/test_tool_initializer.py tests/unit/lib/test_llm_context_generator.py -n auto -v

# Full suite
make test
make type-check
make lint
```
