# Plan: Decouple `LLMContextGenerator` from Semantic Kernel

## Context

`LLMContextGenerator` generates short context snippets for document chunks (Anthropic's contextual retrieval approach). It is currently hardwired to Semantic Kernel's `ChatCompletionClientBase`, meaning:

1. It **cannot work** with the Claude backend — `ClaudeBackend` calls `initialize_tools()` without a `chat_service`, so contextual embeddings silently degrade to no-op.
2. All SK-specific types (`ChatHistory`, `OpenAIChatPromptExecutionSettings`, `get_chat_message_contents()`) are embedded directly in `_call_llm()`.

Since the 021-claude-agent-sdk spec introduced a second backend, we need a lightweight provider-agnostic protocol for one-shot LLM calls.

## Approach

Introduce a `ChatCompletionService` protocol (single `complete(prompt) -> str` method), adapter implementations for **all four providers** (OpenAI, Azure OpenAI, Anthropic, Ollama), and auto-creation logic so contextual embeddings work for **any** backend without manual wiring.

---

## Step 1 — Add `ChatCompletionService` protocol to `base.py`

**File:** `src/holodeck/lib/backends/base.py`

Add after `AgentBackend`:

```python
@runtime_checkable
class ChatCompletionService(Protocol):
    """Lightweight one-shot text-in/text-out LLM call.

    Unlike AgentBackend/AgentSession (multi-turn agent conversations),
    this is for utility LLM calls: contextual embedding generation,
    reranking, summarization, etc.
    """

    async def complete(self, prompt: str) -> str: ...
```

Also export it from `src/holodeck/lib/backends/__init__.py`.

---

## Step 2 — Create adapter implementations

**New file:** `src/holodeck/lib/chat_completion_adapters.py`

### `SKChatCompletionAdapter`
Wraps an existing SK `ChatCompletionClientBase` — extracts the current `_call_llm()` body into an adapter:
- Constructor: `(chat_service, execution_settings=None)`
- `complete()`: Creates `ChatHistory`, calls `get_chat_message_contents()`, returns stripped text

### `OpenAIDirectChatCompletionAdapter`
Uses `openai` SDK directly (already a transitive dependency via SK):
- Constructor: `(model, api_key, endpoint=None, temperature=0.0, max_tokens=150)`
- Handles both standard OpenAI (`AsyncOpenAI`) and Azure OpenAI (`AsyncAzureOpenAI`)
- `complete()`: Calls `client.chat.completions.create()`, returns stripped text

### `AnthropicDirectChatCompletionAdapter`
Uses `anthropic` SDK directly (`anthropic>=0.72.0` is already a production dependency):
- Constructor: `(model, api_key=None, auth_provider=AuthProvider.api_key, temperature=0.0, max_tokens=150)`
- Routes to the correct client class based on `auth_provider`:
  - `api_key` / `oauth_token` → `AsyncAnthropic(api_key=...)`
  - `bedrock` → `AsyncAnthropicBedrock()` (uses AWS credentials from env)
  - `vertex` → `AsyncAnthropicVertex()` (uses GCP credentials from env)
- `complete()`: Calls `client.messages.create()`, extracts `response.content[0].text`, returns stripped text

### `create_chat_completion_service(provider_config: LLMProvider) -> ChatCompletionService`
Factory that routes by `provider_config.provider`:
- `openai` → `OpenAIDirectChatCompletionAdapter`
- `azure_openai` → `OpenAIDirectChatCompletionAdapter` (with endpoint)
- `anthropic` → `AnthropicDirectChatCompletionAdapter` (with auth_provider routing)
- `ollama` → `OpenAIDirectChatCompletionAdapter` using OpenAI-compatible endpoint (Ollama serves an OpenAI-compatible API at `{endpoint}/v1`)

---

## Step 3 — Refactor `LLMContextGenerator`

**File:** `src/holodeck/lib/llm_context_generator.py`

### Constructor
- Change `chat_service` type from `ChatCompletionClientBase` to `ChatCompletionService`
- **Remove** `execution_settings` parameter (now internal to the adapter)

### `_call_llm()`
Replace entire SK-specific body with:
```python
async def _call_llm(self, prompt: str) -> str:
    return await self._chat_service.complete(prompt)
```

### Imports
- Remove all `TYPE_CHECKING` imports of SK types
- Add `TYPE_CHECKING` import of `ChatCompletionService` from `backends.base`

Everything else (retry logic, concurrency, truncation, batch processing) stays identical.

---

## Step 4 — Auto-create service in `tool_initializer.py`

**File:** `src/holodeck/lib/tool_initializer.py`

### Add `_create_context_chat_service(agent) -> ChatCompletionService | None`
Private helper that creates a `ChatCompletionService` from agent config:
- For Anthropic agents with `embedding_provider`: uses `embedding_provider` config
- For Anthropic agents **without** `embedding_provider`: uses `agent.model` directly (Anthropic model for context generation)
- For OpenAI/Azure/Ollama agents: uses `agent.model`
- Returns `None` on failure (logs warning, graceful degradation)

### Update `initialize_tools()`
When `chat_service is None` and hierarchical doc tools exist, auto-create via `_create_context_chat_service(agent)`. This means `ClaudeBackend` (which passes no `chat_service`) will automatically get a working service.

---

## Step 5 — Wrap SK service in `agent_factory.py`

**File:** `src/holodeck/lib/test_runner/agent_factory.py` (line ~978)

Change:
```python
if self._llm_service:
    tool.set_chat_service(self._llm_service)
```
To:
```python
if self._llm_service:
    from holodeck.lib.chat_completion_adapters import SKChatCompletionAdapter
    tool.set_chat_service(SKChatCompletionAdapter(self._llm_service))
```

---

## Step 6 — Update `HierarchicalDocumentTool.set_chat_service()` docstring

**File:** `src/holodeck/tools/hierarchical_document_tool.py` (line 141)

Update docstring from "Semantic Kernel ChatCompletion service instance" to "ChatCompletionService protocol-compatible instance". No logic changes needed — the method already accepts `Any`.

---

## Step 7 — Update tests

### `tests/unit/lib/test_llm_context_generator.py`
- Replace `mock_service.get_chat_message_contents = AsyncMock(...)` pattern with `mock_service.complete = AsyncMock(return_value="...")`
- Remove assertions about `ChatHistory` objects
- Tests become simpler and no longer reference SK types

### New: `tests/unit/lib/test_chat_completion_adapters.py`
- Protocol compliance tests (`isinstance` checks) for all 4 adapters
- `SKChatCompletionAdapter.complete()` delegates to SK service correctly
- `OpenAIDirectChatCompletionAdapter.complete()` delegates to `openai` SDK
- `AnthropicDirectChatCompletionAdapter.complete()` delegates to `anthropic` SDK
- `AnthropicDirectChatCompletionAdapter` auth routing: `api_key` → `AsyncAnthropic`, `bedrock` → `AsyncAnthropicBedrock`, `vertex` → `AsyncAnthropicVertex`
- `create_chat_completion_service()` routes correctly per provider (including `anthropic`)

### Update: `tests/unit/lib/test_tool_initializer.py`
- Test `_create_context_chat_service()` for each provider
- Test `initialize_tools()` auto-creates service when `chat_service=None`
- Test Anthropic agent without `embedding_provider` falls back to main model

---

## Files Modified

| File | Change |
|------|--------|
| `src/holodeck/lib/backends/base.py` | Add `ChatCompletionService` protocol |
| `src/holodeck/lib/backends/__init__.py` | Export `ChatCompletionService` |
| `src/holodeck/lib/chat_completion_adapters.py` | **New** — 4 adapters + factory |
| `src/holodeck/lib/llm_context_generator.py` | Remove SK coupling, use protocol |
| `src/holodeck/lib/tool_initializer.py` | Auto-create service for any backend |
| `src/holodeck/lib/test_runner/agent_factory.py` | Wrap SK service in adapter |
| `src/holodeck/tools/hierarchical_document_tool.py` | Docstring update only |
| `tests/unit/lib/test_llm_context_generator.py` | Update mocks to protocol |
| `tests/unit/lib/test_chat_completion_adapters.py` | **New** — adapter tests |
| `tests/unit/lib/test_tool_initializer.py` | Add auto-creation tests |

## Verification

```bash
# Unit tests (all existing + new)
pytest tests/unit/lib/test_llm_context_generator.py tests/unit/lib/test_chat_completion_adapters.py tests/unit/lib/test_tool_initializer.py -n auto -v

# Full suite to catch regressions
make test

# Type checking
make type-check

# Lint
make lint
```
