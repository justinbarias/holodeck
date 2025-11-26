# Data Model: Ollama Endpoint Support

**Feature**: 009-ollama-endpoint-support
**Date**: 2025-11-26
**Phase**: Phase 1 - Design & Contracts

## Overview

This document defines the data models and entities for Ollama LLM provider support in HoloDeck. The design reuses existing `LLMProvider` model with Ollama-specific validation and error handling.

## Core Entities

### 1. LLMProvider (Existing - Modified)

**Location**: `src/holodeck/models/llm.py:21-81`

**Purpose**: Configuration model for all LLM providers including Ollama

**Status**: EXISTING - No structural changes needed, Ollama already supported

**Schema**:
```python
class LLMProvider(BaseModel):
    provider: ProviderEnum  # Includes OLLAMA enum value
    name: str  # Ollama model name (e.g., "llama3", "phi3", "mistral")
    temperature: float | None = 0.3  # Temperature (0.0-2.0)
    max_tokens: int | None = 1000  # Maximum tokens to generate
    top_p: float | None = None  # Nucleus sampling parameter (0.0-1.0)
    endpoint: str | None = None  # Ollama endpoint URL (REQUIRED for Ollama)
    api_key: str | None = None  # Optional API key for remote Ollama instances
```

**Ollama-Specific Validation** (existing in `check_endpoint_required()` line 74-80):
- `provider == ProviderEnum.OLLAMA` → `endpoint` MUST be non-empty
- `endpoint` format validated by Pydantic (basic string validation)
- Environment variable substitution supported via `python-dotenv`

**Example Configuration**:
```yaml
# Local Ollama instance
model:
  provider: ollama
  name: llama3
  endpoint: http://localhost:11434
  temperature: 0.7
  max_tokens: 1000

# Remote Ollama instance with authentication
model:
  provider: ollama
  name: phi3
  endpoint: ${OLLAMA_ENDPOINT}  # e.g., http://192.168.1.100:11434
  api_key: ${OLLAMA_API_KEY}
  temperature: 0.5
  max_tokens: 2000
  top_p: 0.9
```

**Validation Rules**:
- `name`: Non-empty string (existing validator at line 42-47)
- `temperature`: 0.0 ≤ temperature ≤ 2.0 (existing validator at line 49-55)
- `max_tokens`: max_tokens > 0 (existing validator at line 57-63)
- `top_p`: 0.0 ≤ top_p ≤ 1.0 if provided (existing validator at line 65-71)
- `endpoint`: Required and non-empty for Ollama (existing validator at line 74-80)

**Relationships**:
- Used by `Agent.model` field (`src/holodeck/models/agent.py:68`)
- Consumed by `AgentFactory._create_kernel()` (`src/holodeck/lib/test_runner/agent_factory.py:143-203`)

---

### 2. ProviderEnum (Existing - No Changes)

**Location**: `src/holodeck/models/llm.py:12-18`

**Purpose**: Enumeration of supported LLM providers

**Status**: EXISTING - Ollama already included

**Schema**:
```python
class ProviderEnum(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # Already present
```

**No modifications needed**.

---

### 3. OllamaDefaults (New)

**Location**: `src/holodeck/config/defaults.py` (new constant)

**Purpose**: Default configuration values for Ollama provider

**Status**: NEW

**Schema**:
```python
OLLAMA_DEFAULTS = {
    "endpoint": "http://localhost:11434",  # Standard Ollama endpoint
    "temperature": 0.3,  # Conservative default
    "max_tokens": 1000,  # Reasonable default
    "top_p": None,  # Use model's default behavior
    "api_key": None,  # Local instances don't require auth
}
```

**Usage**: Applied by configuration loader when user doesn't specify values

**Rationale**:
- `endpoint` default matches Ollama's standard installation
- Other defaults align with existing HoloDeck LLM defaults
- No default `name` (model) - forces explicit user selection

---

### 4. OllamaConnectionError (New)

**Location**: `src/holodeck/lib/errors.py` (new exception class)

**Purpose**: Specific error type for Ollama connectivity failures

**Status**: NEW

**Schema**:
```python
class OllamaConnectionError(AgentFactoryError):
    """Error raised when Ollama endpoint is unreachable.

    Attributes:
        endpoint: The Ollama endpoint URL that failed
        message: Human-readable error message with resolution guidance
    """

    def __init__(self, endpoint: str, original_error: Exception | None = None):
        self.endpoint = endpoint
        message = (
            f"Failed to connect to Ollama endpoint at {endpoint}.\n"
            f"Ensure Ollama is running: ollama serve\n"
            f"Check endpoint URL is correct and accessible."
        )
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)
```

**Usage**: Raised by `AgentFactory._create_kernel()` when Ollama connection fails

**Rationale**: Provides actionable error messages per success criteria SC-004

---

### 5. OllamaModelNotFoundError (New)

**Location**: `src/holodeck/lib/errors.py` (new exception class)

**Purpose**: Specific error type when Ollama model is not available

**Status**: NEW

**Schema**:
```python
class OllamaModelNotFoundError(AgentFactoryError):
    """Error raised when requested Ollama model is not found.

    Attributes:
        model_name: The model that was not found
        endpoint: The Ollama endpoint that was queried
        message: Human-readable error message with resolution guidance
    """

    def __init__(self, model_name: str, endpoint: str):
        self.model_name = model_name
        self.endpoint = endpoint
        message = (
            f"Model '{model_name}' not found on Ollama endpoint {endpoint}.\n"
            f"Pull the model first: ollama pull {model_name}\n"
            f"List available models: ollama list"
        )
        super().__init__(message)
```

**Usage**: Raised by `AgentFactory._create_kernel()` when model initialization fails with "not found" error

**Rationale**: Provides specific resolution steps for common user error

---

## Entity Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent                               │
│  ├─ name: str                                               │
│  ├─ model: LLMProvider ───────────┐                         │
│  ├─ instructions: Instructions    │                         │
│  └─ tools: list[ToolUnion]        │                         │
└───────────────────────────────────┴──────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLMProvider                            │
│  ├─ provider: ProviderEnum (includes OLLAMA)                │
│  ├─ name: str (model name)                                  │
│  ├─ endpoint: str (required for Ollama)                     │
│  ├─ api_key: str | None                                     │
│  ├─ temperature: float | None                               │
│  ├─ max_tokens: int | None                                  │
│  └─ top_p: float | None                                     │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     AgentFactory                            │
│  ├─ _create_kernel() ───> Semantic Kernel Integration      │
│  │   └─ OllamaChatCompletion(                              │
│  │        ai_model_id=config.name,                         │
│  │        url=config.endpoint,                             │
│  │        api_key=config.api_key                           │
│  │      )                                                   │
│  ├─ _apply_model_settings() ───> Execution Settings        │
│  └─ invoke() ───> Agent Execution                          │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│           Semantic Kernel (External)                        │
│  └─ OllamaChatCompletion ───> Ollama Endpoint              │
│      (ai_model_id, url, api_key)                           │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Configuration Loading Flow

```
1. User creates agent.yaml
   └─ model:
       provider: ollama
       name: llama3
       endpoint: http://localhost:11434

2. ConfigLoader.load_config(path)
   └─ Parses YAML
   └─ Loads .env variables
   └─ Merges with OllamaDefaults

3. Pydantic validation (LLMProvider)
   ├─ Validates provider == "ollama"
   ├─ Validates endpoint is non-empty (check_endpoint_required)
   ├─ Validates temperature, max_tokens, top_p ranges
   └─ Returns validated LLMProvider instance

4. Agent model instantiated
   └─ agent_config.model: LLMProvider(provider=OLLAMA, ...)
```

### Runtime Execution Flow

```
1. User runs `holodeck chat` or `holodeck test`
   └─ CLI command loads agent config

2. AgentFactory.__init__(agent_config)
   └─ Calls _create_kernel()

3. AgentFactory._create_kernel()
   ├─ Reads model_config.provider (OLLAMA)
   ├─ Creates OllamaChatCompletion service:
   │   service = OllamaChatCompletion(
   │       ai_model_id=config.name,      # "llama3"
   │       url=config.endpoint,          # "http://localhost:11434"
   │       api_key=config.api_key        # None or provided
   │   )
   ├─ Adds service to kernel
   └─ Returns configured kernel

4. AgentFactory._create_agent()
   ├─ Loads instructions
   ├─ Creates ChatCompletionAgent with kernel
   └─ Returns agent instance

5. AgentFactory.invoke(user_input)
   ├─ Lazy tool initialization (_ensure_tools_initialized)
   ├─ Calls agent.invoke() via Semantic Kernel
   ├─ Semantic Kernel calls Ollama endpoint
   ├─ Extracts response and tool calls
   └─ Returns AgentExecutionResult
```

### Error Handling Flow

```
1. Connection Failure Scenario
   └─ Ollama endpoint unreachable
      ├─ Semantic Kernel raises ConnectionError
      ├─ Caught in _create_kernel()
      ├─ Wrapped in OllamaConnectionError
      └─ Surfaced to user with actionable message

2. Model Not Found Scenario
   └─ Model doesn't exist in Ollama instance
      ├─ Ollama API returns 404/error
      ├─ Semantic Kernel raises exception
      ├─ Caught in _create_kernel()
      ├─ Wrapped in OllamaModelNotFoundError
      └─ Surfaced to user with `ollama pull` command

3. Invalid Configuration Scenario
   └─ Missing or malformed endpoint
      ├─ Pydantic validation fails during config load
      ├─ ValidationError raised with field details
      ├─ Caught by ConfigLoader
      └─ User sees clear field-level error messages
```

## Validation Rules Summary

| Field | Required | Type | Range/Format | Default |
|-------|----------|------|--------------|---------|
| `provider` | Yes | ProviderEnum | Must be "ollama" | N/A |
| `name` | Yes | str | Non-empty string | None (no default) |
| `endpoint` | Yes (for Ollama) | str | Valid HTTP/HTTPS URL | `http://localhost:11434` |
| `api_key` | No | str | Any string | None |
| `temperature` | No | float | 0.0 ≤ t ≤ 2.0 | 0.3 |
| `max_tokens` | No | int | > 0 | 1000 |
| `top_p` | No | float | 0.0 ≤ p ≤ 1.0 | None |

## Edge Cases & Constraints

### Edge Case 1: Endpoint URL Variations
**Scenario**: User provides endpoint without http:// prefix
**Handling**: Pydantic validation accepts as string, Semantic Kernel may fail
**Solution**: Add URL format validation in `LLMProvider` validator to ensure http:// or https:// prefix

### Edge Case 2: Environment Variable Not Set
**Scenario**: User specifies `endpoint: ${OLLAMA_ENDPOINT}` but env var doesn't exist
**Handling**: `python-dotenv` leaves as literal string, endpoint validation fails
**Solution**: ConfigLoader should validate env vars are resolved after loading

### Edge Case 3: Ollama Running on Non-Standard Port
**Scenario**: User runs Ollama on port 8080 instead of 11434
**Handling**: User specifies `endpoint: http://localhost:8080`
**Solution**: No special handling - endpoint field supports any port

### Edge Case 4: Model Name Case Sensitivity
**Scenario**: User specifies "Llama3" instead of "llama3"
**Handling**: Ollama API is case-sensitive, model not found error
**Solution**: Error message from OllamaModelNotFoundError guides user to run `ollama list`

### Edge Case 5: Remote Ollama Without Authentication
**Scenario**: Remote endpoint doesn't require api_key but user provides it
**Handling**: Semantic Kernel sends Authorization header, may be ignored by server
**Solution**: No issue - extra header is harmless

## File Modifications Summary

### Existing Files (Modified)
- **`src/holodeck/models/llm.py`**: No changes needed (Ollama already supported)
- **`src/holodeck/lib/test_runner/agent_factory.py`**: Add Ollama case to `_create_kernel()`
- **`src/holodeck/config/defaults.py`**: Add `OLLAMA_DEFAULTS` constant

### New Files
- **`src/holodeck/lib/errors.py`**: Add `OllamaConnectionError` and `OllamaModelNotFoundError`
- **`tests/unit/models/test_llm_ollama.py`**: Unit tests for Ollama validation
- **`tests/unit/lib/test_agent_factory_ollama.py`**: Unit tests for Ollama agent factory
- **`tests/integration/test_ollama_config_loading.py`**: Integration test for end-to-end config
- **`tests/fixtures/ollama/agent_ollama.yaml`**: Sample Ollama agent configuration

### No Changes Required
- `src/holodeck/models/agent.py`: Uses existing `LLMProvider`, no changes
- `src/holodeck/config/loader.py`: Already supports provider-specific configs
- `src/holodeck/config/validator.py`: Existing validation functions sufficient

## Next Steps

1. Create API contracts (if applicable) in `contracts/` directory
2. Generate quickstart.md with Ollama setup and usage guide
3. Update agent context with Ollama-specific technology
4. Proceed to Phase 2 (tasks.md generation) via `/speckit.tasks`
