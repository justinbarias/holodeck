# Contract: Ollama Configuration Schema

**Feature**: 009-ollama-endpoint-support
**Type**: Configuration Schema
**Version**: 1.0.0
**Status**: Draft

## Overview

This contract defines the YAML configuration schema for Ollama LLM provider in HoloDeck agent configurations. It extends the existing `LLMProvider` schema with Ollama-specific requirements and defaults.

## YAML Schema

### Minimal Configuration

```yaml
model:
  provider: ollama
  name: llama3
  endpoint: http://localhost:11434
```

### Full Configuration

```yaml
model:
  provider: ollama
  name: llama3
  endpoint: http://localhost:11434
  api_key: ${OLLAMA_API_KEY}
  temperature: 0.7
  max_tokens: 2000
  top_p: 0.9
```

### Configuration with Defaults Applied

```yaml
model:
  provider: ollama
  name: phi3
  # endpoint defaults to http://localhost:11434
  # temperature defaults to 0.3
  # max_tokens defaults to 1000
  # top_p defaults to None (model default)
  # api_key defaults to None (no auth)
```

## Field Specifications

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `provider` | string (enum) | Must be "ollama" | `ollama` |
| `name` | string | Ollama model identifier | `llama3`, `phi3`, `mistral` |
| `endpoint` | string (URL) | Ollama API endpoint URL | `http://localhost:11434` |

### Optional Fields

| Field | Type | Range/Format | Default | Description |
|-------|------|--------------|---------|-------------|
| `api_key` | string | Any | None | API key for remote Ollama instances (Authorization Bearer token) |
| `temperature` | float | 0.0 - 2.0 | 0.3 | Sampling temperature for generation |
| `max_tokens` | integer | > 0 | 1000 | Maximum tokens to generate in response |
| `top_p` | float | 0.0 - 1.0 | None | Nucleus sampling parameter |

## Validation Rules

### 1. Provider Validation
```python
provider == "ollama"  # Must match ProviderEnum.OLLAMA
```

### 2. Name Validation
```python
len(name.strip()) > 0  # Non-empty string after trimming whitespace
```

### 3. Endpoint Validation
```python
# Required for Ollama provider
endpoint is not None and len(endpoint.strip()) > 0

# Must be valid URL format (enforced at runtime)
endpoint.startswith("http://") or endpoint.startswith("https://")
```

### 4. Parameter Validations
```python
# Temperature: Conservative range for stable generation
0.0 <= temperature <= 2.0

# Max Tokens: Must generate at least 1 token
max_tokens > 0

# Top P: Probability mass for nucleus sampling
0.0 <= top_p <= 1.0
```

## Environment Variable Substitution

### Supported Syntax
```yaml
# Single environment variable
endpoint: ${OLLAMA_ENDPOINT}

# With default fallback (not yet implemented)
endpoint: ${OLLAMA_ENDPOINT:-http://localhost:11434}

# API key from environment
api_key: ${OLLAMA_API_KEY}
```

### Resolution Behavior
- Environment variables resolved by `python-dotenv` during config loading
- If variable not found, literal string `${VAR_NAME}` remains
- Validation will fail for unresolve d endpoint variables (contains `${`)

## Configuration Examples

### Example 1: Local Development (Default)
```yaml
name: my-agent
model:
  provider: ollama
  name: llama3
  endpoint: http://localhost:11434
instructions:
  inline: "You are a helpful assistant."
```

**Use Case**: Local development with standard Ollama installation

**Validation**: ✅ All required fields present, endpoint is localhost

---

### Example 2: Remote Ollama with Authentication
```yaml
name: production-agent
model:
  provider: ollama
  name: phi3
  endpoint: ${OLLAMA_ENDPOINT}
  api_key: ${OLLAMA_API_KEY}
  temperature: 0.5
  max_tokens: 1500
instructions:
  file: instructions/production.txt
```

**.env file**:
```
OLLAMA_ENDPOINT=http://192.168.1.100:11434
OLLAMA_API_KEY=sk-proj-abcd1234
```

**Use Case**: Production deployment with remote Ollama server

**Validation**: ✅ Endpoint and api_key resolved from environment

---

### Example 3: Model Comparison Configuration
```yaml
name: comparison-agent
model:
  provider: ollama
  name: mistral
  endpoint: http://localhost:11434
  temperature: 0.0  # Deterministic for comparison
  max_tokens: 500
test_cases:
  - input: "Explain quantum computing"
    ground_truth: "..."
```

**Use Case**: Testing different models with identical parameters

**Validation**: ✅ Deterministic temperature for reproducible comparisons

---

### Example 4: Invalid Configuration (Missing Endpoint)
```yaml
name: broken-agent
model:
  provider: ollama
  name: llama3
  # endpoint is missing - INVALID
instructions:
  inline: "Test"
```

**Validation Error**:
```
Field 'model.endpoint': endpoint is required for ollama provider
```

**Resolution**: Add `endpoint: http://localhost:11434`

---

### Example 5: Invalid Configuration (Bad Temperature)
```yaml
name: broken-agent
model:
  provider: ollama
  name: llama3
  endpoint: http://localhost:11434
  temperature: 3.0  # INVALID: exceeds max of 2.0
instructions:
  inline: "Test"
```

**Validation Error**:
```
Field 'model.temperature': temperature must be between 0.0 and 2.0 (received: 3.0)
```

**Resolution**: Change temperature to 0.0 - 2.0 range

---

## Error Messages

### Configuration Validation Errors

| Error Condition | Error Message | Resolution |
|----------------|---------------|------------|
| Missing endpoint | `Field 'model.endpoint': endpoint is required for ollama provider` | Add `endpoint: http://localhost:11434` |
| Empty model name | `Field 'model.name': name must be a non-empty string` | Provide valid model name (e.g., `llama3`) |
| Invalid temperature | `Field 'model.temperature': temperature must be between 0.0 and 2.0 (received: 3.0)` | Use temperature in valid range |
| Invalid max_tokens | `Field 'model.max_tokens': max_tokens must be positive` | Use positive integer (e.g., 1000) |
| Unresolved env var | `Field 'model.endpoint': endpoint contains unresolved variable: ${OLLAMA_ENDPOINT}` | Define variable in .env file |

### Runtime Connection Errors

| Error Condition | Error Message | Resolution |
|----------------|---------------|------------|
| Endpoint unreachable | `Failed to connect to Ollama endpoint at http://localhost:11434. Ensure Ollama is running: ollama serve` | Start Ollama server |
| Model not found | `Model 'llama3' not found on Ollama endpoint http://localhost:11434. Pull the model first: ollama pull llama3` | Pull model with `ollama pull` |
| Authentication failure | `Authentication failed for Ollama endpoint http://192.168.1.100:11434. Check OLLAMA_API_KEY environment variable.` | Verify API key is correct |

## Pydantic Model Mapping

### Python Type Annotations

```python
from pydantic import BaseModel, Field
from enum import Enum

class ProviderEnum(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # Ollama provider

class LLMProvider(BaseModel):
    provider: ProviderEnum  # Required, must be ProviderEnum.OLLAMA
    name: str  # Required, model name (e.g., "llama3")
    endpoint: str | None = Field(None)  # Required for Ollama
    api_key: str | None = Field(None)  # Optional authentication
    temperature: float | None = Field(default=0.3)  # Default 0.3
    max_tokens: int | None = Field(default=1000)  # Default 1000
    top_p: float | None = Field(default=None)  # Default None

    @model_validator(mode="after")
    def check_endpoint_required(self) -> "LLMProvider":
        if self.provider == ProviderEnum.OLLAMA and not self.endpoint:
            raise ValueError(f"endpoint is required for {self.provider.value} provider")
        return self
```

## JSON Schema (for tooling)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "OllamaLLMProvider",
  "type": "object",
  "required": ["provider", "name", "endpoint"],
  "properties": {
    "provider": {
      "type": "string",
      "enum": ["ollama"],
      "description": "LLM provider type"
    },
    "name": {
      "type": "string",
      "minLength": 1,
      "description": "Ollama model name (e.g., llama3, phi3, mistral)"
    },
    "endpoint": {
      "type": "string",
      "format": "uri",
      "description": "Ollama API endpoint URL",
      "default": "http://localhost:11434"
    },
    "api_key": {
      "type": ["string", "null"],
      "description": "Optional API key for remote instances"
    },
    "temperature": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 2.0,
      "default": 0.3,
      "description": "Sampling temperature"
    },
    "max_tokens": {
      "type": "integer",
      "minimum": 1,
      "default": 1000,
      "description": "Maximum tokens to generate"
    },
    "top_p": {
      "type": ["number", "null"],
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Nucleus sampling parameter"
    }
  }
}
```

## Backward Compatibility

### Existing Configurations
- All existing agent configurations (OpenAI, Azure OpenAI, Anthropic) remain unchanged
- No breaking changes to `LLMProvider` model structure
- `ProviderEnum.OLLAMA` is an additive change

### Migration Path
- Users with existing configs: No migration needed
- New Ollama users: Create config following examples above
- Testing: Ollama configs can coexist with other provider configs in test suites

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-26 | Initial schema definition |

## References

- Feature Specification: `/specs/009-ollama-endpoint-support/spec.md`
- Data Model: `/specs/009-ollama-endpoint-support/data-model.md`
- LLMProvider Model: `/src/holodeck/models/llm.py`
- Ollama API Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
