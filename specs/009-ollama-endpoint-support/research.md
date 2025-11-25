# Research: Ollama Endpoint Support

**Feature**: 009-ollama-endpoint-support
**Date**: 2025-11-26
**Phase**: Phase 0 - Research & Discovery

## Executive Summary

Research completed to support Ollama LLM provider integration in HoloDeck. Key findings:
1. Ollama provider enum already exists in codebase (ProviderEnum.OLLAMA)
2. Basic endpoint validation already implemented in LLMProvider model
3. Semantic Kernel's OllamaChatCompletion connector is available and compatible
4. AgentFactory requires minimal modifications to support Ollama alongside existing providers
5. No new architecture patterns needed - follow existing OpenAI/Azure OpenAI integration approach

## Research Questions & Findings

### Q1: How does Semantic Kernel support Ollama?

**Decision**: Use Semantic Kernel's `OllamaChatCompletion` connector

**Rationale**:
- Semantic Kernel v1.37.1+ includes native Ollama support via `semantic_kernel.connectors.ai.ollama.OllamaChatCompletion`
- Follows same pattern as OpenAI/Azure/Anthropic connectors already integrated in `agent_factory.py:164-188`
- OpenAI-compatible API means minimal configuration differences
- No additional external dependencies required (already in semantic-kernel package)

**Implementation Pattern** (from Semantic Kernel docs):
```python
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion

service = OllamaChatCompletion(
    ai_model_id="llama3",  # Model name from agent config
    url="http://localhost:11434",  # Host endpoint from agent config
    api_key=None  # Optional for remote instances
)
```

**Alternatives Considered**:
- Direct Ollama REST API integration: Rejected because Semantic Kernel provides abstraction and consistency
- Custom wrapper around Ollama SDK: Rejected due to unnecessary complexity and maintenance burden

### Q2: What configuration schema should Ollama use?

**Decision**: Reuse existing `LLMProvider` model with Ollama-specific validation

**Rationale**:
- `LLMProvider` model (`src/holodeck/models/llm.py:21-81`) already includes:
  - `ProviderEnum.OLLAMA` enum value (line 18)
  - `endpoint` field (line 36-38) for custom URLs
  - `api_key` field (line 39) for optional authentication
  - Endpoint validation for Ollama in `check_endpoint_required()` (line 76-79)
  - Standard execution settings (temperature, max_tokens, top_p)
- Ollama configuration aligns with existing Azure OpenAI pattern (also requires endpoint)
- No new fields needed - existing schema accommodates Ollama requirements

**Configuration Example**:
```yaml
model:
  provider: ollama
  name: llama3
  endpoint: http://localhost:11434
  temperature: 0.7
  max_tokens: 1000
  top_p: 0.9
  api_key: ${OLLAMA_API_KEY}  # Optional, for remote instances
```

**Alternatives Considered**:
- Separate `OllamaConfig` model: Rejected because it duplicates existing fields and violates DRY
- Add `host` field separate from `endpoint`: Rejected because `endpoint` already serves this purpose

### Q3: How should Ollama endpoint validation work?

**Decision**: Configuration validation on model load, lazy connectivity validation on first invoke

**Rationale**:
- Aligns with FR-005 requirement: "validate configuration (URL format, required fields) in the configuration model, and validate endpoint connectivity lazily on first agent invocation"
- Current `LLMProvider.check_endpoint_required()` validates endpoint presence (line 74-80)
- Follows existing pattern: configuration validates schema, runtime validates connectivity
- AgentFactory already implements lazy tool initialization (`_ensure_tools_initialized()` - line 363-372)
- Avoids blocking configuration loading for offline/unreachable endpoints

**Validation Levels**:
1. **Configuration Time** (existing in `llm.py`):
   - Endpoint field is required and non-empty for Ollama provider
   - URL format validation (basic string validation)
   - Standard parameter ranges (temperature 0-2, max_tokens > 0, top_p 0-1)

2. **Runtime** (new in `agent_factory.py`):
   - First invocation attempts connection to Ollama endpoint
   - Semantic Kernel's OllamaChatCompletion handles connection errors naturally
   - Existing retry logic (`_invoke_with_retry()` - line 489-547) catches ConnectionError
   - Clear error messages bubble up through AgentFactoryError

**Alternatives Considered**:
- Eager endpoint validation during config load: Rejected because it blocks offline development
- No validation until execution: Rejected because it delays error detection unnecessarily

### Q4: What error handling patterns should be used?

**Decision**: Follow existing AgentFactoryError pattern with Ollama-specific error messages

**Rationale**:
- AgentFactory already defines `AgentFactoryError` exception (line 71-74)
- Existing error handling in `_create_kernel()` (line 143-203) provides template:
  - Catch provider initialization failures
  - Wrap with AgentFactoryError and clear messages
  - Log errors with context (logger.error with exc_info=True)
- Success criteria SC-004 requires "clear enough that 90% of users can resolve the problem"

**Error Scenarios & Messages**:
1. **Ollama endpoint unreachable**:
   ```
   Failed to connect to Ollama endpoint at http://localhost:11434.
   Ensure Ollama is running: ollama serve
   ```

2. **Model not found**:
   ```
   Model 'llama3' not found on Ollama endpoint http://localhost:11434.
   Pull the model first: ollama pull llama3
   ```

3. **Invalid endpoint URL**:
   ```
   Invalid Ollama endpoint URL: 'localhost:11434'.
   Must be a valid HTTP/HTTPS URL (e.g., http://localhost:11434)
   ```

4. **Authentication failure** (remote instances):
   ```
   Authentication failed for Ollama endpoint http://192.168.1.100:11434.
   Check OLLAMA_API_KEY environment variable.
   ```

**Implementation Location**: `src/holodeck/lib/errors.py` will add Ollama-specific error subclasses

**Alternatives Considered**:
- Generic error messages: Rejected because they don't meet SC-004 (90% self-resolution)
- Silent failures with fallbacks: Rejected because explicit failures are more debuggable

### Q5: How should Ollama integrate with the evaluation framework?

**Decision**: No special handling needed - evaluations use same agent configuration

**Rationale**:
- Evaluations run through the same `AgentFactory` used by chat and test commands
- `EvaluationConfig` model supports flexible model selection (global, per-run, per-metric)
- Ollama models can be used for evaluations by specifying `provider: ollama` in evaluation model config
- No changes needed to evaluation execution (`src/holodeck/lib/test_runner/executor.py`)

**Example Evaluation Config**:
```yaml
evaluations:
  model:  # Override for evaluations (optional)
    provider: ollama
    name: llama3
    endpoint: http://localhost:11434
  metrics:
    - type: groundedness
      model:  # Metric-level override (optional)
        provider: openai
        name: gpt-4o
```

**Alternatives Considered**:
- Separate evaluation-specific Ollama config: Rejected because existing hierarchy already supports this
- Ollama-specific evaluation metrics: Rejected because metrics are provider-agnostic

### Q6: What default values should be used?

**Decision**: Add Ollama defaults to `src/holodeck/config/defaults.py`

**Rationale**:
- Existing `defaults.py` provides default values for all providers
- Default endpoint aligns with Ollama's standard installation (http://localhost:11434)
- Default model should be widely available and performant

**Proposed Defaults**:
```python
OLLAMA_DEFAULTS = {
    "endpoint": "http://localhost:11434",
    "temperature": 0.3,
    "max_tokens": 1000,
    "top_p": None,  # Use model's default
    "api_key": None,  # Not required for local instances
}
```

**Default Model**: Do NOT provide a default model name - require users to specify explicitly
- Rationale: Prevents confusion when model is not pulled; forces intentional model selection

**Alternatives Considered**:
- Default to "llama3": Rejected because model availability varies by user installation
- No defaults (require all fields): Rejected because endpoint is highly predictable

### Q7: How should Ollama support be tested?

**Decision**: Unit tests with mocks + integration tests with live Ollama (optional)

**Rationale**:
- Unit tests validate configuration models and validation logic without external dependencies
- Integration tests require Ollama installation - mark with `@pytest.mark.integration` and `@pytest.mark.skip_if_no_ollama`
- Follows existing test structure (`tests/unit/`, `tests/integration/`)
- Minimum 80% coverage requirement (constitution)

**Test Coverage**:
1. **Unit Tests** (required for CI):
   - `test_llm_ollama.py`: OllamaConfig validation (endpoint required, parameter ranges)
   - `test_validator_ollama.py`: Ollama-specific validation functions
   - `test_agent_factory_ollama.py`: AgentFactory initialization with Ollama (mocked)

2. **Integration Tests** (optional, requires Ollama):
   - `test_ollama_config_loading.py`: End-to-end config loading from YAML
   - `test_ollama_chat_session.py`: Live chat with local Ollama endpoint
   - `test_ollama_test_execution.py`: Test case execution against Ollama

**Test Fixtures**:
- `tests/fixtures/ollama/agent_ollama.yaml`: Sample agent configuration
- `tests/fixtures/ollama/agent_ollama_remote.yaml`: Remote endpoint configuration with auth

**Alternatives Considered**:
- Only integration tests: Rejected because they require Ollama installation and slow down CI
- Only unit tests: Rejected because they don't validate actual Semantic Kernel integration

## Technology Stack

### Primary Dependencies
- **Semantic Kernel** v1.37.1+: Provides `OllamaChatCompletion` connector
- **Pydantic** v2.0+: Configuration validation (existing)
- **PyYAML** v6.0+: YAML parsing (existing)

### Development Dependencies
- **pytest**: Testing framework (existing)
- **pytest-asyncio**: Async test support (existing)
- **pytest-mock**: Mocking for unit tests (existing)

### External Dependencies (User-Managed)
- **Ollama** (local or remote): LLM inference server
  - Installation: https://ollama.com/download
  - Default endpoint: http://localhost:11434
  - Model management: `ollama pull <model>`

## Implementation Checklist

### Configuration Layer
- [ ] Review `src/holodeck/models/llm.py` - confirm ProviderEnum.OLLAMA exists (DONE - line 18)
- [ ] Review `src/holodeck/models/llm.py` - confirm endpoint validation exists (DONE - line 76-79)
- [ ] Add Ollama defaults to `src/holodeck/config/defaults.py`
- [ ] Add Ollama-specific error classes to `src/holodeck/lib/errors.py`

### Agent Engine Integration
- [ ] Modify `src/holodeck/lib/test_runner/agent_factory.py` - add Ollama case to `_create_kernel()`
- [ ] Add Ollama import: `from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion`
- [ ] Implement Ollama service initialization with endpoint and optional api_key
- [ ] Add error handling for Ollama-specific failures (endpoint unreachable, model not found)

### Testing
- [ ] Create `tests/unit/models/test_llm_ollama.py` - test OllamaConfig validation
- [ ] Create `tests/unit/config/test_validator_ollama.py` - test Ollama validation functions
- [ ] Create `tests/integration/test_ollama_config_loading.py` - end-to-end config loading
- [ ] Create `tests/fixtures/ollama/agent_ollama.yaml` - sample Ollama agent configuration
- [ ] Add integration test skip decorator for tests requiring Ollama installation

### Documentation
- [ ] Update README.md with Ollama configuration examples
- [ ] Create quickstart.md with Ollama setup guide
- [ ] Add Ollama troubleshooting section to docs

## Risk Assessment

### Low Risk
- **Configuration schema changes**: Minimal - ProviderEnum.OLLAMA already exists
- **Validation logic**: Reuses existing patterns from Azure OpenAI
- **Test coverage**: Unit tests with mocks ensure CI passes without Ollama

### Medium Risk
- **Semantic Kernel Ollama connector compatibility**: Mitigated by using officially supported connector
- **Error message clarity**: Requires user testing to validate SC-004 (90% self-resolution)

### High Risk
- **None identified**: Implementation follows established patterns with existing dependencies

## Next Steps

1. **Phase 1 - Design & Contracts** (next):
   - Generate `data-model.md` with OllamaConfig and related entities
   - Create API contracts for Ollama configuration (if applicable)
   - Generate `quickstart.md` with Ollama setup and usage examples
   - Update agent context with Ollama-specific technology

2. **Phase 2 - Tasks** (after Phase 1):
   - Break down implementation into atomic, dependency-ordered tasks
   - Use `/speckit.tasks` command to generate `tasks.md`

3. **Implementation** (after planning):
   - Follow task order from `tasks.md`
   - Run code quality checks after each task (`make format lint type-check security`)
   - Achieve 80% test coverage minimum

## References

- Semantic Kernel Ollama Documentation: https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion
- Ollama API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
- HoloDeck Constitution: `/Users/justinbarias/Documents/Git/python/agentlab/.specify/memory/constitution.md`
- Existing LLM Provider Models: `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/llm.py`
- AgentFactory Implementation: `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/test_runner/agent_factory.py`
