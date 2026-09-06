# Feature Specification: Ollama Endpoint Support

**Feature Branch**: `009-ollama-endpoint-support`
**Created**: 2025-11-26
**Status**: Draft
**Input**: User description: "Create a spec to allow holodeck users to use an Ollama endpoint for chat and test"

## Clarifications

### Session 2025-11-26

- Q: What authentication mechanism should be supported for remote Ollama endpoints? → A: API Key in Authorization header (Bearer token)
- Q: What level of operational logging should be provided for Ollama endpoint interactions? → A: Standard - connections, errors, response times
- Q: When should Ollama endpoint connectivity be validated? → A: Validate configuration on model, lazy validation on first invoke

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure Agent with Local Ollama Model (Priority: P1)

A HoloDeck user wants to develop and test AI agents using locally-hosted Ollama models instead of cloud-based API providers, enabling offline development, reduced costs, and data privacy.

**Why this priority**: This is the core value proposition - enabling users to run agents with local LLM infrastructure. Without this, users cannot use Ollama at all with HoloDeck.

**Independent Test**: Can be fully tested by creating an agent YAML file with Ollama configuration pointing to a local endpoint (e.g., localhost:11434) and verifying the agent loads without errors and delivers working chat interactions.

**Acceptance Scenarios**:

1. **Given** a user has Ollama running locally with the `phi3` model, **When** they create an agent.yaml with `provider: ollama`, model name `phi3`, and host `http://localhost:11434`, **Then** the agent configuration loads successfully without errors
2. **Given** an agent configured with Ollama, **When** the user runs `holodeck chat`, **Then** the interactive chat session starts and responds to user messages using the local Ollama model
3. **Given** an agent configured with Ollama endpoint, **When** the configuration includes temperature and max_tokens settings, **Then** these parameters are applied to model interactions

---

### User Story 2 - Test Agent with Ollama Models (Priority: P2)

A HoloDeck user wants to run test cases against agents using Ollama models to validate agent behavior, tool usage, and response quality without incurring cloud API costs.

**Why this priority**: Testing is a core HoloDeck feature, and users need to run evaluations on local models just as they would with cloud providers. This enables cost-effective iterative development.

**Independent Test**: Can be fully tested by defining test cases in agent.yaml with expected outputs/tool usage, running `holodeck test`, and verifying that tests execute against the Ollama endpoint and produce pass/fail results.

**Acceptance Scenarios**:

1. **Given** an agent configured with Ollama and test cases defined in YAML, **When** the user runs `holodeck test`, **Then** all test cases execute against the local Ollama endpoint
2. **Given** a test case with ground truth expectations, **When** the test runs against Ollama, **Then** evaluation metrics (groundedness, relevance) are computed using the responses from Ollama
3. **Given** multiple test cases with different inputs, **When** tests run in sequence, **Then** each test receives fresh context and does not carry over state from previous tests

---

### User Story 3 - Switch Between Multiple Ollama Models (Priority: P3)

A HoloDeck user wants to compare agent performance across different Ollama models (e.g., llama3, phi3, mistral) by switching the model name in configuration without changing code.

**Why this priority**: Model comparison is valuable for optimization but not essential for basic functionality. Users can initially work with a single model and add comparisons later.

**Independent Test**: Can be fully tested by creating multiple agent configurations with different Ollama model names, running chat/test commands with each, and verifying that each uses its specified model.

**Acceptance Scenarios**:

1. **Given** an agent.yaml specifying `model.name: llama3`, **When** the user starts a chat session, **Then** the agent uses the llama3 model from Ollama
2. **Given** the user changes the model name to `phi3` in agent.yaml, **When** they restart the chat, **Then** the agent switches to using the phi3 model
3. **Given** different models have different capabilities, **When** running the same test case against multiple model configurations, **Then** each model processes the test independently with its own characteristics

---

### User Story 4 - Remote Ollama Server Configuration (Priority: P3)

A HoloDeck user wants to connect to an Ollama server running on a remote machine or network endpoint (not localhost) to leverage more powerful hardware or shared infrastructure.

**Why this priority**: While useful for teams and production scenarios, most users will start with local Ollama instances. Remote connectivity is an enhancement for advanced use cases.

**Independent Test**: Can be fully tested by configuring agent.yaml with a remote Ollama URL (e.g., http://192.168.1.100:11434), ensuring network connectivity, and verifying that chat/test commands successfully connect to the remote endpoint.

**Acceptance Scenarios**:

1. **Given** an Ollama server running at `http://192.168.1.100:11434`, **When** the user configures `host: http://192.168.1.100:11434` in agent.yaml, **Then** the agent successfully connects to the remote server
2. **Given** a remote Ollama server requires authentication, **When** the user provides credentials in environment variables, **Then** the agent authenticates and establishes connection
3. **Given** a remote server is unreachable, **When** the agent attempts to initialize, **Then** a clear error message indicates connection failure with the attempted host

---

### Edge Cases

- What happens when the configured Ollama endpoint is not running or unreachable? (System should provide clear error messages indicating connection failure)
- What happens when the specified model name does not exist in the local Ollama instance? (System should report model not found with guidance to pull the model)
- How does the system handle Ollama endpoints that become unavailable mid-conversation? (Should gracefully fail with retry guidance or connection timeout)
- What happens when users specify Ollama-incompatible parameters (e.g., parameters only valid for OpenAI)? (Should validate and warn about unsupported parameters)
- How does the system handle different Ollama API versions? (Should work with standard Ollama OpenAI-compatible v1 endpoint)
- What happens when the Ollama endpoint returns rate limit or resource exhaustion errors? (Should surface these errors clearly to users)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow users to specify Ollama as a model provider in agent configuration files
- **FR-002**: System MUST support configuration of Ollama endpoint URL (default: http://localhost:11434)
- **FR-003**: System MUST allow users to specify Ollama model names (e.g., llama3, phi3, mistral) in agent configuration
- **FR-004**: System MUST support Ollama-specific execution settings including temperature, max_tokens, and top_p parameters
- **FR-005**: System MUST validate Ollama configuration (URL format, required fields) in the configuration model, and validate endpoint connectivity lazily on first agent invocation
- **FR-006**: System MUST provide clear error messages when Ollama endpoints are unreachable or models are unavailable
- **FR-007**: System MUST support the `holodeck chat` command with Ollama-configured agents
- **FR-008**: System MUST support the `holodeck test` command execution against Ollama endpoints
- **FR-009**: System MUST maintain conversation history correctly across multiple turns when using Ollama models
- **FR-010**: System MUST allow environment variable substitution for Ollama endpoint URLs and API key credentials (passed via Authorization Bearer token header)
- **FR-011**: System MUST support both local (localhost) and remote Ollama server endpoints
- **FR-012**: System MUST pass user-defined execution settings (temperature, max_tokens, top_p) to Ollama API calls
- **FR-013**: System MUST integrate Ollama with the evaluation framework for running AI-powered and NLP metrics
- **FR-014**: System MUST handle Ollama API errors gracefully and surface them to users with actionable guidance
- **FR-015**: System MUST support Ollama's OpenAI-compatible API interface (v1 endpoint)
- **FR-016**: System MUST log Ollama endpoint connection attempts, errors, and response times for troubleshooting

### Key Entities *(include if feature involves data)*

- **OllamaConfig**: Configuration model containing provider type (ollama), host URL, model name, optional API key for authentication, and optional execution settings (temperature, max_tokens, top_p)
- **OllamaService**: Service wrapper that manages connection to Ollama endpoints, handles chat completion requests, and processes responses
- **ExecutionSettings**: Settings object containing model parameters like temperature, max_tokens, top_p that are passed to Ollama API
- **ChatHistory**: Conversation context maintained across multiple interactions with Ollama models
- **TestExecution**: Test case execution record that tracks which Ollama endpoint and model were used for evaluation

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can configure an agent to use Ollama and start a chat session in under 2 minutes (assuming Ollama is already installed and running)
- **SC-002**: Chat interactions with Ollama models respond within the same time frame as the underlying Ollama endpoint (no significant overhead introduced by HoloDeck)
- **SC-003**: Test cases execute successfully against Ollama endpoints with a 100% success rate when the Ollama server is running and the model is available
- **SC-004**: Error messages for common issues (endpoint unreachable, model not found) are clear enough that 90% of users can resolve the problem without consulting documentation
- **SC-005**: Users can switch between different Ollama models by changing a single configuration value and restarting the agent
- **SC-006**: The system supports at least 5 different Ollama models commonly used in the community (llama3, phi3, mistral, codellama, gemma)
- **SC-007**: Configuration model validation catches 100% of invalid Ollama endpoint URL formats and missing required fields during configuration loading
- **SC-008**: Users can run evaluations using Ollama models with identical test case definitions as they would use for cloud providers

## Assumptions *(mandatory)*

1. **Ollama Installation**: Users are responsible for installing and running Ollama on their local or remote systems; HoloDeck does not install or manage Ollama
2. **Model Availability**: Users must pull required Ollama models using `ollama pull <model-name>` before configuring agents to use them
3. **API Compatibility**: Ollama provides an OpenAI-compatible API endpoint (v1) that follows standard chat completion patterns
4. **Authentication**: Local Ollama instances typically do not require authentication; remote instances use API key authentication via Authorization Bearer token header, with keys managed through environment variables
5. **Default Endpoint**: The default Ollama endpoint is http://localhost:11434 unless explicitly configured otherwise
6. **Network Accessibility**: For remote Ollama servers, users are responsible for ensuring network connectivity and firewall configurations
7. **Semantic Kernel Integration**: Ollama support will leverage Semantic Kernel's existing OllamaChatCompletion connector
8. **Configuration Format**: Ollama configuration follows the same YAML schema patterns as other providers (OpenAI, Azure OpenAI, Anthropic)

## Dependencies *(if applicable)*

### External Dependencies
- **Ollama**: Users must have Ollama installed and running (local or remote)
- **Semantic Kernel**: The microsoft/semantic-kernel library provides `OllamaChatCompletion` connector that HoloDeck will integrate
- **Network Access**: For remote Ollama servers, network connectivity between HoloDeck and the Ollama endpoint

### Internal Dependencies
- **Configuration System**: Existing HoloDeck configuration loader, validator, and schema must be extended to support Ollama provider
- **LLM Config Models**: `LLMConfig` Pydantic models must include Ollama-specific configuration schema
- **Agent Engine**: The planned agent execution runtime must integrate Semantic Kernel's Ollama connector
- **CLI Commands**: `holodeck chat` and `holodeck test` commands must support Ollama-configured agents
- **Evaluation Framework**: The planned evaluation system must work with Ollama model responses

## Out of Scope *(if applicable)*

- **Ollama Installation/Management**: HoloDeck will not install, update, or manage Ollama software
- **Model Pulling**: HoloDeck will not automatically download Ollama models; users must use `ollama pull` directly
- **Ollama Server Hosting**: HoloDeck will not provide hosted Ollama endpoints or infrastructure
- **Custom Ollama API Extensions**: Only the standard OpenAI-compatible Ollama API will be supported, not custom extensions
- **Ollama-Specific Embedding Models**: Initial implementation focuses on chat completion; embedding support is separate
- **Ollama Version Management**: HoloDeck will target current stable Ollama API versions but will not manage version compatibility
- **Performance Optimization**: While HoloDeck will not add significant overhead, optimizing Ollama's inherent performance is out of scope
- **Multi-Modal Support**: Image/vision models in Ollama are not covered in this initial specification
