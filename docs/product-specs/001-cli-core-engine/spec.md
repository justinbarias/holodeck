# Feature Specification: CLI & Core Agent Engine (v0.1)

**Feature Branch**: `001-cli-core-engine`
**Created**: 2025-10-19
**Status**: Draft
**Input**: Inferred from VISION.md v0.1 roadmap - CLI commands and core agent execution engine

## User Scenarios & Testing _(mandatory)_

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.

  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Define Agent Configuration (Priority: P1)

A developer edits the agent.yaml file to configure their AI agent's behavior: model provider, instructions, tools, and evaluations. The YAML configuration should be intuitive and validate correctness.

**Why this priority**: Core feature required before any agent can run. Enables the no-code vision - users should define agents entirely through YAML.

**Independent Test**: Can be fully tested by editing agent.yaml and running `holodeck test` to verify configuration is parsed and applied correctly. Validates the YAML schema works as intended.

**Acceptance Scenarios**:

1. **Given** an agent.yaml file with model, instructions, and tools defined, **When** HoloDeck parses the configuration, **Then** no validation errors occur and all sections are correctly loaded
2. **Given** an agent.yaml with missing required fields, **When** HoloDeck attempts to load it, **Then** clear validation error messages indicate what's missing
3. **Given** an agent.yaml specifying OpenAI as provider with gpt-4o model, **When** the agent executes, **Then** it uses the configured LLM provider and model

---

### User Story 2 - Initialize New Agent Project (Priority: P1)

A developer wants to quickly bootstrap a new AI agent project without writing code. They use the CLI to create a project template with all necessary files and directory structure.

**Why this priority**: Foundational experience - users cannot proceed without this. Essential for onboarding and MVP viability.

**Independent Test**: Can be fully tested by running `holodeck init <name>` and verifying all expected files/directories are created. Delivers immediate value for project setup.

**Acceptance Scenarios**:

1. **Given** the user has HoloDeck installed, **When** they run `holodeck init customer-support --template conversational`, **Then** a new directory is created with agent.yaml, instructions/, data/, tools/, and tests/ folders
2. **Given** the user initializes a project, **When** they examine the generated agent.yaml, **Then** it contains valid YAML structure with model, instructions, and tools sections ready for customization
3. **Given** the user runs init without specifying a template, **When** the command completes, **Then** a basic default template is used with sensible defaults

---

### User Story 2.5 - Configure Global Settings (Priority: P1)

A developer wants to configure global HoloDeck settings once (API keys, vector store connections, deployment defaults) so they can be reused across multiple agent projects without repeating configuration.

**Why this priority**: Critical for practical multi-project workflows. Developers need centralized credential management before they can use agents across different projects. Essential for MVP user experience.

**Independent Test**: Can be fully tested by creating/updating `~/.holodeck/config.yaml` and verifying that agent projects can load credentials from global config. Validates configuration precedence works correctly.

**Acceptance Scenarios**:

1. **Given** a user creates `~/.holodeck/config.yaml` with provider credentials, **When** they run an agent that references those providers, **Then** the agent successfully uses credentials from global config
2. **Given** global config specifies vectorstore connections (Redis, Postgres), **When** an agent.yaml tool references a vectorstore, **Then** the agent uses the connection details from global config
3. **Given** both global config and agent.yaml specify the same provider, **When** the agent initializes, **Then** agent.yaml settings take precedence over global config
4. **Given** global config specifies environment variables with `${VARIABLE_NAME}` syntax, **When** the config is loaded, **Then** environment variables are properly substituted
5. **Given** a developer sets deployment defaults in global config, **When** they run `holodeck deploy`, **Then** the deployment uses those defaults (unless overridden by CLI flags)

---

### User Story 3 - Execute Agent Against Test Cases (Priority: P1)

A developer runs their agent through predefined test cases to verify behavior. They want to see if the agent produces expected outputs and uses the correct tools, with evaluation metrics applied.

**Why this priority**: Essential for the hypothesis-driven testing promise. Without this, developers can't validate agent behavior before deployment.

**Independent Test**: Can be fully tested by defining test cases and running `holodeck test agent.yaml`. Validates agents execute correctly and evaluations work.

**Acceptance Scenarios**:

1. **Given** test cases with input and ground_truth defined, **When** `holodeck test` executes, **Then** each test case runs and produces agent responses
2. **Given** test cases with expected_tools specified, **When** the agent executes, **Then** HoloDeck tracks which tools were called and compares against expected_tools
3. **Given** evaluations configured in agent.yaml, **When** tests execute, **Then** evaluation metrics (groundedness, relevance, etc.) are calculated and displayed with pass/fail status

---

### User Story 4 - Interactive Agent Testing (Priority: P2)

A developer wants to chat with their agent interactively in the terminal to test behavior in real-time before running formal test cases.

**Why this priority**: Important for developer experience and debugging. Enables rapid iteration and learning but not blocking for MVP.

**Independent Test**: Can be fully tested by running `holodeck chat agent.yaml` and having a conversation session. Validates real-time agent interaction works.

**Acceptance Scenarios**:

1. **Given** an initialized agent, **When** the user runs `holodeck chat agent.yaml`, **Then** an interactive prompt appears accepting user input
2. **Given** user input in the chat interface, **When** the user sends a message, **Then** the agent processes it and returns a response with visible tool execution
3. **Given** an active chat session, **When** the user types 'exit', **Then** the session ends gracefully

---

### User Story 5 - Deploy Agent as Local API (Priority: P2)

A developer wants to deploy their tested agent as a running FastAPI server locally to test it as a service with API endpoints.

**Why this priority**: Enables production validation before cloud deployment. Important for workflow but P2 since initial testing can work without it.

**Independent Test**: Can be fully tested by running `holodeck deploy agent.yaml --port 8000` and hitting the API endpoints with curl/client. Validates local deployment works.

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml, **When** `holodeck deploy agent.yaml --port 8000` is executed, **Then** a FastAPI server starts and listens on port 8000
2. **Given** a running deployed agent, **When** a POST request is sent to `/v1/chat` with a message, **Then** the agent processes the request and returns a JSON response
3. **Given** a deployed agent, **When** a request is sent to `/health`, **Then** the endpoint returns a 200 status indicating the service is healthy

---

### User Story 6 - Execute Tool Operations (Priority: P1)

The core agent engine executes tool calls during agent reasoning. Tools can be vector search, custom functions, MCP-based integrations, or prompt-based semantic functions.

**Why this priority**: Fundamental to agent functionality. Agents are useless without tool execution capability.

**Independent Test**: Can be fully tested by configuring tools in agent.yaml and running a test case that triggers tool usage. Validates tool execution pipeline works end-to-end.

#### US 6.1 - Vector Search Tool Operations (Priority: P1)

A developer defines a vectorstore tool that performs semantic search over grounding data. The agent engine loads data sources (unstructured text or structured data), embeds content, and returns semantically similar results with proper field mapping and metadata handling.

##### US 6.1.1 - Unstructured Data Vectorization (Priority: P1)

The vectorstore tool supports unstructured text data in multiple formats and creates searchable embeddings from full content.

**Acceptance Scenarios**:

1. **Given** a tool defined as `type: vectorstore` with `source: data/faqs.md` in agent.yaml, **When** the agent calls this tool with a query, **Then** the markdown file content is embedded and semantically similar results are returned
2. **Given** a vectorstore tool with `source: data/docs/` pointing to a directory, **When** the agent executes, **Then** all supported text files in the directory are loaded and embedded (supports: .txt, .md, .pdf, .csv, .json)
3. **Given** a vectorstore tool with unstructured data and embedding_model specified (e.g., `text-embedding-3-small`), **When** the agent executes, **Then** embeddings are generated using the specified model
4. **Given** a vectorstore tool searching unstructured data, **When** the agent calls it with a query, **Then** results are ranked by relevance score and returned in order with source file reference
5. **Given** a vectorstore tool with a configured vector database (e.g., Redis, Postgres), **When** the agent executes, **Then** the data source is ingested using the file_processor and stored in the vector database for search

---

##### US 6.1.2 - Structured Data Field Mapping (Priority: P1)

The vectorstore tool supports structured data (CSV, JSON) with flexible field selection for vectorization and metadata handling.

**CSV Data Configuration**:

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with `source: data/products.csv` and `vector_field: description`, **When** the agent calls this tool, **Then** only the "description" column content is embedded and searchable
2. **Given** a CSV with multiple text columns and `vector_fields: [description, specs, reviews]`, **When** the agent executes, **Then** content from all specified columns is combined and embedded as a single vector per row
3. **Given** a CSV with `vector_field: content` and `meta_fields: [id, name, category]`, **When** the agent searches, **Then** results include both the matched content and metadata fields in the response
4. **Given** a CSV with no `vector_field` specified, **When** the agent initializes, **Then** all text-type columns are automatically used for vectorization
5. **Given** a CSV search result, **When** the agent receives it, **Then** both the vector-matched content and metadata are returned for LLM context

---

**JSON Data Configuration**:

**Acceptance Scenarios**:

1. **Given** a vectorstore tool with `source: data/articles.json` and `vector_field: content` (flat structure), **When** the agent calls this tool, **Then** the "content" field from each JSON object is embedded
2. **Given** a nested JSON structure with `record_path: items` and `vector_field: description`, **When** the agent loads the data, **Then** each object in the "items" array is treated as a separate record with its "description" field embedded
3. **Given** a nested JSON with `record_path: data.records` (dot notation for nested access), **When** the agent executes, **Then** the tool correctly navigates to nested arrays using path notation
4. **Given** a nested JSON with `record_path: items` and `record_prefix: record_` and `meta_prefix: meta_`, **When** the agent initializes, **Then** fields from record path are prefixed with "record*" and metadata fields are prefixed with "meta*" in the output
5. **Given** a complex JSON with `meta` specified as `[id, category, tags, author]`, **When** the agent searches, **Then** these fields are stored as metadata and returned with search results without being vectorized
6. **Given** a JSON with multiple text fields and no explicit `vector_field` specified, **When** the agent initializes, **Then** all text-type fields are combined and vectorized per record
7. **Given** a JSON structure with mixed data types, **When** the tool processes records, **Then** numeric, boolean, and null fields are included in metadata without vectorization

---

**Generic Structured Data Scenarios**:

1. **Given** a vectorstore tool with `source: data/structured_data` (auto-detected format: CSV or JSON), **When** the agent executes, **Then** the tool automatically detects and applies appropriate parsing
2. **Given** a vectorstore tool with explicit `chunk_size: 512` and `chunk_overlap: 128`, **When** loading long text fields, **Then** content is split into chunks with specified overlap for better search granularity
3. **Given** a structured data source with `vector_field: [description, summary]` and chunking enabled, **When** the agent searches, **Then** it can match against any chunk and return the full record with chunk reference
4. **Given** a vectorstore tool loading large structured data, **When** multiple records have the same vector embedding, **Then** all matching records are returned with relevance scores

---

**Unified Search Results**:

1. **Given** a vectorstore tool returning structured data matches, **When** the agent receives results, **Then** each result contains: matched_content, metadata_fields, source_reference, relevance_score
2. **Given** a vectorstore tool with both `meta_fields` and chunking enabled, **When** returning results, **Then** metadata is preserved consistently across all chunks from the same record
3. **Given** a search query executed on structured data, **When** results are returned, **Then** the agent can reconstruct the original record structure from the response for context

---

#### US 6.2 - Custom Function Tool Operations (Priority: P1)

A developer defines a function tool that points to a Python function in the tools/ directory. The agent engine loads the function, validates parameters, and executes it with proper error handling.

**Acceptance Scenarios**:

1. **Given** a tool defined as `type: function` with `file: tools/orders.py` and `function: check_order_status` in agent.yaml, **When** the agent calls this tool with parameters (e.g., order_id), **Then** the Python function is loaded and executed
2. **Given** a function tool with parameters schema defined, **When** the agent calls it, **Then** parameters are validated against the schema before execution
3. **Given** a function tool that raises an exception during execution, **When** the agent calls it, **Then** the error is caught, logged, and returned to the agent with a clear error message
4. **Given** a function tool returns a result, **When** the agent receives it, **Then** the result is properly serialized and passed back to the LLM for further reasoning

---

#### US 6.3 - MCP Tool Operations (Priority: P1)

A developer defines an MCP (Model Context Protocol) tool that integrates with standardized MCP servers. The agent engine establishes connections to MCP servers, sends requests through the standard protocol, and handles responses.

**Acceptance Scenarios**:

1. **Given** a tool defined as `type: mcp` with `server: @modelcontextprotocol/server-filesystem` in agent.yaml, **When** the agent calls this tool, **Then** the MCP server is invoked using the standard Model Context Protocol
2. **Given** an MCP tool with custom configuration (e.g., allowed_directories, API tokens), **When** the agent executes, **Then** configuration is properly passed to the MCP server
3. **Given** an MCP server that returns structured data, **When** the agent receives the response, **Then** the response is properly parsed and integrated into agent reasoning
4. **Given** an MCP server connection that fails, **When** the agent attempts to use the tool, **Then** the error is caught and returned with a clear message about the MCP server unavailability

---

#### US 6.4 - Prompt-Based Tool Operations (Priority: P1)

A developer defines a prompt-based tool (semantic function) that uses an LLM to transform input. The agent engine manages the prompt template, substitutes parameters, and returns the LLM result.

**Acceptance Scenarios**:

1. **Given** a tool defined as `type: prompt` with an inline template and parameters in agent.yaml, **When** the agent calls this tool with input, **Then** the template is instantiated with the provided parameters and sent to the specified LLM
2. **Given** a prompt-based tool with `file: tools/prompts/extract_entities.md`, **When** the agent calls it, **Then** the prompt file is loaded and the template is instantiated with parameters
3. **Given** a prompt-based tool with model configuration specified, **When** the agent executes, **Then** the tool uses the configured LLM provider and model (not the agent's model unless unspecified)
4. **Given** a prompt-based tool returns structured output (JSON), **When** the agent receives it, **Then** the output is properly parsed and passed back to the agent for reasoning

---

### User Story 7 - Load System Instructions (Priority: P1)

The agent engine loads instructions from a file (typically system-prompt.md) or inline YAML configuration that guides agent behavior.

**Why this priority**: Core to agent behavior - instructions define the agent's role and constraints. Required for any meaningful agent execution.

**Independent Test**: Can be fully tested by specifying instructions in agent.yaml and verifying they're passed to the LLM. Validates instruction loading works.

**Acceptance Scenarios**:

1. **Given** instructions specified as `file: instructions/system-prompt.md` in agent.yaml, **When** the agent initializes, **Then** the file content is loaded and used as system prompt
2. **Given** instructions specified inline in agent.yaml, **When** the agent initializes, **Then** the inline instruction text is used as system prompt
3. **Given** an agent with instructions, **When** it makes LLM calls, **Then** the instructions are included in the prompt sent to the model

### Edge Cases

- What happens when an agent.yaml file contains invalid YAML syntax?
- What happens when a referenced tool file (tools/custom.py) doesn't exist?
- What happens when the LLM provider API key is missing or invalid?
- What happens when the global config file `~/.holodeck/config.yaml` doesn't exist? (Should gracefully fall back to env vars or agent.yaml credentials)
- What happens when environment variable substitution in global config references an undefined variable?
- What happens when both global config and agent.yaml specify conflicting credentials? (agent.yaml takes precedence)
- What happens when a test case references a tool that's not defined in agent.yaml?
- What happens when an evaluation metric fails to calculate (e.g., LLM call times out)?
- How does the system handle very long agent conversations in chat mode (memory management)?
- What happens when multiple tools are called simultaneously by the agent?
- How are concurrent requests handled in the deployed API?
- When a user exits chat mode and re-runs it, should previous conversations be available? (Clarified: No persistence by default in v0.1 to keep implementation lightweight)

## Requirements _(mandatory)_

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

**CLI & Project Initialization**

- **FR-001**: System MUST provide `holodeck init <project_name>` command that creates project directory with agent.yaml, instructions/, tools/, tests/, and data/ folders
- **FR-002**: System MUST support multiple project templates (e.g., `--template conversational`, `--template research`, `--template customer-support`) with sensible defaults for each
- **FR-003**: System MUST provide `holodeck --version` command showing the installed HoloDeck version

**Global Configuration**

- **FR-003a**: System MUST support global configuration file at `~/.holodeck/config.yaml` for provider credentials, vector store connections, and deployment defaults
- **FR-003b**: System MUST support provider configuration for LLM providers (openai, azure_openai, anthropic) with API keys and endpoints
- **FR-003c**: System MUST support vectorstore configuration (Redis, Postgres) for vector database connections at the global level
- **FR-003d**: System MUST support deployment defaults configuration (default_port, rate_limit, auth settings) in global config
- **FR-003e**: System MUST support environment variable substitution in global config using `${VARIABLE_NAME}` syntax
- **FR-003f**: System MUST load provider credentials from global config if not explicitly specified in agent.yaml or environment variables (precedence: agent.yaml > env vars > global config)

**Agent Configuration & Loading**

- **FR-004**: System MUST parse and validate agent.yaml files against a defined schema with clear error messages for invalid configurations
- **FR-005**: System MUST support agent.yaml specifying model provider (openai, azure_openai, anthropic) and model name with configuration options (temperature, max_tokens)
- **FR-006**: System MUST load agent instructions from either file path or inline YAML specification
- **FR-007**: System MUST support tool definitions as vectorstore, function, or MCP types in agent.yaml

**Agent Execution & Tool System**

- **FR-008**: System MUST execute agent reasoning loops that process user input, call LLMs with instructions, and invoke tools as needed
- **FR-009**: System MUST support vectorstore tools (type: vectorstore) that embed queries and return semantically similar results from unstructured text data (single files, directories, multiple formats: .txt, .md, .pdf)
- **FR-009a**: System MUST support vectorstore tools for structured data (CSV, JSON) with field mapping configuration: `vector_field`, `vector_fields`, `meta_fields`, `chunk_size`, `chunk_overlap`
- **FR-009b**: System MUST support JSON data normalization for nested structures using: `record_path`, `record_prefix`, `meta_prefix`, `meta` array specification (pandas.json_normalize pattern)
- **FR-009c**: System MUST automatically detect data format and parse CSV/JSON correctly; for unspecified `vector_field`, MUST use all text-type columns/fields
- **FR-009d**: System MUST preserve and return metadata fields with search results including matched_content, metadata_fields, source_reference, and relevance_score
- **FR-014**: System MUST support custom function tools (type: function) loaded from Python files with parameter validation and error handling
- **FR-015**: System MUST support MCP tools (type: mcp) that invoke standardized Model Context Protocol servers with proper request/response handling
- **FR-016**: System MUST support prompt-based tools (type: prompt) that use LLM templates with parameter substitution and configurable model providers
- **FR-017**: System MUST track which tools are invoked during agent execution for validation against expected_tools

**Testing & Evaluation**

- **FR-018**: System MUST provide `holodeck test agent.yaml` command that runs all test cases defined in agent.yaml against the agent
- **FR-019**: System MUST support test cases with input, expected_tools, and ground_truth fields for validation
- **FR-020**: System MUST execute configured evaluation metrics (AI-powered like groundedness/relevance, and NLP-based like F1/BLEU/ROUGE) on test results
- **FR-020a**: System MUST handle failed evaluation metrics gracefully; failed metrics show "ERROR" status with logged details; test continues to validate agent reasoning even if metrics can't be calculated
- **FR-021**: System MUST compare actual tool usage against expected_tools and report pass/fail for each test case
- **FR-022**: System MUST provide clear test output showing which tests passed, which failed, and why (evaluation scores, tool mismatches, metric errors, etc.)
- **FR-023**: System MUST support ground_truth comparison when provided to calculate accuracy-based metrics

**Interactive Testing**

- **FR-024**: System MUST provide `holodeck chat agent.yaml` command that starts an interactive terminal session with the agent
- **FR-025**: System MUST accept user input in chat mode and display agent responses with visible tool execution trace
- **FR-026**: System MUST gracefully handle session termination (e.g., 'exit' or Ctrl+C) in chat mode
- **FR-026a**: System MUST maintain conversation history in chat mode; v0.1 CLI defaults to in-memory storage of last N messages (configurable, e.g., 20 messages)
- **FR-026b**: System MUST support memory strategy configuration for API deployments (in-memory, Redis, database); CLI mode uses in-memory by default

**Local Deployment**

- **FR-027**: System MUST provide `holodeck deploy agent.yaml` command that starts a local FastAPI server with the agent
- **FR-028**: System MUST support `--port` option to specify which port the API server runs on (default: 8000)
- **FR-029**: System MUST provide `/v1/chat` POST endpoint accepting JSON with message and optional session_id, returning agent response
- **FR-029a**: System MUST isolate session state per session_id; requests to the same session_id are serialized to prevent race conditions
- **FR-029b**: System MUST generate and return a session_id in the response if not provided in the request
- **FR-030**: System MUST provide `/health` GET endpoint returning 200 status when service is running
- **FR-031**: System MUST include structured JSON logging for all requests/responses in deployed agents with fields: timestamp, request_id, session_id, message_input, tool_calls, response, duration_ms, error
- **FR-031a**: System MUST rotate logs daily with 7-day retention by default; log location: `~/.holodeck/logs/agent-<name>.log`
- **FR-031b**: System MUST support log configuration in agent.yaml (log_level, rotation_days, retention_days) to override defaults

**Error Handling & Validation**

- **FR-032**: System MUST provide clear validation error messages when agent.yaml is malformed or missing required fields
- **FR-033**: System MUST provide clear error messages when referenced files (instructions, tools, data) don't exist
- **FR-034**: System MUST provide clear error messages when LLM API calls fail (auth errors, rate limits, model not found)
- **FR-035**: System MUST provide clear error messages when evaluation metrics fail to calculate
- **FR-036**: System MUST NOT block agent execution when tool calls fail; failed tools MUST log errors and return empty string context to the agent for continued reasoning
- **FR-037**: System MUST support middleware hooks for extensible tool error recovery (e.g., custom retry logic, fallback strategies) without blocking default graceful-degradation behavior

### Key Entities

- **Agent**: Represents a single AI agent instance defined by agent.yaml, with model provider, instructions, tools, and evaluation configuration
- **Tool**: Represents an agent's capability - can be vectorstore search (type: vectorstore), custom Python function (type: function), MCP integration (type: mcp), or prompt-based semantic function (type: prompt)
- **TestCase**: Represents a single test scenario with input, optional ground_truth, expected_tools, and evaluation requirements
- **EvaluationMetric**: Represents a metric that measures test result quality (groundedness, relevance, F1, etc.) with configurable threshold
- **Project**: Represents the directory structure created by `holodeck init` containing agent.yaml and supporting files
- **SearchResult** (NEW): Represents a vectorstore search result with structured fields: `matched_content` (string), `metadata_dict` (dict), `source_reference` (string), and `relevance_score` (float). Must be modeled as a concrete type/class in implementation for consistent handling across tool execution pipeline

## Clarifications

### Session 2025-10-19

- Q: How should vectorstore search results be structured to support agent reasoning and test assertions? → A: Each result returns {matched_content, metadata_dict, source_reference, relevance_score} as a concrete SearchResult type/class for consistent handling across tool execution pipeline.
- Q: When tools fail (network errors, exceptions, timeouts), should errors block agent execution? → A: Errors should not block execution by default. Failed tool errors are logged, but return an empty string in the context to the agent. Error recovery/retry behavior must be extensible through middleware hooks for custom strategies.
- Q: How should chat mode manage conversation memory and session state? → A: Chat memory strategy is configurable. v0.1 CLI experimentation mode defaults to in-memory (last N messages). For API deployment, strategy becomes configurable via agent.yaml (Redis cache, database, or in-memory).
- Q: How should the API handle concurrent requests and session state isolation? → A: Session isolation via session_id. Each request includes session_id; API maintains separate conversation history per session. Requests to the same session_id are serialized (queued) to prevent race conditions.
- Q: When evaluation metrics fail (LLM timeout, rate limit), should failed metrics fail the entire test? → A: Soft failure: Test continues; failed metrics show "ERROR" status with logged error details. Agent reasoning is validated even if some metrics can't be calculated. Prevents external API issues from breaking feedback loops.
- Q: How should global configuration interact with agent.yaml and environment variables? → A: Three-level precedence: agent.yaml (highest) > environment variables > global config (lowest). If agent.yaml specifies a provider, it overrides global config. If neither agent.yaml nor env vars specify credentials, system loads from global config. Missing global config file is not an error; system gracefully falls back to env vars or agent.yaml settings.
- Q: When agents are deployed as local APIs, how should request/response logging be implemented? → A: Structured JSON logs to file with rotation policy. Logs MUST include: timestamp, request_id, session_id, message_input, tool_calls, response, duration_ms, error (if any). Logs are rotated daily with 7-day retention by default. Format enables easy parsing for monitoring/aggregation and supports future OpenTelemetry integration.
- Q: What should be the default retry behavior when tools fail? → A: No automatic retries by default. Errors are logged and returned as empty string context to the agent for continued reasoning. This aligns with graceful degradation principle and keeps initial implementation simple. Users can define custom retry logic via middleware hooks if needed.
- Q: For chat mode in CLI, should conversation history persist across sessions? → A: No persistence by default. Conversation history resets each session. This keeps v0.1 implementation lightweight, prevents unbounded memory growth, and aligns with CLI as an experimentation/exploration tool. Users can enable persistence via agent.yaml configuration if needed in future versions.

## Success Criteria _(mandatory)_

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Developers can initialize a new agent project and have a working agent.yaml in under 30 seconds
- **SC-002**: Agent execution against a test case completes in under 5 seconds per test (including LLM calls)
- **SC-003**: Developers can define and test agents entirely through YAML without writing any Python code
- **SC-004**: Test results clearly show which evaluation metrics passed/failed with specific scores (e.g., "Groundedness: 4.2/5.0 ✓")
- **SC-005**: Tool invocation tracking works with 100% accuracy (all called tools are correctly identified and compared against expected_tools)
- **SC-006**: Chat mode provides real-time interaction with response time under 3 seconds per message
- **SC-007**: Deployed API handles at least 10 concurrent requests without errors
- **SC-008**: Error messages are actionable - developer can understand and fix issues without consulting documentation
- **SC-009**: All CLI commands (`init`, `test`, `chat`, `deploy`) are available and functional
- **SC-010**: Documentation/help for all CLI commands is accessible via `--help` or command examples

---

## Assumptions

1. **LLM Provider Setup**: Users have valid API keys for their chosen LLM provider (OpenAI, Azure, etc.) configured in environment variables
2. **Python Environment**: Users have Python 3.10+ installed with pip/Poetry available
3. **Local Development Only**: v0.1 deployment is local only; cloud deployment comes in later versions
4. **Single Agent Focus**: v0.1 focuses on single-agent execution; multi-agent orchestration comes later
5. **Tool System Scope**: MCP, vectorstore, and function tools are in scope; external API integrations use MCP pattern, not custom API tool type
6. **Evaluation Models**: AI-powered evaluations use the same LLM provider/model as the agent unless overridden in agent.yaml
7. **File Paths**: All file references in agent.yaml are relative to the project directory
8. **Test Case Structure**: Test cases follow the predefined schema with input as required field, others optional

---

## Out of Scope (v0.2+)

- Cloud deployment (Azure, AWS, GCP)
- Multi-agent orchestration patterns (sequential, concurrent, handoff, group chat, magentic)
- Web UI for no-code agent creation
- Advanced RAG with chunking strategies and reranking
- Rate limiting and authentication on deployed APIs
- Enterprise features (SSO, audit logs, RBAC)
- Plugin marketplace/registry
- Advanced observability/monitoring (OpenTelemetry integration)
- Multi-file test case inputs (images, PDFs in test cases)
