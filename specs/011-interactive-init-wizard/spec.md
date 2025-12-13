# Feature Specification: Interactive Init Wizard

**Feature Branch**: `011-interactive-init-wizard`
**Created**: 2025-11-29
**Status**: Draft
**Input**: User description: "Plan a spec for an improvement to the config init command. Make it into an interactive experience where a user will: get prompted for agent name, which llm provider to use, which vectorstore to use, which evals to enable, and which mcp servers they want."

## Clarifications

### Session 2025-11-29

- Q: What is the order of wizard prompts? → A: Agent Name → LLM Provider → Vector Store → Evals → MCP Servers
- Q: What are the default MCP servers? → A: brave-search[web-search], @modelcontextprotocol/server-memory, @modelcontextprotocol/server-sequentialthinking
- Q: What are the default evals? → A: rag-faithfulness, rag-answer_relevancy
- Q: What is the default LLM provider and model? → A: Ollama with gpt-oss:20b model
- Q: What is the default vector store? → A: ChromaDB at http://localhost:8000

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Quick Start with Defaults (Priority: P1)

A new user runs the `holodeck init` command and wants to quickly get started with sensible defaults. They provide an agent name when prompted and accept all default selections (Ollama for LLM with gpt-oss:20b, ChromaDB for vector store, default evals, and the three default MCP servers) and have a working project scaffolded in under 30 seconds.

**Why this priority**: This is the most common use case - users want to get started quickly without making decisions. Providing smart defaults reduces friction and time-to-first-agent.

**Independent Test**: Can be fully tested by running `holodeck init` and pressing Enter at each prompt (after entering agent name), then verifying the generated configuration files contain the expected defaults.

**Acceptance Scenarios**:

1. **Given** a user runs `holodeck init` in an empty directory, **When** they enter an agent name and press Enter at each interactive prompt without making selections, **Then** the project is created with the specified name, Ollama (gpt-oss:20b), ChromaDB (http://localhost:8000), default evals, and the three default MCP servers configured.
2. **Given** a user completes the interactive wizard with all defaults, **When** they examine the generated `agent.yaml` file, **Then** all default selections are correctly reflected in the configuration.
3. **Given** a user runs the init command, **When** each prompt is displayed, **Then** the default option is clearly indicated with visual highlighting.

---

### User Story 2 - Custom LLM Provider Selection (Priority: P1)

A user wants to use a specific LLM provider (OpenAI, Azure OpenAI, or Anthropic) instead of the default Ollama. They select their preferred provider from the list and receive appropriate follow-up prompts for provider-specific configuration (such as API key environment variable names).

**Why this priority**: LLM provider choice is fundamental to agent functionality and directly impacts cost, performance, and capabilities. Users need this flexibility from day one.

**Independent Test**: Can be tested by running `holodeck init`, selecting a non-default LLM provider, and verifying the generated configuration references the correct provider settings.

**Acceptance Scenarios**:

1. **Given** a user runs `holodeck init`, **When** they reach the LLM provider prompt, **Then** they see a list of available providers: Ollama (default, gpt-oss:20b), OpenAI, Azure OpenAI, and Anthropic.
2. **Given** a user selects OpenAI as their provider, **When** the selection is confirmed, **Then** the generated configuration includes OpenAI-specific settings and references the appropriate environment variable for the API key.
3. **Given** a user selects Azure OpenAI, **When** the selection is confirmed, **Then** the generated configuration includes Azure-specific settings (endpoint, deployment name placeholders).

---

### User Story 3 - Custom Vector Store Selection (Priority: P2)

A user wants to use a specific vector store for their agent's memory and retrieval needs. They select from the available options (ChromaDB, Redis, or In-Memory) based on their deployment requirements.

**Why this priority**: Vector store choice affects data persistence, scalability, and deployment complexity. Important for production use but defaults work well for getting started.

**Independent Test**: Can be tested by selecting a non-default vector store and verifying the configuration includes appropriate vector store settings.

**Acceptance Scenarios**:

1. **Given** a user runs `holodeck init`, **When** they reach the vector store prompt, **Then** they see a list of available options: ChromaDB (default, http://localhost:8000), Redis, and In-Memory.
2. **Given** a user selects Redis, **When** the selection is confirmed, **Then** the generated configuration includes Redis connection settings and appropriate environment variable references.
3. **Given** a user selects In-Memory, **When** the selection is confirmed, **Then** the generated configuration indicates ephemeral storage with a warning about data loss on restart.

---

### User Story 4 - Evaluation Metrics Selection (Priority: P2)

A user wants to configure which evaluation metrics their agent will use for testing. They see a list of available evaluation metrics and can select multiple metrics, with rag-faithfulness and rag-answer_relevancy pre-selected by default.

**Why this priority**: Evaluations are critical for agent quality assurance. Providing sensible defaults enables testing from day one while allowing customization.

**Independent Test**: Can be tested by modifying the default eval selection (adding or removing evals) and verifying the generated configuration reflects exactly the chosen evals.

**Acceptance Scenarios**:

1. **Given** a user runs `holodeck init`, **When** they reach the evals prompt, **Then** they see a multi-select list with rag-faithfulness and rag-answer_relevancy pre-selected.
2. **Given** a user adds additional metrics like rag-context_precision, **When** the selection is confirmed, **Then** the generated configuration includes all selected evaluation metrics.
3. **Given** a user views the evals list, **When** they examine each option, **Then** each eval displays its name and a brief description of what it measures.
4. **Given** a user completes eval selection, **When** the configuration is generated, **Then** each selected eval has proper configuration in the agent.yaml file.

---

### User Story 5 - MCP Server Selection (Priority: P2)

A user wants to configure which MCP (Model Context Protocol) servers their agent can use. They see a list of available MCP servers with descriptions and can select multiple servers, with three pre-selected by default (brave-search[web-search], memory, sequentialthinking).

**Why this priority**: MCP servers extend agent capabilities significantly. Providing a curated list with smart defaults enables powerful functionality while allowing customization.

**Independent Test**: Can be tested by modifying the default MCP server selection (adding or removing servers) and verifying the generated configuration reflects exactly the chosen servers.

**Acceptance Scenarios**:

1. **Given** a user runs `holodeck init`, **When** they reach the MCP server prompt, **Then** they see a multi-select list with brave-search[web-search], memory, and sequentialthinking pre-selected.
2. **Given** a user deselects the memory server and adds filesystem server, **When** the selection is confirmed, **Then** the generated configuration includes only brave-search[web-search], sequentialthinking, and filesystem MCP servers.
3. **Given** a user views the MCP server list, **When** they examine each option, **Then** each server displays its name and a brief description of its capabilities.
4. **Given** a user completes MCP selection, **When** the configuration is generated, **Then** each selected MCP server has proper configuration stubs in the agent.yaml file.

---

### User Story 6 - Non-Interactive Mode (Priority: P3)

A user wants to run the init command in a CI/CD pipeline or automated script where interactive prompts are not possible. They use command-line flags to specify all choices upfront.

**Why this priority**: Supports automation and reproducible project setup, important for teams and DevOps workflows but not critical for initial user experience.

**Independent Test**: Can be tested by running `holodeck init` with all configuration flags and verifying no prompts appear while configuration is correctly generated.

**Acceptance Scenarios**:

1. **Given** a user runs `holodeck init --name my-agent --llm openai --vectorstore redis --evals rag-faithfulness --mcp filesystem,brave-search`, **When** the command executes, **Then** no interactive prompts appear and the project is created with the specified configuration.
2. **Given** a user runs `holodeck init --non-interactive`, **When** no other flags are provided, **Then** all default values are used without prompting.
3. **Given** a user provides invalid flag values, **When** the command executes, **Then** a clear error message is displayed listing valid options.

---

### Edge Cases

- What happens when the user cancels mid-way through the wizard (Ctrl+C)? No partial files should be created.
- How does the system handle when a user's terminal doesn't support interactive prompts? Falls back to non-interactive mode with defaults.
- What happens if the target directory already contains HoloDeck configuration files? Prompts user to confirm overwrite or exit.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: System MUST display prompts in this order: (1) Agent name, (2) LLM provider, (3) Vector store, (4) Evals, (5) MCP servers.
- **FR-002**: System MUST display an interactive prompt for agent name with validation (alphanumeric, hyphens, underscores only).
- **FR-003**: System MUST display an interactive prompt for LLM provider selection with Ollama (gpt-oss:20b) as the default option.
- **FR-004**: System MUST display an interactive prompt for vector store selection with ChromaDB (http://localhost:8000) as the default option.
- **FR-005**: System MUST display a multi-select prompt for evals with rag-faithfulness and rag-answer_relevancy pre-selected by default.
- **FR-006**: System MUST display a multi-select prompt for MCP server selection with brave-search[web-search], memory, and sequentialthinking pre-selected by default.
- **FR-007**: System MUST clearly indicate default selections in each prompt using visual highlighting or markers.
- **FR-008**: System MUST generate configuration files reflecting all user selections upon completion.
- **FR-009**: System MUST support a non-interactive mode via command-line flags for automated/scripted usage.
- **FR-010**: System MUST validate all selections before generating configuration files.
- **FR-011**: System MUST display descriptive help text for each option (LLM providers, vector stores, evals, MCP servers).
- **FR-012**: System MUST clean up any partial files if the user cancels the wizard before completion.
- **FR-013**: System MUST warn users if configuration files already exist in the target directory and prompt for confirmation.
- **FR-014**: System MUST include provider-specific configuration stubs (API key references, endpoint placeholders) based on selections.
- **FR-015**: System MUST fall back to non-interactive mode with defaults when terminal does not support interactive prompts.

### Key Entities

- **Agent Configuration**: Represents the agent with name and description.
- **LLM Provider Configuration**: Represents the selected LLM provider with provider-specific settings (name, API key environment variable, model defaults, endpoint for Azure).
- **Vector Store Configuration**: Represents the selected vector store with connection settings and persistence options.
- **Eval Configuration**: Represents selected evaluation metrics with their configuration.
- **MCP Server Configuration**: Represents selected MCP servers with package references and initialization settings.
- **Init Wizard State**: Tracks user progress through the wizard and accumulated selections.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Users can complete the interactive init wizard in under 60 seconds when accepting all defaults.
- **SC-002**: 95% of users successfully generate a valid configuration on their first attempt.
- **SC-003**: Users can identify the default option in each prompt within 2 seconds of viewing.
- **SC-004**: Non-interactive mode completes project setup in under 5 seconds.
- **SC-005**: Users report the wizard as "easy to understand" in at least 85% of feedback.
- **SC-006**: Zero partial/corrupted configuration files result from cancelled wizard sessions.
- **SC-007**: All generated configurations pass validation without errors.

## Assumptions

- Users have a terminal that supports basic input (stdin) for interactive mode; color/styling is optional enhancement.
- MCP servers are from a predefined list (no registry lookup required).
- Default MCP servers are: brave-search[web-search], @modelcontextprotocol/server-memory, @modelcontextprotocol/server-sequentialthinking.
- Default evals are: rag-faithfulness, rag-answer_relevancy.
- Default LLM provider is Ollama with gpt-oss:20b model.
- Default vector store is ChromaDB at http://localhost:8000.
- API keys and credentials are stored as environment variable references, not actual values, in generated configurations.
- The existing `holodeck init` command structure and template system will be extended rather than replaced.

## Out of Scope

- Custom/user-defined LLM providers beyond the four specified options.
- Custom/user-defined vector stores beyond the three specified options.
- Automatic API key validation or credential testing during init.
- GUI or web-based configuration wizard.
- Dynamic MCP registry lookup (using predefined list instead).
- Custom/user-defined evaluation metrics beyond the predefined list.
