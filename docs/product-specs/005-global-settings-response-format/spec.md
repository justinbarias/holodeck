# Feature Specification: Global Settings and Response Format Configuration

**Feature Branch**: `005-global-settings-response-format`
**Created**: 2025-10-25
**Status**: Draft
**Input**: User description: "Create a spec for User Story 2.5 - implementing Global Settings. Also, add into the spec the ability to specify a field called response_format in the Agent Config. This is optional. The field can either be a raw yaml-converted JSON schema definition (strict). Or a file path like schemas/format.json in JSON format."

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

### User Story 1 - Configure Global Settings (Priority: P1)

A developer wants to define project-wide settings that apply to all agents in their HoloDeck project. They want a single configuration file where they can set defaults (like model provider, API keys, default temperature) that individual agents can inherit, reducing duplication and making projects easier to maintain.

**Why this priority**: Foundational for multi-agent projects and simplifies configuration management. Without global settings, users must repeat configuration across every agent file, creating maintenance burden and inconsistency. Essential for real-world project organization.

**Independent Test**: Can be fully tested by creating a global settings configuration file, loading it, and verifying that agents inherit and apply those settings correctly. Validates that configuration hierarchy works and reduces duplication.

**Acceptance Scenarios**:

1. **Given** a HoloDeck project exists, **When** the user creates a `config.yaml` global configuration file at project root, **Then** the file is recognized as the global settings file and loaded by the HoloDeck engine
2. **Given** global settings define a default model provider and api_key, **When** an agent.yaml omits these fields, **Then** the agent inherits and uses the global settings values
3. **Given** an agent.yaml specifies model provider and an api_key, **When** the agent is loaded, **Then** the agent's settings override the global settings (explicit config takes precedence)
4. **Given** global settings are updated, **When** agents are reloaded, **Then** agents that don't have explicit overrides automatically use the updated global settings
5. **Given** an agent explicitly disables a global setting (e.g., sets `inherit_global: false`), **When** the agent is loaded, **Then** only that agent's explicit configuration is used, ignoring the global setting

---

### User Story 2 - Define Response Format (Priority: P1)

A developer wants to specify the exact structure of the agent's response using a JSON schema. This ensures the LLM generates responses in a consistent, structured format that their application can reliably parse. They want to define this either inline in agent.yaml or reference an external JSON schema file.

**Why this priority**: Critical for production use cases where structured outputs are required. Enables reliable integration with downstream systems and APIs. Users need this from day one for real-world applications. Without structured response formats, responses are unpredictable and difficult to process programmatically.

**Independent Test**: Can be fully tested by defining response_format in an agent configuration (both inline and file-based), running the agent, and verifying the output matches the specified schema. Validates that response constraints are properly enforced.

**Acceptance Scenarios**:

1. **Given** an agent.yaml includes a `response_format` field with an inline JSON schema definition, **When** the agent processes a request, **Then** the LLM is constrained to generate responses matching that schema structure
2. **Given** a response_format references an external file (e.g., `response_format: schemas/qa_response.json`), **When** the agent is loaded, **Then** the schema file is loaded and used to constrain the response format
3. **Given** an agent's response_format specifies required fields (e.g., `answer`, `confidence`, `sources`), **When** the agent responds, **Then** the response includes all required fields in the specified structure
4. **Given** a response_format is not specified in agent.yaml, **When** the agent processes a request, **Then** the agent generates responses without format constraints (natural language format)
5. **Given** a response_format file is invalid or missing, **When** the agent is loaded, **Then** HoloDeck displays a clear error message indicating which file is problematic and exits gracefully

---

### User Story 3 - Apply Response Format at Global Level (Priority: P2)

A developer wants to set a default response format at the global settings level that all agents inherit, while allowing individual agents to override with their own response_format. This reduces configuration duplication when most agents use the same response structure.

**Why this priority**: Improves configuration reusability for projects with consistent response patterns. P2 since individual agent-level response formats work independently, but valuable for reducing repetition in larger projects.

**Independent Test**: Can be fully tested by setting a global response_format, verifying agents inherit it, then overriding it in specific agents and confirming the override takes effect. Validates inheritance and override mechanism work correctly.

**Acceptance Scenarios**:

1. **Given** global settings define a `response_format` field, **When** an agent.yaml omits response_format, **Then** the agent inherits and uses the global response_format
2. **Given** global settings define a response_format and an agent.yaml defines its own response_format, **When** the agent is loaded, **Then** the agent's response_format takes precedence over the global setting
3. **Given** an agent has `response_format: null` or `response_format: ~` in its config, **When** the agent is loaded, **Then** the agent disables response format constraints despite the global setting
4. **Given** global settings are updated with a new response_format, **When** agents that inherit from global settings are reloaded, **Then** they automatically use the updated response_format

---

### User Story 4 - Validate Schema Syntax and Structure (Priority: P1)

A developer wants HoloDeck to validate that response_format schemas are syntactically correct JSON and semantically valid according to JSON Schema specification. Invalid schemas should be caught at configuration load time, not during agent execution.

**Why this priority**: Essential for preventing runtime failures and providing good developer experience. Early validation catches configuration errors before they impact agent behavior. Prevents cryptic failures during response generation.

**Independent Test**: Can be fully tested by providing valid and invalid response schemas in different formats and verifying validation works correctly, catching errors at config load time. Validates error detection and user feedback.

**Acceptance Scenarios**:

1. **Given** a response_format contains invalid JSON syntax, **When** the configuration is loaded, **Then** HoloDeck displays a clear error message indicating the JSON parsing error and line number
2. **Given** a response_format violates JSON Schema specification (e.g., invalid property constraints), **When** the configuration is loaded, **Then** HoloDeck validates against JSON Schema spec and reports the specific violation
3. **Given** a response_format file path is specified but the file doesn't exist, **When** the configuration is loaded, **Then** HoloDeck displays a clear error with the expected file path and suggests checking the path
4. **Given** response_format is correctly specified (valid JSON schema), **When** the configuration is loaded, **Then** no validation errors occur and the configuration loads successfully

---

### Edge Cases

- What happens when global settings reference a file that's outside the project directory? (Should accept absolute paths but document the security implications)
- What happens when an agent's response_format overrides a global setting with a conflicting structure? (Override takes precedence; no conflict resolution needed)
- What happens if a response_format schema is very deeply nested or extremely complex? (System should still function; no limit specified, assume reasonable limits based on LLM capabilities)
- What happens when a developer tries to apply response_format to multiple agents with slightly different structures? (They must define separate schemas; inheritance handles the common case)
- What happens if the global settings file is malformed or missing required fields? (Should use defaults for unspecified fields and warn about missing optional fields)

## Requirements _(mandatory)_

### Functional Requirements

**Global Settings File**

- **FR-001**: System MUST support user-level global settings configuration file at `~/.holodeck/config.yml` or `~/.holodeck/config.yaml` that defines settings applicable to all agents across all projects
- **FR-002**: System MUST support project-level global settings configuration file at `config.yml` or `config.yaml` in the project/folder root that overrides user-level settings for that project
- **FR-003**: Configuration precedence MUST follow: user-level (`~/.holodeck/config.yml|config.yaml`) → project-level (`config.yml|config.yaml`) → individual agent configs (explicit agent settings always win)
- **FR-003a**: System MUST check for both `.yml` and `.yaml` file extensions and use either (prefer `.yml` if both exist; log informational message if both present)
- **FR-004**: System MUST load global settings files automatically when agents are loaded, checking both levels in precedence order
- **FR-005**: System MUST validate that global settings files follow valid YAML syntax
- **FR-006**: System MUST support the following global settings fields: `model` (provider, name, temperature, max_tokens), `api_keys` (dictionary of provider-specific keys), `response_format` (JSON schema or file path), and `default_tools` (list of tools all agents inherit)
- **FR-007**: System MUST provide a mechanism for agents to explicitly disable inheritance of global settings using `inherit_global: false` flag

**Response Format Configuration**

- **FR-008**: System MUST support optional `response_format` field in agent.yaml
- **FR-009**: System MUST allow `response_format` to be specified as inline YAML-converted JSON schema object or as a file path (e.g., `schemas/response.json`)
- **FR-010**: System MUST support both `.json` file extension and other text formats for schema files (treated as JSON content)
- **FR-011**: System MUST load external schema files relative to the project root when a file path is provided
- **FR-012**: System MUST validate JSON schema syntax at configuration load time, not during agent execution
- **FR-013**: System MUST validate that schemas conform to Basic JSON Schema specification (supporting `type`, `properties`, `required`, `additionalProperties` features only)
- **FR-014**: System MUST log a warning if response_format is specified but the configured LLM provider does not natively support structured output constraints; assume OpenAI-API compliant providers will be used in practice
- **FR-015**: System MUST pass validated response_format to the LLM as a structured output constraint during request processing

**Inheritance & Override Behavior**

- **FR-016**: System MUST apply global settings (both user-level and project-level) as defaults for all agents unless explicitly overridden in agent.yaml
- **FR-017**: System MUST prioritize agent-level configuration over project-level settings, which take precedence over user-level settings (explicit agent config always wins)
- **FR-018**: System MUST allow agents to inherit response_format from global settings (user or project level) when not explicitly defined in agent.yaml
- **FR-019**: System MUST allow agents to override global response_format with their own definition, completely replacing (not merging with) the global setting
- **FR-020**: System MUST allow agents to explicitly disable response_format constraints and global inheritance using `inherit_global: false` flag

**Configuration Validation & Error Handling**

- **FR-021**: System MUST display clear error messages when global settings files (user-level or project-level) are malformed with details about the specific syntax error and file location
- **FR-022**: System MUST display clear error messages when a referenced schema file is missing, showing the expected file path
- **FR-023**: System MUST display clear error messages when response_format JSON is invalid, indicating the parsing error and location
- **FR-024**: System MUST display clear error messages when response_format violates Basic JSON Schema specification, indicating which constraint is violated

### Key Entities

- **User-Level Global Settings**: Configuration file at `~/.holodeck/config.yml` or `~/.holodeck/config.yaml` containing defaults for model, API keys, response format, and tools. Applies to all agents across all projects unless overridden.
- **Project-Level Global Settings**: Configuration file at `config.yml` or `config.yaml` in project/folder root containing defaults that override user-level settings for that specific project.
- **Response Format Schema**: JSON Schema definition (supporting basic JSON Schema features: type, properties, required, additionalProperties) that constrains the structure of agent responses. Can be inline YAML or external JSON file.
- **Agent Configuration**: Individual agent.yaml file that can inherit from or override both user-level and project-level global settings. Agent configuration always takes final precedence.

## Clarifications

### Session 2025-10-25

- Q: What level of JSON Schema features should response_format support? → A: Basic JSON Schema only (type, properties, required, additionalProperties)
- Q: How should HoloDeck handle LLM providers without native structured output support? → A: Warn & Continue; assume OpenAI-API compliant providers
- Q: Should agents support granular per-field override of global settings? → A: No; agent configuration always overrides global settings (all-or-nothing)
- Q: Which global config file(s) should the system support? → A: User-level only at `~/.holodeck/config.yml` (not project-level)
- Q: Should project-level settings override user-level settings? → A: Yes; support both user-level (`~/.holodeck/config.yml`) and project/folder level (`config.yml`) with hierarchy: user → project → agent
- Q: What should be the standardized global settings filename? → A: Always `config.yml` or `config.yaml` (both extensions supported); check both levels for both extensions; prefer `.yml` if both exist

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Developers can define a global settings file and have all agents automatically inherit its configuration without modification to individual agent files
- **SC-002**: Developers can specify response_format inline in agent.yaml and the system correctly constrains LLM output to that schema
- **SC-003**: Developers can specify response_format as a file path and the system correctly loads, validates, and applies the schema
- **SC-004**: Invalid response_format schemas are caught at configuration load time with clear error messages (not during runtime)
- **SC-005**: Configuration errors are resolved within 2 minutes by developers following error messages and documentation
- **SC-006**: Global settings reduce configuration file size by at least 30% in projects with 3+ agents using shared settings
- **SC-007**: Agents correctly inherit global settings and can selectively override them without side effects
- **SC-008**: Response format constraints are successfully applied to LLM requests in 100% of cases where specified
- **SC-009**: All error messages are actionable and include the specific file/line causing the issue

## Assumptions

- JSON Schema validation will use a standard JSON Schema library (Python: `jsonschema` package) supporting Basic JSON Schema features (type, properties, required, additionalProperties)
- File paths in response_format are relative to project root
- LLM providers will be OpenAI-API compliant with native structured output support (OpenAI JSON mode, Anthropic structured generation, etc.)
- When response_format is specified but LLM provider lacks native support, a warning is logged; best-effort formatting is attempted
- Settings inheritance is unidirectional: agents inherit from global (user-level → project-level → agent), not the reverse
- Global settings files (user-level and project-level) are optional; agents work independently without them
- User-level settings (`~/.holodeck/config.yml` or `~/.holodeck/config.yaml`) apply across all projects; project-level settings (named `config.yml` or `config.yaml` in project root) override only for that specific project
- Both `.yml` and `.yaml` file extensions are supported; `.yml` is preferred if both exist
- Agent configuration always takes final precedence over all global settings (no partial inheritance; complete override at agent level)
