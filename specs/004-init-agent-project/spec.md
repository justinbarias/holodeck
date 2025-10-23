A# Feature Specification: Initialize New Agent Project (v0.1)

**Feature Branch**: `004-init-agent-project`
**Created**: 2025-10-22
**Status**: Draft
**Input**: Holodeck-Initialize New Agent Project. Reference: @specs/001-cli-core-engine/spec.md

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

### User Story 1 - Create Basic Agent Project (Priority: P1)

A developer wants to quickly bootstrap a new AI agent project without writing code. They run the HoloDeck CLI to create a project template with all necessary files and directory structure ready for customization.

**Why this priority**: Foundational experience - users cannot proceed with any agent development without this. Essential for onboarding and MVP viability. Users need a working project structure immediately.

**Independent Test**: Can be fully tested by running `holodeck init <project_name>` and verifying all expected files/directories are created with valid YAML structure and proper folder hierarchy. Delivers immediate value for project setup.

**Acceptance Scenarios**:

1. **Given** the user has HoloDeck installed and is in a development directory, **When** they run `holodeck init customer-support`, **Then** a new directory named `customer-support` is created with agent.yaml, instructions/, data/, tools/, and tests/ folders
2. **Given** the user initializes a project, **When** they examine the generated agent.yaml, **Then** it contains valid YAML structure with placeholder values for model, instructions, and tools sections ready for customization
3. **Given** the user initializes a project without specifying a template, **When** the command completes, **Then** a basic default template is used with sensible defaults for a general-purpose conversational agent
4. **Given** a project already exists with the same name, **When** the user runs init with that name, **Then** HoloDeck prompts the user to confirm overwrite or choose a different name

---

### User Story 2 - Select Project Templates (Priority: P1)

A developer wants to start with a project template tailored to their use case (conversational, research, customer-support) instead of editing from a blank slate. Different templates provide starter configurations and examples appropriate for each domain.

**Why this priority**: Dramatically improves developer experience and reduces setup time. Templates guide users toward best practices for their use case. Essential for MVP viability as it demonstrates the no-code vision.

**Independent Test**: Can be fully tested by running `holodeck init <name> --template <template_type>` for each template type and verifying the generated agent.yaml and supporting files contain appropriate defaults and examples for that domain. Validates users can choose their domain immediately.

**Acceptance Scenarios**:

1. **Given** the user runs `holodeck init chat-bot --template conversational`, **When** the project is created, **Then** the agent.yaml includes default instructions for a conversational assistant and example tools suitable for chat interactions
2. **Given** the user runs `holodeck init research-tool --template research`, **When** the project is created, **Then** the agent.yaml includes default instructions for research/analysis and vector search tool examples
3. **Given** the user runs `holodeck init support-bot --template customer-support`, **When** the project is created, **Then** the agent.yaml includes default instructions for customer support and examples of function tools for ticket/order systems
4. **Given** the user runs init without `--template` flag, **When** the project is created, **Then** the conversational template is used as the default

---

### User Story 3 - Generate Sample Files and Examples (Priority: P1)

A developer wants to see working examples within their new project - sample instructions, example test cases embedded in agent.yaml, and starter data files - so they understand how to configure and use HoloDeck without consulting external documentation.

**Why this priority**: Critical for self-service learning and reducing friction. Examples demonstrate the no-code vision and show best practices. Users can learn by examining the template files.

**Independent Test**: Can be fully tested by running `holodeck init <name> --template <type>` and verifying generated files contain appropriate example content (sample instructions, test cases embedded in agent.yaml, data loading examples). Validates examples are present and properly formatted.

**Acceptance Scenarios**:

1. **Given** a project is initialized with a template, **When** the user examines `instructions/system-prompt.md`, **Then** it contains a well-written system prompt appropriate for the template type that the agent can immediately use
2. **Given** a project is initialized with a template, **When** the user examines `agent.yaml`, **Then** it includes a `test_cases` field with 2-3 sample test cases with input, expected_tools, and ground_truth fields demonstrating the test structure
3. **Given** a project is initialized with a template, **When** the user examines `data/` folder, **Then** it contains example data files (sample CSV, JSON, or markdown) with clear field names and structure appropriate to the template
4. **Given** the `tools/` directory is created, **When** the user examines it, **Then** it includes a `README.md` or inline comments in `__init__.py` explaining how to add custom Python functions
5. **Given** the template includes a vector search example, **When** the user examines the sample tool configuration in agent.yaml, **Then** it shows proper syntax with file paths, vector_field, and meta_fields

---

### User Story 4 - Validate Project Structure (Priority: P1)

After creating a project, a developer wants to verify the initialization succeeded without errors. HoloDeck should validate that the created project structure is correct and all files are properly formatted.

**Why this priority**: Provides confidence that initialization succeeded and catches any errors early. Necessary for reliability of the onboarding experience.

**Independent Test**: Can be fully tested by running `holodeck init <name>` followed by examining the created files and validating they parse without errors. Validation output should be clear about success or any issues.

**Acceptance Scenarios**:

1. **Given** a project has been successfully initialized, **When** the user examines the created agent.yaml, **Then** the YAML is syntactically valid and parses without errors
2. **Given** HoloDeck creates a new project, **When** initialization completes, **Then** the user sees a success message indicating the project name, location, and next steps (e.g., "edit agent.yaml, then run 'holodeck test'")
3. **Given** a project structure is created, **When** all required directories exist (agent.yaml, instructions/, tools/, data/, tests/), **Then** the initialization is considered successful

---

### User Story 5 - Specify Project Metadata (Priority: P2)

A developer wants to provide descriptive metadata about their agent (name, description) during or immediately after initialization so the project is properly documented and identifiable.

**Why this priority**: Improves project organization and documentation. P2 since initialization works without it, but valuable for managing multiple projects.

**Independent Test**: Can be fully tested by running `holodeck init <name>` with optional `--description` flag and verifying metadata appears in agent.yaml. Validates metadata is stored and retrievable.

**Acceptance Scenarios**:

1. **Given** the user runs `holodeck init my-agent --description "Customer support chatbot for order inquiries"`, **When** the project is created, **Then** agent.yaml includes the description field with the provided text
2. **Given** the user initializes a project without providing a description, **When** the project is created, **Then** agent.yaml includes a description field with a template placeholder like "TODO: Add agent description"

---

### Edge Cases

- What happens when the user runs `holodeck init` in a directory where a project with the same name already exists? (Should prompt to confirm overwrite or choose a different name)
- What happens if the user lacks write permissions in the target directory? (Should show a clear error message about permissions)
- What happens if an invalid template name is provided (e.g., `--template invalid`)? (Should list available templates and show an error)
- What happens if the user provides a project name with special characters or spaces? (Should either sanitize the name or show a validation error with allowed characters)
- What happens if HoloDeck encounters a network error while accessing template resources? (Should gracefully degrade to bundled templates with an informational message)
- What happens when the user initializes multiple projects in quick succession? (Should handle concurrency gracefully without conflicts)

## Requirements _(mandatory)_

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

**CLI Command & Project Creation**

- **FR-001**: System MUST provide `holodeck init <project_name>` command that creates a new project directory with the specified name
- **FR-002**: System MUST create the following directory structure inside the project directory: `agent.yaml` (configuration file), `instructions/` (folder for system prompts), `tools/` (folder for custom functions), `data/` (folder for data files)
- **FR-003**: System MUST support multiple project templates via `--template <template_name>` option with at least three templates: `conversational`, `research`, and `customer-support`
- **FR-004**: System MUST use `conversational` as the default template when `--template` is not specified
- **FR-005**: System MUST validate the project name and reject invalid names (alphanumeric characters, hyphens, and underscores only; no leading numbers)

**Generated Files & Content**

- **FR-006**: System MUST generate a valid agent.yaml file with placeholder sections for model, instructions, tools, and test_cases
- **FR-007**: System MUST include a sample system prompt file at `instructions/system-prompt.md` appropriate to the selected template
- **FR-008**: System MUST include 2-3 sample test cases in the `test_cases` field within agent.yaml showing structure (input, expected_tools, ground_truth)
- **FR-009**: System MUST include sample data files in the `data/` folder appropriate to the template type
- **FR-010**: System MUST include a `tools/README.md` explaining how to add custom Python functions
- **FR-011**: System MUST include a `.gitignore` file excluding standard Python artifacts (.venv/, __pycache__/, *.pyc, .env)

**Template Customization**

- **FR-012**: System MUST support template-specific default configurations with appropriate defaults for model temperature, instructions, and suggested tool types
- **FR-013**: System MUST load templates from bundled package templates
- **FR-014**: System MUST gracefully handle missing template resources by falling back to default template with warning message

**Metadata & Configuration**

- **FR-015**: System MUST support `--description <text>` optional flag to set agent description in agent.yaml
- **FR-016**: System MUST populate agent.yaml with project name in the `name` field
- **FR-017**: System MUST support `--author <name>` optional flag to set project creator metadata

**User Feedback & Validation**

- **FR-018**: System MUST display success message indicating project name, location, and next recommended steps
- **FR-019**: System MUST validate all created files are syntactically correct (YAML parses without errors)
- **FR-020**: System MUST detect existing project directories and prompt user to confirm overwrite or choose different name
- **FR-021**: System MUST provide clear error messages for initialization failures (permissions, invalid name, write errors) with actionable guidance

**Version Support**

- **FR-022**: System MUST include HoloDeck version in generated agent.yaml for compatibility tracking

### Key Entities

- **Project**: A directory structure created by `holodeck init` containing agent.yaml and supporting folders (instructions/, tools/, data/, tests/) representing a single agent application
- **Template**: A predefined project structure and default configuration (conversational, research, customer-support) providing starter files, examples, and best practices for specific use case
- **Agent Configuration**: The agent.yaml file defining agent's model provider, instructions, tools, and evaluations - initialized with template defaults

## Success Criteria _(mandatory)_

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Developers can initialize a new agent project with a single command and have a working project structure in under 30 seconds
- **SC-002**: Generated agent.yaml is immediately parseable and contains no validation errors
- **SC-003**: All generated files (instructions, embedded test cases, data samples) are present and properly formatted
- **SC-004**: At least 80% of first-time users complete project initialization without errors on their first attempt
- **SC-005**: Users can identify how to customize generated project files within 2 minutes of opening the template
- **SC-006**: Sample test cases embedded in agent.yaml follow HoloDeck test case schema perfectly (0 schema validation errors)
- **SC-007**: Users can run `holodeck test` immediately after `holodeck init` without additional setup (leveraging generated example test cases in agent.yaml)

---

## Clarifications

### Session 2025-10-22

- Q: Should users be able to specify a custom output directory for project creation? → A: No. Projects are always created in the current working directory. Users navigate to desired location via `cd` before running `holodeck init`.

## Assumptions

1. **Template Bundling**: Templates are bundled with HoloDeck installation; no external download required for v0.1
2. **File Paths**: Projects are always created in the current working directory; no `--path` or `--output-dir` flag supported in v0.1
3. **Default Model**: Generated agent.yaml uses OpenAI as default provider; users can override in environment or update file
4. **Single Agent Focus**: v0.1 initializes single-agent projects; multi-agent projects are out of scope
5. **Python Version**: Users have Python 3.10+ installed as per project requirements
6. **No Dependency Installation**: Project initialization does not automatically install Python dependencies
7. **Case Sensitivity**: Project names are case-sensitive on Unix-like systems, validated consistently across platforms
8. **Template Stability**: Built-in templates are stable and tested; custom templates are out of scope for v0.1

---

## Out of Scope (v0.2+)

- Web UI for project creation
- Custom/user-defined templates (template marketplace)
- Interactive project wizard with prompts
- Automatic dependency installation during initialization
- Cloud project initialization or remote project management
- Project migration tools or versioning utilities
