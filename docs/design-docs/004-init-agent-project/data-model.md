# Data Model: Initialize New Agent Project (Phase 1)

**Date**: 2025-10-22
**Status**: Complete

---

## Core Entities

### 1. AgentConfig (from 001-cli-core-engine)

**Purpose**: Master configuration schema for agent definition. Generated agent.yaml MUST conform to this schema.

**Key Fields** (as specified in 001-cli-core-engine):

- `name`: str - Project/agent name
- `description`: str - Agent description
- `model`: ModelConfig - LLM provider and model selection
  - `provider`: str (openai, azure_openai, anthropic)
  - `name`: str (e.g., gpt-4o)
  - `temperature`: float (0.0-2.0)
  - `max_tokens`: int
- `instructions`: InstructionsConfig - System prompt definition
  - `file`: str (path to markdown file) OR
  - `inline`: str (inline prompt text)
- `tools`: List[ToolConfig] - Available tools
  - Each tool: type, name, configuration per tool type
- `evaluations`: EvaluationConfig - Test evaluation setup
- `test_cases`: List[TestCase] - Test scenarios

**Validation Rules**:

- Required fields: name, model, instructions
- Project name: alphanumeric, hyphens, underscores; no leading digits
- Model provider must match installed SDK
- Temperature range: 0.0-2.0
- Max tokens: positive integer

**Used By**: Init command validates generated agent.yaml against this schema before writing

---

### 2. TemplateManifest

**Purpose**: Metadata and validation rules for template rendering

**Fields**:

- `name`: str - Template identifier (conversational, research, customer-support)
- `display_name`: str - Human-readable name for CLI output
- `description`: str - One-line description of template purpose
- `category`: str - Use case category (e.g., "conversational-ai", "research-analysis")
- `version`: str - Template version (semver format)
- `created_at`: datetime - Template creation timestamp
- `variables`: Dict[str, VariableSchema] - Allowed template variables
  - Each variable:
    - `type`: str (string, number, boolean, enum)
    - `description`: str - What this variable controls
    - `default`: Any - Default value if not provided
    - `required`: bool - Whether variable must be provided
    - `allowed_values`: List[str] - For enum type, allowed choices
- `defaults`: Dict[str, Any] - Template-specific default values
  - E.g., `model.temperature: 0.7`, `model.provider: "openai"`
- `files`: Dict[str, FileMetadata] - Files in template
  - Each file:
    - `path`: str - Relative path in generated project
    - `template`: bool - True if Jinja2 template (.j2)
    - `required`: bool - True if always included

**Validation Rules**:

- Name must match directory name (conversational, research, customer-support)
- All variables must be used in at least one template file
- Default values must satisfy type constraints
- No variable name conflicts across templates
- Version format: MAJOR.MINOR.PATCH (semver)

**Location**: Each template includes manifest.yaml at root

**Example**:

```yaml
# conversational/manifest.yaml
name: conversational
display_name: Conversational Agent
description: AI assistant for multi-turn conversations
category: conversational-ai
version: 1.0.0
variables:
  project_name:
    type: string
    description: Name of the agent project
    required: true
  description:
    type: string
    description: Brief description of agent purpose
    required: false
    default: "TODO: Add agent description"
  model_temperature:
    type: number
    description: Temperature for model creativity
    default: 0.7
    allowed_values: [0, 0.3, 0.5, 0.7, 0.9, 1.0]
defaults:
  model.provider: "openai"
  model.name: "gpt-4o"
files:
  agent.yaml:
    template: true
    required: true
  instructions/system-prompt.md:
    template: true
    required: true
  tools/README.md:
    template: true
    required: true
  data/faqs.md:
    template: false
    required: false
```

---

### 3. ProjectInitInput

**Purpose**: User-provided input for project initialization

**Fields**:

- `project_name`: str - Name of project to create
- `template`: str - Template choice (conversational, research, customer-support)
- `description`: Optional[str] - Agent description
- `author`: Optional[str] - Project creator name
- `output_dir`: str - Target directory (currently always CWD, but model allows future extension)
- `overwrite`: bool - Whether to overwrite existing project (default: False)

**Validation Rules**:

- project_name: required, must be valid per AgentConfig.name constraints
- template: required, must be one of available templates
- output_dir: must be writable directory
- If project already exists and overwrite=False, raise validation error

**Used By**: CLI init command

---

### 4. ProjectInitResult

**Purpose**: Outcome of project initialization

**Fields**:

- `success`: bool - Whether initialization completed successfully
- `project_name`: str - Name of created project
- `project_path`: str - Absolute path to created project directory
- `template_used`: str - Which template was applied
- `files_created`: List[str] - Relative paths of created files
- `warnings`: List[str] - Non-blocking issues (e.g., permission notes)
- `errors`: List[str] - Blocking errors that prevented creation
- `duration_seconds`: float - Time taken for initialization

**Used By**: CLI command to display user feedback

---

### 5. TemplateRenderer

**Purpose**: Renders Jinja2 templates and validates output

**Key Methods**:

- `render_template(template_path: str, variables: Dict[str, Any]) -> str`

  - Renders Jinja2 template with provided variables
  - Returns rendered content as string
  - Raises TemplateError if rendering fails

- `validate_agent_config(yaml_content: str) -> AgentConfig`

  - Parses YAML string and validates against AgentConfig schema
  - Returns parsed AgentConfig object
  - Raises ValidationError with helpful message if invalid
  - This is the CRITICAL validation gate

- `render_and_validate(template_path: str, variables: Dict[str, Any]) -> str`
  - Combines render + validate for YAML files
  - Returns rendered content only if it validates
  - Safe to write to disk after this call

**Constraints**:

- Jinja2 environment uses restricted filters (no arbitrary Python execution)
- Variables must be explicitly whitelisted per template manifest
- Output YAML must validate against AgentConfig or declared schema

---

## Relationships & Workflows

```
User Input (ProjectInitInput)
    ↓
Load TemplateManifest from selected template
    ↓
Validate input against manifest variables
    ↓
For each file in manifest:
    ├─ If template (.j2): Render + Validate
    │   ├─ TemplateRenderer.render_template()
    │   ├─ If YAML: TemplateRenderer.validate_agent_config() [VALIDATION GATE]
    │   └─ Write to disk only if validation passes
    └─ If static: Copy directly to disk
    ↓
Create project directories (if not exist)
    ↓
Return ProjectInitResult with status
    ↓
Display success message or errors to user
```

---

## State Transitions

**Project Lifecycle**:

1. **Pre-init**: Project directory does not exist
2. **Initializing**: Directory created, files being written
3. **Post-init (Success)**: All files created, agent.yaml valid, project ready for use
4. **Post-init (Failure)**: Partial files created OR agent.yaml invalid → cleanup and report error

**Failure Handling**:

- If any template file fails to render: STOP, don't write any files
- If generated agent.yaml fails validation: STOP, don't write files
- If file write fails (permissions): STOP, cleanup created files, report error
- All-or-nothing semantics: either complete success or no files created

---

## Data Constraints

| Entity            | Field            | Type           | Constraint                                                    |
| ----------------- | ---------------- | -------------- | ------------------------------------------------------------- |
| ProjectInitInput  | project_name     | str            | `^[a-zA-Z0-9_-]+$`, length 1-64, no leading digits            |
| ProjectInitInput  | template         | str            | Must match one of: conversational, research, customer-support |
| ProjectInitInput  | author           | str (optional) | Max length 256                                                |
| ProjectInitInput  | description      | str (optional) | Max length 1000                                               |
| TemplateManifest  | version          | str            | Semver format (MAJOR.MINOR.PATCH)                             |
| TemplateManifest  | variables        | dict           | No reserved variable names (e.g., no "\_\_\*")                |
| ProjectInitResult | duration_seconds | float          | Positive number, measured in seconds                          |

---

## Implementation Notes

1. **AgentConfig Reuse**: Do NOT duplicate AgentConfig definition. Import from core models package (001-cli-core-engine). If AgentConfig doesn't exist yet, create it as shared model in `src/holodeck/models/agent_config.py` for reuse.

2. **Validation Order**: Always validate BEFORE writing to disk:

   - Manifest structure validation
   - Input validation
   - Template rendering validation
   - Schema validation (AgentConfig)
   - File write validation (permissions)

3. **Template Variables**: Make template variables explicit and documented (via manifest) so users can understand what values are customizable.

4. **Extensibility**: TemplateManifest design allows future support for:
   - Custom templates (v0.2+)
   - Template composition (v0.3+)
   - Conditional template sections (v0.4+)

Without changing core init logic.
