# Quickstart: Initialize New Agent Project

**Goal**: Get developers from "zero to working agent project" in under 30 seconds

---

## For End Users

### Installation

```bash
pip install holodeck
```

### Create Your First Agent Project

```bash
# Create a conversational chatbot (default template)
holodeck init my-chatbot

# Or choose a different template
holodeck init research-tool --template research
holodeck init support-bot --template customer-support

# Add metadata
holodeck init my-agent --description "My AI assistant" --author "Alice"
```

### What You Get

```
my-chatbot/
├── agent.yaml                    # Main agent configuration with embedded test cases
├── instructions/
│   └── system-prompt.md          # Agent system prompt (edit this!)
├── tools/
│   └── README.md                 # How to add custom functions
├── data/
│   └── faqs.md                   # Sample data for vector search
└── .gitignore                    # Standard Python ignores
```

### Next Steps

```bash
# Test your agent
cd my-chatbot
holodeck test agent.yaml

# Chat interactively
holodeck chat agent.yaml

# Deploy locally
holodeck deploy agent.yaml --port 8000
```

---

## For Developers (Implementation Guide)

### Module Overview

**CLI Command** (`src/holodeck/cli/commands/init.py`):
- Entry point: `@click.command('init')`
- Parse arguments (project_name, template, description, author, force)
- Validate inputs
- Call `ProjectInitializer.initialize()` from `project_init.py`
- Format and display result

**Core Logic** (`src/holodeck/cli/utils/project_init.py`):
- `ProjectInitializer` class:
  - `validate_inputs(ProjectInitInput)` → ValidationError
  - `load_template(template_name)` → TemplateManifest
  - `initialize(ProjectInitInput)` → ProjectInitResult
- Returns structured result with success/errors

**Template Rendering** (`src/holodeck/lib/template_engine.py`):
- `TemplateRenderer` class:
  - `render_template(path, variables)` → str
  - `validate_agent_config(yaml_content)` → AgentConfig (VALIDATION GATE)
  - `render_and_validate(path, variables)` → str (safe to write after this)
- Strict validation: render → validate → write

**Data Models** (`src/holodeck/models/`):
- `ProjectInitInput`: Pydantic model for CLI args
- `ProjectInitResult`: Pydantic model for outcome
- `TemplateManifest`: Pydantic model for template metadata
- `AgentConfig`: Imported from holodeck.core (reused, not duplicated)

**Templates** (`src/holodeck/templates/`):
- `conversational/`, `research/`, `customer-support/` directories
- Each template:
  - `manifest.yaml`: Template metadata and variables
  - `agent.yaml.j2`: Jinja2 template for agent config (includes test_cases field)
  - `instructions/system-prompt.md.j2`: System prompt template
  - `tools/README.md.j2`: Tools directory guidance
  - `data/*`: Sample data files
  - `.gitignore`: Standard Python ignores

### Key Implementation Patterns

#### Pattern 1: Template Manifest Validation

```python
# Load and validate template manifest
manifest = TemplateManifest.model_validate(
    yaml.safe_load(open(f'templates/{template_name}/manifest.yaml'))
)

# Manifest automatically validates:
# - Required fields present (name, version, variables)
# - Variable types correct
# - No invalid variable names
# - Version is valid semver
```

#### Pattern 2: Template Rendering with Strict Validation

```python
# Render Jinja2 template
template = env.get_template('agent.yaml.j2')
rendered_yaml = template.render(variables)

# CRITICAL: Validate rendered YAML against schema BEFORE writing
config = AgentConfig.model_validate(
    yaml.safe_load(rendered_yaml)
)

# Only now is it safe to write to disk
with open(f'{project_dir}/agent.yaml', 'w') as f:
    f.write(rendered_yaml)
```

#### Pattern 3: All-or-Nothing Project Creation

```python
try:
    # Create directory
    project_dir.mkdir(parents=True, exist_ok=False)

    # Render and validate each file
    for file_spec in manifest.files:
        content = render_and_validate(file_spec, variables)
        (project_dir / file_spec.path).write_text(content)

    # Success!
    return ProjectInitResult(success=True, ...)

except Exception as e:
    # CLEANUP: Remove partially created directory
    shutil.rmtree(project_dir)
    return ProjectInitResult(success=False, errors=[str(e)])
```

#### Pattern 4: Input Validation

```python
# Validate project name
if not re.match(r'^[a-zA-Z0-9_-]+$', project_name):
    raise ValidationError(f"Invalid project name: {project_name}")

# Validate template exists
if template not in AVAILABLE_TEMPLATES:
    raise ValidationError(f"Unknown template: {template}")

# Validate directory doesn't exist (unless force=True)
if project_dir.exists() and not force:
    raise ValidationError(f"Directory already exists: {project_dir}")
```

### File Structure for Templates

Each template directory must contain `manifest.yaml`:

```yaml
name: conversational
display_name: Conversational Agent
description: Multi-turn AI conversation assistant
category: conversational-ai
version: 1.0.0

variables:
  project_name:
    type: string
    required: true
    description: "Name of the agent project"
  description:
    type: string
    required: false
    default: "TODO: Add agent description"
  author:
    type: string
    required: false
    default: ""
  model_temperature:
    type: number
    default: 0.7

defaults:
  model.provider: "openai"
  model.name: "gpt-4o"

files:
  agent.yaml:
    path: agent.yaml
    template: true
    required: true
  instructions/system-prompt.md:
    path: instructions/system-prompt.md
    template: true
    required: true
  tools/README.md:
    path: tools/README.md
    template: true
    required: true
  data/faqs.md:
    path: data/faqs.md
    template: false
    required: false
  .gitignore:
    path: .gitignore
    template: false
    required: true
```

### Testing Strategy

**Unit Tests** (`tests/unit/`):
- `test_project_init.py`: ProjectInitializer logic (isolation, mocking filesystem)
- `test_cli_init_command.py`: CLI argument parsing and error handling
- `test_template_engine.py`: Jinja2 rendering and validation

**Integration Tests** (`tests/integration/`):
- `test_init_templates.py`: Full flow (render → validate → create for each template)
- `test_agent_config_compliance.py`: Verify generated agent.yaml matches schema

**Test Fixtures** (`tests/fixtures/`):
- `generated_projects/`: Temporary directories for integration tests
- Template manifests and example variables for testing

**Coverage Target**: 80% minimum (per Constitution)

### Error Handling

| Scenario | Error Message | Exit Code | Cleanup |
|----------|---------------|-----------|---------|
| Invalid name | "Invalid project name: 'bad!name'" | 1 | None |
| Template not found | "Unknown template: 'badtemplate'" | 1 | None |
| Dir exists (no force) | "Project 'my-agent' already exists" | 1 | None |
| Permission denied | "Cannot write to directory; check permissions" | 1 | Remove partial dir |
| Disk full | "Insufficient disk space" | 1 | Remove partial dir |
| Invalid agent.yaml | "agent.yaml validation error: [details]" | 1 | Remove full dir |
| Ctrl+C | "Initialization cancelled" | 130 | Remove partial dir |

---

## Integration Points

### Dependency on 001-cli-core-engine

- **Reuses**: `AgentConfig` Pydantic model (schema source of truth)
- **Depends on**: `holodeck` CLI command structure (Click setup)
- **Provides to**: Generated agent.yaml files that feed into Agent Engine

### Future Integration (v0.2+)

- Custom templates: Template registry/marketplace
- Interactive mode: Prompts for customization
- Git integration: Auto-initialize git repos
- Environment setup: Auto-install dependencies

---

## Success Metrics

✓ **SC-001**: < 30 seconds initialization time
✓ **SC-002**: Generated agent.yaml is parseable (0 validation errors)
✓ **SC-003**: All template files present and formatted
✓ **SC-004**: 80%+ of first-time users succeed without errors
✓ **SC-005**: Users find customization patterns within 2 minutes
✓ **SC-006**: Sample test cases validate perfectly against schema
✓ **SC-007**: Can run `holodeck test` immediately after init

---

## Troubleshooting

### "Permission denied: Cannot write to current directory"
- Solution: Run `cd` to a directory you own or have write permissions for
- Check: `ls -ld .` to see directory permissions

### "Project 'my-agent' already exists"
- Solution 1: Use `holodeck init my-agent-v2` (different name)
- Solution 2: Use `holodeck init my-agent --force` to overwrite

### "Unknown template: 'mytemplate'"
- Solution: Use one of: `conversational`, `research`, `customer-support`
- List: `holodeck init --help` to see available templates

### Generated agent.yaml fails validation
- This should not happen if template is valid
- Solution: Reinstall HoloDeck: `pip install --upgrade holodeck`
- Report bug with template name if issue persists

---

## Architecture Diagram

```
User: holodeck init my-chatbot --template conversational
   ↓
CLI Command (init.py)
   ├─ Validate: name, template, permissions
   ├─ Parse: ProjectInitInput
   └─ Call: ProjectInitializer.initialize()
   ↓
ProjectInitializer (project_init.py)
   ├─ Load: TemplateManifest from conversational/manifest.yaml
   ├─ For each file in manifest:
   │  └─ Render + Validate via TemplateRenderer
   ├─ Create: project directory + files
   └─ Return: ProjectInitResult
   ↓
TemplateRenderer (template_engine.py)
   ├─ Render: agent.yaml.j2 with variables
   ├─ Validate: rendered YAML against AgentConfig [GATE]
   └─ Return: validated content
   ↓
Result
   ├─ Success → Display next steps
   └─ Error → Show validation errors + cleanup
```

This architecture ensures:
- ✅ Template flexibility (Jinja2)
- ✅ Strict validation (AgentConfig model)
- ✅ User-friendly errors (clear messages)
- ✅ Safety (all-or-nothing semantics)
