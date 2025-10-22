# CLI Contract: holodeck init Command

**Type**: CLI Command Interface
**Command**: `holodeck init <project_name> [OPTIONS]`

---

## Command Signature

```
holodeck init <project_name> [OPTIONS]
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_name` | string | Yes | Name of the new agent project. Must be alphanumeric with hyphens/underscores; no leading digits |

### Options

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--template` | string | No | conversational | Project template: conversational, research, or customer-support |
| `--description` | string | No | (empty) | Description of the agent for agent.yaml |
| `--author` | string | No | (empty) | Name of project creator for agent.yaml |
| `--force` / `--no-force` | boolean | No | False | Overwrite existing project directory without prompting |
| `--help` | boolean | No | False | Show help message and exit |
| `--version` | boolean | No | False | Show HoloDeck version and exit |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success: Project created successfully |
| 1 | Error: General failure (invalid input, permission error, template not found, etc.) |
| 2 | Error: Invalid arguments (incompatible option combinations, missing required arg) |
| 130 | Error: User interrupted (Ctrl+C) during initialization |

---

## Output Examples

### Success Case

```
$ holodeck init my-chatbot --template conversational

✓ Created project 'my-chatbot' from 'conversational' template
  Location: /Users/alice/projects/my-chatbot/

Next steps:
  1. Edit /Users/alice/projects/my-chatbot/agent.yaml to customize your agent
  2. Run: holodeck test my-chatbot/agent.yaml
  3. See examples in my-chatbot/instructions/ and my-chatbot/tools/
```

### Error: Project Already Exists

```
$ holodeck init my-chatbot

✗ Project 'my-chatbot' already exists in current directory
  Use --force to overwrite, or choose a different name
```

### Error: Invalid Name

```
$ holodeck init invalid!name

✗ Invalid project name: 'invalid!name'
  Project names must contain only letters, numbers, hyphens, and underscores
  Must not start with a number

Example: my-chatbot, research_tool, chat-v2
```

### Error: Unknown Template

```
$ holodeck init my-chatbot --template unknown

✗ Unknown template: 'unknown'
  Available templates:
    - conversational: Multi-turn conversation AI assistant
    - research: Research analysis and document processing
    - customer-support: Customer service and issue resolution

Usage: holodeck init <name> --template {conversational|research|customer-support}
```

### Error: Permission Denied

```
$ holodeck init my-chatbot

✗ Permission denied: Cannot write to current directory
  Please run from a directory where you have write permissions
```

---

## Behavior Specification

### Happy Path

1. User runs: `holodeck init my-chatbot --template conversational --description "My first agent"`
2. System validates inputs:
   - project_name matches pattern `^[a-zA-Z0-9_-]+$` (1-64 chars, no leading digit)
   - template is one of (conversational, research, customer-support)
   - description is <= 1000 chars
3. System checks if directory `./my-chatbot` exists:
   - If exists and `--force` not set: prompt user "Overwrite? (y/N)"
   - If user says no: exit with code 0 (user chose to cancel)
   - If exists and `--force` set: remove directory (with warning)
4. System creates project directory `./my-chatbot`
5. System loads conversational template manifest
6. For each file in manifest:
   - Render Jinja2 template with variables: `{project_name: "my-chatbot", description: "My first agent", ...}`
   - If YAML file: validate rendered output against AgentConfig schema
   - If validation fails: **STOP, cleanup created files, show error**
   - If validation passes: write to disk
   - If non-template file: copy directly
7. Create `.gitignore` file with standard Python ignores
8. Display success message with next steps
9. Exit with code 0

### Overwrite Behavior (with --force)

- User runs: `holodeck init existing-project --template research --force`
- System does NOT prompt; immediately removes existing directory
- System creates new project with research template
- Display warning: "Overwritten existing project at ./existing-project"

### Interrupted Behavior (Ctrl+C)

- If user presses Ctrl+C during initialization
- System attempts to cleanup partial files (best effort)
- Display message: "Initialization cancelled"
- Exit with code 130

### Template Variable Substitution

Variables available in templates:

| Variable | Source | Example | Used in |
|----------|--------|---------|---------|
| `project_name` | User input | my-chatbot | all files |
| `description` | --description flag | "My first agent" | agent.yaml, README |
| `author` | --author flag | "Alice" | agent.yaml |
| `template_name` | Template choice | conversational | agent.yaml |
| `holodeck_version` | Package version | 0.1.0 | agent.yaml |
| `created_date` | System time | 2025-10-22 | agent.yaml comments |

Templates can reference: `{{ project_name }}`, `{{ description }}`, etc.

---

## Constraints & Edge Cases

### Directory Creation Constraints
- Projects always created in current working directory
- No `--path` or `--output-dir` option (v0.2+)
- Users navigate via `cd` before running init

### Name Validation
- Rejects: `my project` (spaces), `my.agent` (dots), `_myagent` (leading underscore rejected), `123agent` (leading digit)
- Accepts: `my-agent`, `my_agent`, `MyAgent`, `my-agent-v2`
- Length: 1-64 characters

### Template Bundling
- All templates bundled with package (no download required)
- If template file corrupted: show error "Template corrupted, reinstall HoloDeck"
- Custom templates not supported in v0.1 (deferred to v0.2+)

### Validation Failures
- If rendered agent.yaml fails schema validation: show validation error with line number
- If file write fails (disk full): show error "Insufficient disk space"
- If permission denied: show error "Cannot write to directory; check permissions"

### Performance
- Target: < 30 seconds for complete initialization (SC-001)
- Template rendering: < 1 second
- File I/O: < 5 seconds
- Validation: < 2 seconds

---

## Version Notes

- **v0.1.0**: Initial implementation (this spec)
- **v0.2.0** (planned): Custom templates, interactive wizard, --output-dir flag
- **v0.3.0** (planned): Template composition, conditional sections
