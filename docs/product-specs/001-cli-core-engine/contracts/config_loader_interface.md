# ConfigLoader Interface Contract

**User Story**: US1 - Define Agent Configuration
**Interface**: `holodeck.config.loader.ConfigLoader`

## Purpose

Core interface for loading and validating agent.yaml files. This is the primary contract between:

- YAML file on disk
- Validated Python Agent object
- Error handling system

## Method Contracts

### `ConfigLoader.load(file_path: str) -> Agent`

**Purpose**: Load and validate an agent.yaml file.

**Parameters**:

- `file_path` (str): Absolute path to agent.yaml file

**Returns**:

- `Agent`: Validated agent configuration object

**Raises**:

- `ConfigFileNotFoundError`: If file_path doesn't exist
- `InvalidYAMLError`: If YAML syntax is malformed
- `ValidationError`: If configuration fails schema validation
- `EnvironmentVariableError`: If environment variable can't be resolved

**Example**:

```python
loader = ConfigLoader()
agent = loader.load("/path/to/agent.yaml")
# agent is fully validated and ready to use
```

**Acceptance Criteria**:

- ✅ Loads valid agent.yaml without errors
- ✅ Raises clear error if file doesn't exist
- ✅ Raises clear error if YAML is malformed
- ✅ Raises clear error if schema validation fails
- ✅ Returns Agent object with all fields populated

---

### `ConfigLoader.load_string(yaml_content: str, base_path: str = None) -> Agent`

**Purpose**: Load agent configuration from a YAML string (useful for testing and programmatic config).

**Parameters**:

- `yaml_content` (str): YAML configuration as string
- `base_path` (str, optional): Base directory for relative file references

**Returns**:

- `Agent`: Validated agent configuration object

**Raises**: Same as `load()`

**Acceptance Criteria**:

- ✅ Parses YAML string correctly
- ✅ Uses base_path for file resolution if provided
- ✅ Falls back to current directory if base_path not provided

---

### `ConfigLoader.validate(agent: Agent) -> ValidationResult`

**Purpose**: Validate an already-loaded Agent object (useful after programmatic construction).

**Parameters**:

- `agent` (Agent): Agent object to validate

**Returns**:

- `ValidationResult`:
  - `is_valid` (bool): True if valid
  - `errors` (List[str]): Human-readable error messages

**Acceptance Criteria**:

- ✅ Returns ValidationResult with is_valid flag
- ✅ Error messages are actionable
- ✅ All validation rules from data-model.md are checked

---

## Error Handling Contract

All ConfigLoader errors MUST:

1. **Be actionable**: Include enough info for developer to fix
2. **Be human-readable**: Translate Pydantic errors to plain English
3. **Include file references**: Show which field in agent.yaml caused error
4. **Suggest fixes**: Where applicable (e.g., "model provider 'foo' not supported; use 'openai', 'azure_openai', or 'anthropic'")

### Error Message Format

```
[ERROR] {severity}: {problem}
  Location: {file}:{line} (or "agent.yaml" if line not applicable)
  Field: {yaml_path} (e.g., "tools[0].file")
  Details: {explanation}
  Suggestion: {how_to_fix}
```

### Example Error Messages

✅ **GOOD**:

```
[ERROR] Missing required field: instructions
  Location: agent.yaml
  Field: instructions
  Details: Either 'file' or 'inline' instruction must be provided
  Suggestion: Add either:
    instructions:
      file: instructions/system-prompt.md
    OR
    instructions:
      inline: "You are a helpful assistant..."
```

✅ **GOOD**:

```
[ERROR] Invalid tool configuration: vectorstore tool 'search_docs' references missing file
  Location: agent.yaml:15
  Field: tools[0].source
  Details: File does not exist: data/documents.txt
  Suggestion: Create file at data/documents.txt or update 'source' to correct path
```

❌ **BAD**:

```
[ERROR] Validation error in __root__
```

---

## Loading Process Contract

ConfigLoader MUST follow this process:

1. **Read file** (or parse string)

   - File must exist and be readable
   - YAML must be syntactically valid

2. **Parse YAML** → Dict

   - Use PyYAML or equivalent

3. **Resolve environment variables**

   - Pattern: `${VAR_NAME}` or `$VAR_NAME`
   - Replace with environment variable value
   - Raise error if variable not found

4. **Validate against schema** (Pydantic)

   - Check types
   - Check constraints
   - Check relationships

5. **Validate file references**

   - Resolve relative paths from agent.yaml directory
   - Check files exist
   - Raise error with suggested fixes if missing

6. **Return validated Agent object**
   - All fields set and validated
   - Ready for agent execution

---

## Testing Expectations

ConfigLoader tests MUST cover:

- ✅ Valid agent.yaml → successful load
- ✅ Missing file → clear error
- ✅ Invalid YAML → clear error
- ✅ Missing required fields → clear errors
- ✅ Invalid field values → clear errors with suggestions
- ✅ File references (instructions, tools, data) → validation
- ✅ Environment variables → interpolation and error handling
- ✅ Tool validation (type-specific rules) → comprehensive
- ✅ Relative path resolution → correct directory context

---

## Performance Contract

- Load time: < 100ms for typical agent.yaml (< 50 tools, < 100 test cases)
- Measured in tests with `pytest-benchmark` or equivalent
- Must be synchronous (no async operations)
