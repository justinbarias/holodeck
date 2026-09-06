# CLI API Contract: `holodeck test`

**Date**: 2025-11-01
**Feature**: Execute Agent Against Test Cases
**Phase**: 1 - Design

## Overview

This document defines the command-line interface for the `holodeck test` command. The CLI follows Click framework patterns and integrates with the existing `holodeck` CLI structure.

**Key Design Principle**: All CLI options can be configured in agent.yaml via the `execution` section (`ExecutionConfig` model). CLI flags override agent.yaml settings, which override global defaults.

## Configuration Hierarchy

```
CLI Flags  >  agent.yaml execution  >  ~/.holodeck/config.yaml  >  Built-in Defaults
(highest)                                                          (lowest)
```

---

## ExecutionConfig Model

**Location**: `src/holodeck/models/config.py` (NEW)

The `ExecutionConfig` model defines test execution settings that can be specified in agent.yaml:

```python
class ExecutionConfig(BaseModel):
    """Test execution configuration.

    Defines timeouts, caching, and output settings for test execution.
    All fields are optional - unspecified values use defaults.
    """

    model_config = ConfigDict(extra="forbid")

    # Timeout settings
    file_timeout: int | None = Field(
        None,
        description="Timeout for file processing in seconds (1-300)",
        ge=1,
        le=300
    )
    llm_timeout: int | None = Field(
        None,
        description="Timeout for LLM API calls in seconds (1-600)",
        ge=1,
        le=600
    )
    download_timeout: int | None = Field(
        None,
        description="Timeout for remote file downloads in seconds (1-300)",
        ge=1,
        le=300
    )

    # Cache settings
    cache_enabled: bool | None = Field(
        None,
        description="Enable file caching for remote URLs (default: true)"
    )
    cache_dir: str | None = Field(
        None,
        description="Cache directory path (default: .holodeck/cache)"
    )

    # Output settings
    verbose: bool | None = Field(
        None,
        description="Enable verbose output (default: false)"
    )
    quiet: bool | None = Field(
        None,
        description="Suppress progress indicators (default: false)"
    )
```

### Usage in agent.yaml

```yaml
name: Customer Support Agent
model:
  provider: azure_openai
  name: gpt-4o
instructions:
  file: instructions/system.md
tools: []
evaluations:
  metrics:
    - metric: groundedness
      threshold: 0.7
test_cases:
  - name: Business hours query
    input: "What are your business hours?"
    ground_truth: "We're open Monday-Friday 9AM-5PM EST"

# NEW: Execution configuration
execution:
  file_timeout: 60 # Allow 60s for large file processing
  llm_timeout: 120 # Allow 120s for complex LLM calls
  download_timeout: 45 # Allow 45s for remote file downloads
  cache_enabled: true # Enable caching (default)
  cache_dir: .cache # Custom cache directory
  verbose: false # Standard output (default)
```

---

## Command Signature

```bash
holodeck test [OPTIONS] AGENT_CONFIG
```

### Arguments

**AGENT_CONFIG** (required)

- **Type**: File path
- **Description**: Path to agent.yaml configuration file
- **Validation**:
  - File must exist
  - File must be valid YAML
  - File must pass schema validation (Agent model with ExecutionConfig)
- **Example**: `agent.yaml`, `./configs/customer-support.yaml`

### Options

All CLI options override corresponding `execution` settings in agent.yaml.

#### Output Options

**`--output`, `-o` <file_path>**

- **Type**: File path
- **Default**: None (results printed to stdout)
- **Description**: Save test report to file (format auto-detected from extension)
- **Supported Extensions**: `.json`, `.md`
- **Agent Config**: Not configurable (CLI-only option)
- **Example**: `--output report.json`, `-o results.md`

**`--format`, `-f` <format>**

- **Type**: Choice [`json`, `markdown`]
- **Default**: Inferred from `--output` extension, or `json` if no output file
- **Description**: Report format (overrides extension-based detection)
- **Agent Config**: Not configurable (CLI-only option)
- **Example**: `--format markdown`

#### Execution Options

**`--verbose`, `-v`**

- **Type**: Flag
- **Default**: False
- **Description**: Enable verbose output with detailed stack traces and debug info
- **Agent Config Mapping**: `execution.verbose`
- **Override Behavior**: CLI flag overrides agent.yaml setting
- **Example**: `--verbose`

**`--quiet`, `-q`**

- **Type**: Flag
- **Default**: False
- **Description**: Suppress progress indicators (for CI/CD environments)
- **Mutually Exclusive**: Cannot use with `--verbose`
- **Agent Config Mapping**: `execution.quiet`
- **Override Behavior**: CLI flag overrides agent.yaml setting
- **Example**: `--quiet`

#### Timeout Options

**`--file-timeout` <seconds>**

- **Type**: Integer (1-300)
- **Default**: 30
- **Description**: Timeout for file processing operations (markitdown) in seconds
- **Agent Config Mapping**: `execution.file_timeout`
- **Override Behavior**: CLI option overrides agent.yaml value
- **Example**: `--file-timeout 60`

**`--llm-timeout` <seconds>**

- **Type**: Integer (1-600)
- **Default**: 60
- **Description**: Timeout for LLM API calls (agent execution and evaluation) in seconds
- **Agent Config Mapping**: `execution.llm_timeout`
- **Override Behavior**: CLI option overrides agent.yaml value
- **Example**: `--llm-timeout 120`

**`--download-timeout` <seconds>**

- **Type**: Integer (1-300)
- **Default**: 30
- **Description**: Timeout for remote file downloads in seconds
- **Agent Config Mapping**: `execution.download_timeout`
- **Override Behavior**: CLI option overrides agent.yaml value
- **Example**: `--download-timeout 45`

#### Filtering Options

**`--test`, `-t` <test_name>**

- **Type**: String (can be specified multiple times)
- **Default**: None (run all tests)
- **Description**: Run only tests matching the specified name(s)
- **Agent Config**: Not configurable (CLI-only option for ad-hoc filtering)
- **Example**: `--test "Business hours query" --test "Order status"`

**`--filter` <pattern>**

- **Type**: String (glob pattern)
- **Default**: None (run all tests)
- **Description**: Run only tests matching glob pattern
- **Agent Config**: Not configurable (CLI-only option for ad-hoc filtering)
- **Example**: `--filter "customer_*"`, `--filter "*_error"`

#### Cache Options

**`--no-cache`**

- **Type**: Flag
- **Default**: False (caching enabled)
- **Description**: Disable file caching for remote URLs (always re-download)
- **Agent Config Mapping**: Sets `execution.cache_enabled = false`
- **Override Behavior**: CLI flag overrides agent.yaml setting
- **Example**: `--no-cache`

**`--clear-cache`**

- **Type**: Flag
- **Default**: False
- **Description**: Clear file cache before running tests
- **Agent Config**: Not configurable (CLI-only action)
- **Example**: `--clear-cache`

**`--cache-dir` <directory>**

- **Type**: Directory path
- **Default**: `.holodeck/cache`
- **Description**: Custom cache directory for remote files
- **Agent Config Mapping**: `execution.cache_dir`
- **Override Behavior**: CLI option overrides agent.yaml value
- **Example**: `--cache-dir /tmp/holodeck-cache`

---

## Configuration Resolution Examples

### Example 1: CLI Override

**agent.yaml**:

```yaml
execution:
  file_timeout: 30
  llm_timeout: 60
```

**Command**:

```bash
holodeck test agent.yaml --llm-timeout 120
```

**Resolved Configuration**:

- `file_timeout`: 30 (from agent.yaml)
- `llm_timeout`: 120 (CLI override)
- `download_timeout`: 30 (built-in default)

---

### Example 2: All from agent.yaml

**agent.yaml**:

```yaml
execution:
  file_timeout: 60
  llm_timeout: 120
  download_timeout: 45
  cache_enabled: true
  verbose: true
```

**Command**:

```bash
holodeck test agent.yaml
```

**Resolved Configuration**:

- All settings from agent.yaml
- No CLI overrides

---

### Example 3: Built-in Defaults

**agent.yaml**:

```yaml
# No execution section
```

**Command**:

```bash
holodeck test agent.yaml
```

**Resolved Configuration**:

- `file_timeout`: 30 (built-in default)
- `llm_timeout`: 60 (built-in default)
- `download_timeout`: 30 (built-in default)
- `cache_enabled`: true (built-in default)
- `verbose`: false (built-in default)

---

### Example 4: Global Config Fallback (Future)

**~/.holodeck/config.yaml** (global settings):

```yaml
execution:
  llm_timeout: 90
  cache_dir: ~/.holodeck/global-cache
```

**agent.yaml**:

```yaml
execution:
  file_timeout: 45
```

**Command**:

```bash
holodeck test agent.yaml
```

**Resolved Configuration**:

- `file_timeout`: 45 (from agent.yaml)
- `llm_timeout`: 90 (from global config)
- `cache_dir`: ~/.holodeck/global-cache (from global config)
- `download_timeout`: 30 (built-in default)

---

## Exit Codes

| Code | Meaning             | Description                                                          |
| ---- | ------------------- | -------------------------------------------------------------------- |
| 0    | Success             | All tests passed                                                     |
| 1    | Test Failure        | One or more tests failed (metric threshold not met or tool mismatch) |
| 2    | Configuration Error | Invalid agent.yaml or configuration validation failed                |
| 3    | Execution Error     | Error during test execution (agent failure, timeout, etc.)           |
| 4    | Evaluation Error    | All evaluation metrics failed for one or more tests                  |

---

## Output Formats

### Standard Output (Default)

When no `--output` is specified, results are printed to stdout with progress indicators:

```
🧪 Running HoloDeck Tests...

✅ Test 1/10: What are your business hours? [PASSED] (3.5s)
   Groundedness: 0.95 / 0.70 ✅
   Relevance: 0.82 / 0.70 ✅

❌ Test 2/10: Where is my order #12345? [FAILED] (4.2s)
   Groundedness: 0.65 / 0.70 ❌ (score below threshold)
   Relevance: 0.88 / 0.70 ✅

...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary: 8/10 tests passed (80% pass rate)
Duration: 45.2s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Quiet Mode Output (CI/CD)

When `--quiet` is specified or `execution.quiet: true` in agent.yaml:

```
Test 1/10: PASSED (3.5s)
Test 2/10: FAILED (4.2s)
...
Summary: 8/10 passed (80%)
```

### Verbose Mode Output

When `--verbose` is specified or `execution.verbose: true` in agent.yaml:

```
DEBUG: Loading agent config from agent.yaml
DEBUG: Execution config: file_timeout=30s, llm_timeout=60s, cache_enabled=true
DEBUG: Found 10 test cases
DEBUG: Processing file: sample.pdf (1.2 MB)
DEBUG: markitdown extraction took 1.5s
...
✅ Test 1/10: What are your business hours? [PASSED] (3.5s)
   DEBUG: Agent response: "We're open Monday-Friday 9AM-5PM EST"
   DEBUG: Tool calls: ['get_hours']
   DEBUG: Expected tools: ['get_hours'] ✅ matched
   Groundedness: 0.95 / 0.70 ✅
     DEBUG: Evaluated with gpt-4o in 1.2s (retry_count=0)
   Relevance: 0.82 / 0.70 ✅
     DEBUG: Evaluated with gpt-4o-mini in 0.8s (retry_count=0)
...
```

### JSON Output

```json
{
  "agent_name": "Customer Support Agent",
  "agent_config_path": "./agent.yaml",
  "execution_config": {
    "file_timeout": 60,
    "llm_timeout": 120,
    "download_timeout": 30,
    "cache_enabled": true,
    "cache_dir": ".holodeck/cache"
  },
  "results": [...],
  "summary": {...}
}
```

---

## Error Handling

### Configuration Errors

**Invalid ExecutionConfig**:

```
ERROR: Configuration validation failed
  Cause: execution.file_timeout must be between 1 and 300 (got: 500)
  File: agent.yaml:25
  Suggestion: Set file_timeout to a value between 1 and 300 seconds
```

**Conflicting Settings**:

```
ERROR: Configuration validation failed
  Cause: execution.verbose and execution.quiet cannot both be true
  File: agent.yaml:30
  Suggestion: Choose either verbose or quiet mode, not both
```

---

## Usage Examples

### Example 1: Use agent.yaml Settings

agent.yaml includes execution configuration:

```yaml
execution:
  file_timeout: 60
  llm_timeout: 120
  verbose: true
```

Run tests with these settings:

```bash
holodeck test agent.yaml
```

### Example 2: Override with CLI Flags

Override specific settings from agent.yaml:

```bash
holodeck test agent.yaml --llm-timeout 180 --quiet
```

Result: Uses `file_timeout: 60` from agent.yaml, but overrides `llm_timeout` to 180 and enables quiet mode (overriding `verbose: true`)

### Example 3: No Execution Config in agent.yaml

agent.yaml has no execution section - use built-in defaults:

```bash
holodeck test agent.yaml
```

Result: Uses all built-in defaults (file_timeout=30, llm_timeout=60, etc.)

### Example 4: Custom Cache Directory

Configure custom cache in agent.yaml:

```yaml
execution:
  cache_dir: /tmp/my-cache
```

Or override via CLI:

```bash
holodeck test agent.yaml --cache-dir /tmp/my-cache
```

---

## Model Integration

### Agent Model Update

The `Agent` model in `src/holodeck/models/agent.py` will be updated to include ExecutionConfig:

```python
from holodeck.models.config import ExecutionConfig

class Agent(BaseModel):
    """Agent configuration entity."""

    name: str
    model: LLMProvider
    instructions: Instructions
    tools: list[Any] | None = None
    evaluations: EvaluationConfig | None = None
    test_cases: list[TestCaseModel] | None = None

    # NEW: Execution configuration
    execution: ExecutionConfig | None = Field(
        None,
        description="Test execution configuration (timeouts, caching, output)"
    )
```

### Configuration Loader Integration

The test command will merge configurations in priority order:

```python
def load_execution_config(agent: Agent, cli_args: dict) -> ExecutionConfig:
    """Merge execution config from agent.yaml and CLI args.

    Priority: CLI args > agent.yaml > built-in defaults
    """
    # Start with built-in defaults
    config = ExecutionConfig(
        file_timeout=30,
        llm_timeout=60,
        download_timeout=30,
        cache_enabled=True,
        cache_dir=".holodeck/cache",
        verbose=False,
        quiet=False
    )

    # Apply agent.yaml execution settings
    if agent.execution:
        config = config.model_copy(update=agent.execution.model_dump(exclude_none=True))

    # Apply CLI overrides
    cli_overrides = {k: v for k, v in cli_args.items() if v is not None}
    config = config.model_copy(update=cli_overrides)

    return config
```

---

## Environment Variables

The following environment variables can be used as defaults (lowest priority):

| Variable                    | Default           | Description                               |
| --------------------------- | ----------------- | ----------------------------------------- |
| `HOLODECK_FILE_TIMEOUT`     | 30                | Default file processing timeout (seconds) |
| `HOLODECK_LLM_TIMEOUT`      | 60                | Default LLM API timeout (seconds)         |
| `HOLODECK_DOWNLOAD_TIMEOUT` | 30                | Default file download timeout (seconds)   |
| `HOLODECK_CACHE_DIR`        | `.holodeck/cache` | Cache directory for remote files          |
| `HOLODECK_VERBOSE`          | `false`           | Enable verbose output by default          |

**Priority**: CLI options > agent.yaml > environment variables > built-in defaults

---

## TTY Detection

The CLI automatically detects TTY environments and adjusts output:

**TTY (Interactive Terminal)**:

- Progress indicators with spinners
- Colored output (✅ green, ❌ red)
- Real-time status updates

**Non-TTY (CI/CD Pipelines)**:

- Plain text output (no spinners)
- No ANSI color codes
- Line-buffered output for log parsing

Detection logic:

```python
import sys

is_tty = sys.stdout.isatty()
```

---

## Compatibility

### Python Version

- Minimum: Python 3.10+

### Operating Systems

- Linux (Ubuntu 20.04+, RHEL 8+)
- macOS (11.0+)
- Windows (10+, with WSL recommended)

---

## Summary: Agent Config vs CLI Options

| Setting                 | Agent Config Field           | CLI Option             | Default         |
| ----------------------- | ---------------------------- | ---------------------- | --------------- |
| File processing timeout | `execution.file_timeout`     | `--file-timeout`       | 30s             |
| LLM API timeout         | `execution.llm_timeout`      | `--llm-timeout`        | 60s             |
| Download timeout        | `execution.download_timeout` | `--download-timeout`   | 30s             |
| Enable caching          | `execution.cache_enabled`    | `--no-cache` (inverts) | true            |
| Cache directory         | `execution.cache_dir`        | `--cache-dir`          | .holodeck/cache |
| Verbose output          | `execution.verbose`          | `--verbose`            | false           |
| Quiet mode              | `execution.quiet`            | `--quiet`              | false           |
| Output file             | N/A                          | `--output`             | stdout          |
| Report format           | N/A                          | `--format`             | json            |
| Test filter             | N/A                          | `--test`, `--filter`   | all tests       |
| Clear cache             | N/A                          | `--clear-cache`        | false           |

**Key Principle**: Configuration is declarative in agent.yaml, CLI flags provide runtime overrides.
