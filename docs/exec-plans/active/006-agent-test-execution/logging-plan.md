## Plan: Implement Best Practice Python Logging for HoloDeck

### Overview

Implement comprehensive, production-ready logging following Python best practices across the test command execution path.

### Key Findings

- **Current State**: Only 4 files have minimal logging (config/loader.py, agent_factory.py, schema.py, merge.py)
- **Critical Gaps**: test.py, executor.py, file_processor.py, evaluators have NO logging
- **No Global Config**: No centralized logging setup or standardization

### Implementation Plan

#### Phase 1: Centralized Logging Infrastructure

1. **Create logging configuration module** (`src/holodeck/lib/logging_config.py`)

   - Structured logging format with timestamps, levels, module names
   - Support for console and file handlers
   - Integration with ExecutionConfig (verbose/quiet modes)
   - Environment variable support (HOLODECK_LOG_LEVEL, HOLODECK_LOG_FILE)
   - Log rotation for production use

2. **Initialize logging in package entry point** (`src/holodeck/__init__.py`)
   - Call setup_logging() on package import
   - Ensure consistent configuration across all modules

#### Phase 2: Add Logging to Critical Paths

3. **Test Command** (`cli/commands/test.py`)

   - Log command invocation with parameters
   - Log config loading success/failure
   - Log test execution start/completion with timing
   - Log report generation and file saving
   - Enhanced error logging for all exception handlers

4. **Test Executor** (`lib/test_runner/executor.py`)

   - Log component initialization (FileProcessor, AgentFactory, Evaluators)
   - Log test loop progress at DEBUG level
   - Log individual test execution with timing
   - Log file processing errors
   - Log agent invocation timeouts and errors
   - Log tool validation results
   - Log evaluation execution and failures
   - Log pass/fail determination with reasoning

5. **File Processor** (`lib/file_processor.py`)

   - Log file type detection (local vs remote)
   - Log large file warnings
   - Log download attempts and retries
   - Log cache hits/misses
   - Log conversion errors

6. **Evaluators** (`lib/evaluators/`)
   - Log evaluator initialization
   - Log metric execution start/completion
   - Log metric errors (currently caught silently)
   - Log threshold comparisons

#### Phase 3: Enhancement & Testing

7. **Enhance existing logging** in agent_factory.py

   - Add timing information to retry logs
   - Add structured context (attempt numbers, delays)

8. **Add logging utilities** for common patterns

   - Context managers for operation timing
   - Structured context helpers (test_id, metric_name, etc.)

9. **Update documentation**
   - Add logging section to README.md
   - Document log levels and environment variables
   - Add troubleshooting guide with log examples

### Log Levels Strategy

- **DEBUG**: Detailed execution flow, variable values, all decisions
- **INFO**: Test progress, component initialization, successful operations
- **WARNING**: Retries, fallbacks, large files, cache issues
- **ERROR**: Failed operations, exceptions, validation errors
- **CRITICAL**: Fatal errors preventing execution

### Configuration Integration

- Respect `--verbose` flag: Set DEBUG level
- Respect `--quiet` flag: Set ERROR level only
- Default: INFO level
- File output optional via env var or config

### Testing Strategy

- Unit tests for logging configuration
- Integration tests verifying logs are generated
- Test log output format and content
- Test verbose/quiet mode integration
