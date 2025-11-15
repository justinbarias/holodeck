# Utilities and Support API

HoloDeck provides several utility modules for logging, file processing, template rendering,
and error handling across the platform.

## Error Hierarchy

HoloDeck defines a custom exception hierarchy for consistent error handling.

::: holodeck.lib.errors.HoloDeckError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.errors.ConfigError
    options:
      docstring_style: google

::: holodeck.lib.errors.ValidationError
    options:
      docstring_style: google

::: holodeck.lib.errors.FileNotFoundError
    options:
      docstring_style: google

## Logging

Configure and manage HoloDeck's logging system with automatic initialization.

::: holodeck.lib.logging_config.setup_logging
    options:
      docstring_style: google

::: holodeck.lib.logging_config.get_logger
    options:
      docstring_style: google

### Logging Utilities

Utilities for error handling and retries with logging.

::: holodeck.lib.logging_utils.log_and_raise
    options:
      docstring_style: google

::: holodeck.lib.logging_utils.retry_with_backoff
    options:
      docstring_style: google

## Template Engine

Jinja2-based template rendering for dynamic configuration and instruction generation.

::: holodeck.lib.template_engine.TemplateEngine
    options:
      docstring_style: google
      show_source: true
      members:
        - render
        - render_file
        - render_from_string

## File Processing

Multimodal file handling for images, PDFs, Office documents, and data files.

::: holodeck.lib.file_processor.FileProcessor
    options:
      docstring_style: google
      show_source: true
      members:
        - process_file
        - extract_text
        - load_image

### Supported File Types

- **Images**: JPG, PNG with OCR support via Tesseract
- **Documents**:
  - PDF: Full documents or specific page ranges
  - Word (.docx): Text extraction
  - PowerPoint (.pptx): Slide text and notes extraction
- **Data**:
  - Excel (.xlsx): Sheet selection with row/column ranges
  - CSV: Full or partial content
  - Text (.txt): Raw content
- **Remote Files**: URL-based inputs with caching

## CLI Utilities

Utilities for project scaffolding and CLI operations.

::: holodeck.cli.utils.project_init.initialize_project
    options:
      docstring_style: google

::: holodeck.cli.utils.project_init.create_project_structure
    options:
      docstring_style: google

## CLI Exceptions

CLI-specific exception handling.

::: holodeck.cli.exceptions.CLIError
    options:
      docstring_style: google

::: holodeck.cli.exceptions.ProjectInitError
    options:
      docstring_style: google

## Usage Examples

### Logging Setup

```python
from holodeck import setup_logging

# Logging is automatically initialized on import
# Get a logger for your module
import logging
logger = logging.getLogger(__name__)

logger.info("Application started")
logger.error("An error occurred")
```

### Template Rendering

```python
from holodeck.lib.template_engine import TemplateEngine

engine = TemplateEngine()

# Render inline template
result = engine.render_from_string(
    "Hello {{ name }}!",
    {"name": "Alice"}
)

# Render from file
result = engine.render_file(
    "instructions/system_prompt.jinja2",
    {"agent_name": "Assistant"}
)
```

### File Processing

```python
from holodeck.lib.file_processor import FileProcessor

processor = FileProcessor()

# Process an image
image_data = processor.process_file("path/to/image.jpg")

# Extract text from PDF (pages 1-3)
text = processor.extract_text(
    "path/to/document.pdf",
    pages=(1, 3)
)

# Load Excel data
data = processor.load_data(
    "path/to/spreadsheet.xlsx",
    sheet="Sheet1",
    range="A1:D10"
)
```

### Error Handling

```python
from holodeck.lib.errors import ConfigError, ValidationError

try:
    # Configuration loading
    config = load_config("agent.yaml")
except ConfigError as e:
    print(f"Configuration error: {e}")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Related Documentation

- [Configuration Loading](config-loader.md): Using template engine with configs
- [Test Runner](test-runner.md): File processing in test execution
- [CLI Commands](../user-guide/cli.md): Project initialization
