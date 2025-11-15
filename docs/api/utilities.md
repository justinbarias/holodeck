# Utilities and Support API

HoloDeck provides several utility modules for template rendering and error handling.

## Template Engine

Jinja2-based template rendering for dynamic configuration and instruction generation.

::: holodeck.lib.template_engine.TemplateRenderer
    options:
      docstring_style: google
      show_source: true

## Usage Examples

### Template Rendering

```python
from holodeck.lib.template_engine import TemplateRenderer

renderer = TemplateRenderer()

# Render inline template
result = renderer.render_template(
    "template_string",
    {"name": "Alice"}
)
```

## Related Documentation

- [Configuration Loading](config-loader.md): Using template engine with configs
- [CLI Commands](cli.md): Project initialization and CLI API
