# Research: Initialize New Agent Project (Phase 0)

**Date**: 2025-10-22
**Status**: Complete - No critical unknowns identified

## Technical Investigation Summary

All technical decisions are well-grounded based on HoloDeck's existing architecture and Python ecosystem best practices. No critical clarifications remained after specification review.

---

## Decision: CLI Framework - Click

**Decision**: Use Click (Pallets) for CLI framework
**Rationale**:
- Click is the de-facto standard for Python CLI tools (used by Flask, Pip, and hundreds of mature projects)
- Excellent type-safe command decorators and parameter validation
- Rich help text and error handling
- Lightweight (minimal dependencies)
- Aligned with Python community standards

**Alternatives Considered**:
- Typer: Newer, uses Python type hints, but adds AsyncIO complexity not needed for init
- Argparse: Python stdlib, but verbose and harder to test
- Invoke: Task-focused, not command-focused; overkill for simple init

**Verdict**: Click provides the right balance of simplicity, maturity, and power for the init command.

---

## Decision: Template Rendering - Jinja2

**Decision**: Use Jinja2 for template rendering with strict schema validation
**Rationale**:
- Jinja2 is the industry standard for Python templating (used by Flask, Django, Ansible)
- Powerful yet safe template syntax (sandboxed by default)
- Supports complex conditional logic for template-specific variations
- Can be configured with strict mode for safety
- Enables DRY principle: store template defaults once in manifest.yaml, apply consistently

**Validation Approach**:
- Each rendered template output is validated against Pydantic AgentConfig model BEFORE file write
- Template manifests define allowed variable substitutions (whitelist approach)
- Jinja2 filters/tests restricted to safe built-ins only
- Validation failure blocks file creation with clear error message

**Alternatives Considered**:
- String replacement/formatting: Simple but error-prone, no validation
- Template inheritance files: Harder to validate and version
- Python code generation: Violates No-Code principle; hard to inspect for users

**Verdict**: Jinja2 with post-render validation provides ideal balance of flexibility and safety.

---

## Decision: Template Distribution - Bundled Resources

**Decision**: Bundle templates as Python package resources (using importlib.resources)
**Rationale**:
- Templates always available (no download/network call required)
- Easier versioning (templates versioned with package)
- Simplified distribution (single pip install provides complete feature)
- Users can't accidentally delete templates

**Alternatives Considered**:
- Download from GitHub releases: Adds network dependency, complexity
- User-provided templates directory: Adds setup friction
- Both bundled + user directory: Extra complexity for v0.1 (deferred to v0.2+)

**Verdict**: Bundled templates satisfy v0.1 MVP requirements; user template support deferred to v0.2+.

---

## Decision: Template Validation Model - TemplateManifest

**Decision**: Create TemplateManifest Pydantic model to validate template structure and metadata
**Rationale**:
- Ensures consistency across templates (name, description, defaults)
- Defines allowed template variables (whitelist for Jinja2)
- Enables future template marketplace without code changes
- Declarative validation (spec, not code)

**Fields** (TBD in design phase):
- name: str (e.g., "conversational")
- description: str
- category: str (for grouping in v0.2+)
- variables: Dict[str, VariableSchema] (allowed template variables with types)
- defaults: Dict[str, Any] (default values for optional variables)

**Verdict**: Manifest model provides extensible, declarative template system aligned with no-code vision.

---

## Decision: Project Configuration - Reuse AgentConfig Model

**Decision**: Generated agent.yaml MUST be valid per AgentConfig Pydantic model (from 001-cli-core-engine)
**Rationale**:
- Single source of truth for agent configuration schema
- Ensures generated projects are immediately usable
- Prevents init from creating invalid configurations
- Maintains consistency across HoloDeck (no duplicate schema definitions)

**Validation Integration**:
- TemplateRenderer calls AgentConfig.model_validate() on rendered YAML
- Validation errors surface immediately with helpful messages
- Failed validation blocks project creation

**Verdict**: Strict schema validation ensures generated projects are always valid.

---

## Decision: File Organization - Functional Grouping

**Decision**: Organize source code by function (cli/, models/, templates/, lib/) not by feature
**Rationale**:
- Clear separation of concerns: CLI handling, data models, templates, template logic
- Easier to reuse components (e.g., template_engine can be used by future deploy command)
- Templates as first-class resources (own directory)
- Aligns with holodeck/src structure for consistency

**Verdict**: Functional organization scales better than feature-based organization.

---

## Decision: Template File Naming - .j2 Extension

**Decision**: Use `.j2` file extension for Jinja2 templates
**Rationale**:
- Industry standard convention (used by Ansible, etc.)
- Signals to users/editors that file contains template syntax
- IDE/editor plugins recognize `.j2` extension
- Differentiates templates from static files (e.g., data/faqs.md vs instructions/system-prompt.md.j2)

**Verdict**: Clear naming convention improves user experience and maintainability.

---

## No Unknowns Remaining

All technical decisions are grounded in:
1. HoloDeck's constitutional principles (No-Code-First, AgentConfig as source of truth)
2. Python ecosystem best practices (Click, Jinja2, Pydantic maturity)
3. v0.1 MVP scope constraints (bundled templates, no custom templates, no network calls)

Next phase: Design data model and contracts based on these decisions.
