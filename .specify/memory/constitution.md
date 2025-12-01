<!-- SYNC IMPACT REPORT
Version Change: 1.0.0 → 1.0.1 (PATCH - Python target version clarification)

Principles: No changes to core principles - all 5 principles fully aligned with VISION.md and CLAUDE.md
Modified Sections:
  - Code Quality & Testing Discipline: Python target confirmed as 3.10+ (explicitly stated in CLAUDE.md)

Added Sections: None
Removed Sections: None

Templates Updated:
  ✅ plan-template.md - Constitution Check section references verified (no changes needed)
  ✅ spec-template.md - No constitution-dependent sections (no changes needed)
  ✅ tasks-template.md - No constitution-dependent sections (no changes needed)
  ✅ agent-file-template.md - Not checked (MCP-focused integration doc)

Follow-up TODOs: None - constitution now fully synchronized with VISION.md and CLAUDE.md
-->

# HoloDeck Constitution

## Core Principles

### I. No-Code-First Agent Definition

Every agent configuration MUST be defined through declarative YAML files.
Users SHOULD NOT write Python code to define agents, tools, evaluations, or
test cases. All agent behavior specification MUST be expressible through
YAML configuration without custom code implementation.

**Rationale**: Lowering barriers to entry and enabling non-technical users
to build production agents. This principle drives the entire HoloDeck design
and differentiates the platform from code-first frameworks.

### II. MCP for API Integrations

External API integrations MUST use Model Context Protocol (MCP) servers,
not custom API tool types. When a new API integration is needed, the team
MUST either (1) use an existing MCP server, (2) create a new MCP server,
or (3) document why MCP is unsuitable for this specific case.

**Rationale**: MCP provides standardized, interoperable tool integration.
Building custom API tools creates maintenance burden and prevents code
reuse across projects. MCP-first design ensures agents remain portable and
composable.

### III. Test-First with Multimodal Support

All new agent behaviors MUST be validated through test cases BEFORE
deployment. Test cases MUST support multimodal inputs (images, PDFs,
Office documents, text, CSV, URLs) and SHOULD validate agent tool
selection via `expected_tools`. Each test case MUST include either ground
truth data or expected evaluation scores for validation.

**Rationale**: Multimodal test support is core to HoloDeck's value
proposition—agents must handle real-world document-heavy workflows.
Test-first approach ensures quality and provides regression detection.

### IV. OpenTelemetry-Native Observability

Observability MUST follow OpenTelemetry semantic conventions for
generative AI (GenAI Semantic Conventions) from day one. Every LLM call,
tool invocation, and evaluation MUST be automatically instrumented with
traces, metrics, and logs. Cost tracking and alerting MUST be built-in,
not bolted-on.

**Rationale**: Observability-as-a-default prevents production surprises
and enables data-driven optimization. Following GenAI semantic conventions
ensures compatibility with industry observability platforms (Jaeger,
Datadog, Honeycomb, LangSmith, Prometheus).

### V. Evaluation Flexibility with Model Overrides

Evaluations MUST support model configuration at three levels: (1) global
default model for all metrics, (2) per-evaluation-run model override, and
(3) per-metric model override (e.g., GPT-4o for critical groundedness
checks, GPT-4o-mini for others). AI-powered metrics MUST follow Azure AI
Evaluation patterns. NLP metrics (F1, BLEU, ROUGE, METEOR) MUST not
require LLM calls.

**Rationale**: Teams need cost-quality tradeoffs: critical metrics may
warrant expensive models while basic metrics can use cheaper alternatives.
This flexibility enables production agents to optimize for business value,
not just evaluation perfection.

## Architecture Constraints

HoloDeck MUST maintain three distinct, decoupled engines:

1. **Agent Engine**: LLM interactions, tool execution, memory, vector stores
2. **Evaluation Framework**: AI-powered metrics + NLP metrics with flexible model selection
3. **Deployment Engine**: FastAPI conversion, Docker packaging, cloud deployment

Each engine SHOULD be independently testable and evolvable. Cross-engine
communication MUST use well-defined contracts (not tight coupling).

## Code Quality & Testing Discipline

- **Language Target**: Python 3.10+
- **Style Guide**: Google Python Style Guide (enforced via Black, Ruff)
- **Type Checking**: MyPy strict mode (all new code must pass)
- **Testing Framework**: pytest with markers (`@pytest.mark.unit`,
  `@pytest.mark.integration`, `@pytest.mark.slow`)
- **Minimum Coverage**: 80% (measured by pytest-cov)
- **Security Scanning**: Bandit, Safety, detect-secrets (pre-commit required)
- **Pre-commit Hooks**: Format, lint, type-check, and security scans MUST
  pass before commit

All test cases MUST be independently runnable. Integration tests SHOULD
validate agent contracts and inter-service communication. Unit tests MUST
cover edge cases explicitly.

## Governance

**Amendment Procedure**:

1. Proposed amendment MUST justify why current principle(s) are insufficient
2. Impact assessment MUST cover all five core principles and dependent architecture
3. Migration path MUST be documented if amendment breaks existing compliance
4. Approval MUST include verification that templates (spec, plan, tasks,
   checklist) are updated or explicitly exempted

**Versioning Policy**:

- MAJOR bump: Principle removal or redefinition (backward incompatible)
- MINOR bump: New principle added or existing principle scope expanded
- PATCH bump: Clarifications, wording refinements, non-semantic fixes

**Compliance Review**:

- Every feature specification (spec.md) MUST include Constitution Check
  section (see plan-template.md)
- Every pull request MUST verify compliance before merge
- Justified exceptions (Complexity Tracking table) MUST reference specific
  principles and explain why constraints cannot be met

**Runtime Guidance**:
Development workflow details (make commands, git workflow, pre-commit
configuration, etc.) are documented in CLAUDE.md and Makefile—NOT in this
Constitution. This Constitution establishes _principles_; CLAUDE.md
operationalizes them.

**Version**: 1.0.1 | **Ratified**: 2025-10-19 | **Last Amended**: 2025-10-19
