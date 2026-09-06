# Feature 035 reconciliation — 2026-09-06

**Next work:** The [completion execution plan](2026-09-06-complete-035.md) turns this audit into the active delivery checklist.
The dispositions and counts below describe the audit baseline.

## Scope and evidence

This audit reconciles all six plans and TODO records against `b585e80` on `main`.
PR #338 (`3805e80`) merged the native backend, SK cleanup, and LiteLLM migration.
The audit changes documentation only. It does not implement pending features.

Checked implementation items mean the code or document exists with supporting inspection evidence.
They do not imply that all live, deployment, security, or full-suite acceptance gates passed.
Historical validation remains labelled separately from the commands run during this audit.

## Record dispositions

| Record | Reconciled disposition |
| --- | --- |
| [MVP](plan-mvp.md) | Shipped. Later parity work supersedes function-only tooling, empty thinking, unconditional tracing disable, and retained SK inference services. |
| [Full parity plan](plan-full.md) | Partial implementation. Task evidence and outstanding acceptance gaps are reconciled with the TODO. |
| [Full parity TODO](todo-full.md) | Current parity backlog. Implementation, partial tasks, deferred scope, and unverified checkpoints are separated. |
| [SK cleanup](plan-sk-decouple.md) | Dead-code and configuration removal implemented. Original broad validation gates are not re-established by this audit. |
| [LiteLLM migration](plan-litellm-embeddings-contextgen.md) | Inference cutover implemented. Dimension plumbing, promised exception wrapping, telemetry acceptance, and documentation have unresolved items. |
| [Documentation](plan-docs.md) | Substantially implemented. Quick-start prerequisites, guide structure, stale claims, and validation remain. |

The feature remains `pending` because full parity is incomplete.
Completed code must not be repeated just because its original checkbox was empty.
Records stay together in the active feature directory while acceptance and follow-up work remains.

## Remaining work

### Full parity

- Subagent handoffs, skill tools, and AG-UI handoff events (D).
- YAML hook models and observation/rejection behavior (E).
- Hosted-tool models, factories, and safety gates (G).
- OpenAI serving capacity enforcement, startup/readiness acceptance, and deployment sizing (I).
- Tool-output credential guardrails and subprocess environment scrubbing (J).
- Integration acceptance and accurate documentation for those surfaces (K).
- Full acceptance evidence for implemented fallback and tracing behavior, as detailed in the parity records.

Existing retrieval, MCP, budget, effort, structured-output, and tracing code is not evidence that these remaining features work.
The shared HTTP server and some packaging behavior exist. Broad claims that all serving/deployment code is absent are inaccurate.

### LiteLLM and documentation

The embedding shim accepts a dimension override, but the shared factory does not supply it.
Provider exceptions propagate rather than following the error-wrapping promise in the draft.
These are differences from planned acceptance, not failures demonstrated by the current passing tests.
The migration also lacks the planned semantic-convention pin and complete exporter-level acceptance evidence.

Documentation gaps include provider tabs, the warehouse example's missing tool implementation, guide-template differences, and stale changelog scope.
The documentation plan records each affected page and its disposition.

### Final Semantic Kernel removal

Final decommission is deferred, not complete and not covered by a dedicated implementation spec.
It requires replacement of vector-store records/connectors and the text splitter, followed by telemetry, test, documentation, and dependency cleanup.
The [SK inventory](plan-sk-decouple.md#remaining-sk-inventory-and-follow-ups) identifies the remaining surfaces.
The [technical debt tracker](../../tech-debt-tracker.md) records the exit criteria.

## Verification in this audit

- `uv run pytest tests/unit/lib/backends/test_openai_agents_backend.py tests/unit/lib/backends/test_openai_agents_tool_adapters.py tests/unit/lib/backends/test_selector.py -n auto -q`: **128 passed**.
- `uv run pytest tests/unit/lib/test_litellm_support.py tests/unit/lib/test_tool_initializer.py tests/unit/lib/test_llm_context_generator.py tests/unit/lib/backends/test_openai_agents_tracing.py tests/unit/lib/backends/test_openai_agents_output.py tests/unit/lib/backends/test_openai_agents_cost.py tests/unit/lib/backends/test_openai_agents_fallback.py tests/unit/lib/backends/test_openai_agents_mcp.py tests/unit/lib/backends/test_openai_agents_permissions.py tests/unit/lib/backends/test_otel_redaction.py -n auto -q`: **196 passed**.
- `make harness-check`: passed (structure, links, discovery, and schema inventory).
- `uv run mkdocs build --strict`: passed. An initial build ran before this report existed and failed on its two incoming links. The completed report resolved both links.
- `git diff --check`: passed.

No live model calls, deployment validation, full test suite, isolated installation, visual inspection, or comparative coverage measurement ran.
