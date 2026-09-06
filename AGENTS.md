# AGENTS.md

HoloDeck defines agents in YAML and runs them through provider backends.
Start here. Read linked documents only when they apply to the task.

## Map

| Need | Source of truth |
| --- | --- |
| Architecture, boundaries, code locations | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Product principles | [Constitution](docs/design-docs/constitution.md), [product sense](docs/PRODUCT_SENSE.md) |
| Design decisions and invariants | [Design](docs/DESIGN.md), [design index](docs/design-docs/index.md) |
| Feature requirements and status | [Product specs](docs/product-specs/index.md), [spec inventory](docs/product-specs/inventory.md) |
| Plans, decisions, and technical debt | [Plans](docs/PLANS.md) |
| Setup, commands, Python standards | [Contributing](docs/contributing.md) |
| Dashboard and interaction design | [Frontend](docs/FRONTEND.md) |
| Test strategy and known gaps | [Quality score](docs/QUALITY_SCORE.md) |
| Runtime diagnostics and deployment validation | [Reliability](docs/RELIABILITY.md) |
| Credentials, input boundaries, security checks | [Security](docs/SECURITY.md) |
| Published schemas and stored artifacts | [Schema inventory](docs/generated/db-schema.md) |
| User-facing behavior and examples | [Documentation](docs/index.md), [guides](docs/guides), [examples](docs/examples) |

## Working agreement

- Infer scope from the request and prior conversation. Complete authorized work through verification.
- Make reasonable assumptions for reversible choices. State assumptions that affect the result.
- Ask only when missing information materially changes the outcome or an action exceeds authorization.
- Before requesting approval, prepare the concrete result that existing authorization permits.
- User instructions take precedence over repository and skill guidance, subject to system and developer instructions.
- If a skill blocks progress, identify its path and exact instruction. Explain why it applies.
- Treat new messages as steering. Preserve the original objective unless the user changes it.
- Keep changes surgical. Reuse existing patterns and remove only dead code introduced by the change.
- Keep unrelated working-tree edits intact. Avoid speculative abstractions, configuration, and dependencies.
- Use concise, plain language. Report the result, verification evidence, and remaining limitations.

## Work loop

1. Inspect the working tree and relevant code before editing.
2. Define the expected behavior and the smallest useful verification.
3. For complex work, maintain an execution plan using [PLANS.md](docs/PLANS.md).
4. Implement the smallest complete change, with regression tests for changed behavior.
5. Run focused checks, review the diff, and correct problems introduced by the change.
6. Update affected documentation and the plan with evidence and unresolved work.

For independent research, implementation, or review tasks, use available subagents when parallel work saves time or improves verification.
Give each subagent a bounded task and clear file ownership. Review its results before integration.
Keep tightly coupled or trivial work local. Use readable messages between agents.

## Invariants

- Keep agent configuration declarative and validated with Pydantic.
- Route runtime backends through `BackendSelector`. Consumers use contracts from `lib/backends/base.py`.
- Preserve Claude as a first-class backend. OpenAI/Azure route to OpenAI Agents, Anthropic/Ollama to Claude.
- Use MCP for external API tools. Document justified exceptions under the constitution.
- Preserve multimodal tests, evaluation model overrides, and OpenTelemetry GenAI instrumentation.
- Keep LLM calls and I/O outside Temporal workflow execution. Use the documented deterministic surface.
- Use concrete domain types. Convert untyped SDK data at the boundary.
- Do not use `Any` or `getattr` to bypass known models or protocols.
- Keep blocking I/O out of async execution paths.

## Verification

Use `uv run` for Python commands. Run pytest with `-n auto`.

```bash
uv run pytest tests/unit/lib/backends/test_selector.py -n auto -q
make harness-check
```

Choose relevant tests using the [quality map](docs/QUALITY_SCORE.md).
For Python changes, run focused Black, Ruff, and MyPy checks.
For agent model changes, run `make schema` and `make schema-check`.
For documentation changes, validate links and structure with `make harness-check`.
When docsite navigation or rendering changes, also run `uv run mkdocs build`.
Avoid tests that repeat implementation details and repeated broad checks after focused checks pass.
Broaden verification for shared contracts, failures, or unresolved risks. Preserve required CI and commit checks.
Report unavailable dependencies or unrelated failures explicitly. Never claim an unrun check passed.

## Keep the harness current

Keep this file at most 100 lines. Put detailed guidance in its mapped document.
Keep `CLAUDE.md` as the single import of this file.
When boundaries change, update architecture, design evidence, and relevant tests together.
When a repeated failure reveals a missing guard, add a focused mechanical check and a useful remediation message.
Track gaps in [technical debt](docs/exec-plans/tech-debt-tracker.md).
Run only deployment validation authorized by the user. See [Reliability](docs/RELIABILITY.md).
