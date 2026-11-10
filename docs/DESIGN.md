# Engineering design

Start with [architecture](../ARCHITECTURE.md) for code locations and runtime boundaries.
Use the [design index](design-docs/index.md) for decisions and their evidence.
The [reference index](references/index.md) records the external guidance used by this harness.

## Change contracts

- Preserve declarative agent configuration and published schemas.
- Keep provider-specific behavior behind backend contracts and adapters.
- Preserve independent agent execution, evaluation, and deployment engines.
- Keep deterministic workflow helpers separate from model execution.
- Validate external values at the boundary before domain logic uses them.
- Reach vector stores only through `holodeck.lib.retrieval`: the adapter contract, the registry, and the manifest and fingerprint guard. Do not import a store client in a tool.
- Keep evaluation on DeepEval, deterministic NLP metrics and code graders. Do not add Microsoft AI evaluation or a retrieval framework.
- Keep features in install extras. The base install must start without them, and a missing extra must name its install command.
- Do not depend on Semantic Kernel. A test fails if it becomes importable or named in `src/`.

The [constitution](design-docs/constitution.md) defines product principles.
The [core beliefs](design-docs/core-beliefs.md) explain engineering tradeoffs.

## Decision records

For a boundary or contract change, record the problem, alternatives, selected approach, and verification in an execution plan.
Use [PLANS.md](PLANS.md) for the plan lifecycle.
Link durable decisions from [design-docs/index.md](design-docs/index.md).
Link requirements from [product-specs/index.md](product-specs/index.md).
The dependency stack decisions (native adapters, extras, evaluation) are in [spec 042](product-specs/042-dependency-stack-revamp/index.md) and its [selection evidence](design-docs/042-dependency-stack-revamp/selection-evidence.md).

Code and named tests establish current implementation. Specs describe intent and history.
When those disagree, record the discrepancy instead of silently assuming that the spec shipped.
Update the relevant guide in the same change as its behavior.

## Mechanical enforcement

`make harness-check` validates the harness layout, local link targets, document discovery,
entry-point size, Claude import, and generated schema inventory.
It does not prove that prose matches runtime behavior.

Schema synchronization and Temporal import/sandbox tests enforce selected architecture constraints.
See [QUALITY_SCORE.md](QUALITY_SCORE.md) for their locations and remaining enforcement gaps.
