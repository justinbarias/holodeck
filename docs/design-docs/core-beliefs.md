# Core engineering beliefs

The [constitution](constitution.md) remains the authority for product principles.
These beliefs guide implementation within those principles.

1. **Repository evidence is discoverable.** Keep decisions beside code and link them from an index.
2. **Small entry points preserve attention.** Keep rules brief and load task-specific detail through links.
3. **Contracts stabilize change.** Use typed boundaries, published schemas, and focused regression tests.
4. **Failures improve the environment.** Turn repeated mistakes into a useful check or clearer documentation.
5. **Verification follows risk.** Exercise the changed behavior and shared contracts it can affect.
6. **Existing code is evidence, not automatic precedent.** Preserve useful patterns and record known exceptions.
7. **Humans set intent.** Agents complete authorized work and surface decisions that require human judgment.

Keep new rules here only when they express a durable tradeoff.
Put commands in [contributing](../contributing.md) and known gaps in [technical debt](../exec-plans/tech-debt-tracker.md).
