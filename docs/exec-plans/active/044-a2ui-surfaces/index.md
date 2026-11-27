# Native A2UI Surfaces — execution plans

Feature: `044-a2ui-surfaces`. Product contract: [spec 044](../../../product-specs/044-a2ui-surfaces/index.md).

**Status:** Not started. Plan written 2026-11-27.

## Documents

- [2026-11-27-native-a2ui-surfaces.md](2026-11-27-native-a2ui-surfaces.md) — execution checklist: 24 tasks (T0 to T23), six checkpoints, six PRs, dependency graph, risks, acceptance table
- [command-center.toml](command-center.toml) — task dependencies, feature (PR) order, and handoff rules for the command center

## Command center

The status board for this plan is published as a private Claude artifact: [HoloDeck 044 command center](https://claude.ai/artifact/Do4CzpdF4vodorHzq23kSk).
It shows only plan 044, so the build reads a staging copy that holds `AGENTS.md` and this folder.
Rebuild it after each change to the task index, then publish the output to the same artifact.

```bash
CC=~/.claude/skills/agent-harness/command-center/command_center.py
STAGE=$(mktemp -d)
mkdir -p "$STAGE/docs/exec-plans/active" && cp AGENTS.md "$STAGE/"
cp -R docs/exec-plans/active/044-a2ui-surfaces "$STAGE/docs/exec-plans/active/"
python3 "$CC" --root "$STAGE" --config "$STAGE/docs/exec-plans/active/044-a2ui-surfaces/command-center.toml" --check
python3 "$CC" --root "$STAGE" --config "$STAGE/docs/exec-plans/active/044-a2ui-surfaces/command-center.toml" --artifact --out "$STAGE/command-center.html"
```
