---
name: port-python-to-dotnet
description: Port a Python repository to C# on .NET with coding agents, using a judge-first, structure-preserving process. Use when the user asks to port, migrate, rewrite or translate Python code to C# or .NET, to plan such a port, or to resume one that has a port/ folder. Covers the parity judge, the Python-to-C# rulebook, the library map, the on-disk work queue, adversarial review and the cutover. Not for .NET version upgrades or for ports to other languages.
---

# Port Python to .NET

This skill runs a port as a series of gated phases. The state lives in files under `port/`, so any session can pick up where the last one stopped.

## Standing rules

These hold in every phase. If the conversation was compacted, read this section again.

1. Read `port/STATE.md` and `port/queue.json` before you do anything. Update both before you stop.
2. Never edit the judge, its golden files or the Python source to make a check pass. If a check looks wrong, stop and tell the user.
3. A unit is done when the commands say so. Your own reading of the code does not count as proof.
4. Keep the structure of the source. One Python module becomes one C# file with the same names in the same order. Make no improvements before phase 9.
5. Do not write stubs or simplified versions, and do not call the Python code from product code. If you cannot port something, mark it `// TODO(port): <reason>`, set the unit to `blocked` and move on.
6. When the same mistake shows up twice, change `port/RULES.md` and redo the units it affects. Do not patch the output by hand.
7. Stop at every human gate and wait for the user.

## Files

| File | Holds | Who may change it |
| --- | --- | --- |
| `port/STATE.md` | Current phase, last action, next action, open questions | Any session |
| `port/DECISIONS.md` | Scope, target framework, library choices, gap choices | Only after the user agrees |
| `port/RULES.md` | The rulebook: how each Python pattern becomes C# | The lead session, never an implementer |
| `port/GAPS.md` | Python constructs with no direct C# form, and the chosen answer | The lead session |
| `port/queue.json` | Every module in port order, with a status | Implementers change `status`, `target` and `notes` only |
| `port/judge/` | Parity checks and golden files recorded from Python | Nobody after gate 2, unless the user approves |

Statuses in the queue are `todo`, `in_progress`, `built`, `tests_pass`, `parity_pass`, `reviewed` and `blocked`.

## Phases

### 1. Scope and gaps

Read the repository. Run the queue builder to see its size and its third-party imports:

```bash
python .claude/skills/port-python-to-dotnet/scripts/build_queue.py src <package> --out port/queue.json
```

Go through `external_packages` in the output and pick a .NET answer for each one, starting from [references/library-map.md](references/library-map.md). Check current package versions before you pin them. Some Python libraries have no official .NET version, and for each of those the user must choose to rebuild it, use a community package, or keep that part in Python behind a process boundary.

Write `port/DECISIONS.md` with the target framework (.NET 10 unless the user says otherwise), what is in and out of scope, and every library choice.

Gate 1: the user approves `port/DECISIONS.md`.

### 2. Build the judge

Build the parity checks before any C# exists. Follow [references/judge.md](references/judge.md). The judge must be able to run against both the Python program and the .NET program from the outside.

Then prove it works. It must pass against Python. Break the Python code on purpose in three or more places, one at a time, and confirm that the judge fails each time. Undo each break. Record what you broke and what the judge reported in `port/STATE.md`.

Gate 2: the user reviews the judge and the break test results. After this gate `port/judge/` is read-only.

### 3. Prepare the source

Complete the type hints in the Python code where they are missing, because every C# member needs a type. Keep these changes in their own commits and rerun the Python tests and the judge after them.

Write `port/GAPS.md`. List each place where Python does something C# cannot say directly: duck-typed calls that need an interface, decorators, monkeypatching in tests, module-level state, generators that use `send()`, and import cycles. The queue marks cycles with `"cycle": true`. Decide the answer for each gap.

### 4. Rulebook and scaffold

Copy [references/rules.md](references/rules.md) to `port/RULES.md` and extend it with rules for this repository: naming, file layout, error types, logging and every answer from `port/GAPS.md`.

Create the .NET solution from [references/scaffold.md](references/scaffold.md). Build the small `PyCompat` helper library it describes, with its tests, before any other code.

### 5. Pilot

Pick three to five modules of different kinds, for example a model file, an async service and a CLI command. Port them using only `port/RULES.md`. Read the results closely with the user, fix the rules, then delete the pilot C#. Repeat if the rules changed a lot.

Gate 3: the user approves `port/RULES.md`.

### 6. Translate

Port the type and signature skeleton first so that everything compiles against fixed shapes. Then fill in bodies in queue order. The queue lists modules after the modules they import. Port every module in a cycle group together.

For each unit:

1. Set its status to `in_progress` and fill in `target`.
2. Port the module and its unit tests by following `port/RULES.md`.
3. Run `dotnet build -warnaserror`. On success set `built`.
4. Run `dotnet test` for the unit. On success set `tests_pass`.
5. Run the judge checks that cover the unit. On success set `parity_pass`.
6. Send the unit to review (below). On success set `reviewed`.

When you run implementers as subagents, give each one a single unit, the rulebook, the Python file and the list of files it owns. Tell it the standing rules. Implementers may not change `port/RULES.md` or `port/judge/`.

Review each unit with two reviewers that start with a fresh context. Give each one the Python file, the C# file and `port/RULES.md`, and ask for every difference in behaviour, with the input that shows it. Ask them to check empty, zero, negative and non-ASCII inputs. If the two disagree, a third reviewer decides. A finding that points to a missing rule goes back to the rulebook under standing rule 6.

### 7. Match behaviour

Fix build errors in groups that share a cause. Start the CLI and the server and smoke-test them. Run the whole judge and work the failing list down to zero. Then run the scanner:

```bash
python .claude/skills/port-python-to-dotnet/scripts/scan_port.py <dotnet-src> --exclude /obj/ --exclude /bin/
```

It must report no findings. Add the judge and the scanner to CI as merge checks. The count of judge checks must never go down between runs.

Gate 4: the user sees the full judge pass and the clean scan.

### 8. Cut over

Switch callers to the .NET build at a process boundary: the CLI, the HTTP interface or a queue. Parts that stayed in Python under `port/DECISIONS.md` keep running behind that boundary. Keep the Python version runnable and keep the judge able to compare both for as long as the user wants a way back.

### 9. Make it idiomatic

Start only after the cutover holds. Change one thing at a time and run the judge after each change. This is the phase for records in place of property bags, dependency injection in place of module state, and middleware in place of decorator chains.

## Skills to use alongside this one

Use these if they are installed. This skill works without them.

| Need | Skill or plugin |
| --- | --- |
| An independent proof verdict on a finished module | `modernize-verify` in the `code-modernization` plugin, `anthropics/claude-plugins-official` |
| C# style, nullable types, concurrency | `Aaronontheweb/dotnet-skills` |
| Test writing and the phase 9 refactor | `dotnet/skills` |
| Plans, one task per subagent, evidence before claims | `obra/superpowers` |
| Current .NET documentation | Microsoft Learn MCP server, `https://learn.microsoft.com/api/mcp` |

## What to report

At each gate and at the end of each session, tell the user the phase, the queue counts by status, the judge result with its command output, every blocked unit with its reason, and any rule you changed. Say which checks you did not run.
