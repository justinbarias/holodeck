# Native A2UI surfaces (feature 044)

**Status:** Not started. Plan written 2026-11-27. All spec decisions (D1 to D11) are settled. Tasks marked **ASK** stop for a maintainer checkpoint before code is written.
**Product contract:** [Spec 044](../../../product-specs/044-a2ui-surfaces/spec.md).
**Code baseline:** `bab06fe`.
**Debt opened by this plan:** [H-032](../../tech-debt-tracker.md#h-032).

## Objective and observable acceptance criteria

Add an `a2ui` tool type. HoloDeck validates A2UI 1.0 component trees in Python and paints surfaces on the `holodeck serve` AG-UI stream in the wire shape of `@ag-ui/a2ui-middleware` 0.0.16. The OpenAI Agents AG-UI path gains parity with the Claude path first.

The plan is complete when spec success criteria 1 to 6 pass:

1. The CopilotKit sample, with only `a2ui={{ catalog }}` added and no Node middleware, renders a surface from `holodeck serve` on Claude and on OpenAI.
2. In the raw `curl -N` stream, the first `ACTIVITY_SNAPSHOT` for a surface arrives before `TOOL_CALL_END` for that call, on both backends.
3. An invalid component tree gets `status: "invalid"`, and a later call renders in the same run, with one `RUN_STARTED` and one `RUN_FINISHED`.
4. A button click reaches the agent as an action message, and the next turn updates the same `surfaceId`.
5. The contract test passes against the recorded middleware fixtures.
6. `holodeck test run` passes on a sample agent with the `a2ui` tool.

## Scope and constraints

In scope: spec sections 2 to 5. Out of scope: the spec's out-of-scope list, surface assertions in `holodeck test` ([H-032](../../tech-debt-tracker.md#h-032)), and REST changes (D11).

Constraints that apply to every task:

- Follow the invariants in [AGENTS.md](../../../../AGENTS.md): Pydantic models, no `Any` or `getattr` at new boundaries, no blocking I/O on async paths.
- `src/holodeck/lib/a2ui/` imports nothing from `ag_ui`, `serve`, or `lib/backends`.
- Tests first. Each behavior change has a regression test that fails before the change.
- The projector never emits an `ACTIVITY_SNAPSHOT` with components that did not validate (R8).
- Live provider calls follow [Reliability](../../../RELIABILITY.md). Missing credentials leave an acceptance item open. They do not pass it.

## Code map

| Area | Files | Note |
| --- | --- | --- |
| AG-UI handler | `src/holodeck/serve/protocols/agui.py` (1,015 lines) | `handle_request` has a Claude branch (`execute_turn_agui`) and an OpenAI branch (`_stream_turn_events`, text only) |
| Claude AG-UI | `src/holodeck/lib/backends/claude_backend.py` | Frontend tools at line 1351, `forwardedProps` allowlist at line 130, `TOOL_CALL_ARGS` from `input_json_delta` at line 1738 |
| OpenAI events | `src/holodeck/lib/backends/openai_agents_events.py`, `openai_agents_backend.py` | Emits tool start and end only. No argument deltas, no AG-UI input |
| Executor | `src/holodeck/chat/executor.py:927` | `execute_turn_agui` needs a session with `send_agui` |
| Backend contract | `src/holodeck/lib/backends/base.py` | `ToolEvent.kind` literal |
| Tool config | `src/holodeck/models/tool.py:1450` | `ToolUnion` discriminated union |
| Tool adapters | `src/holodeck/lib/backends/tool_adapters.py`, `openai_agents_tool_adapters.py` | One builder per tool type and backend |
| Serve tests | `tests/unit/serve/test_protocols_agui.py`, `test_agui_openai_streaming.py`, `test_agui_tool_events.py` | Extend, do not duplicate |
| Middleware reference | `@ag-ui/a2ui-middleware` 0.0.16, `src/index.ts` | Wire shape source for T0 fixtures |

## Model and effort per task

From the user-level model guidance. Escalate in order: add a self-check, raise effort one level, then move up a model at a lower effort.

| Kind of task | Model | Effort |
| --- | --- | --- |
| Contracts, ASK items, cross-boundary integration | Fable (main session), inline | session default |
| Backend AG-UI parity, projector, run binding | Opus 5.5 subagent | `medium`, then `high` |
| Well-scoped work against a fixed contract: validator, parser, ops, sample, docs | Sonnet 5.5 subagent | `medium`, then `high` |
| Log reading, wide searches | Haiku 5.5 subagent | default |
| Adversarial review, once per PR layer | Codex gpt-6-astra `high`, or Opus 5.5 `high` | maintainer's choice |

Each subagent spawn needs maintainer approval first, as the user-level guidance requires.

## Ordered work and verification

Sizes: S is 1 to 2 files, M is 3 to 5, L is 5 to 8. Every task also runs focused Black, Ruff, MyPy and pytest (`-n auto`) on the files it changes.

### Phase 0: Evidence

#### T0. Record middleware wire fixtures

Run `@ag-ui/a2ui-middleware` 0.0.16 once, in a scratch Node project outside the repository, over scripted AG-UI input streams. Commit only the JSON input and output.

- [ ] Four cases under `tests/fixtures/a2ui/middleware/`: a valid surface with data, an invalid tree then a valid tree, data that streams in parts, and an action request.
- [ ] Each case has `input.json` (AG-UI events into the middleware) and `expected.json` (events out).
- [ ] A `README.md` in the fixture folder names the package version, the commit `ed94ca4`, and the script used.

**Verify:** the fixtures parse as AG-UI `BaseEvent` lists with `ag-ui-protocol` 1.0.0.
**Depends on:** none. **Size:** S. **Model:** Sonnet `high`. **Spec:** R6 to R8, success criterion 5.

#### T1. Spike: OpenAI frontend-tool stop and resume — ASK on failure

Prove D9 against a live model before 044.1 starts.

- [ ] A throwaway script runs an OpenAI Agents agent with `tool_use_behavior=StopAtTools([...])` and one frontend tool, stops at the call, then resumes from the history plus a `tool` message.
- [ ] The resumed run uses the tool output and does not call the tool again.
- [ ] The result and the exact SDK calls are recorded under "Progress and decisions".

**Verify:** the script output, recorded in this plan.
**Depends on:** none. **Size:** S. **Model:** Opus `medium`. **Spec:** D9.
If resume fails, stop and ask the maintainer: defer OpenAI frontend tools, or choose another resume design.

### Phase 1: PR 044.1, OpenAI Agents AG-UI parity

#### T2. Tool-argument deltas on the OpenAI path

- [ ] `ToolEvent.kind` gains `"args_delta"` with a `text` delta. The Claude backend is unchanged.
- [ ] The OpenAI stream maps Responses function-call argument deltas to `ToolEvent(kind="args_delta")`.
- [ ] `_tool_event_to_agui` emits `TOOL_CALL_ARGS` for it, between `TOOL_CALL_START` and `TOOL_CALL_END` of the same call.

**Verify:** a unit test feeds a recorded OpenAI stream and asserts the event order. Extend `tests/unit/serve/test_agui_openai_streaming.py`.
**Depends on:** none. **Size:** M. **Model:** Sonnet `high`. **Spec:** D5, section 4.6.

#### T3. OpenAI `send_agui`: context, state, and `forwardedProps`

- [ ] The OpenAI session implements `send_agui(input_data, message_override)` with the same contract as the Claude session.
- [ ] AG-UI `context` and `state` go into the run instructions with the Claude addendum text. Extract the addendum builder to one shared function, so both backends use it.
- [ ] An OpenAI `forwardedProps` allowlist, with a warning for unknown keys.

**Verify:** unit tests for the addendum, the allowlist warning, and one AG-UI turn through a fake model.
**Depends on:** T2. **Size:** M. **Model:** Opus `medium`. **Spec:** section 4.6.

#### T4. OpenAI frontend tools

- [ ] AG-UI `tools` become OpenAI function tools that the run stops at, with the T1 design.
- [ ] The run emits the frontend tool's `TOOL_CALL_*` events and ends without a tool result.
- [ ] A request whose latest message has role `tool` resumes the run, as on Claude.

**Verify:** unit tests for stop, resume, and no repeated call. R10.
**Depends on:** T1, T3. **Size:** M. **Model:** Opus `medium`. **Spec:** R10.

#### T5. One `handle_request` path for both providers — ASK

Present the merged handler before the edit. Both providers go through `execute_turn_agui`. Remove the OpenAI text-only branch and the post-hoc fallback branch only if no backend still needs them.

- [ ] `handle_request` has no branch on the provider.
- [ ] Existing serve tests pass unchanged, or each changed assertion is explained in "Progress and decisions".

**Verify:** `uv run pytest tests/unit/serve -n auto -q`.
**Depends on:** T4. **Size:** M. **Model:** Fable inline. **Spec:** section 4.6.

#### Checkpoint A

- [ ] Live loop step 2 on an OpenAI sample: two threads, two turns each, one frontend tool call and resume, argument deltas in the SSE stream.
- [ ] Adversarial review of PR 044.1. Findings resolved or accepted by the maintainer.

### Phase 2: PR 044.2, A2UI core (independent of Phase 1)

#### T6. Catalog contract and resolution — ASK

Present the models before the code: `CatalogEntry`, `ResolvedCatalog`, `ClientCatalogEntry`, `CatalogRejected`.

- [ ] `resolve_catalog(tool, client_entry)` implements the D1 rules and the working assumptions: allowlist, client wins over YAML, `max_catalog_bytes`.
- [ ] The basic catalog ships at `lib/a2ui/resources/basic_catalog_v1_0.json` and loads with `importlib.resources`.
- [ ] Parsing of the client context entry, found by the middleware `description` constant.

**Verify:** unit tests for allowed, rejected, oversize, client-wins, and YAML-fallback cases. R2, R3, R12.
**Depends on:** none. **Size:** M. **Model:** Fable inline. **Spec:** D1, R2, R3, R12.

#### T7. Component validation

- [ ] `validate_components(components, catalog) -> list[A2UIIssue]` with `jsonschema`. One issue per error, each with a JSON pointer into the tool arguments.
- [ ] Checks: unknown component type, property schema, duplicate IDs, missing child references, `max_components`.

**Verify:** unit tests per check, with the expected pointer and message text. R5.
**Depends on:** T6. **Size:** S. **Model:** Sonnet `high`. **Spec:** D8, R5.

#### T8. Incremental argument parser

- [ ] Promote `jiter` to a direct dependency, with the version range already in `uv.lock`. `uv lock --check` passes with no new package.
- [ ] `SurfaceStreamParser.feed(delta)` uses `jiter` partial mode and returns a step when `surfaceId` is known, when `components` closes, and when `data` closes.

**Verify:** a test splits each T0 payload at every byte offset and gets the same steps in the same order.
**Depends on:** none. **Size:** S. **Model:** Sonnet `high`. **Spec:** D4, D8.

#### T9. A2UI 1.0 operation builders

- [ ] `build_operations` returns `createSurface`, `updateComponents`, and `updateDataModel` messages with `version: "1.0"`, in the order of the middleware's `streamedOps`.

**Verify:** unit tests compare output to the operations in the T0 fixtures.
**Depends on:** T0, T6. **Size:** S. **Model:** Sonnet `medium`. **Spec:** section 4.5.

#### Checkpoint B

- [ ] `lib/a2ui` imports nothing from `ag_ui`, `serve`, or `lib/backends`. Add a focused structural test that fails if it does.
- [ ] Adversarial review of PR 044.2.

### Phase 3: PR 044.3, tool configuration and adapters

#### T10. `A2UITool` configuration — ASK

Present the model and the loader rules before the code. This changes the published schema.

- [ ] `A2UITool` (`type: a2ui`) in `ToolUnion`, with `catalogs`, `default_catalog`, `accept_client_catalogs`, and `limits`, as in spec section 4.3.
- [ ] Validation: at least one catalog, unique IDs, `default_catalog` in the list, at most one `a2ui` tool per agent.
- [ ] `make schema` and `make schema-check` pass.

**Verify:** model tests for each rule. R1.
**Depends on:** T6. **Size:** M. **Model:** Fable inline. **Spec:** R1.

#### T11. Run binding and tool adapters on both backends

- [ ] `A2UIRunBinding` holds the resolved catalog and the repair count for one run. Serve sets it per AG-UI request. `test` and `chat` set it from the YAML catalog.
- [ ] Both adapter modules build `render_a2ui` from `A2UITool` and the binding. The description lists the catalog components.
- [ ] Results: `rendered`, `invalid` with issues, and `failed` after `max_repair_attempts` for one `surfaceId`.
- [ ] Record in "Progress and decisions" how the binding reaches the in-process tool on each backend, with evidence that it does not leak between concurrent sessions.

**Verify:** unit tests on both backends with a fake model: valid, invalid then valid, and three invalid calls. Two concurrent sessions with different catalogs. R4, R5.
**Depends on:** T7, T9, T10. **Size:** L. **Model:** Opus `medium`. **Spec:** D3, R4, R5.

#### T12. `holodeck test` and `holodeck chat` support

- [ ] Load fails with `a2ui tool '<name>': default_catalog '<id>' needs a schema for test and chat` when the default catalog has no schema.
- [ ] A sample agent under `sample/` with the `a2ui` tool and a local catalog, with test cases that assert the `render_a2ui` tool call.

**Verify:** unit test for the load error. `cd sample/<a2ui-sample> && uv run holodeck test run agent.yaml` on both backends. R11.
**Depends on:** T11. **Size:** M. **Model:** Sonnet `medium`. **Spec:** R11, success criterion 6.

#### Checkpoint C

- [ ] Live loop step 1 passes for the sample on Claude and OpenAI.
- [ ] Adversarial review of PR 044.3.

### Phase 4: PR 044.4, projector and actions in serve

#### T13. `A2UIProjector`

- [ ] `serve/protocols/agui_a2ui.py`: an async generator over AG-UI events that applies spec section 4.5 rules 1 to 6.
- [ ] It uses `SurfaceStreamParser`, `validate_components`, and `build_operations` from the core. It keeps no backend state.

**Verify:** the contract test in `tests/unit/serve/test_agui_a2ui.py` replays each T0 input and compares the output to `expected.json`, ignoring only fields the fixture README marks as volatile (timestamps, generated IDs). R6, R7, R8.
**Depends on:** T0, T8, T9. **Size:** M. **Model:** Opus `medium`, then `high`. **Spec:** D2, D6, R6 to R8.

#### T14. Action ingestion

- [ ] Serve validates `forwardedProps.a2uiAction.userAction`, removes it before the backend allowlists, and sends `A2UI user action:\n<json>` as the turn message.
- [ ] The action message is not echoed as a `TEXT_MESSAGE_*` event.
- [ ] An invalid action gives `RUN_ERROR` with a sanitized message.

**Verify:** unit tests for a valid action on both providers, an invalid action, and no allowlist warning. R9.
**Depends on:** T5. **Size:** S. **Model:** Sonnet `high`. **Spec:** R9.

#### T15. Wire binding, projector, and actions into `handle_request`

- [ ] `handle_request` builds the run binding, applies action ingestion, and wraps the backend events in the projector when the agent has an `a2ui` tool.
- [ ] Agents without an `a2ui` tool produce the same events as before.

**Verify:** integration tests in `tests/integration/serve` with a fake backend stream on both providers. Existing serve tests unchanged.
**Depends on:** T5, T11, T13, T14. **Size:** M. **Model:** Opus `medium`. **Spec:** sections 4.1, 4.5, 4.7.

#### Checkpoint D

- [ ] Live loop step 2 on Claude and OpenAI: two threads, two turns each, one surface with data, one invalid-then-valid repair, one action. Record the SSE excerpts for success criteria 2, 3, and 4.
- [ ] Adversarial review of PR 044.4.

### Phase 5: PR 044.5, sample, guide, and live evidence

#### T16. CopilotKit sample with an A2UI catalog

- [ ] `sample/copilotkit` registers a small catalog with `a2ui={{ catalog }}` and talks to `holodeck serve /awp` with no A2UI middleware in the route.

**Verify:** `npm run build` and `npm run lint` in `sample/copilotkit`.
**Depends on:** T15. **Size:** M. **Model:** Sonnet `medium`. **Spec:** success criterion 1.

#### T17. User guide

- [ ] `docs/guides/a2ui.md`: configuration, catalog sources and the allowlist, the event sequence, actions, limits, and troubleshooting.
- [ ] Links from the tools guide, the serve guide, and the docsite navigation.

**Verify:** `make harness-check` and `uv run mkdocs build` with no new warnings.
**Depends on:** T15. **Size:** S. **Model:** Sonnet `medium`. **Spec:** section 7.

#### T18. Live validation and closeout

- [ ] Live loop step 3: the maintainer drives the CopilotKit sample on Claude and on OpenAI and confirms the surface, the repair, and the action. Record what the UI showed.
- [ ] Fill the acceptance table below. Update the spec status and the inventory row.
- [ ] Move this plan to `completed/` only when all six criteria pass.

**Verify:** the acceptance table has evidence for every row.
**Depends on:** T16, T17. **Size:** S. **Model:** Fable inline. **Spec:** section 11.

## Delivery

| PR | Tasks | Base |
| --- | --- | --- |
| 044.1 OpenAI AG-UI parity | T1 to T5 | `main` |
| 044.2 A2UI core | T0, T6 to T9 | `main` |
| 044.3 Tool configuration and adapters | T10 to T12 | 044.2 |
| 044.4 Projector and actions | T13 to T15 | 044.3, rebased on 044.1 after it merges |
| 044.5 Sample, guide, evidence | T16 to T18 | 044.4 |

044.1 and 044.2 do not depend on each other and target `main` in parallel. 044.4 needs both. Use `gh stack` for 044.2 to 044.5. Run one integrated adversarial pass over the whole stack before it merges.

## Acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| 1. CopilotKit renders on both backends, no middleware | T18 | Open |
| 2. First snapshot before `TOOL_CALL_END` | Checkpoint D | Open |
| 3. Repair in one run | Checkpoint D | Open |
| 4. Action round trip | Checkpoint D, T18 | Open |
| 5. Contract test passes | T13 | Open |
| 6. `holodeck test run` passes | T12, Checkpoint C | Open |

## Progress and decisions

- 2026-11-27: Spec decisions D1 to D11 settled with the maintainer. `jsonschema` 4.26 is already a core dependency. `jiter` 0.19 is in `uv.lock` through the Anthropic and OpenAI SDKs. `openai-agents` 0.24 exports `StopAtTools` and `Agent(tool_use_behavior=...)`, which T1 must still prove for resume.

## Validation results and remaining blockers

None yet. No task has started.
