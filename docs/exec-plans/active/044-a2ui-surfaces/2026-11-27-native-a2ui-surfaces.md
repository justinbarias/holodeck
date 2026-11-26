# Native A2UI surfaces (feature 044)

**Status:** Not started. Plan written 2026-11-27 and revised the same day to the task breakdown standard (files per task, dependency graph, risks). All spec decisions (D1 to D11) are settled. Tasks marked **ASK** stop for a maintainer checkpoint before code is written.
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
- Each task leaves `main` (or its stack branch) green. No task depends on a later task to pass its own tests.
- Live provider calls follow [Reliability](../../../RELIABILITY.md). Missing credentials leave an acceptance item open. They do not pass it.

## Code map

| Area | Files | Note |
| --- | --- | --- |
| AG-UI handler | `src/holodeck/serve/protocols/agui.py` (1,015 lines) | `handle_request` has a Claude branch (`execute_turn_agui`) and an OpenAI branch (`_stream_turn_events`, text only) |
| Claude AG-UI | `src/holodeck/lib/backends/claude_backend.py` | Frontend tools at line 1351, `forwardedProps` allowlist at line 130, `TOOL_CALL_ARGS` from `input_json_delta` at line 1738 |
| OpenAI events | `src/holodeck/lib/backends/openai_agents_events.py`, `openai_agents_backend.py` | Emits tool start and end only. No argument deltas, no AG-UI input |
| Executor | `src/holodeck/chat/executor.py:927` | `execute_turn_agui` needs a session with `send_agui` |
| Backend contract | `src/holodeck/lib/backends/base.py` | `ToolEvent.kind` literal |
| Tool config | `src/holodeck/models/tool.py:1450`, `models/agent.py:179` | `ToolUnion` discriminated union. Tool-name uniqueness validator |
| Tool adapters | `src/holodeck/lib/backends/tool_adapters.py`, `openai_agents_tool_adapters.py` | One builder per tool type and backend |
| Serve tests | `tests/unit/serve/test_protocols_agui.py`, `test_agui_openai_streaming.py`, `test_agui_tool_events.py` | Extend, do not duplicate |
| Middleware reference | `@ag-ui/a2ui-middleware` 0.0.16, `src/index.ts` | Wire shape source for T0 fixtures |

## Dependency graph

```text
T0 fixtures ──────────────┬───────────────────────────────┐
                          │                               │
T6 catalog ──┬─ T7 validate ─┐                            │
             ├─ T9 ops ◀─────┼── T0                       │
             └─ T10 config   │                            │
T8 parser ───────────────────┤                            │
                             ▼                            ▼
             T11 binding ─▶ T12 adapters ─▶ T13 test/chat  T14 projector ◀─ T8, T9
                                   │                       │
T1 spike ─▶ T4 frontend tools      │                       │
T2 deltas ─▶ T3 send_agui ─▶ T4 ─▶ T5 one path ─▶ T15 actions
                                   │                       │
                                   └──────────▶ T16 wiring ◀┘
                                                   │
T17 API check ─▶ T18 catalog ◀─ T13                │
             │      └─▶ T19 surface rows ─▶ T20 actions
             └─▶ T21 route                         │
                                 Checkpoint E ◀────┘ (needs T16 for the browser check)
                                      │
                         T22 guide ─▶ T23 closeout
```

The two backend roots are independent: OpenAI parity (T1 to T5) and the A2UI core (T0, T6 to T9). They meet at T16. The frontend tasks T17 to T21 need only fixtures and the sample catalog, so they can run against the T0 fixtures before serve is done. Only Checkpoint E needs a running serve.

## Parallel work

- Safe in parallel: PR 044.1 (T1 to T5) and PR 044.2 (T0, T6 to T9). Inside 044.2, T7, T8 and T9 each own one module once T6 fixes the catalog types.
- Safe in parallel: the frontend tasks T17 to T21 and backend Phases 3 and 4, once T0 and T13 exist. T17 can start on day one.
- Safe in parallel: T21 (route) and T19 (surface rows). Both follow T17 and touch different files.
- Sequential: T10 before T11 and T12, because the schema is the contract. T16 after T5, T12, T14 and T15.
- Coordination point: T6 fixes the core types that T7, T9, T11 and T14 consume. Do not start those before the T6 checkpoint.

## Slicing decision

The skill default is vertical slices. This plan keeps layered PRs (core, configuration, serve, sample), because the user-level PR strategy puts foundations at the bottom of a stack and asks each layer to pass its own gates. Each layer still ends in a usable result: 044.2 validates trees, 044.3 runs the tool in `holodeck test`, and 044.4 paints surfaces in serve. The early spike (T1) and fixture recording (T0) put the two largest unknowns first.

## Model and effort per task

From the user-level model guidance. Escalate in order: add a self-check, raise effort one level, then move up a model at a lower effort.

| Kind of task | Model | Effort |
| --- | --- | --- |
| Contracts, ASK items, cross-boundary integration | Fable (main session), inline | session default |
| Backend AG-UI parity, projector, run binding, adapters | Opus 5.5 subagent | `medium`, then `high` |
| Well-scoped work against a fixed contract: validator, parser, ops, sample, docs | Sonnet 5.5 subagent | `medium`, then `high` |
| Log reading, wide searches | Haiku 5.5 subagent | default |
| Adversarial review, once per PR layer | Codex gpt-6-astra `high`, or Opus 5.5 `high` | maintainer's choice |

Each subagent spawn needs maintainer approval first, as the user-level guidance requires.

## Ordered work and verification

Sizes: XS is one function, S is 1 to 2 files, M is 3 to 5 files. No task is L or larger. Every task also runs focused Black, Ruff, MyPy and pytest (`-n auto`) on the files it changes.

### Phase 0: Evidence (highest risk first)

#### T0. Record middleware wire fixtures

Run `@ag-ui/a2ui-middleware` 0.0.16 once, in a scratch Node project outside the repository, over scripted AG-UI input streams. Commit only the JSON input and output.

- [ ] Four cases: a valid surface with data, an invalid tree then a valid tree, data that streams in parts, and an action request. Each has `input.json` and `expected.json`.
- [ ] A fixture `README.md` names the package version, the commit `ed94ca4`, the script, and the volatile fields (timestamps, generated IDs).

**Verify:** a test loads each fixture as a list of AG-UI `BaseEvent` with `ag-ui-protocol` 1.0.0.
**Files:** `tests/fixtures/a2ui/middleware/<case>/{input,expected}.json`, `tests/fixtures/a2ui/middleware/README.md`, `tests/unit/lib/a2ui/test_fixtures.py`.
**Depends on:** none. **Size:** S. **Model:** Sonnet `high`. **Spec:** R6 to R8, criterion 5.

#### T1. Spike: OpenAI frontend-tool stop and resume — ASK on failure

Prove D9 against a live model before the parity work starts.

- [ ] A throwaway script stops an OpenAI Agents run with `tool_use_behavior=StopAtTools([...])` at one frontend tool, then resumes from the history plus a `tool` message.
- [ ] The resumed run uses the tool output and does not call the tool again.
- [ ] The result and the exact SDK calls are recorded under "Progress and decisions".

**Verify:** the script output, recorded in this plan.
**Files:** this plan only. The script stays in the session scratchpad.
**Depends on:** none. **Size:** XS. **Model:** Opus `medium`. **Spec:** D9.
If resume fails, stop and ask the maintainer: defer OpenAI frontend tools, or choose another resume design.

### Phase 1: PR 044.1, OpenAI Agents AG-UI parity

#### T2. Tool-argument deltas on the OpenAI path

- [ ] `ToolEvent.kind` gains `"args_delta"` with a `text` delta. The Claude backend is unchanged.
- [ ] The OpenAI stream maps Responses function-call argument deltas to `ToolEvent(kind="args_delta")`.
- [ ] `_tool_event_to_agui` emits `TOOL_CALL_ARGS` between `TOOL_CALL_START` and `TOOL_CALL_END` of the same call.

**Verify:** `uv run pytest tests/unit/serve/test_agui_openai_streaming.py -n auto -q` with a recorded stream that asserts the event order.
**Files:** `lib/backends/base.py`, `lib/backends/openai_agents_events.py`, `serve/protocols/agui.py`, `tests/unit/serve/test_agui_openai_streaming.py`.
**Depends on:** none. **Size:** M. **Model:** Sonnet `high`. **Spec:** D5, section 4.6.

#### T3. OpenAI `send_agui`: context, state, and `forwardedProps`

- [ ] The OpenAI session implements `send_agui(input_data, message_override)` with the Claude session contract.
- [ ] AG-UI `context` and `state` go into the run instructions. The addendum builder moves to one shared module that both backends call.
- [ ] An OpenAI `forwardedProps` allowlist, with a warning for unknown keys.

**Verify:** unit tests for the shared addendum, the allowlist warning, and one AG-UI turn through a fake model. Claude AG-UI tests unchanged.
**Files:** `lib/backends/agui_context.py` (new), `lib/backends/claude_backend.py`, `lib/backends/openai_agents_backend.py`, `tests/unit/lib/backends/test_openai_agents_agui.py` (new).
**Depends on:** T2. **Size:** M. **Model:** Opus `medium`. **Spec:** section 4.6.

#### T4. OpenAI frontend tools

- [ ] AG-UI `tools` become OpenAI function tools that the run stops at, with the T1 design.
- [ ] The run emits the frontend tool's `TOOL_CALL_*` events and ends without a tool result.
- [ ] A request whose latest message has role `tool` resumes the run, as on Claude.

**Verify:** unit tests for stop, resume, and no repeated call. R10.
**Files:** `lib/backends/openai_agents_backend.py`, `lib/backends/openai_agents_tool_adapters.py`, `tests/unit/lib/backends/test_openai_agents_agui.py`.
**Depends on:** T1, T3. **Size:** M. **Model:** Opus `medium`. **Spec:** R10.

#### T5. One `handle_request` path for both providers — ASK

Present the merged handler before the edit. Both providers go through `execute_turn_agui`.

- [ ] `handle_request` has no branch on the provider.
- [ ] The OpenAI text-only branch and the post-hoc fallback branch are removed only if no backend still needs them.
- [ ] Existing serve tests pass unchanged, or each changed assertion is explained in "Progress and decisions".

**Verify:** `uv run pytest tests/unit/serve -n auto -q`.
**Files:** `serve/protocols/agui.py`, `chat/executor.py`, `tests/unit/serve/test_protocols_agui.py`.
**Depends on:** T4. **Size:** M. **Model:** Fable inline. **Spec:** section 4.6.

#### Checkpoint A (after T2 to T5)

- [ ] `make verify` passes.
- [ ] Live loop step 2 on an OpenAI sample: two threads, two turns each, one frontend tool call and resume, argument deltas in the SSE stream.
- [ ] Adversarial review of PR 044.1. Findings resolved or accepted by the maintainer.

### Phase 2: PR 044.2, A2UI core (independent of Phase 1)

#### T6. Catalog contract and resolution — ASK

Present the models before the code: `CatalogEntry`, `ResolvedCatalog`, `ClientCatalogEntry`, `CatalogRejected`.

- [ ] `resolve_catalog(tool, client_entry)` implements D1 and the working assumptions: allowlist, client wins over YAML, `max_catalog_bytes`.
- [ ] The basic catalog ships as a package resource and loads with `importlib.resources`.
- [ ] The client context entry is found by the middleware `description` constant and parsed into `ClientCatalogEntry`.

**Verify:** unit tests for allowed, rejected, oversize, client-wins, and YAML-fallback cases. R2, R3, R12.
**Files:** `lib/a2ui/__init__.py`, `lib/a2ui/catalog.py`, `lib/a2ui/resources/basic_catalog_v1_0.json`, `tests/unit/lib/a2ui/test_catalog.py`.
**Depends on:** none. **Size:** M. **Model:** Fable inline. **Spec:** D1, R2, R3, R12.

#### T7. Component validation

- [ ] `validate_components(components, catalog) -> list[A2UIIssue]` with `jsonschema`. One issue per error, each with a JSON pointer into the tool arguments.
- [ ] Checks: unknown component type, property schema, duplicate IDs, missing child references, `max_components`.

**Verify:** unit tests per check, with the expected pointer and message text. R5.
**Files:** `lib/a2ui/validate.py`, `tests/unit/lib/a2ui/test_validate.py`.
**Depends on:** T6. **Size:** S. **Model:** Sonnet `high`. **Spec:** D8, R5.

#### T8. Incremental argument parser

- [ ] `jiter` becomes a direct dependency within the version range already in `uv.lock`. `uv lock --check` passes with no new package.
- [ ] `SurfaceStreamParser.feed(delta)` uses `jiter` partial mode and returns a step when `surfaceId` is known, when `components` closes, and when `data` closes.

**Verify:** a test splits each T0 payload at every byte offset and gets the same steps in the same order.
**Files:** `lib/a2ui/stream.py`, `pyproject.toml`, `uv.lock`, `tests/unit/lib/a2ui/test_stream.py`.
**Depends on:** T0 for the payloads. **Size:** S. **Model:** Sonnet `high`. **Spec:** D4, D8.

#### T9. A2UI 1.0 operation builders

- [ ] `build_operations` returns `createSurface`, `updateComponents`, and `updateDataModel` messages with `version: "1.0"`, in the order of the middleware's `streamedOps`.

**Verify:** unit tests compare output to the operations in the T0 fixtures.
**Files:** `lib/a2ui/ops.py`, `tests/unit/lib/a2ui/test_ops.py`.
**Depends on:** T0, T6. **Size:** S. **Model:** Sonnet `medium`. **Spec:** section 4.5.

#### Checkpoint B (after T6 to T9)

- [ ] A structural test fails if `lib/a2ui` imports from `ag_ui`, `serve`, or `lib/backends`.
- [ ] `make verify` passes.
- [ ] Adversarial review of PR 044.2.

### Phase 3: PR 044.3, tool configuration and adapters

#### T10. `A2UITool` configuration — ASK

Present the model and the loader rules before the code. This changes the published schema.

- [ ] `A2UITool` (`type: a2ui`) in `ToolUnion`, with `catalogs`, `default_catalog`, `accept_client_catalogs`, and `limits`, as in spec section 4.3.
- [ ] Validation: at least one catalog, unique IDs, `default_catalog` in the list, at most one `a2ui` tool per agent.
- [ ] `make schema` and `make schema-check` pass.

**Verify:** model tests for each rule. R1.
**Files:** `models/tool.py`, `models/agent.py`, `schemas/agent.schema.json` (generated), `tests/unit/models/test_tool_a2ui.py`.
**Depends on:** T6. **Size:** M. **Model:** Fable inline. **Spec:** R1.

#### T11. Run binding

- [ ] `A2UIRunBinding` holds the resolved catalog and the per-`surfaceId` repair count for one run.
- [ ] A binding provider gives the in-process tool the binding of its own session. Record the mechanism on each backend under "Progress and decisions".
- [ ] Two concurrent sessions with different catalogs never see each other's binding.

**Verify:** unit tests, including a concurrency test with two sessions.
**Files:** `lib/a2ui/binding.py`, `tests/unit/lib/a2ui/test_binding.py`.
**Depends on:** T6, T10. **Size:** S. **Model:** Opus `medium`. **Spec:** D3.

#### T12. `render_a2ui` tool adapters on both backends

- [ ] Both adapter modules build `render_a2ui` from `A2UITool` and the binding. The description lists the catalog components.
- [ ] Results: `rendered`, `invalid` with issues, and `failed` after `max_repair_attempts` for one `surfaceId`.
- [ ] One shared function computes the result, so both backends return the same JSON.

**Verify:** unit tests on both backends with a fake model: valid, invalid then valid, and three invalid calls. R4, R5.
**Files:** `lib/a2ui/tool.py` (new, shared result logic), `lib/backends/tool_adapters.py`, `lib/backends/openai_agents_tool_adapters.py`, `tests/unit/lib/backends/test_a2ui_tool_adapters.py`.
**Depends on:** T7, T9, T11. **Size:** M. **Model:** Opus `medium`. **Spec:** D3, R4, R5.

#### T13. `holodeck test` and `holodeck chat` support

- [ ] Load fails with `a2ui tool '<name>': default_catalog '<id>' needs a schema for test and chat` when the default catalog has no schema.
- [ ] A sample agent with the `a2ui` tool, a local catalog, and test cases that assert the `render_a2ui` tool call.

**Verify:** unit test for the load error. `cd sample/a2ui-booking && uv run holodeck test run agent.yaml` on both backends. R11.
**Files:** `models/tool.py` or the loader that builds the binding, `sample/a2ui-booking/agent.yaml`, `sample/a2ui-booking/catalogs/booking.json`, `tests/unit/models/test_tool_a2ui.py`.
**Depends on:** T12. **Size:** M. **Model:** Sonnet `medium`. **Spec:** R11, criterion 6.

#### Checkpoint C (after T10 to T13)

- [ ] `make verify`, `make schema-check` pass.
- [ ] Live loop step 1 passes for the sample on Claude and OpenAI.
- [ ] Adversarial review of PR 044.3.

### Phase 4: PR 044.4, projector and actions in serve

#### T14. `A2UIProjector`

- [ ] `serve/protocols/agui_a2ui.py`: an async generator over AG-UI events that applies spec section 4.5 rules 1 to 6.
- [ ] It uses `SurfaceStreamParser`, `validate_components`, and `build_operations`. It keeps no backend state.

**Verify:** the contract test replays each T0 input and compares the output to `expected.json`, ignoring only the volatile fields in the fixture README. R6, R7, R8.
**Files:** `serve/protocols/agui_a2ui.py`, `tests/unit/serve/test_agui_a2ui.py`.
**Depends on:** T0, T7, T8, T9. **Size:** S. **Model:** Opus `medium`, then `high`. **Spec:** D2, D6, R6 to R8.

#### T15. Action ingestion

- [ ] Serve validates `forwardedProps.a2uiAction.userAction`, removes it before the backend allowlists, and sends `A2UI user action:\n<json>` as the turn message. It is not echoed as `TEXT_MESSAGE_*`.
- [ ] An invalid action gives `RUN_ERROR` with a sanitized message.

**Verify:** unit tests for a valid action on both providers, an invalid action, and no allowlist warning. R9.
**Files:** `serve/protocols/agui_a2ui.py`, `serve/protocols/agui.py`, `tests/unit/serve/test_agui_a2ui.py`.
**Depends on:** T5. **Size:** S. **Model:** Sonnet `high`. **Spec:** R9.

#### T16. Wire binding, projector, and actions into `handle_request`

- [ ] `handle_request` builds the run binding, applies action ingestion, and wraps the backend events in the projector when the agent has an `a2ui` tool.
- [ ] Agents without an `a2ui` tool produce the same events as before.

**Verify:** integration tests with a fake backend stream on both providers. Existing serve tests unchanged.
**Files:** `serve/protocols/agui.py`, `tests/integration/serve/test_agui_a2ui_flow.py` (new).
**Depends on:** T5, T12, T14, T15. **Size:** S. **Model:** Opus `medium`. **Spec:** sections 4.1, 4.5, 4.7.

#### Checkpoint D (after T14 to T16)

- [ ] Live loop step 2 on Claude and OpenAI: two threads, two turns each, one surface with data, one invalid-then-valid repair, one action. Record the SSE excerpts for criteria 2, 3, and 4.
- [ ] Adversarial review of PR 044.4.

### Phase 5: PR 044.5, native A2UI in the CopilotKit sample

The sample does not use `CopilotChat`. `src/lib/conversation.ts` builds transcript rows (user, tool, group, answer) from AG-UI messages, and `src/components/chat/Conversation.tsx` renders them. So the sample needs its own surface row, as well as the catalog registration. File paths in this phase are relative to `sample/copilotkit/`. The sample's commands are `npm test` (Node test runner over `src/**/*.test.ts`), `npm run lint`, and `npm run build`.

#### T17. CopilotKit A2UI API check — ASK on surprise

Read the installed `@copilotkit/react-core`, `@copilotkit/runtime`, and `@copilotkit/a2ui-renderer` 1.87.1 types. Record the answers under "Progress and decisions" before T18 starts.

- [ ] How the app registers a catalog (`<CopilotKit a2ui={{ catalog }}>` or another prop), and whether the client sends the `A2UI_SCHEMA_CONTEXT_DESCRIPTION` context entry that HoloDeck reads (D1).
- [ ] Which component renders one `a2ui-surface` activity message outside `CopilotChat`, and how it reports a user action.
- [ ] Whether `CopilotRuntime` wraps agents in the A2UI middleware or injects `render_a2ui` by default, and how to turn that off.

**Verify:** the three answers, with the type names and file paths that prove them, are in this plan.
**Files:** this plan only. `package.json` and the lock file do not change.
**Depends on:** none. Can run any time before T18. **Size:** XS. **Model:** Sonnet `medium`. **Spec:** D1, D2.
If the client does not send the context entry, or the renderer cannot run outside `CopilotChat`, stop and ask the maintainer.

#### T18. Catalog and registration

- [ ] `src/lib/a2ui-catalog.ts` defines the booking catalog with the same ID and component names as `sample/a2ui-booking/catalogs/booking.json`.
- [ ] `src/app/layout.tsx` registers it with the API from T17.
- [ ] A test fails if the TypeScript catalog and the agent catalog differ in ID or component names.

**Verify:** `npm test` and `npm run lint`.
**Files:** `src/lib/a2ui-catalog.ts` (new), `src/lib/a2ui-catalog.test.ts` (new), `src/app/layout.tsx`.
**Depends on:** T13, T17. **Size:** S. **Model:** Sonnet `medium`. **Spec:** D1, criterion 1.

#### T19. Surface rows in the transcript

- [ ] `buildRows` in `src/lib/conversation.ts` adds a `surface` row for each activity message with `activityType` `"a2ui-surface"`, in run order, keyed by `messageId` so a new snapshot replaces the old one.
- [ ] `src/components/a2ui/SurfaceCard.tsx` renders the row with the T17 renderer, and shows the `building`, `retrying`, and `failed` states.
- [ ] Rows for other activity types and existing tool rows do not change.

**Verify:** `npm test` with a new `conversation.test.ts` that feeds messages from the T0 fixtures. `npm run build`.
**Files:** `src/lib/conversation.ts`, `src/lib/conversation.test.ts` (new), `src/components/a2ui/SurfaceCard.tsx` (new), `src/components/chat/Conversation.tsx`.
**Depends on:** T0, T18. **Size:** M. **Model:** Sonnet `high`. **Spec:** R6, R7, criterion 1.

#### T20. User actions from surfaces

- [ ] A user action from a surface starts a new run on the same thread with `forwardedProps.a2uiAction.userAction` set, using the T17 mechanism or `useAgent`.
- [ ] The UI does not show the action as a user bubble, because serve does not echo it (section 4.7).

**Verify:** `npm test` with a unit test for the `forwardedProps` builder. The browser check is in T24.
**Files:** `src/lib/a2ui-action.ts` (new), `src/lib/a2ui-action.test.ts` (new), `src/components/a2ui/SurfaceCard.tsx`.
**Depends on:** T19. **Size:** S. **Model:** Sonnet `high`. **Spec:** R9, criterion 4.

#### T21. Runtime route and sample docs

- [ ] `src/app/api/copilotkit/route.ts` keeps a plain `HttpAgent` to `holodeck serve /awp`, with no A2UI middleware and no second `render_a2ui` tool, using the T17 answer.
- [ ] `README.md` explains how to run the sample against `sample/a2ui-booking`, with `HOLODECK_BACKEND_URL` and `AGENT_ID`.

**Verify:** `npm run build`, `npm run lint`, `npm test`. A request to the route forwards `context` and `forwardedProps` unchanged (check with `holodeck serve` debug logging).
**Files:** `src/app/api/copilotkit/route.ts`, `README.md`.
**Depends on:** T17. **Size:** S. **Model:** Sonnet `medium`. **Spec:** D2, criterion 1.

#### Checkpoint E (after T17 to T21)

- [ ] `npm test`, `npm run lint`, and `npm run build` pass in `sample/copilotkit`.
- [ ] Against a local `holodeck serve` on the `a2ui-booking` sample, the browser pane shows one surface and the action reaches the agent. A screenshot is recorded.
- [ ] Adversarial review of PR 044.5.

### Phase 6: PR 044.6, guide and live evidence

#### T22. User guide

- [ ] `docs/guides/a2ui.md`: configuration, catalog sources and the allowlist, the event sequence, actions, limits, and troubleshooting.
- [ ] Links from the tools guide, the serve guide, and the docsite navigation.

**Verify:** `make harness-check` and `uv run mkdocs build` with no new warnings.
**Files:** `docs/guides/a2ui.md` (new), `docs/guides/tools.md`, `docs/guides/serve.md`, `mkdocs.yml`.
**Depends on:** T16. Write the frontend section after T21. **Size:** M. **Model:** Sonnet `medium`. **Spec:** section 7.

#### T23. Live validation and closeout

- [ ] Live loop step 3: the maintainer drives the CopilotKit sample on Claude and on OpenAI and confirms the surface, the repair, and the action. Record what the UI showed.
- [ ] Fill the acceptance table. Update the spec status, the inventory row, and `docs/PLANS.md`.
- [ ] Move this plan to `completed/` only when all six criteria pass.

**Verify:** the acceptance table has evidence for every row. `make harness-check`.
**Files:** this plan, `docs/product-specs/044-a2ui-surfaces/spec.md`, `docs/product-specs/inventory.md`, `docs/PLANS.md`.
**Depends on:** Checkpoint E, T22. **Size:** S. **Model:** Fable inline. **Spec:** section 11.

#### Checkpoint F (complete)

- [ ] All six acceptance criteria have evidence.
- [ ] One integrated adversarial pass over `git diff main...<top of stack>` before the stack merges.

## Delivery

| PR | Tasks | Base |
| --- | --- | --- |
| 044.1 OpenAI AG-UI parity | T1 to T5 | `main` |
| 044.2 A2UI core | T0, T6 to T9 | `main` |
| 044.3 Tool configuration and adapters | T10 to T13 | 044.2 |
| 044.4 Projector and actions | T14 to T16 | 044.3, rebased on `main` after 044.1 merges |
| 044.5 Native A2UI in the CopilotKit sample | T17 to T21 | 044.4 |
| 044.6 Guide and live evidence | T22, T23 | 044.5 |

044.5 is its own PR because its reviewers are frontend reviewers. Use `gh stack` for 044.2 to 044.6. Remote writes (submit, push, merge) need maintainer approval each time.

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| `StopAtTools` cannot resume from a `tool` message with the same history | High: blocks R10 and the one-path handler | T1 spike runs first. On failure, the maintainer picks between deferring OpenAI frontend tools and another design. A2UI itself does not need frontend tools. |
| The Responses stream does not expose function-call argument deltas through the Agents SDK | High: no progressive paint on OpenAI (criterion 2) | T2 starts with a recorded stream. Fallback: paint on `TOOL_CALL_END` for OpenAI only, recorded as a known gap with the maintainer's approval. |
| The run binding leaks between concurrent sessions, because tools run in-process | High: one client sees another's catalog | T11 has a concurrency test with two sessions and different catalogs. |
| The middleware wire shape changes in a later 0.0.x release | Medium: CopilotKit stops rendering | T0 pins fixtures to 0.0.16 and commit `ed94ca4`. The guide names the tested version. |
| Large catalogs in the tool description cost prompt tokens every turn | Medium: cost and latency | `max_catalog_bytes` limit. Deferred loading is listed in spec section 13. |
| T5 changes event order for existing OpenAI clients | Medium: regressions in current users | ASK checkpoint. Existing serve tests must pass, and any changed assertion is explained. |
| `jiter` partial mode reports a closed array before the closing bracket arrives | Low: paint of incomplete components | T8 every-byte split test. The projector validates before it paints (R8). |
| `CopilotRuntime` adds its own A2UI middleware, so `render_a2ui` is injected twice | High: duplicate tools and double surfaces | T17 finds the default. T21 turns it off in the route. |
| The CopilotKit A2UI renderer works only inside `CopilotChat` | High: the custom transcript cannot draw surfaces | T17 checks first and stops for the maintainer. Fallback: render with `@a2ui/web_core` directly in `SurfaceCard`. |
| The TypeScript catalog drifts from the agent catalog | Medium: the agent renders components the app does not have | T18 parity test fails on any ID or component-name difference. |

## Acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| 1. CopilotKit renders on both backends, no middleware | Checkpoint E, T23 | Open |
| 2. First snapshot before `TOOL_CALL_END` | Checkpoint D | Open |
| 3. Repair in one run | Checkpoint D | Open |
| 4. Action round trip | Checkpoint D, T20, T23 | Open |
| 5. Contract test passes | T14 | Open |
| 6. `holodeck test run` passes | T13, Checkpoint C | Open |

## Progress and decisions

- 2026-11-27: Spec decisions D1 to D11 settled with the maintainer. `jsonschema` 4.26 is already a core dependency. `jiter` 0.19 is in `uv.lock` through the Anthropic and OpenAI SDKs. `openai-agents` 0.24 exports `StopAtTools` and `Agent(tool_use_behavior=...)`, which T1 must still prove for resume.
- 2026-11-27: Plan revised to the task breakdown standard. T11 (L) split into T11 run binding and T12 adapters. Later tasks renumbered by one. Added files per task, the dependency graph, parallel-work notes, the slicing decision, a risk table, and Checkpoint E.
- 2026-11-27: Maintainer asked for native A2UI support in the CopilotKit sample as part of this plan. The sample renders its own transcript (`src/lib/conversation.ts`), so the one sample task became T17 to T21 (API check, catalog, surface rows, actions, route) in a new PR 044.5. The guide and closeout moved to PR 044.6 as T22 and T23. Checkpoint E covers the sample; the final checkpoint is now F.

## Validation results and remaining blockers

None yet. No task has started.
