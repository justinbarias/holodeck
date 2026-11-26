# SPEC: Native A2UI Surfaces

**Status:** Draft 2026-11-27. Decisions D1 to D11 were settled with the maintainer on 2026-11-27. Nothing is built. Plan: [execution plan](../../exec-plans/active/044-a2ui-surfaces/index.md).
**Code baseline:** `bab06fe`.
**Sources:** [A2UI v1.0 specification](https://a2ui.org/specification/v1.0-a2ui/), `@ag-ui/a2ui-middleware` 0.0.16 (AG-UI repository, commit `ed94ca4`, `middlewares/a2ui-middleware/src`).
**Related:** [017 agent local server](../017-agent-local-server/index.md), [035 OpenAI Agents backend](../035-openai-agents-backend/index.md), [Server guide](../../guides/serve.md).

## 1. Objective

Let a HoloDeck agent draw interactive UI in a frontend app with A2UI 1.0, with no Node middleware between the app and `holodeck serve`. A2UI is a declarative UI format: the agent sends component trees and data, and the client renders them with its own component catalog. A catalog is the set of components, with their property schemas, that a renderer can draw.

Today a HoloDeck user who wants A2UI must put `@ag-ui/a2ui-middleware` in a Node runtime in front of `holodeck serve`. That middleware adds the `render_a2ui` tool as a frontend tool, validates in TypeScript, and runs the agent again after each render. It works on the Claude path only, because the OpenAI Agents path of `holodeck serve` ignores AG-UI frontend tools and context.

This feature makes A2UI a HoloDeck tool type. HoloDeck validates components in Python, returns errors to the model in the same run, and emits the same AG-UI events that the TypeScript middleware emits. A CopilotKit app that sets `a2ui={{ catalog }}` works against `holodeck serve` with no other change.

Users:

- Frontend developers, who own the component catalog and the renderer.
- Agent authors, who add one `a2ui` tool to `agent.yaml`.
- Agents, which call `render_a2ui` and receive user actions as messages.
- Operators, who run `holodeck serve` and control which catalogs it accepts.

User stories:

1. As a frontend developer, I register my catalog in `<CopilotKit a2ui={{ catalog }}>`, point the app at `holodeck serve /awp`, and the agent renders surfaces with my components.
2. As an agent author, I enable A2UI with one tool block and no Python.
3. As an agent author, my agent works the same on the Claude and OpenAI backends.
4. As a user, I see the surface build while the model streams it, not only after the tool call ends.
5. As a user, I click a button on a surface and the agent receives the action in the same thread.
6. As an agent, when I send an invalid component tree I get the validation errors as the tool result and I can fix them in the same run.
7. As an operator, the server rejects a client catalog whose ID is not in the allowlist in `agent.yaml`.
8. As an agent author, `holodeck test` and `holodeck chat` run an agent that has the `a2ui` tool, with the catalog from `agent.yaml`.

## 2. Scope

| Area | In the MVP | Notes |
| --- | --- | --- |
| A2UI protocol | 1.0 only | `createSurface`, `updateComponents`, `updateDataModel`, `deleteSurface` from the agent. `action` from the renderer. |
| Catalog sources | YAML and client | YAML lists allowed catalog IDs, with an optional local schema. The client sends its catalog in AG-UI context. |
| Execution | Backend tool | Python validation in the tool and in the stream projector. |
| Wire | Middleware-compatible | `ACTIVITY_SNAPSHOT`, `activityType` `"a2ui-surface"`, `content.a2ui_operations`, actions in `forwardedProps.a2uiAction`. |
| Streaming | Progressive paint | Paint when the `components` array closes, then stream data. |
| Backends | Claude and OpenAI Agents | The OpenAI path gains AG-UI parity first (section 4.6). |
| Hosts | `serve` AG-UI, `test`, `chat` | REST `serve` returns operations in the tool result only. |
| Frontend sample | `sample/copilotkit` | Registers the catalog, draws `a2ui-surface` activity messages in its own transcript, and sends actions. No A2UI middleware in the runtime route. |

Out of scope: A2UI v0.9, `callRendererFunction`, `callAgentFunction` and the function-response messages, A2A and MCP bindings of A2UI, a HoloDeck renderer, server-side catalog hosting, Temporal activities, and new `holodeck test` assertion types for surfaces.

## 3. Decisions

| ID | Decision | Reason |
| --- | --- | --- |
| D1 | Catalogs come from both `agent.yaml` and the client. `agent.yaml` is the allowlist of catalog IDs. | The app owns its components. The operator controls what the server accepts. `test` and `chat` have no client, so they need a YAML schema. |
| D2 | The AG-UI wire shape matches `@ag-ui/a2ui-middleware`. | CopilotKit's A2UI renderer and other AG-UI clients work with no adapter. |
| D3 | `render_a2ui` is a backend tool that HoloDeck runs in Python. | The model gets validation errors in the same run. The same tool works on both backends. No second agent run. |
| D4 | The MVP covers A2UI 1.0 with progressive paint. | One protocol version keeps the validator small. Progressive paint is the main UX gain of A2UI. |
| D5 | The OpenAI Agents AG-UI path gains parity with the Claude path before A2UI lands: frontend tools, context, state, `forwardedProps`, and tool-argument deltas. | Maintainer request on 2026-11-27. D3 and D4 need tool-argument deltas on both paths. |
| D6 | One stream projector turns `render_a2ui` tool events into activity events, on both backend paths. | One implementation of the wire shape. Backends do not know about A2UI. |
| D7 | The default tool name is `render_a2ui`. | The same name as the middleware default, so client code and prompts carry over. |
| D8 | Validation uses `jsonschema`, which is already a core dependency. The stream parser uses `jiter` partial mode, and `jiter` becomes a direct dependency. | `jiter` is already in the lock file through the Anthropic and OpenAI SDKs. No new package. |
| D9 | The OpenAI path stops at frontend tools with `StopAtTools` (openai-agents 0.24) and resumes from history plus the `tool` message. 044.1 starts with a live spike that proves resume. | The API exists. Resume behavior is the risk, so it is proved first. |
| D10 | `holodeck test` gets no surface assertion in the MVP. Tool-call assertions on `render_a2ui` cover it. | Keeps the test case schema unchanged. Tracked as technical debt. |
| D11 | REST `serve` returns A2UI operations only inside the `render_a2ui` tool result. | No change to the REST contract. |

Decisions D8 to D11 were settled with the maintainer on 2026-11-27.

### Working assumptions

- A client catalog wins over the YAML schema for the same ID, because the client renders it. The YAML schema is the fallback when the client sends none.
- The client sends its catalog as the AG-UI context entry whose `description` equals the middleware constant `A2UI_SCHEMA_CONTEXT_DESCRIPTION`. Its `value` is JSON `{ "catalogId", "components" }`.
- The basic catalog `https://a2ui.org/specification/v1_0/basic_catalog.json` ships as a package resource, so an agent can use it with no file.

## 4. Architecture

### 4.1 Request and stream path

```text
CopilotKit app ──POST /awp RunAgentInput──▶ AGUIProtocol.handle_request
   context[A2UI schema]   forwardedProps.a2uiAction
                │
                ▼
   A2UIRunBinding.from_request(input, tool_config)      (catalog + action)
                │
                ├─▶ executor turn (Claude send_agui | OpenAI stream)
                │        └─ render_a2ui tool: validate(args, binding.catalog)
                │             └─ result: {"status":"rendered"} or errors
                ▼
   AG-UI event stream (TOOL_CALL_START/ARGS/END/RESULT, TEXT_*, …)
                │
                ▼
   A2UIProjector.project(events, binding)
                │  adds ACTIVITY_SNAPSHOT "a2ui-surface" after each paint
                ▼
   AGUIEventStream.encode ──SSE──▶ client renderer
```

### 4.2 Ports and seams

| Seam | Location | Contract |
| --- | --- | --- |
| Tool config | `models/tool.py` | New `A2UITool` (`type: a2ui`) in `ToolUnion`. |
| A2UI core | `lib/a2ui/` (new) | Pure Python. No AG-UI, no backend imports. |
| Catalog resolution | `lib/a2ui/catalog.py` | `resolve_catalog(tool, client_entry) -> ResolvedCatalog` or `CatalogRejected`. |
| Validation | `lib/a2ui/validate.py` | `validate_components(components, catalog) -> list[A2UIIssue]`. |
| Incremental parse | `lib/a2ui/stream.py` | `SurfaceStreamParser.feed(delta) -> list[SurfaceStep]`. |
| Operations | `lib/a2ui/ops.py` | `build_operations(surface_id, catalog_id, components, data) -> list[A2UIMessage]`. |
| Run binding | `lib/a2ui/binding.py` | `A2UIRunBinding` holds the catalog and the action for one run. The tool adapter reads it. |
| Backend tool | `lib/backends/tool_adapters.py`, `openai_agents_tool_adapters.py` | Both build `render_a2ui` from `A2UITool` and the run binding. Same validation, same result shape. |
| Tool-argument deltas | both backends, `ToolEvent` | New `ToolEvent.kind` `"args_delta"`, so the OpenAI path emits `TOOL_CALL_ARGS`. Claude already emits them. |
| Projection | `serve/protocols/agui_a2ui.py` (new) | `A2UIProjector.project(events) -> AsyncIterator[BaseEvent]`. Wraps both AG-UI paths in `handle_request`. |
| Action ingestion | `serve/protocols/agui.py` | Reads `forwardedProps.a2uiAction`, removes it before backend allowlists, and turns it into the turn message. |

The core owns no I/O and no event types. The projector and the tool adapters are the only consumers of the core.

### 4.3 Configuration

```yaml
tools:
  - name: render_a2ui
    type: a2ui
    description: Render interactive UI for the user.
    catalogs:
      - id: https://acme.dev/catalogs/booking/v1
        schema: ./catalogs/booking.json   # optional; used when the client sends none
      - id: https://a2ui.org/specification/v1_0/basic_catalog.json
    default_catalog: https://acme.dev/catalogs/booking/v1
    accept_client_catalogs: true          # default true
    limits:
      max_catalog_bytes: 262144
      max_components: 200
      max_repair_attempts: 3
```

Rules:

- `catalogs` must have at least one entry. IDs are unique.
- A client catalog with an ID that is not in `catalogs` is rejected. The run continues without A2UI, and the server logs a warning with the ID.
- If `accept_client_catalogs` is false, the server ignores client catalogs and uses YAML schemas only.
- A catalog entry without `schema` needs a client catalog at run time. `holodeck test` and `holodeck chat` fail at load time when `default_catalog` has no `schema`, with the message `a2ui tool '<name>': default_catalog '<id>' needs a schema for test and chat`.
- An agent can have at most one `a2ui` tool.

### 4.4 Tool contract

The tool input schema is the middleware schema, so prompts carry over:

```json
{
  "surfaceId": "string",
  "components": [ { "id": "string", "component": "string", "...": "..." } ],
  "data": { "path": "/", "value": {} }
}
```

`surfaceId` and `components` are required. The tool description includes the catalog's component names and property schemas for the run. The binding sets the description per run on the Claude AG-UI path and per session on other paths.

Results:

- Valid: `{"status": "rendered", "surfaceId": "<id>", "catalogId": "<id>"}`.
- Invalid: `{"status": "invalid", "issues": [{"path": "/components/3/component", "message": "Unknown component 'Chart'"}]}`. The model can call the tool again. After `max_repair_attempts` invalid calls for one `surfaceId` in one run, the result is `{"status": "failed"}` and the projector paints nothing for that surface.

### 4.5 Projection rules

The projector reproduces the middleware wire shape:

1. `TOOL_CALL_START` for the A2UI tool opens a stream entry and emits a lifecycle snapshot with `status: "building"`.
2. `TOOL_CALL_ARGS` deltas feed `SurfaceStreamParser`. When the `components` array closes and validates, the projector emits `ACTIVITY_SNAPSHOT` with `messageId` `a2ui-surface-<toolCallId>`, `activityType` `"a2ui-surface"`, `replace: true`, and `content.a2ui_operations` set to `createSurface` then `updateComponents`.
3. When `data` closes, the projector emits a new snapshot on the same `messageId` that adds `updateDataModel`.
4. If the components do not validate, the projector emits a snapshot with `status: "retrying"` and paints nothing.
5. `TOOL_CALL_RESULT` with `status: "failed"` gives `status: "failed"` on the same `messageId`.
6. All other events pass through unchanged and in order.

Snapshots for one tool call always reuse one `messageId`, so the client replaces the surface in place.

### 4.6 OpenAI Agents AG-UI parity (prerequisite)

The OpenAI path of `AGUIProtocol.handle_request` passes only the text message to the executor. This phase adds a `send_agui` path to the OpenAI Agents session with the same contract as Claude's:

- AG-UI `tools` become frontend tools. The run stops at a frontend tool call, emits its `TOOL_CALL_*` events, and resumes on the next request whose latest message has role `tool`.
- AG-UI `context` and `state` go into the instructions for the run, with the same addendum as Claude.
- An allowlist for `forwardedProps`, with a warning for unknown keys.
- Function-call argument deltas from the Responses stream become `ToolEvent(kind="args_delta")` and then `TOOL_CALL_ARGS`.

After this phase, `handle_request` has one code path for both providers.

### 4.7 Actions

A request with `forwardedProps.a2uiAction.userAction` carries an A2UI `action` message: `name`, `surfaceId`, `sourceComponentId`, `timestamp`, `context`. The server validates it, removes `a2uiAction` from `forwardedProps`, and sends the turn message `A2UI user action:\n<json>`. This matches the middleware text, so prompts carry over. The server does not echo this message as a `TEXT_MESSAGE_*` event.

## 5. Requirements (EARS)

| ID | Requirement |
| --- | --- |
| R1 | When `agent.yaml` has a tool with `type: a2ui`, the loader shall validate it as `A2UITool` and the published schema shall include it. |
| R2 | When an AG-UI request has the A2UI schema context entry with an allowed catalog ID, the server shall use that catalog for the run. |
| R3 | When an AG-UI request has a catalog ID that is not allowed, the server shall not use it and shall log `a2ui: rejected client catalog '<id>'`. |
| R4 | When the model calls the A2UI tool with components that validate, the tool shall return `status: "rendered"`. |
| R5 | When the model calls the A2UI tool with components that do not validate, the tool shall return `status: "invalid"` with one issue per error, each with a JSON pointer. |
| R6 | While a valid A2UI tool call streams on either backend, the server shall emit an `ACTIVITY_SNAPSHOT` with `createSurface` and `updateComponents` before `TOOL_CALL_END`. |
| R7 | When the `data` argument closes, the server shall emit a snapshot on the same `messageId` that adds `updateDataModel`. |
| R8 | The server shall never emit an `ACTIVITY_SNAPSHOT` that contains components that did not validate. |
| R9 | When a request has `forwardedProps.a2uiAction`, the agent shall receive the action as the turn message on both backends. |
| R10 | When the OpenAI backend serves an AG-UI request with frontend tools, the server shall emit their `TOOL_CALL_*` events and stop the run at the call. |
| R11 | When `holodeck test` runs an agent with an `a2ui` tool, the tool shall validate against the YAML schema of `default_catalog`. |
| R12 | If a client catalog is larger than `max_catalog_bytes`, the server shall reject it with the R3 log message. |

## 6. Commands

```bash
uv run pytest tests/unit/lib/a2ui -n auto -q
uv run pytest tests/unit/serve/test_agui_a2ui.py tests/unit/lib/backends -n auto -q
uv run black src/holodeck/lib/a2ui src/holodeck/serve/protocols
uv run ruff check src/holodeck/lib/a2ui src/holodeck/serve/protocols
uv run mypy src/holodeck/lib/a2ui src/holodeck/serve/protocols
make schema
make schema-check
make harness-check
```

Live validation follows the loop in `AGENTS.md`: `holodeck test run`, raw AG-UI `curl -N` on two threads with two turns each, then the CopilotKit sample with an A2UI catalog.

## 7. Project structure

```text
src/holodeck/lib/a2ui/
  __init__.py
  catalog.py        catalog models, resolution, basic catalog resource
  validate.py       component validation against a catalog
  stream.py         incremental JSON parser for tool arguments
  ops.py            A2UI 1.0 message builders
  binding.py        per-run catalog and action binding
  resources/basic_catalog_v1_0.json
src/holodeck/serve/protocols/agui_a2ui.py   projector and action ingestion
tests/unit/lib/a2ui/
tests/unit/serve/test_agui_a2ui.py
tests/fixtures/a2ui/                         catalogs and recorded event streams
sample/copilotkit/                           catalog, surface rows, actions; no A2UI middleware
docs/guides/a2ui.md                          user guide
```

## 8. Code style

Pure core functions take typed models and return typed models. SDK and wire data become models at the boundary.

```python
class A2UIIssue(BaseModel):
    path: str  # JSON pointer into the tool arguments
    message: str


def validate_components(
    components: list[ComponentNode], catalog: ResolvedCatalog
) -> list[A2UIIssue]:
    """Return one issue per catalog violation. An empty list means valid."""
```

No `Any` or `getattr` across the core boundary. The projector is an async generator that yields AG-UI `BaseEvent` objects and owns no backend state.

## 9. Testing strategy

| Level | What | Where |
| --- | --- | --- |
| Unit | Catalog resolution, allowlist, size limits, validation issues, operation builders | `tests/unit/lib/a2ui` |
| Unit | Incremental parser on deltas split at every byte of a recorded payload | `tests/unit/lib/a2ui/test_stream.py` |
| Contract | Projector output equals the event sequence that `@ag-ui/a2ui-middleware` 0.0.16 emits for the same recorded input, as JSON fixtures | `tests/unit/serve/test_agui_a2ui.py` |
| Unit | OpenAI `send_agui`: frontend tool stop and resume, context addendum, argument deltas | `tests/unit/lib/backends` |
| Integration | `holodeck serve` AG-UI with a fake backend stream, both providers | `tests/integration/serve` |
| Live | Claude and OpenAI sample agents through serve and CopilotKit | Exec plan evidence |

Fixtures for the contract test come from one run of the TypeScript middleware, recorded once and committed. The test does not run Node.

## 10. Boundaries

- Always: keep the A2UI core free of AG-UI and backend imports. Run the contract test for any projector change. Regenerate the schema for any `A2UITool` change.
- Ask first: a runtime dependency other than the `jiter` promotion in D8. Any change to the wire shape in D2. Supporting v0.9.
- Never: paint unvalidated components. Trust a client catalog ID that is not in the allowlist. Send A2UI activity content back to the model.

## 11. Success criteria

1. The CopilotKit sample, with no A2UI middleware in its runtime route, renders a surface from `holodeck serve` on Claude and on OpenAI. An app that uses `CopilotChat` needs only the catalog registration.
2. The first `ACTIVITY_SNAPSHOT` for a surface arrives before `TOOL_CALL_END` for that call, on both backends, in the raw `curl -N` stream.
3. An invalid component tree gets `status: "invalid"`, and the model's second call renders in the same run, with one `RUN_STARTED` and one `RUN_FINISHED`.
4. A button click reaches the agent as an action message, and the next turn updates the same `surfaceId`.
5. The contract test passes against the recorded middleware fixtures.
6. `holodeck test run` passes on a sample agent with the `a2ui` tool.

## 12. Delivery

A stack of six PRs, bottom first:

| PR | Layer | Depends on |
| --- | --- | --- |
| 044.1 | OpenAI Agents AG-UI parity (section 4.6) | none |
| 044.2 | A2UI core: catalog, validation, parser, operations | none |
| 044.3 | `A2UITool` config, schema, run binding, tool adapters for both backends | 044.2 |
| 044.4 | Projector and action ingestion in serve | 044.1, 044.3 |
| 044.5 | Native A2UI in the CopilotKit sample: catalog, surface rows, actions, route | 044.4 |
| 044.6 | User guide, live validation | 044.5 |

044.1 and 044.2 are independent and can target `main` in parallel.

## 13. Trade-offs and what to revisit

- Validation runs twice, in the projector and in the tool. The projector needs it to avoid painting bad trees before the tool runs. Both call one function.
- Matching the middleware wire shape ties HoloDeck to a 0.0.x package convention. Revisit when AG-UI or A2UI publishes a standard activity type.
- The tool description carries the catalog, so large catalogs cost prompt tokens on every turn. Revisit with deferred loading.

## 14. Open questions

None. D8 to D11 closed the four questions of the first draft. If the 044.1 spike shows that `StopAtTools` cannot resume from a `tool` message, the maintainer decides between deferring OpenAI frontend tools and another resume design.
