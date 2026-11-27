# SPEC: Backend-Agnostic Chat Thread Store

**Status:** Draft. Decisions in section 3 were settled with the maintainer on 2026-11-15. Not built.
**Date:** 2026-11-15.
**Code baseline:** `a903683`.
**SDK baseline:** `claude-agent-sdk` 0.2.179, `openai-agents` 0.24.0.
**Related:** [017 agent local server](../017-agent-local-server/index.md), [021 Claude Agent SDK](../021-claude-agent-sdk/index.md), [034 production hardening](../034-production-hardening/index.md), [035 OpenAI Agents backend](../035-openai-agents-backend/index.md), [040 Temporal](../040-holodeck-temporal/index.md). Design: [thread store design](../../design-docs/045-thread-store/design.md).

## 1. Objective

Give `holodeck serve` durable chat threads that work the same way on the Claude backend and the OpenAI Agents backend.

Today a thread lives in process memory. `SessionStore` in `serve/session_store.py` maps a thread id to a live `AgentExecutor` and drops it after 30 minutes without activity. The Claude backend keeps history in a JSONL file that the CLI owns. The OpenAI backend keeps history in an in-memory `SQLiteSession`. A restart, a TTL expiry or a second replica loses the thread. No endpoint lists threads or returns their messages.

This feature adds one store, one schema and one API for both backends.

Users:

- Chat users, who return to a conversation after a restart or on another replica.
- UI developers, who list threads and reload their history.
- Operators, who choose a storage engine and a retention period in `agent.yaml`.

User stories:

1. As a chat user, I continue a thread after `holodeck serve` restarts and the agent remembers the earlier turns, including tool results.
2. As a chat user, I continue a thread on a different replica from the one that started it.
3. As a UI developer, I list one owner's threads and load a thread's messages in one shape for both backends.
4. As an operator, I enable a durable store with one `threads:` block and no Python.
5. As an operator, I set a retention period and old threads are deleted.
6. As a chat user, I see a short title on each thread without typing one.

## 2. Scope

In scope:

- A `ThreadStore` port in `lib/` with in-memory, SQLite and PostgreSQL adapters.
- Native state persistence for both SDKs through the port.
- A canonical message log in a HoloDeck model.
- Per-thread leases, so that replicas do not run two turns on one thread.
- REST endpoints for threads and an AG-UI `MESSAGES_SNAPSHOT` event.
- A `threads:` block in `agent.yaml`.
- Automatic thread titles.

Non-goals:

- Moving a thread between backends.
- Forking or branching a thread.
- Editing or regenerating an earlier message.
- Application-level encryption of message content.
- Authentication. The store filters on an owner string that the caller supplies.
- `holodeck chat`, `holodeck test` and Temporal as consumers. The port is designed for Temporal activities, but v1 ships no Temporal wiring.
- Search across threads and thread export.

## 3. Decisions

| # | Decision | Choice |
| --- | --- | --- |
| D1 | Drivers | Survive restart and TTL expiry; list and reload history; multiple `serve` replicas; a port that Temporal activities can call. |
| D2 | Ownership of history | Hybrid. HoloDeck owns the thread record and a canonical log. Each backend also stores its native state through an SDK bridge in the same store. |
| D3 | Surfaces | `holodeck serve` only. The port lives in `lib/`. |
| D4 | Backend portability | None. A thread is pinned to the backend that created it. |
| D5 | Engines | In-memory, SQLite, PostgreSQL. |
| D6 | History shape | A HoloDeck Pydantic model that maps one-to-one to AG-UI 1.0 messages. |
| D7 | Ownership scope | An optional opaque `owner` string. List and get filter on it. |
| D8 | Temporal | Contract only: async calls, idempotent appends, no process-local state in the port. |
| D9 | Concurrent turns | A per-thread lease in the store. The second turn is rejected. |
| D10 | Canonical log write | Written by HoloDeck from normalised events: the user message at turn start, the rest at turn end. |
| D11 | Authority on an AG-UI run | The store. For a known thread, only the new user message and tool results are read from the request. |
| D12 | Database access | Native drivers and hand-written SQL: `psycopg` 3 and `aiosqlite`. |
| D13 | Schema changes | Versioned DDL in code, applied at startup under a lock, recorded in `schema_version`. |
| D14 | Retention | Threads persist. The 30-minute TTL evicts only the live executor. `retention_days` is optional. DELETE removes the log and the native state. |
| D15 | Configuration | A validated `threads:` block in `agent.yaml`. |
| D16 | Read API | REST endpoints under `/threads`, and AG-UI `MESSAGES_SNAPSHOT`. `/sessions` stays as an alias. |
| D17 | Failed turn | The user message stays, the turn is marked `failed`, native state is kept, the lease expires. |
| D18 | Store outage | The run fails with a clear error. Readiness reports the store as down. |
| D19 | Attachments | The canonical log stores name, MIME type and size. It does not store bytes. |
| D20 | Thread metadata | `title` and a `metadata` dict, both editable. A title is generated automatically. |
| D21 | Configuration drift | Resume is allowed after `agent.yaml` changes. Each turn records the agent configuration hash. A different backend or agent name is rejected. |
| D22 | Default | In-memory when there is no `threads:` block. Behaviour does not change for current users. |
| D23 | Replica proof | An integration test with two `serve` processes on one PostgreSQL database, for both backends. |
| D24 | Title generation | In the background after a completed turn on a thread with no title (normally the first), on the thread's own backend. A model override and an off switch exist. A client-set title is never overwritten. |
| D25 | Owner source | A trusted header (default `X-HoloDeck-Owner`). AG-UI forwarded props are the fallback. |
| D26 | Thread id | The client's AG-UI `thread_id` is the key and is globally unique. REST creates mint a ULID. |
| D27 | Deliverable | This spec and a design document. The execution plan follows review. |
| D29 | Recovery after a Claude mirror failure | On the next turn, a replica that still has the local transcript file sends its entries to the store again before it resumes. On any other replica the loss is accepted and the turn stays `failed`. |
| D30 | Snapshot size | `MESSAGES_SNAPSHOT` carries the newest `threads.history.snapshot_messages` messages (default 200). When older messages exist, the server also sends a cursor for them. Older messages are read over REST. |
| D31 | Turn status on messages | Each message in a REST response carries `turn_status` (`running`, `completed` or `failed`). |
| D32 | Client pattern | The server does not assume a client uses only REST or only AG-UI, and cannot control a client's own store. `sample/copilotkit` is the reference pattern: load over REST on page load, treat the snapshot as authoritative, page older messages over REST. |
| D33 | Backend interface | `AgentBackend.create_session` gains an optional `thread: ThreadBinding` argument. `None` keeps today's behaviour. Approved 2026-11-15. |
| D34 | Compaction | Both backends compact long threads in v1. Claude: the CLI compacts by itself, and the store and resume must carry that through. OpenAI: the bridge is wrapped in the SDK's `OpenAIResponsesCompactionSession`, which replaces old native items with a compacted item. The canonical log is never compacted. |
| D35 | Unknown thread with history | The server creates the thread and imports the client's earlier messages once. After that the store is authoritative. |
| D36 | Owner fallback on deploy | `holodeck deploy` sets `threads.owner.allow_forwarded_props` to `false` unless the agent file sets it, and warns when it is `true`. |
| D37 | Lost title task | No retry. A thread without a title gets one after its next completed turn. |
| D28 | OpenAI item limit | A turn loads at most `threads.history.max_items` native items (default 200). The window always starts at a user message. With compaction on (D34) the limit is a safety bound that is rarely reached; it is the only bound where compaction is unavailable. The Claude backend is not limited: its CLI needs the full transcript and compacts by itself. |

### Working assumptions

The maintainer did not decide these. Each can change in review.

| # | Assumption |
| --- | --- |
| A1 | Lease expiry is 60 seconds. The holder renews it every 20 seconds while a turn runs. |
| A2 | List and message endpoints use cursor paging. Threads sort by last activity, newest first. |
| A3 | Each store operation has an OpenTelemetry span. Message content follows the existing redaction rules. |
| A4 | `holodeck deploy` warns when the provider is `sqlite`, because a container file system is not durable. |
| A5 | The Claude bridge uses flush mode `eager`. |
| A6 | The forwarded-props owner fallback can be disabled with `threads.owner.allow_forwarded_props: false`. See D36 for the deploy default. |
| A7 | Promoted to D35. |
| A8 | OpenAI compaction starts when 40 or more candidate items are stored (`threads.compaction.openai_min_items`). The SDK default of 10 would compact almost every turn. |

## 4. Architecture

```text
            AG-UI /awp            REST /threads
                 |                     |
                 v                     v
        +-------------------------------------+
        |  serve: ThreadService               |
        |  lease, turn lifecycle, owner check |
        +-------------------------------------+
           |                |              |
           v                v              v
   canonical log     native state      thread record
   (ThreadMessage)   (SDK entries)     (owner, backend, title)
           \                |              /
            v               v             v
        +-------------------------------------+
        |  ThreadStore port (lib/threads)     |
        +-------------------------------------+
           |            |             |
        memory        SQLite      PostgreSQL

   SDK bridges (implement an SDK protocol, call the port):
   ClaudeSession --> ClaudeNativeStore (SDK SessionStore protocol) --+
   OpenAIAgentsSession --> OpenAINativeSession (SDK Session protocol) +--> ThreadStore
```

Three kinds of data share one store:

- **Thread record.** Id, owner, backend, agent name, title, metadata, timestamps, lease.
- **Canonical log.** `ThreadMessage` rows that clients read. HoloDeck writes them.
- **Native state.** Opaque JSON entries that each SDK writes and reads back to resume a thread.

Native state is authoritative for resume. The canonical log is authoritative for reading. The two can differ after a compaction, and that is expected: the model sees the compacted native state, and the user sees the full log.

### How each SDK plugs in

| | Claude Agent SDK | OpenAI Agents SDK |
| --- | --- | --- |
| Protocol | `SessionStore` (`append`, `load`, `delete`, `list_subkeys`) | `Session` (`get_items`, `add_items`, `pop_item`, `clear_session`) |
| How it is set | `ClaudeAgentOptions.session_store` | `Runner.run(..., session=...)` |
| What it stores | Transcript lines, one JSON object each | Response input items |
| Write model | A mirror. The CLI writes local disk first, then the SDK calls `append`. | Direct. The SDK calls `add_items` during the run. |
| Resume | `resume=<id>` loads from the store into a temporary file. | `get_items` on each run. |
| Write failure | Retried three times, then dropped and reported as `MirrorErrorMessage`. The run continues. | The exception fails the run. |

The Claude write model has one consequence. The SDK does not fail a turn when a mirror write is lost. HoloDeck must do that itself (TS-09).

### Configuration

```yaml
threads:
  provider: postgres            # memory (default) | sqlite | postgres
  connection: ${THREADS_DSN}    # sqlite: a file path; postgres: a connection string
  retention_days: 90            # optional; omitted means keep
  lease_seconds: 60
  history:
    max_items: 200              # OpenAI backend only; null means no limit
    snapshot_messages: 200      # newest messages in MESSAGES_SNAPSHOT
  compaction:
    enabled: true               # OpenAI backend; the Claude CLI always compacts by itself
    openai_min_items: 40
    openai_model: null          # null means the agent's own model
  owner:
    header: X-HoloDeck-Owner
    allow_forwarded_props: true
  title:
    enabled: true
    model: null                 # null means the agent's own model
```

### REST API

| Method and path | Result |
| --- | --- |
| `GET /threads?cursor=&limit=` | The caller's threads, newest activity first. |
| `GET /threads/{id}` | One thread record. |
| `GET /threads/{id}/messages?cursor=&limit=` | Canonical messages in order. |
| `PATCH /threads/{id}` | Sets `title` or `metadata`. |
| `DELETE /threads/{id}` | Removes the thread record, the canonical log and the native state. |

`DELETE /sessions/{id}` stays and calls the same code.

## 5. Requirements (EARS)

"The store" is a `ThreadStore` adapter. "The server" is `holodeck serve`.

| ID | Pattern | Requirement |
| --- | --- | --- |
| TS-01 | Ubiquitous | The server and both backends shall reach thread storage only through the `ThreadStore` port. |
| TS-02 | Ubiquitous | Every adapter shall pass the same contract test suite. |
| TS-03 | Ubiquitous | Every port method shall be async and shall hold no state that is local to one process. |
| TS-04 | Event-driven | When native entries with an id that the store already holds are appended, the store shall ignore the duplicates. |
| TS-05 | Event-driven | When a turn starts on a thread, the server shall take that thread's lease before it calls the backend. |
| TS-06 | Unwanted | If the lease is held by another turn, then the server shall reject the request with HTTP 409 on REST and `RUN_ERROR` with code `thread_busy` on AG-UI, and shall not call the backend. |
| TS-07 | State-driven | While a turn runs, the server shall renew the lease before it expires. |
| TS-08 | Unwanted | If a writer no longer holds the lease, then the store shall reject its canonical writes. |
| TS-09 | Unwanted | If the Claude SDK reports a mirror failure during a turn, then the server shall mark the turn `failed` with reason `native_state_lost` and emit `RUN_ERROR`. |
| TS-30 | Event-driven | When a turn starts on a Claude thread whose last turn failed with reason `native_state_lost`, and the local transcript file is present, the server shall append that file's entries to the store before it resumes. |
| TS-10 | Unwanted | If the store is unreachable when a turn arrives, then the server shall return HTTP 503 or `RUN_ERROR` with code `thread_store_unavailable` and shall not call the backend. |
| TS-11 | State-driven | While the store is unreachable, the readiness endpoint shall report not ready and name the store. |
| TS-12 | Event-driven | When a turn starts, the server shall write the user message and a turn row with status `running`. |
| TS-13 | Event-driven | When a turn completes, the server shall write the assistant messages, tool calls and tool results, and set the turn status to `completed`, in one transaction. |
| TS-14 | Unwanted | If a turn fails, then the server shall set the turn status to `failed` with the error text, keep the user message and keep the native state. |
| TS-15 | Event-driven | When a request names a known thread, the server shall take history from the store and shall read only the new user message and tool results from the request. |
| TS-16 | Unwanted | If a request names a thread whose backend or agent name differs from the running agent, then the server shall reject it with code `thread_backend_mismatch` or `thread_agent_mismatch`. |
| TS-17 | Ubiquitous | Each turn row shall record the hash of the agent configuration in use. |
| TS-18 | Unwanted | If the caller's owner differs from the thread's owner, then every read, write and delete shall respond as if the thread does not exist. |
| TS-19 | Ubiquitous | The server shall read the owner from the configured header, and from AG-UI forwarded props only when the header is absent and the fallback is enabled. |
| TS-20 | Event-driven | When a thread is deleted, the store shall remove its record, its turns, its canonical messages and its native entries, including subagent entries. |
| TS-21 | Event-driven | When the live-executor TTL expires, the server shall close the executor and shall keep the thread. |
| TS-22 | Optional | Where `retention_days` is set, the server shall delete threads whose last activity is older than that period. |
| TS-23 | Event-driven | When an AG-UI run starts on a known thread, the server shall emit `MESSAGES_SNAPSHOT` with the newest `snapshot_messages` canonical messages. |
| TS-31 | Event-driven | When a snapshot leaves out older messages, the server shall emit a `CUSTOM` event named `holodeck.thread_history` with the cursor that `GET /threads/{id}/messages` accepts for the older messages. |
| TS-32 | Ubiquitous | Each message that `GET /threads/{id}/messages` returns shall carry the `turn_status` of its turn. |
| TS-24 | Ubiquitous | A canonical message shall store an attachment's name, MIME type and size, and shall not store its bytes. |
| TS-25 | Event-driven | When a turn completes and the thread has no title, the server shall generate a title in the background on the thread's backend. |
| TS-26 | Unwanted | If title generation fails, then the title shall stay empty and the turn result shall not change. |
| TS-27 | Ubiquitous | The server shall never replace a title that a client set. |
| TS-28 | Optional | Where `threads.history.max_items` is set, the OpenAI bridge shall return at most that many of the newest native items to the SDK. |
| TS-29 | Ubiquitous | The items that the OpenAI bridge returns shall start at a user message. If the limit would cut before one, the bridge shall load back to the nearest earlier user message. |
| TS-33 | Optional | Where `threads.compaction.enabled` is true, the OpenAI backend shall wrap its bridge in `OpenAIResponsesCompactionSession` and compact when at least `openai_min_items` candidate items are stored. |
| TS-34 | Ubiquitous | The store shall replace a thread's native items in one transaction. After a compaction a reader shall see the old list or the new list, and never an empty or partial list. |
| TS-35 | Ubiquitous | A compaction on either backend shall not change the canonical log. |
| TS-36 | Event-driven | When a Claude thread whose transcript the CLI compacted is resumed on another replica, the agent shall recall content from the compacted summary and from the turns after it. |
| TS-37 | Event-driven | When a compaction happens during a turn, the server shall set `compacted` on that turn row. |
| TS-38 | Unwanted | If an OpenAI compaction call fails, then the native items shall stay unchanged and the turn shall continue. |
| CFG-01 | Ubiquitous | The agent schema shall accept a `threads` block validated by a Pydantic model with `extra="forbid"`. |
| CFG-02 | Unwanted | If `provider` is `sqlite` or `postgres` and `connection` is missing, then loading shall fail and name the field. |
| CFG-03 | Ubiquitous | With no `threads` block, the server shall use the in-memory adapter. |
| CFG-04 | Unwanted | If `threads.provider` is durable and the Claude agent enables file checkpointing, then loading shall fail and name both settings. The SDK rejects that combination. |
| CFG-05 | Ubiquitous | `holodeck deploy` shall set `threads.owner.allow_forwarded_props` to `false` when the agent file does not set it, and shall warn when it is `true`. |
| MIG-01 | Event-driven | When a durable adapter starts, it shall apply every DDL step newer than the recorded `schema_version`, under a lock. |
| MIG-02 | Unwanted | If the recorded schema version is newer than the code knows, then startup shall fail and state both versions. |

## 6. Commands

```bash
uv run pytest tests/unit/lib/threads -n auto -q
uv run pytest tests/integration/threads -n auto -q      # needs PostgreSQL
make schema && make schema-check
make harness-check
```

## 7. Project structure

```text
src/holodeck/models/threads.py            # ThreadsConfig, ThreadRecord, ThreadMessage, TurnRecord
src/holodeck/lib/threads/
  port.py                                 # ThreadStore protocol, errors
  memory.py  sqlite.py  postgres.py       # adapters
  migrations.py                           # ordered DDL steps
  claude_native.py                        # Claude SessionStore over the port
  openai_native.py                        # OpenAI Session over the port
  projection.py                           # backend events to ThreadMessage
  titles.py
src/holodeck/serve/thread_service.py      # lease, turn lifecycle, owner check
src/holodeck/serve/protocols/threads.py   # REST endpoints
tests/unit/lib/threads/contract.py        # one suite, run against every adapter
tests/integration/threads/test_two_replicas.py
```

`serve/session_store.py` stays. It becomes the cache of live executors only.

## 8. Testing strategy

- **Contract suite.** One set of tests for the port. It runs against the memory, SQLite and PostgreSQL adapters.
- **SDK bridges.** Each SDK protocol is tested for round trip, idempotent append, delete cascade and, for OpenAI, `pop_item` and the item limit.
- **Compaction.** OpenAI: a thread past `openai_min_items` is compacted, the native list shrinks, the canonical log is unchanged, and the next turn recalls an early fact. Claude: the same check after a CLI compaction, with the resume on a second replica (TS-36).
- **Restart.** A thread is created, the server process is replaced, and the next turn recalls a fact and a tool result from the first turn. Both backends, SQLite.
- **Two replicas.** Two `serve` processes share one PostgreSQL database. A thread created on A continues on B. Two concurrent turns on one thread produce exactly one `thread_busy`. Both backends.
- **Live validation.** The loop in `AGENTS.md`: a sample run, raw AG-UI requests on two threads with two turns each, then the CopilotKit flow with a page reload.

## 9. Boundaries

- Always: reach storage through the port; convert SDK entries at the adapter boundary; keep blocking I/O out of async paths.
- Ask first: the `threads` schema block; any change to `AgentBackend.create_session` beyond D33; any change to the DDL after the first release.
- Never: read the Claude transcript format in HoloDeck code outside the projection tests; store attachment bytes in the canonical log; let a client-supplied history replace stored history.

## 10. Success criteria

1. With `provider: sqlite`, a thread survives a server restart on both backends.
2. With `provider: postgres`, the two-replica test passes on both backends.
3. `GET /threads/{id}/messages` returns the same message shape for both backends.
4. The CopilotKit sample shows earlier messages after a page reload, marks a failed turn, and loads older messages over REST on a thread longer than one snapshot.
5. With no `threads` block, the existing serve test suite passes unchanged.

## 11. Delivery

Small stacked PRs, bottom first. The execution plan will name the model and effort for each task.

1. Models, `threads` configuration and published schema.
2. Port, memory adapter and contract suite.
3. SQLite adapter and migrations.
4. PostgreSQL adapter and lease.
5. OpenAI SDK bridge.
6. Claude SDK bridge.
7. Serve integration: thread service, turn lifecycle, authority, errors.
8. REST endpoints, `MESSAGES_SNAPSHOT`, titles.
9. Two-replica test, guides and the CopilotKit sample as the reference client pattern (D32).

## 12. Trade-offs and what to revisit

- **Two copies of history.** The hybrid stores each turn twice. It costs space and can drift. It buys full-fidelity resume and a stable read shape. Revisit if drift appears in practice.
- **Reject, do not queue.** A busy thread returns an error. This is simple and visible. Revisit if UIs retry badly.
- **Owner without authentication.** The owner filter is only as strong as the gateway that sets the header.
- **Hand-written SQL for two engines.** There is no ORM dependency, but each query exists twice. The contract suite is the guard.
- **Compaction costs a model call.** OpenAI compaction calls `responses.compact` when the threshold is reached. It runs inside the turn, so that turn is slower.
- **Compaction depends on SDK internals for atomic replace.** See the design, seam 4.
- **Title cost.** Each new thread makes one extra model call unless titles are off.

## 13. Open questions

1. **Resolved (D29).** Recovery after a Claude mirror failure: the replica that has the local file sends it again; other replicas accept the loss.
2. **Resolved (D34).** Compaction is in v1 for both backends.
3. **Resolved (D35).** An unknown thread that arrives with history imports it once.
4. **Resolved by a spike on 2026-11-15** (`openai-agents` 0.24.0, fake model, recording session). The SDK did not call `pop_item` in any run with a local session, including runs where the model failed. `pop_item` is reached only from `rewind_session_items`, which runs when a server-managed conversation retries a model request. The canonical log is not affected. The spike also showed that a failed run leaves its partial items in the session: the user message, and any tool call and result that finished before the failure. This matches D17.
5. **Claude native entries before a compaction.** The store keeps every transcript entry, so `load` still returns the entries from before the CLI's compaction. Can the bridge drop them safely? That needs a statement from the SDK about what resume reads.
6. **Azure and `responses.compact`.** `provider: azure_openai` uses the Responses API on the Azure v1 surface. Whether that surface supports compaction is not verified. Where it does not, TS-38 applies and D28 is the bound.
