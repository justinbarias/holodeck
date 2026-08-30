# SPEC: File and Bytestream Inputs for Temporal Agents

**Status:** Draft for review
**Date:** 2026-08-30
**Depends on:** `specs/040-holodeck-temporal/spec.md` (activity library, worker CLI, plugin)
**Method note:** Post-speckit format, same as 040. Decisions in §4 were settled
in a design interview on 2026-08-30; the rationale lines record why.

## 1. Objective

Files become first-class inputs to HoloDeck agents running as Temporal
activities, reusing the existing document-parsing → markdown stack
(`holodeck.lib.file_processor.FileProcessor`, markitdown) and the existing
`FileInput` model. The HoloDeck agent already supports direct file inputs in
the test runner and file bytestreams over the API/AG-UI serve surface; this
spec extends the same capability to the Temporal path without inventing a
second parsing pipeline.

Out of scope: a blob-store/claim-check seam (acknowledged end state, §6), and
any human-in-the-loop machinery (settled as docs-only in 040's T17 — Temporal
signals/updates/`wait_condition`/timers are sufficient and control flow is
user-authored).

## 2. Deliverables

### D1 — `parse_document` activity

A built-in Temporal activity shipped by `holodeck.temporal`:

- **Input:** one `FileInput` per call (`path` or `url` ref + type + extraction
  params `pages`/`sheet`/`range`/`cache`). One file per call so retry,
  timeout, and the payload budget apply per document; workflows fan out with
  `asyncio.gather` for many files.
- **Output:** typed result carrying the markdown plus metadata (source name,
  type, extraction params applied) — the `ProcessedFileInput` shape adapted to
  a payload model.
- **Implementation:** `FileProcessor.process_file` verbatim — no parallel
  parsing code. It is an activity (not workflow code, not inlined into the
  agent activity) because parsing may involve model calls in the future:
  non-deterministic and billable work belongs in an activity.
- **Registration:** automatic. Both `holodeck worker` and `HoloDeckPlugin`
  always register it alongside node activities under a fixed well-known name.
  Zero configuration; there are no knobs to configure.
- **Error taxonomy:** inherits 040's split. Missing file, unsupported type,
  bad extraction params → `ApplicationError(non_retryable=True)` (authoring
  faults). Download failures and transient I/O → plain exceptions, retryable.

### D2 — Pass-through attachments on the agent activity

`AgentActivityInput` gains `files: list[FileInput] | None = None`
(`extra="forbid"` intact, backward compatible). These are **pass-through
attachments only** — images handed raw to a multimodal backend through the
existing multimodal message path. Documents never parse inside the agent
activity. Each file type has exactly one route:

| File type | Route |
| --- | --- |
| image | `AgentActivityInput.files` → backend multimodal input |
| pdf, text, excel, word, powerpoint, csv | `parse_document` activity → markdown → workflow embeds in `message`/`context` |

### D3 — Gate-schema codegen (moved from 040's T14)

`holodeck generate models --config worker.yaml`: emit a module with one typed
Pydantic model per edge-node gate schema (datamodel-code-generator), pairing
with `AgentActivityResult.output_as()` for typed workflow code. Deterministic
output (stable ordering, no timestamps). Moved here 2026-08-30 — it pairs with
the file-input work and 040 shipped without it.

### D4 — Serve/AG-UI bridge documentation

The existing bytes → temp file → `FileInput` bridge in
`holodeck.serve.file_utils` (`create_temp_file_from_bytes`,
`convert_binary_dict_to_file_input`) is the entry point for bytestreams. This
spec documents its interaction with worker locality (below); it does not
rebuild it.

## 3. Constraints

1. **Refs only across workflow history.** `FileInput.path`/`url` cross the
   boundary; raw bytes never do. No inline-bytes field: Temporal's ~2MB
   payload limit makes it a footgun.
2. **Path locality.** A `path` ref written by the workflow client resolves on
   the worker's filesystem. In a distributed topology (client and worker on
   different hosts — already true in 040's T16 test) `path` requires a shared
   volume; otherwise use `url`. Documented loudly, not papered over.
3. **Payload limit on parse output.** A parsed document's markdown crosses
   history and can exceed the 2MB payload cap. Position: accept and document.
   The activity fails loud with an error naming the fix; `FileInput`'s
   `pages`/`sheet`/`range` params are the pressure valve. Chunked output and
   store+ref were considered and rejected for v1 (§6).

## 4. Decisions

| # | Decision |
| --- | --- |
| 1 | Transport is refs only (`FileInput` as-is). Inline bytes rejected (payload limit); claim-check deferred (needs a storage seam that does not exist). |
| 2 | Parsing runs in a dedicated `parse_document` activity, not inside the agent activity. Rationale: future parsing may involve model calls — non-deterministic, retryable, billable. Also keeps one parsing path. |
| 3 | `AgentActivityInput.files` exists but carries pass-through attachments (images) only. Split by type: every file type has exactly one route. |
| 4 | Parse output size: accept + document the 2MB payload limit; extraction params are the mitigation; fail loud. |
| 5 | `parse_document` is auto-registered by worker CLI and plugin under a fixed name. Opt-in config rejected: configuration for something with no knobs. |
| 6 | One file per `parse_document` call. Batch rejected: one bad file would fail/retry the whole batch and compound the payload cap. |
| 7 | HITL ships nothing in code. 040's T17 documents the pattern: `@workflow.signal` + `workflow.wait_condition` + timer race for SLA escalation. |
| 8 | Gate-schema codegen (040 T14) lands in this spec — `holodeck generate models`, deterministic output. |

## 5. Acceptance criteria

- AC-1: A workflow calls `parse_document` with a `url` ref to a PDF; the
  agent activity receives the markdown via `message`/`context` and produces a
  gated output. No bytes in workflow history (history inspection).
- AC-2: An image `FileInput` on `AgentActivityInput.files` reaches a
  multimodal backend and influences the gated output.
- AC-3: A document whose markdown exceeds the payload cap fails the parse
  activity with an error that names `pages`/`sheet`/`range` as the fix.
- AC-4: A missing file / unsupported type is non-retryable; a transient
  download failure retries under the caller's retry policy.
- AC-5: `holodeck worker` and `HoloDeckPlugin` both serve `parse_document`
  with zero configuration.

## 6. Deferred

- **Claim-check storage seam:** parse activity writes markdown (or the client
  writes bytes) to a blob store; history carries an opaque ref. The right end
  state for large documents; requires a storage abstraction HoloDeck does not
  have. Revisit when a concrete deployment hits the payload cap with real
  documents.
- **Chunked parse output:** superseded by claim-check if/when needed.

## 7. Open questions

None at draft time. §4 records the settled interview.
