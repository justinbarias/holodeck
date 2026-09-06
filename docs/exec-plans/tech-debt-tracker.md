# Technical debt tracker

Review date: 2026-09-06. Owner identifies the responsible module maintainers, not an assigned individual.
Update an entry when its evidence or status changes. Close it only with a linked result.

| ID | Gap / evidence | Owner | Exit criterion | Status |
| --- | --- | --- | --- | --- |
| H-001 | [Spec inventory](../product-specs/inventory.md) has historical task drift and inconsistent status vocabulary, including `complete`. | Feature maintainers | Reconcile each affected row against merge and acceptance evidence. | Open |
| H-002 | [Architecture](../../ARCHITECTURE.md) has no global dependency-layer enforcement. Some models import library logic. | Architecture maintainers | Agree actual allowed edges, record exceptions, and add focused structural checks without inventing a new runtime architecture. | Open |
| H-003 | [Makefile](../../Makefile) includes serial test aliases, placeholder package helpers, and Sphinx-style docs targets despite MkDocs. | Developer tooling | Align affected helpers with current package and documented commands, with command-level verification. | Open |
| H-004 | No automatically isolated per-worktree logs/metrics/traces environment. | Observability maintainers | Provide a reproducible local setup and a demonstrated query against a representative request. | Open |
| H-005 | Legacy context generation could repopulate root instruction files. | Repository harness | Removed command scaffolding and generators; the harness checker validates both entry points. | Resolved in workflow cleanup |
| H-006 | Repository-source links broke when published as relative docsite links. | Documentation maintainers | [Publishing hook](../../scripts/mkdocs_hooks.py) resolves existing source links. [Regression tests](../../tests/unit/test_mkdocs_harness.py) and strict MkDocs build passed. | Resolved in harness adoption |
| H-007 | NLTK 3.10.3 has no patch for PYSEC-2026-3740. The [scoped audit exception](../SECURITY.md#nltk-model-artifact-exception) records inspected evaluator paths and does not cover arbitrary custom tools. | Evaluation and dependency maintainers | Lock a patched release and remove the exception; re-review by 2026-10-06 or before evaluator/model persistence changes. | Open; affected APIs absent from inspected paths |

## Feature 035 follow-ups — reconciled 2026-09-06

| ID | Gap / evidence | Owner | Exit criterion | Status |
| --- | --- | --- | --- | --- |
| <a id="h-008"></a>H-008 | [Final SK inventory](active/035-openai-agents-backend/plan-sk-decouple.md#remaining-sk-inventory-and-follow-ups): connectors, chunking, telemetry defaults, and tests still depend on SK. | Retrieval and backend maintainers | Replace remaining runtime/test dependencies, preserve retrieval behavior, remove package/lock constraints, and verify SK-free installation. Create a dedicated implementation spec before this work. | Deferred; migration of agent execution and inference is complete |
| <a id="h-009"></a>H-009 | [LiteLLM acceptance gaps](active/035-openai-agents-backend/plan-litellm-embeddings-contextgen.md): dimension plumbing, error contract, telemetry evidence, and provider documentation. | Retrieval and observability maintainers | Resolve each documented contract difference and supply focused and provider acceptance evidence. | Open |

The [035 completion plan](active/035-openai-agents-backend/2026-09-06-complete-035.md) schedules H-009 and retains H-008 as a separate decommission effort.
The [035 reconciliation](active/035-openai-agents-backend/reconciliation.md) audits the 035 portion of H-001.
Other feature inventories remain outside this audit.

<a id="deferred-035-scope"></a>

## Deferred 035 scope — T0 destinations

These entries preserve the existing exclusions in the [completion plan](active/035-openai-agents-backend/2026-09-06-complete-035.md).
They are durable backlog destinations, not new implementation authorization or delivered capabilities.
FR-084 still forbids silently bypassing approval gates. FR-089 still requires managed child-process scrubbing.
H-013 and H-021 cover the broader interactive-approval and arbitrary-execution boundaries.

| ID | Deferred scope | Owner | Exit criterion | Status |
| --- | --- | --- | --- | --- |
| <a id="h-010"></a>H-010 | Hardened Envoy profile (FR-090–093, SC-010) | Deployment/security maintainers | Define a cross-backend spec and verify credential-free agent containers, provider/embedding/MCP allowlists, proxy routing, and invalid-environment rejection on an authorized target. | Deferred from 035 |
| <a id="h-011"></a>H-011 | Sandbox mode and remote clients (FR-094–099, SC-012–013), including Modal | Backend/deployment maintainers | Specify local/remote boundaries, safety opt-in, manifests, cleanup, redundant-tool validation, and credential handling; demonstrate local and remote workspace isolation for each supported client. | Deferred from 035 |
| <a id="h-012"></a>H-012 | Computer-use harness | Backend/tool maintainers | Define the Computer/AsyncComputer adapter and safety contract; demonstrate controlled computer actions, failure handling, and lifecycle tests before accepting ComputerTool configuration. | Deferred from 035 |
| <a id="h-013"></a>H-013 | Interactive human tool approval | Backend/serve maintainers | Specify interruption, persisted RunState, authenticated approve/reject, resume, timeout, and cancellation; prove an unapproved call cannot execute. | Deferred from 035 |
| <a id="h-014"></a>H-014 | Guardrails for SDK-built MCP-server tools | Backend/MCP maintainers | Establish a supported interception point; prove input rejection, output redaction, and failure hooks for stdio/SSE/HTTP tools without bypassing SDK lifecycle or tool identity. | Deferred from 035 |
| <a id="h-015"></a>H-015 | Executable modify hooks | Backend/model maintainers | Define input/output mutation and ordering across supported backends; verify observable mutation, permission boundaries, and failure behavior before removing the inert-action warning. | Deferred from 035 |
| <a id="h-016"></a>H-016 | Prompt-tool execution | Backend/tool maintainers | Specify runtime prompt-tool semantics and result/event contracts; implement adapters and acceptance tests before replacing skip-with-warning behavior. | Deferred from 035 |
| <a id="h-017"></a>H-017 | Cross-backend namespace unification | Model/configuration maintainers | Specify a migration from backend-specific hooks/subagents/permissions with validated aliases and conflict rules; prove backward compatibility and update schemas/docs. | Deferred from 035 |
| <a id="h-018"></a>H-018 | VoicePipeline audio lifecycle | Backend/multimodal maintainers | Define audio configuration, input/output contracts, streaming, cancellation, and provider coverage; demonstrate an end-to-end speech-agent-speech fixture. | Deferred from 035 |
| <a id="h-019"></a>H-019 | RealtimeAgent transport | Backend/serve maintainers | Specify bidirectional realtime transport, session lifecycle, events, and authorization; demonstrate reconnect, cancellation, and end-to-end interaction. | Deferred from 035 |
| <a id="h-020"></a>H-020 | Ambient skill discovery on OpenAI Agents | Backend/configuration maintainers | Specify deterministic discovery roots, precedence, permission restrictions, and provenance; prove compatibility without silently loading ambient files. | Deferred from 035 |
| <a id="h-021"></a>H-021 | Per-session ephemeral containers and untrusted Python containment | Runtime/security maintainers | Specify kernel/process boundaries, resource limits, credential isolation, filesystem lifecycle, and escape tests. Prove arbitrary child processes cannot escape the declared boundary on the chosen runtime. | Deferred from 035 |

The harness checker detects layout, link-target, discovery, and generated-inventory drift.
Semantic freshness still requires code inspection and named test evidence.
No scheduled cleanup service is configured by this change.
