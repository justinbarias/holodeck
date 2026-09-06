# SPEC: HoloDeck Dependency Stack Revamp

**Status:** Draft for review; no dependency changes implemented.
**Date:** 2026-09-06.
**Code baseline:** `3e59e1d` on `feat/035-t2-litellm-acceptance`; working tree clean at audit start.
**Owners:** Backend, retrieval, packaging, evaluation, and observability maintainers for their respective boundaries.
**Related:** [035](../035-openai-agents-backend/spec.md), [040](../040-holodeck-temporal/spec.md), [041](../041-temporal-file-inputs/spec.md), [H-008](../../exec-plans/tech-debt-tracker.md#h-008).

## 1. Objective

Revamp the dependencies that define HoloDeck's agent hosting stack. Establish clear ownership for agent execution, inference, retrieval, parsing, evaluation, serving, deployment, and telemetry. Remove Semantic Kernel completely and consolidate retrieval integrations around selected open-source LlamaIndex packages.

The target users are agent authors, operators building deployment images, and maintainers upgrading provider integrations. Agent authors retain declarative YAML. Operators install the features they use. Maintainers work through typed HoloDeck contracts instead of framework objects spread through the runtime.

Success is an SK-free, reproducible installation with preserved agent behavior, stored-data compatibility, and verified provider capabilities. A smaller dependency count alone does not establish success.

### Working assumptions

- Native Claude Agent SDK and OpenAI Agents SDK remain the execution foundations. OpenAI/Azure use OpenAI Agents; Anthropic/Ollama use Claude through `BackendSelector`.
- LiteLLM remains the embedding and contextual-generation inference service established by 035. Claude-specific context generation remains supported.
- LlamaIndex is the preferred retrieval integration layer, subject to operation-level acceptance. Native adapters are permitted when a connector cannot satisfy the contract.
- DeepEval remains the existing evaluation framework. Replacing DeepEval or its evaluators is outside this revamp.
- Remove all Microsoft AI evaluation libraries and the deprecated Azure AI evaluation paths. This retirement is explicitly in scope; DeepEval owns LLM-based evaluation.
- Existing YAML, result artifacts, deployment behavior, and persisted collections must have an explicit compatibility path.
- This request produces a product spec and inventory. Version selection, an implementation plan, dependency edits, and data migration follow review of this draft.

## 2. Scope and current evidence

The [dependency inventory](dependency-inventory.md) is the canonical package-by-package audit. It separates runtime requirements, optional integrations, development/build tooling, security constraints, and important transitive dependencies. The [selection evidence](../../design-docs/042-dependency-stack-revamp/selection-evidence.md) records external sources and connector gaps.

The current package mixes hosting dependencies with document conversion, three evaluation families, optimizer dependencies, documentation building, and package publishing. Several security floors are deliberately direct requirements. Removing a source import does not establish that those constraints are redundant.

SK remains in vector record definitions, eleven collection connectors, text splitting, telemetry configuration, and tests. HoloDeck already implements native filtered reads for five stores and substantial Qdrant behavior outside SK. The eleven configured provider names do not establish eleven fully working implementations of every retrieval operation.

### Included

- Disposition of every direct dependency, extra, dependency group, build requirement, override, and security constraint.
- Complete SK removal from executable source, tests, fixtures, installation metadata, and the resolved dependency graph.
- LlamaIndex integration selection, HoloDeck retrieval contracts, provider adapters, chunking, and parser consolidation assessment.
- Feature-based installation, import isolation, Docker/worker packaging, and migration documentation.
- Evaluation, optimization, transport, and telemetry dependency rationalization while preserving their product contracts.
- Complete Microsoft AI evaluation dependency removal and retirement of legacy Azure AI metrics, with explicit configuration migration to DeepEval.

### Excluded

- Replacing native agent SDKs with LlamaIndex agents or another agent framework.
- Replacing DeepEval with LlamaIndex evaluators or another evaluation framework.
- Replacing Temporal with LlamaIndex workflows, or moving I/O into deterministic workflow execution.
- Requiring LlamaCloud, hosted parsing, a new SaaS account, or paid services for existing local capabilities.
- Dropping providers, file formats, metrics, or security floors merely to make dependency resolution easier.
- Implementing unfinished 035 hosted tools, hooks, or serving features through this migration.

The explicitly requested retirement of Microsoft AI evaluation and legacy Azure AI metrics is an exception to metric preservation. Retained deterministic/NLP metrics, code graders, and saved-result readers remain supported. Azure OpenAI hosting, authentication, storage, and deployment dependencies remain governed by their own usage; their Microsoft provenance alone is not a reason for removal.

## 3. Proposed stack and ownership

| Responsibility | Target owner | Dependency direction |
| --- | --- | --- |
| YAML/configuration and shared results | HoloDeck Pydantic models, PyYAML, schema validation | SDK types converted at adapters; no vendor types in public YAML/results |
| Agent execution | Native Claude and OpenAI agent SDKs | Consumers continue through `BackendSelector` and backend protocols |
| Embedding/context inference | Existing LiteLLM service; retained Claude context path | LlamaIndex receives computed embeddings or an explicit per-agent adapter |
| Storage/retrieval integrations | Selected LlamaIndex core/integration packages | HoloDeck-owned interface contains node/filter/result conversion |
| Provider-specific capabilities | Native client inside the relevant adapter | Qdrant compatibility behavior remains available without SK |
| Document conversion | Existing `FileProcessor` contract | Consolidate readers only after proving extraction parity |
| Chunking/hierarchical semantics | HoloDeck policies with selected LlamaIndex splitter where suitable | Retain domain-specific structure; replace SK splitting |
| Serving and streaming | FastAPI, Uvicorn, AG-UI, MCP | Preserve existing HTTP, lifecycle, and tool-event contracts |
| Telemetry | OpenTelemetry and HoloDeck redaction/routing | Adapt retrieval events without duplicate inference spans |
| Evaluation and optimization | DeepEval, existing metric contracts, and Optuna | Retain DeepEval; optional feature installation must preserve evaluation behavior |
| Durable execution and deployment | Temporal and existing deployers | Optional packages; no new deployment target inferred from an extra |

Use `llama-index-core` plus selected integrations, not the `llama-index` starter bundle. The bundle adds model-provider integrations that would duplicate established inference ownership. Core itself has substantial transitive dependencies, including workflow packages; installed presence does not authorize using a second workflow engine.

LlamaIndex consolidation must reduce duplicated integration responsibility. It does not require adopting every component from that ecosystem. Retained dependencies and exceptions require explicit rationale in the inventory.

## 4. Functional requirements

### Inventory, packaging, and supply chain

- **FR-001:** Every declared dependency has a current usage/constraint, proposed disposition, owner, and verification gate. Classify it as retain, replace/consolidate, relocate to an optional feature/tooling group, remove after proof, or unresolved with an owner and exit criterion.
- **FR-002:** Produce before/after resolved graphs for each supported installation profile. Trace direct imports to declared distributions, including dependencies currently obtained through SK. Report packages still reachable through other paths after SK removal.
- **FR-003:** Remove `semantic-kernel` from all supported installation profiles and their lock resolution after replacing runtime and test consumers. Historical documentation may name SK; executable imports, dynamic connector strings, and test helpers must not require it. Close H-008 only with this evidence.
- **FR-004:** Separate documentation/release tooling from production installation. Make heavyweight evaluation, optimizer, parser, search-service, cloud, and dashboard packages feature-scoped where contracts permit. Retain security requirements even when their scope moves.
- **FR-005:** Preserve existing extra names through aliases or documented migration. Define exact new profile names and a release compatibility policy before editing package metadata. A missing optional feature yields an actionable install command when selected; unrelated commands and backends must still import and start.
- **FR-006:** Preserve Python 3.10 support and publish a tested Python/platform matrix. Resolve any new upper bound, binary-wheel limitation, or driver conflict explicitly. Do not silently narrow the existing `requires-python` promise.
- **FR-007:** Reconcile all security constraints, license obligations, overrides, and prerelease policy against the new graph. Removing SK is not evidence that shared transitive constraints are obsolete. Existing exceptions must be re-reviewed when newly selected packages introduce new affected code paths.
- **FR-008:** Measure installed distributions, installed bytes, clean import/startup time, and built-image size for identical old/new profiles and environments. Record regressions and their causes; do not claim consolidation reduces footprint without measurements. Documentation/release and unselected feature packages must be absent from the lean hosting profile unless a documented runtime dependency requires them.

### Runtime and inference boundaries

- **FR-009:** Preserve provider routing, single/multi-turn execution, streaming, multimodal inputs, structured output, tools, budget/fallback behavior, and backend cleanup. SDK changes must preserve the shared result and event contracts.
- **FR-010:** Keep retrieval model inference behind existing HoloDeck services. Preserve dimensions, model overrides, errors, retries, truncation, concurrency, Azure endpoints, and OpenAI/Azure/Ollama embedding support. No implicit LlamaIndex default model, embedding call, credential requirement, or remote service is permitted.
- **FR-011:** Use per-agent configuration and explicit dependencies. Do not mutate LlamaIndex global `Settings`, shared clients, or process-global telemetry to select an agent's embedding model or storage configuration. Concurrent agents with different providers/dimensions must remain isolated.
- **FR-012:** Restrict LlamaIndex/node/filter objects and native SDK clients to integration adapters. Define concrete HoloDeck record, query, result, capability, and error types. New boundaries must not use `Any` or `getattr` to bypass known contracts.

### Retrieval and provider contracts

- **FR-013:** Cover collection inspection/creation, upsert, get/delete by ID, exact metadata filtering, paginated enumeration without vectors, source replacement, vector/lexical/hybrid search, and client lifecycle. Capabilities must be explicit per provider and operation.
- **FR-014:** Assess all eleven existing provider names using the matrix in the selection evidence. Preserve working operations. Resolve current unsupported combinations through an implementation or a precise capability-validation error before ingestion; do not silently turn failures into successful empty results, skipped deletion, or repeated ingestion. Provider retirement requires an explicit product decision and migration path.
- **FR-015:** Preserve plain, structured, and hierarchical record families: IDs, configurable dimensions, vector names, cosine semantics, dynamic metadata types, provenance, source keys, and hierarchical relationships. Convert scores into a documented HoloDeck convention; raw distance and similarity values must not be confused.
- **FR-016:** Preserve incremental ingestion for local mtime and remote content-hash sources. Replacing a source must remove stale chunks, including when its chunk count shrinks. Interrupted ingestion must be recoverable. Cold restart must support persisted retrieval without reembedding unchanged sources, including rebuilding fallback lexical indexes where required.
- **FR-017:** Preserve Qdrant UUIDv5 IDs, the `embedding` vector name, payload layout, index configuration, filtered pagination, and reusable async-client ownership. Consolidate existing native code into the adapter instead of rebuilding it around SK objects.
- **FR-018:** Preserve Qdrant dense-plus-full-text retrieval: weighted `searchable_text`, `MatchText` token matching, OR filters, candidate pool `max(4 * top_k, 20)`, RRF fusion, and punctuation-only fallback. Dense-plus-sparse LlamaIndex hybrid is a different retrieval policy; enabling it requires a declared schema/index migration and quality comparison. It must not silently replace the current path.
- **FR-019:** Select each LlamaIndex connector only after verifying the required operations. A native extension must use supported client APIs and declare its capability/ownership; vendor-private attribute access must not leak into tools. No placeholder SK-backed adapter may remain at final cutover.
- **FR-020:** Keep async execution responsive. Audit actual connector methods rather than trusting an `async` method name. Use native async calls or bounded offloading for synchronous parsing/storage, with cancellation, backpressure, timeouts, and deterministic client closure.

### Parsing, chunking, and stored data

- **FR-021:** Preserve `FileProcessor` behavior for PDF, Word, PowerPoint, Excel, CSV, text, images, URLs, caching, and extraction options. Readers must preserve selected pages, sheet/range behavior, headings, provenance, and multimodal routing. Keep MarkItDown/pypdf behind the boundary where LlamaIndex readers do not provide parity; optionalize unused format dependencies.
- **FR-022:** Replace SK's text splitter and retain HoloDeck's structured/hierarchical chunker. The current plain splitter ignores accepted overlap/separator settings; the replacement must implement their declared behavior or reject unsupported combinations explicitly. Any changed chunk boundaries require a new pipeline fingerprint and controlled reindexing, not mixed old/new chunks.
- **FR-023:** Fingerprint embedding model/dimensions, parser/splitter policy, record layout, vector names, and retrieval mode. Detect incompatible existing collections before writes. Produce a migration dry run reporting collection identity, record counts, compatibility, reembedding need, and estimated work/cost assumptions. Migration must be resumable, verified, and reversible via retained source collections/snapshots. Never silently recreate a collection.
- **FR-024:** Preserve hierarchy, section IDs, ancestry, definitions, cross references, contextual content, token budgets, and source attribution. Existing JSON-encoded relationship fields require decoding compatibility or an explicit migration. New ingestion must not rely on old SK decorators or dynamic record-class mutation.

### Evaluation, observability, and hosting

- **FR-025:** Retain DeepEval and all existing DeepEval evaluators; do not replace them with LlamaIndex evaluators or another framework. Audit retained evaluation packaging and inference/telemetry dependencies across DeepEval, Hugging Face Evaluate, and direct NLP libraries. Optional packaging must preserve DeepEval availability in the evaluation and compatibility profiles, inputs, scores, scales, thresholds, explanations, model overrides, and errors. Any consolidation of retained metric libraries requires a metric-by-metric compatibility mapping; NLP evaluation remains free of LLM calls. Microsoft AI evaluation and legacy metrics follow the removal requirements in FR-030–031.
- **FR-026:** Preserve optimizer scoring, candidate artifacts, and deterministic FEEL/schema gates. Optuna and `bkflow-feel` are not replaced solely for vendor consolidation. Preserve exact pins that encode verified correctness until equivalent verification supports a change.
- **FR-027:** Preserve OpenTelemetry GenAI events, cost/token accounting, provider-upload policy, content-capture opt-out, and redaction. Add retrieval spans without duplicating embedding/model spans. Remove SK-specific instrumentation only after its retained responsibilities have owners. No new telemetry destination or upload is enabled implicitly.
- **FR-028:** Verify chat, test, REST/AG-UI serve, tool-init, Docker/ACA packaging, and Temporal activities under their selected profiles. Preserve readiness/capacity contracts owned by 035; measurements after dependency changes must expose memory regressions. Parsing, retrieval, and LLM imports/I/O must not enter the Temporal deterministic surface.
- **FR-029:** Include non-Python prerequisites in installation and supply-chain evidence: Node.js, SDK executables, uv, image/system packages, and selected MCP launchers. Base/agent/worker images must install the intended HoloDeck candidate and reviewed feature dependency graph. Record image digest, package manifest, system/runtime versions, and build inputs. Generated extras installation must not silently upgrade the candidate or resolve an unaudited prerelease graph; review NodeSource and uv bootstrap provenance. Missing prerequisites must produce actionable startup errors.
- **FR-030:** Remove all Microsoft AI evaluation libraries from direct requirements, optional extras, compatibility profiles, and resolved dependency graphs, including `azure-ai-evaluation`. Inventory any additional Microsoft evaluation packages discovered transitively and remove the paths that introduce them. Delete `lib/evaluators/azure_ai.py`, its exports, factory/executor branches, SDK configuration conversion, and associated mocks/fixtures and implementation tests. Remove dependencies and security constraints used solely by this retired stack only after checking reverse dependencies; retain shared packages and security floors needed elsewhere. No optional legacy evaluation installation or fallback may reintroduce the removed stack.
- **FR-031:** Retire the legacy Azure AI metric configuration and execution surface. Remove accepted legacy names such as `groundedness`, `relevance`, `coherence`, and `fluency`; account for the `similarity` adapter and documented `safety` path even where current schema support differs. Update models, generated schemas, templates, examples, guides, and tests together. Rejected old YAML must give an actionable migration error before agent execution or model calls. Map groundedness to DeepEval faithfulness, relevance to answer relevancy, and other legacy criteria to reviewed G-Eval configurations. Document input, score, threshold, and criteria changes; do not assume score equivalence or silently translate configurations. Preserve historical result readability without importing or executing removed evaluators. Deterministic/NLP metrics and code graders remain supported.

## 5. Installation profiles and compatibility

These are proposed capability boundaries, not existing extra names or runnable installation commands. The implementation plan must map them to package metadata and documented upgrade instructions.

| Profile | Required capabilities | Must not require simply to start |
| --- | --- | --- |
| Base hosting | Configuration, CLI, current default Claude backend, MCP, serving, OTel | SK, docs/release tools, evaluators, optimizer, dashboard, cloud SDKs |
| OpenAI hosting | Base plus OpenAI Agents; OpenAI/Azure routing | Retrieval/parsing or evaluation unless selected |
| Retrieval | Selected LlamaIndex components, LiteLLM inference where used, selected store, required parser/chunker | Every store driver or hosted LlamaCloud account |
| Evaluation / optimization | Retained DeepEval and selected existing metric families; optimizer as separately selected capability | Unselected providers or dashboard |
| Worker / deployment / sources | Temporal, Docker/cloud/source packages as selected | Unimplemented AWS/GCP deployment merely because extras exist |
| Development / docs / release | Test/style/security tools, site build, package publication | Inclusion in production wheel requirements |

Any base-install capability removed by this split is a packaging compatibility change. Release notes must give old-to-new commands and a compatibility aggregate extra for retained functionality where needed. That aggregate must contain neither SK nor Microsoft AI evaluation libraries and must not restore retired legacy metrics. CLI help and configuration inspection must not eagerly import every optional feature.

## 6. Acceptance criteria

| ID | Observable acceptance | Owner |
| --- | --- | --- |
| AC-01 | Every declared requirement/group/constraint is accounted for; candidate lock graphs and direct-import declarations match all supported profiles. | Packaging |
| AC-02 | Clean wheel installs for base, OpenAI, each store, evaluation, worker, deploy, and compatibility aggregate resolve without SK; import/run checks pass with SK unavailable. Source and executable test scan has no SK requirement. | Packaging/backend |
| AC-03 | Claude/Ollama and OpenAI/Azure backend regressions pass across chat, test, streaming, serve, and Temporal consumers. Live provider evidence is recorded separately from mocks/skips. | Backend |
| AC-04 | Each of eleven stores has a completed operation matrix with tested support or actionable early rejection. Existing working behavior is preserved; no provider is silently removed. | Retrieval |
| AC-05 | Qdrant golden fixtures retain IDs/payloads, lexical edge cases, dense/full-text ranking, source attribution, filtered pagination, and client lifecycle. Both old and newly created collections pass. | Retrieval |
| AC-06 | Source replacement, shrinking files, failures/retry, cold restart, and concurrent agent isolation pass. Unsupported operations cannot masquerade as successful empty retrieval. | Retrieval |
| AC-07 | Parser fixtures retain page/sheet/range and multimodal semantics. Splitter boundary changes have versioned fingerprints and explicit migration tests. | Retrieval/backend |
| AC-08 | Migration dry run is read-only; interrupted migration resumes; verified rollback restores the old collection. Compatible collections need no reembedding. | Retrieval/deployment |
| AC-09 | Evaluation and compatibility installs include DeepEval and run all existing DeepEval evaluators through the DeepEval implementation. All retained metrics preserve result contracts/model overrides and thresholds on fixed fixtures; NLP metrics issue no model calls. Optimizer and Temporal replay/determinism checks pass. | Evaluation/workflow |
| AC-10 | In-memory exporter and authorized collector checks prove span routing, no duplicate inference spans, redaction, capture-disabled behavior, and no new default uploads. | Observability |
| AC-11 | Security/license and Python/platform resolution checks pass with reviewed exceptions; non-Python prerequisites and missing-runtime errors are verified. Base/agent/worker images resolve the intended candidate and reviewed extras, with digests and Python/system manifests recorded. Unselected features are absent or have documented unavoidable transitive ownership. Footprint/startup/RSS results compare identical profiles. | Packaging/security |
| AC-12 | YAML/templates, schema, installation guides, provider capability docs, Docker manifests, architecture and dependency inventory agree; H-008 closes only after AC-01–11. | Documentation/backend |
| AC-13 | All supported installation graphs exclude Microsoft AI evaluation libraries, including `azure-ai-evaluation`; imports and retained evaluations pass with those libraries unavailable. Executable source and test scans find no retired SDK imports, adapters, or dispatch paths. Regression tests prove legacy metric YAML fails with migration guidance before agent/model calls; documented DeepEval replacements run through DeepEval, and historical result fixtures remain readable. Deterministic/NLP metrics and code graders still pass. | Evaluation/packaging |

Golden retrieval fixtures must cover definitions, headings, section identifiers, contextual text, negative/no-result and punctuation queries. Freeze expected IDs/order or justified relevance thresholds before adapting connectors. Record ranking changes; a successful API call does not establish retrieval quality.

## 7. Verification commands and source ownership

Implementation stays in the existing backend, tool, file-processing, telemetry, serve, and Temporal areas shown in [Architecture](../../../ARCHITECTURE.md). A proposed `src/holodeck/lib/retrieval/` package may own contracts/adapters; it is not present or required by this draft. Tests belong in the corresponding unit, integration, and installed-package suites. Fixtures must be tracked, not sourced from ignored `sample/` directories.

Use Pydantic for serializable public models and typed protocols at internal boundaries. Follow [Contributing](../../contributing.md) and existing naming, Black/Ruff, and MyPy conventions. For example, the proposed boundary shape is:

```python
from typing import Protocol

from pydantic import BaseModel


class StoredDocument(BaseModel):
    id: str
    content: str
    source_path: str


class DocumentReader(Protocol):
    async def get_by_id(self, record_id: str) -> StoredDocument | None: ...
```

This illustrates style, not the complete record schema or a mandate to add these exact classes.

Existing focused checks for retrieval changes:

```bash
uv run pytest tests/unit/lib/test_vector_store.py tests/unit/lib/test_collection_filter.py tests/unit/lib/test_keyword_search.py tests/unit/lib/test_text_chunker.py tests/unit/lib/test_structured_chunker.py tests/unit/tools/test_hierarchical_document_tool.py -n auto -q
uv run black --check src/holodeck tests
uv run ruff check src/holodeck tests
uv run mypy src/holodeck
make schema-check
make harness-check
uv run mkdocs build --strict
```

Run focused style/type checks on changed Python files during implementation; the broad forms above are final-integration checks. Run `make schema` if configuration models change. Preserve repository CI and security checks, plus `uv run pytest -n auto --cov --cov-branch --cov-report=xml` on the final candidate. Add installed-wheel/profile and migration tests; `uv run` from a fully populated developer environment is insufficient evidence of optional-dependency isolation.

Live databases, cloud deployment, data migration, and provider calls require the authorization rules in [Reliability](../../RELIABILITY.md). Missing credentials remain unresolved acceptance, not passed tests.

## 8. Relationship to other work and release gates

This spec provides the dedicated replacement scope requested by H-008. It does not reopen 035's requirement checklist or claim that 035 T2 removes SK storage. Coordinate shared edits after the relevant 035 inference/tracing contracts stabilize. Final 042 acceptance must rerun affected 035 consumer tests against the assembled candidate, regardless of earlier passes.

Implementation planning follows review of this product contract. The expected release gates are: baseline/profile inventory; connector and version selection; typed boundary plus Qdrant parity; remaining provider/parser/package migration; installed-package and stored-data acceptance; SK removal and documentation closure. These are gates, not a second task checklist.

Do not wait for every unrelated 035 feature to audit dependencies. Do not bypass unresolved 035 live evidence by citing this spec. Spec 041 must reuse the retained file-processing boundary and explicitly account for any parser/package change.

## 9. Open decisions with owners and exit criteria

| Decision | Draft direction | Owner / evidence required before implementation |
| --- | --- | --- |
| LlamaIndex package versions | Selective core/integrations; dated candidates in selection evidence | Packaging: compatible locked resolver output, Python/platform wheels, license/security review |
| Qdrant extension | LlamaIndex adapter with native compatibility operations where necessary | Retrieval: compare against existing payload/ranking fixtures and public native APIs |
| SQL Server and incomplete store operations | Preserve provider intent; no verified universal connector | Retrieval: choose supported native/integration implementation, certify operations, or request explicit scope change |
| Parsing and plain chunking | Evaluate LlamaIndex components; retain format-specific processing as needed | Retrieval: extraction/chunk golden results, fingerprint and reindex policy |
| Evaluation migration and packaging | DeepEval retention and Microsoft/legacy evaluation removal are fixed; assess replacement criteria and retained metric-library overhead | Evaluation: certify migration guidance, early rejection, absence of the removed stack, and DeepEval availability; map compatibility for changes to retained metric libraries |
| Profile names and release compatibility | Lean host plus selected features and compatibility aggregate excluding retired evaluators | Packaging/product: exact extra mapping, missing-extra UX, release/version policy |
| Performance and ranking gates | Fixed old/new environments and fixtures | Retrieval/serving: freeze baseline, permitted changes and thresholds before migration; explain any regression |

These are bounded validation decisions, not permission to leave SK dependencies behind in a release declared complete.

## 10. Constitution check and boundaries

- **No-code-first:** Keep agent YAML and public models; no user-authored LlamaIndex pipeline required.
- **MCP:** External API tools remain MCP. Native database clients implement the existing internal retrieval service boundary, not a new API-tool category.
- **Multimodal tests:** Preserve file/image paths and tracked test fixtures across backends.
- **OTel-native:** Keep existing instrumentation, provider routing, and redaction ownership.
- **Evaluation flexibility:** Preserve global/run/metric overrides and deterministic NLP metrics.
- **Decoupled engines:** Retrieval consolidation must not couple deployment/evaluation to an agent SDK or LlamaIndex runtime.

Always preserve working data and unrelated changes, validate models, and verify changed contracts. Resolve dependency additions in the reviewed implementation plan. Obtain explicit authorization for destructive migration, live targets, provider retirement, or public compatibility changes beyond that plan. Never commit credentials, silently discard collections, or weaken failing acceptance tests. No constitution exception is requested.

## 11. Draft validation

Source and manifest inspection plus upstream documentation/package-metadata research establish this draft. No candidate dependency has been installed or benchmarked; no live provider, database, or migration acceptance is claimed.

Draft checks passed after the evaluation scope updates: `make harness-check`, `uv run mkdocs build --strict`, and `git diff --check`. Requirement structure checks cover 31 functional requirements and 13 acceptance criteria. The initial inventory audit accounted for all 50 core dependencies, 21 optional dependency groups, development/build requirements, and security constraints, with 35 local links and anchors checked. Independent dependency and retrieval-contract reviews informed the draft, including the non-Python runtime and reproducible deployment-image requirements.
