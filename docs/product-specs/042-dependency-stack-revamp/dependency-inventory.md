# Dependency inventory

Status: audit baseline for [spec 042](spec.md). Inspected 2026-09-06 at commit `3e59e1d`. This is a declaration/import audit, not a measured clean-install footprint. Versions below come from the repository lockfile, not claims about latest upstream releases.

## Direct runtime declarations

All **50** `[project].dependencies` declarations are accounted for. “No direct source import” is evidence for review, not proof a dependency is safe to remove: dynamic loading, upstream requirements and security constraints must be checked. Paths in this inventory are relative to `src/holodeck/` unless stated otherwise.

| Dependency constraint | Locked version | Observed role | Proposed disposition |
| --- | --- | --- | --- |
| `requests>=2.32.5,<3.0.0` | `2.33.0` | HTTP fetches in `lib/file_processor.py`, `services/mcp_registry.py`. | Review consolidation with HTTPX; retain until sync/async, retries and TLS behavior match. |
| `pydantic>=2.11.0` | `2.13.4` | 45 source files: `models/`, config validation, backend output, workflow and serving. | Retain as HoloDeck schema/contract boundary. |
| `pyyaml>=6.0.0` | `6.0.3` | `config/loader.py`, `config/manager.py`, templates, workflows, optimizer output. | Retain. |
| `click>=8.3.3,<9.0.0` | `8.3.3` | 12 CLI source files, including `cli/main.py`. | Retain; preserve security lower bound. |
| `jinja2>=3.0.0,<4.0.0` | `3.1.6` | `lib/template_engine.py`, `deploy/dockerfile.py`. | Retain; not superseded by LlamaIndex. |
| `python-dotenv>=1.0.0` | `1.2.2` | `cli/main.py`. | Retain. |
| `python-dateutil>=2.8.0` | `2.9.0.post0` | No direct source import found. | Review historical/transitive rationale; remove direct declaration only after clean-install tests. |
| `urllib3>=2.6.0,<3.0.0` | `2.7.0` | No direct source import; Requests transport dependency. | Review security constraint placement; do not lose patched lower bound. |
| `cryptography>=50.0.0,<51.0.0` | `50.0.1` | No direct source import; manifest records security fix. | Review security constraint placement, not blind removal. |
| `mkdocs-material>=9.7.7,<10.0.0` | `9.7.7` | Documentation tooling; no runtime source import. | Move out of runtime into docs tooling. |
| `twine>=6.2.0,<7.0.0` | `6.2.0` | Release tooling; no runtime source import. | Move out of runtime into release tooling. |
| `jsonschema[format-nongpl]>=4.18.0,<5.0.0` | `4.25.1` | `lib/workflow/edge.py`, native backend output validation. | Retain format-nongpl extra and >=4.18 registry semantics. |
| `referencing>=0.36.0,<1.0.0` | `0.36.2` | `lib/workflow/edge.py` imports specification/registry APIs. | Retain explicit direct dependency. |
| `lark>=1.1.0,<2.0.0` | `1.3.1` | `lib/workflow/feel.py` uses parser nodes. | Retain explicit direct dependency. |
| `semantic-kernel>=1.39.4,<2.0.0` | `1.43.0` | `lib/vector_store.py` record contracts and 11 dynamically imported connectors; `lib/text_chunker.py`. | Replace with HoloDeck contracts plus selected LlamaIndex integration packages/native extensions; remove SK hooks and package only after parity. |
| `markitdown[all]>=0.1.4,<0.2.0` | `0.1.5b1` | `lib/file_processor.py`; all extra includes numerous document/audio/cloud loaders. | Evaluate selective readers behind existing file-processing contract; preserve formats, heading/page extraction and multimodal behavior. |
| `pypdf>=6.14.2` | `6.16.2` | `lib/pdf_processor/page_extractor.py`. | Retain until PDF/page parity demonstrated; preserve security floor. |
| `azure-ai-evaluation>=1.13.0,<2.0.0` | `1.13.7` | `lib/evaluators/azure_ai.py`. | Remove completely from all profiles and resolved graphs, alongside legacy Azure AI evaluators/configuration. Migrate users to retained DeepEval; no optional legacy fallback (FR-030–031, AC-13). |
| `evaluate>=0.4.6,<0.5.0` | `0.4.6` | `lib/evaluators/nlp_metrics.py` lazy loader. | Evaluate removing framework overhead using dedicated metric libraries, or optionalize; preserve metric behavior. |
| `aiofiles>=25.1.0,<26.0.0` | `25.1.0` | No direct source import found. | Review/remove direct declaration if no runtime need survives. |
| `rouge-score>=0.1.2,<0.2.0` | `0.1.2` | Metric backend used through evaluate loader, not a direct import. | Retain in NLP evaluation capability until equivalent metric implementation verified. |
| `absl-py>=2.3.1,<3.0.0` | `2.3.1` | No direct source import; support dependency for ROUGE ecosystem. | Review transitive placement in NLP evaluation capability. |
| `sacrebleu>=2.5.1,<3.0.0` | `2.5.1` | `lib/evaluators/nlp_metrics.py`, direct BLEU backend. | Retain/optionalize with NLP metrics; preserve smoothing and normalization. |
| `anthropic>=0.72.0,<0.73.0` | `0.72.1` | No direct source import found; runtime uses Claude Agent SDK. | Review/remove direct declaration after provider/DeepEval compatibility tests; package-name absence alone does not mean provider removal. |
| `ollama>=0.4,<1.0` | `0.6.1` | No direct source import found; runtime provider routes through Claude, inference through LiteLLM. | Review/remove direct declaration after Ollama runtime and inference tests. |
| `deepeval>=3.8.9,<3.9.0` | `3.8.9` | Eight evaluator modules under `lib/evaluators/deepeval/`. | Retain DeepEval and all existing evaluators; replacement is out of scope. Optional packaging must keep DeepEval available in evaluation and compatibility profiles with unchanged behavior (FR-025, AC-09). |
| `python-ulid>=3.1.0,<4.0.0` | `3.1.0` | Claude backend and serve sessions/protocols/models. | Retain. |
| `ml-dtypes>=0.4.0,<0.5.0` | `0.4.1` | No direct source import; current manifest caps <0.5. | Review legacy numeric compatibility cap and reverse dependency graph. |
| `mcp>=1.28.1,<2` | `1.28.1` | Consumed by native SDKs; tool bridges rely on their MCP APIs. | Retain 1.x compatibility cap until Claude SDK API compatibility established. |
| `inquirerpy>=0.3.4,<0.4.0` | `0.3.4` | `cli/utils/wizard.py`. | Retain CLI capability; optionalize only with workable noninteractive install UX. |
| `fastapi>=0.115.0,<1.0.0` | `0.136.1` | `serve/server.py`, REST protocol, middleware and tool init routes. | Retain hosting capability; consider explicit serving profile. |
| `starlette>=1.3.1` | `1.3.1` | `serve/middleware.py`; also FastAPI/MCP dependency. | Retain hosting compatibility and security floor. |
| `uvicorn[standard]>=0.34.0,<1.0.0` | `0.38.0` | `cli/commands/serve.py`. | Retain serving runner; review standard extra costs by platform. |
| `ag-ui-protocol>=0.1.18,<1.0.0` | `0.1.18` | `chat/executor.py`, Claude backend, AG-UI server/protocol. | Retain protocol contract. |
| `werkzeug>=3.1.6` | `3.1.6` | No direct source import; manifest security floor. | Review as transitive/dashboard constraint. |
| `opentelemetry-sdk>=1.20.0,<2.0.0` | `1.39.1` | Observability providers, exporters, serving, optimizer and SDK tracing. | Retain portable observability core. |
| `opentelemetry-exporter-otlp-proto-grpc>=1.20.0,<2.0.0` | `1.39.1` | `lib/observability/exporters/otlp.py`. | Retain supported transport; optionalize only with explicit compatibility decision. |
| `opentelemetry-exporter-otlp-proto-http>=1.20.0,<2.0.0` | `1.39.1` | `lib/observability/exporters/otlp.py`. | Retain supported transport; optionalize only with explicit compatibility decision. |
| `rank-bm25>=0.2.2,<0.3.0` | `0.2.2` | `lib/keyword_search.py` sparse fallback. | Retain until candidate sparse retrieval preserves ranking semantics. |
| `opensearch-py>=2.0.0,<3.0.0` | `2.8.0` | `lib/keyword_search.py` production keyword index. | Optionalize provider capability; preserve existing keyword search. |
| `tiktoken>=0.12.0` | `0.12.0` | `lib/structured_chunker.py`, contextual generation paths. | Retain token budgets/chunking parity; do not duplicate tokenization ownership. |
| `aiohttp>=3.13.4` | `3.14.3` | No direct source import; LiteLLM and other SDK transport dependency; stricter uv security constraint exists. | Review direct vs constraint scope while preserving security floor. |
| `httpx>=0.27` | `0.28.1` | `lib/source_resolver.py` async remote resolution. | Retain; candidate common HTTP client. |
| `authlib>=1.6.11` | `1.7.0` | No direct source import; security-sensitive transitive support. | Review necessity/constraint scope; preserve auth behavior. |
| `azure-core>=1.38.0` | `1.38.2` | `deploy/deployers/azure_containerapps.py` imports exception types. | Move with Azure deployment capability if core no longer needs it; retain direct ownership where imported. |
| `claude-agent-sdk==0.2.82` | `0.2.82` | `lib/backends/claude_backend.py`, hooks, tool/MCP adapters, `lib/claude_context_generator.py`. | Retain native agent execution and pin. |
| `python-frontmatter>=1.1,<2.0` | `1.1.0` | `lib/prompt_version.py`. | Retain prompt versioning. |
| `optuna>=4.8.0` | `4.9.0` | `optimizer/proposers/numeric.py`. | Optionalize optimization capability; retain proposer semantics. |
| `litellm>=1.80.0,<1.89.0` | `1.88.1` | `lib/litellm_support.py`, `lib/llm_context_generator.py`, telemetry instrumentation. | Retain provider-neutral embedding/context inference boundary from 035; LlamaIndex must not add competing provider configuration. |
| `bkflow-feel==1.2.0` | `1.2.0` | `lib/workflow/feel.py`. | Retain exact pin; shared transformer correctness depends on verified 1.2.0 statelessness. |

## Optional capabilities

| Extra | Declared packages | Evidence and disposition |
| --- | --- | --- |
| `dev` | `pytest>=9.0.3`, `pytest-cov>=4.1.0`, `pytest-asyncio>=0.21.0`, `pytest-mock>=3.11.0`, `pytest-xdist>=3.8.0`, `black>=26.3.1`, `ruff>=0.14.7`, `mypy>=1.19.0`, `pre-commit>=3.3.0`, `tox>=4.0.0`, `bandit[toml]>=1.8.0`, `pip-audit>=2.7.0`, `detect-secrets>=1.4.0`, `types-PyYAML>=6.0.12`, `mkdocstrings[python]>=0.25.0`, `mkdocs-llmstxt>=0.4` | Tests, linters, type checking, security and docs tooling; keep separate from production profiles. |
| `pinecone` | `pinecone[asyncio,grpc]~=7.0` | SK connector mapping exists; retain provider coverage through migration. |
| `postgres` | `psycopg[binary,pool]~=3.2` | SK connector plus native filtered reads in `lib/collection_filter.py`; preserve record layout/filter behavior. |
| `qdrant` | `qdrant-client>=1.18,<2.0` | Native async client use spans four files as well as SK connector; preserve bespoke full-text hybrid/RRF and payload indexes. |
| `chromadb` | `chromadb>=0.5,<1.1` | SK connector plus native collection handling; migration parity required. |
| `vectorstores` | `pinecone[asyncio,grpc]~=7.0`, `psycopg[binary,pool]~=3.2`, `qdrant-client>=1.18,<2.0`, `chromadb>=0.5,<1.1` | Convenience union covers only four provider extras, despite eleven connector mappings. |
| `otel-prometheus` | `opentelemetry-exporter-prometheus>=0.60b0,<1.0.0`, `prometheus-client>=0.17.0,<1.0.0` | Declared/configured; no exporter implementation module found. `lib/observability/config.py` tracks enablement; must not claim installation yields working exporter. |
| `otel-azmon` | `azure-monitor-opentelemetry-exporter>=1.0.0b24,<2.0.0` | Declared/configured; no exporter implementation module found. Resolve supported status before packaging promise. |
| `otel-all` | `opentelemetry-exporter-prometheus>=0.60b0,<1.0.0`, `prometheus-client>=0.17.0,<1.0.0`, `azure-monitor-opentelemetry-exporter>=1.0.0b24,<2.0.0` | Convenience group repeats declared Prometheus/Azure Monitor exporters; implementation gap above applies. |
| `claude-otel` | `otel-instrumentation-claude-agent-sdk>=0.0.6,<0.1.0` | Lazily imported by Claude backend; retain tracing/redaction behavior. |
| `openai-agents` | `openai-agents==0.17.4` | Native OpenAI/Azure backend; retain lazy optional install and exact pin. |
| `dashboard` | `dash>=3.0,<4.0`, `plotly>=5.20,<7.0`, `pandas>=2.0` | Actual Dash/Plotly/pandas implementation under `dashboard/`; retain separate profile. pandas also currently arrives through file parsing/evaluation. |
| `deploy` | `docker>=7.0.0` | Docker SDK used by `deploy/builder.py`. |
| `deploy-aws` | `boto3>=1.42.0` | AWS deployer explicitly unimplemented in `deploy/deployers/__init__.py`; boto3 actually used by S3 source resolver. |
| `s3` | `boto3>=1.42.0` | Actual boto3 source resolution in `lib/source_resolver.py`. |
| `azure-blob` | `azure-storage-blob>=12.19` | Actual Azure Blob source resolution in `lib/source_resolver.py`. |
| `all-sources` | `boto3>=1.42.0`, `azure-storage-blob>=12.19` | Convenience union of implemented source resolvers. |
| `deploy-gcp` | `google-cloud-run>=0.13.0` | GCP deployer explicitly unimplemented in `deploy/deployers/__init__.py`; no google-cloud-run source import found. |
| `deploy-azure` | `azure-mgmt-appcontainers>=4.0.0`, `azure-identity>=1.15.0` | Implemented `deploy/deployers/azure_containerapps.py`; preserve Azure credential flow. |
| `deploy-all` | `docker>=7.0.0`, `boto3>=1.42.0`, `google-cloud-run>=0.13.0`, `azure-mgmt-appcontainers>=4.0.0`, `azure-identity>=1.15.0` | Includes implemented Docker/Azure and unimplemented AWS/GCP deployment declarations; resolve advertised support. |
| `temporal` | `temporalio==1.32.0` | Actual durable activity/worker implementation; keep exact pin and workflow sandbox compatibility. |

## Build and dependency groups

- Build: `hatchling`, `hatch-vcs`; retain VCS version/wheel behavior. These are not runtime dependencies.
- `[dependency-groups].dev`: `pyasn1>=0.6.4`, `types-requests>=2.32.4.20250913`, `virtualenv>=20.36.1`; distinct from the published `dev` extra. Reconcile contributor/CI install commands rather than silently dropping either group.
- `tool.uv.prerelease = "allow"` is justified in the manifest by SK Azure agent dependencies. Reassess after SK removal; current `markitdown` lock is also a prerelease.
- `tool.uv.override-dependencies`: `pytz>=2024` overrides bkflow-feel’s old cap and influences Optuna resolution. Preserve FEEL conformance evidence.
- Security constraints in `tool.uv.constraint-dependencies` must be reevaluated against the new graph, not discarded when a direct declaration moves. Verify constraints in built-wheel/non-uv consumer installations as well as development resolution.

## Hidden direct imports and transitive ownership

- Runtime imports not individually declared in core include `openai`, `typing_extensions`, `exceptiongroup`, `grpc`, `openpyxl`, `pandas`, `pdfminer` and `pptx`. Some are intentionally supplied by named extras or an SDK, but each must have an explicit owning capability in the new graph. `openai` belongs with the native OpenAI capability, document libraries with selected ingestion formats, and `grpcio` with OTLP gRPC.
- `psycopg` and `qdrant_client` have provider extras. `weaviate` is imported in `lib/collection_filter.py` but has no HoloDeck extra. Azure AI Search, Weaviate, FAISS, Cosmos NoSQL/MongoDB and SQL Server have connector mappings but no matching provider extras here. A mapping alone does not prove a fresh install supports the provider.
- `semantic_kernel` occurs as direct imports in two source files, but also as dynamically imported connector paths, telemetry helper names, logging namespaces and tests. Import counting alone understates removal scope.

## High-impact locked transitive clusters

| Root (locked) | Selected dependencies and implication |
| --- | --- |
| `semantic-kernel 1.43.0` | `aiortc`, `azure-ai-agents`, `azure-ai-projects`, `azure-identity`, `numpy`, `scipy`, `openapi-core`, `prance`, `pybars4`, `openai`, MCP and OTel. These illustrate agent/cloud/media/schema baggage unrelated to remaining SK storage/text uses. Shared descendants may remain after removal. |
| `markitdown 0.1.5b1` with `all` | Base includes `magika`, `onnxruntime`, BeautifulSoup and Requests; all adds Azure Document Intelligence, pandas, openpyxl, pdfminer/pdfplumber, audio/speech, PowerPoint and YouTube parsing. Selective reader policy can materially affect footprint; benchmark rather than assume savings. |
| `azure-ai-evaluation 1.13.7` | Retire this evaluation root and any other Microsoft AI evaluation packages discovered in its graph. Remove exclusive transitives; retain Azure Identity/Storage, NLTK, pandas, OpenAI, HTTPX/AIOHTTP or YAML packages only where other retained features need them. Verify reverse dependencies and applicable security constraints (FR-030). |
| `evaluate 0.4.6` | `datasets`, Hugging Face Hub, pandas, multiprocess, NumPy, fsspec; metrics framework has a broad graph. |
| `deepeval 3.8.9` | OpenAI/OTel plus pytest, pytest-asyncio, pytest-repeat, pytest-rerunfailures and pytest-xdist, PostHog/Sentry and build utilities. Production graph currently includes testing/telemetry dependencies through evaluation. |
| `litellm 1.88.1` | OpenAI, tokenizers/tiktoken, HTTPX/AIOHTTP, Jinja2, JSON Schema and Pydantic. Keep one inference-owner boundary. |
| `claude-agent-sdk 0.2.82` / `openai-agents 0.17.4` | Claude depends on AnyIO/MCP; OpenAI Agents depends on OpenAI/MCP/Pydantic/Requests/websockets and types-requests. Retaining both native runtimes preserves some shared packages after SK removal. |

## Audit limits and acceptance implications

- This audit inspects declarations, AST import sites, dynamic connector paths and selected lock entries. It does not certify every provider installed, licensed, vulnerability-free or behavior-compatible.
- Fresh isolated installs must exercise CLI, provider initialization, tool ingestion/retrieval, protocol serving and evaluation profiles. The current developer environment can hide missing direct/provider dependencies.
- A “complete revamp” should mean an explicit disposition for every dependency and install capability, with evidence. It should not force replacement of stable Pydantic, native SDK, protocol, workflow, OTel or deployment boundaries by LlamaIndex.

## Non-Python hosting and build prerequisites

Sources: [base image](../../../docker/Dockerfile), [local-wheel image](../../../docker/Dockerfile.local), [image generator](../../../src/holodeck/deploy/dockerfile.py), [runtime validation](../../../src/holodeck/lib/backends/validators.py), and [Claude backend](../../../src/holodeck/lib/backends/claude_backend.py).

| Dependency | Inspected behavior | Disposition / owner |
| --- | --- | --- |
| Python and Linux base image | Both Dockerfiles use `python:3.10-slim`; published base defaults to latest PyPI HoloDeck when no version argument is supplied. | Packaging: record supported Python/OS/architecture and immutable build inputs; test actual images. |
| Claude executable | Claude Agent SDK bundles a CLI binary; the backend runs it as a subprocess. Node is not required solely because the provider is Anthropic. | Backend: retain SDK executable provisioning, platform wheels, startup and lifecycle checks. |
| Node.js / NodeSource | Generator conditionally installs Node 22 through NodeSource; validator requires Node >=18 for MCP commands `node`, `npx`, `yarn`, or `pnpm`. Some generator comments still incorrectly attribute the need to Claude itself. | Backend/packaging: preserve conditional installation, verify supported versions and bootstrap provenance; correct comments during implementation. |
| MCP launchers and server packages | Selected stdio tools may require Node package managers, `uv`/`uvx`, Python or other configured executables and server packages. These are agent-specific dependencies, not all universal requirements. | Tool/deployment maintainers: inventory the selected agent's command/args and package versions, provide missing-launcher errors, do not install arbitrary user tool dependencies globally. |
| `uv` | Base images install via `https://astral.sh/uv/install.sh`; generated extras install uses `--prerelease=allow`. | Packaging: pin/review bootstrap and resolver inputs; record the actual installed graph. |
| `curl`, `ca-certificates`, `bubblewrap`, shell utilities | Both base images install the first three. Curl supplies bootstrap and health checks; Dockerfile documents bubblewrap for Claude subprocess environment scrubbing. | Deployment/security: verify runtime need and platform support; preserve TLS, health and credential isolation. |
| Docker daemon / build engine and registry | Docker SDK drives image builds; generated images default to `ghcr.io/justinbarias/holodeck-base:latest`. | Deployment: retain supported workflow, record base/agent digest, candidate version, extras and package/system manifest. |
| External services | Configured model APIs, vector stores, MCP servers, OTel collectors, optional OpenSearch/Temporal and cloud source/deployment services. | Respective adapter owners: compatibility and availability tests; distinguish service dependency from installed library. |

The generator currently installs unversioned `holodeck-ai[extras]` independently of the audited lock. This can change the candidate graph even after a successful local wheel test. FR-029/AC-11 require reproducible base, agent, and worker image identity and resolved dependency evidence. There is no repository-owned Node `package.json` outside ignored samples in this baseline; sample frontend dependencies are not universal hosting dependencies.

## Ownership and disposition gates

Owners apply to the rows in their domain; the spec acceptance IDs are the exit gates. Unresolved dispositions stay open until those gates provide evidence.

| Domain | Owner | Gate |
| --- | --- | --- |
| Config, CLI, native SDKs, MCP and shared utilities | Backend maintainers | AC-01–03; isolated imports and both native backend contracts |
| Retrieval, readers, tokenization and provider extras | Retrieval maintainers | AC-04–08; all-provider operations, stored data and format fixtures |
| HTTP, serving and deployment/source packages | Serving/deployment maintainers | AC-03/06/11; lifecycle, async I/O, actual deployer support |
| Evaluation and Optuna | Evaluation/optimizer maintainers | AC-09, AC-13; retain DeepEval and deterministic/code graders, remove Microsoft/legacy evaluation, verify migration and optional installs |
| OTel, exporter extras and telemetry dependencies | Observability maintainers | AC-10; actual exporter implementation, routing and redaction |
| Security floors, transitive-only declarations, docs/release/build/dev and profile resolution | Packaging/security maintainers | AC-01/02/11/12; reverse graph, clean wheels and reviewed constraints |
| FEEL/Temporal/schema primitives | Workflow maintainers | AC-09; conformance, replay and sandbox safety |

## Complete resolver constraints

These are current declarations, not newly verified vulnerability claims. Preserve or revise them based on the resolved candidate graph and [Security](../../SECURITY.md). `uv` constraints alone do not constrain a downstream pip installation of a published wheel.

| Constraint | Proposed disposition / gate |
| --- | --- |
| `sentry-sdk>=2.0.0,<3.0.0` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `filelock>=3.20.3` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `jaraco-context>=6.1.0` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `nltk>=3.9.3` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `pyopenssl>=26.0.0` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `pillow>=12.3.0` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `protobuf>=5.29.6` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `python-multipart>=0.0.31` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `pyarrow>=23.0.1` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `markdown>=3.10.2` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `pygments>=2.20.0` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `aiohttp>=3.14.1` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `pyjwt>=2.13.0` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `pip>=26.1.2` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `msgpack>=1.2.1` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `pydantic-settings>=2.14.2` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `joserfc>=1.6.8` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `soupsieve>=2.8.4` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `setuptools>=83.0.0` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |
| `pymdown-extensions>=11.0.0` | Packaging/security review of reverse dependencies and patched compatibility; retain protection until AC-11 establishes replacement/removal. |

The declaration above is an audit snapshot. `pyproject.toml` and `uv.lock` remain the executable source of truth. Candidate changes must update this inventory with measured evidence.
