# Dependency inventory

Status: [state after implementation](#state-after-implementation-2026-11-10) first, then the audit baseline for [spec 042](spec.md). Re-inspected 2026-11-10 at commit `00d1ed4`, after the dependency upgrades in #419–#431. The first audit was 2026-09-06 at `3e59e1d`. This is a declaration, import and lockfile audit. It is not a measured clean-install footprint. Versions come from `uv.lock`; upstream "latest" claims are dated and sourced in the [selection evidence](../../design-docs/042-dependency-stack-revamp/selection-evidence.md).

## State after implementation (2026-11-10)

Written from `pyproject.toml` and `uv.lock` at commit `536cad9`, the tip of the 042 stack after T23. The sections below this one record the audit that led to the work and are not rewritten. Where they disagree with this section, this section is current.

Result: Semantic Kernel, `azure-ai-evaluation` and `evaluate` are in no install profile. The lock has **329** `[[package]]` entries (376 before the work, 347 after the Semantic Kernel removal). The base install resolves 143 packages (264 before). Measured sizes and start-up times are in the [baseline measurements](../../exec-plans/active/042-dependency-stack-revamp/baseline-measurements.md#result-after-spec-042). `requires-python` is `>=3.11`.

### Base install (`[project].dependencies`, 35 declarations)

| Dependency | Purpose |
| --- | --- |
| `pydantic`, `pyyaml`, `jinja2`, `python-dotenv`, `click`, `inquirerpy` | Agent models and validation, YAML, templates, environment files, the CLI and its prompts |
| `jsonschema[format-nongpl]`, `referencing`, `lark`, `bkflow-feel`, `python-ulid` | Schema gates, FEEL decision tables, identifiers |
| `claude-agent-sdk` (pinned) | Claude backend (Anthropic and Ollama providers) |
| `litellm` | Embeddings and contextual-retrieval chat |
| `mcp` (`>=1.30.0,<3`) | MCP servers and bridges |
| `fastapi`, `starlette`, `uvicorn[standard]`, `ag-ui-protocol`, `werkzeug` | `holodeck serve` and the AG-UI protocol |
| `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc`, `opentelemetry-exporter-otlp-proto-http` | Telemetry and OTLP export |
| `markitdown[pdf,docx,pptx,xlsx]`, `openpyxl`, `pandas`, `pdfminer.six`, `python-pptx`, `pypdf` | File parsing for retrieval and multimodal inputs |
| `tiktoken` | Token counts for the plain text splitter |
| `rank-bm25` | The in-process BM25 index used for lexical and hybrid search |
| `httpx`, `aiohttp`, `urllib3`, `authlib`, `azure-core`, `cryptography` | HTTP clients, authentication and transport (several carry security lower bounds) |

The base install carries no vector store client, no evaluation library, no optimizer and no docs or release tooling.

### Extras (`[project.optional-dependencies]`)

| Extra | Installs | Feature |
| --- | --- | --- |
| `eval` | `deepeval`, `anthropic`, `ollama` | DeepEval metrics and judge models |
| `nlp` | `nltk`, `rouge-score`, `sacrebleu` | BLEU, ROUGE and METEOR |
| `optimizer` | `optuna` | `holodeck test optimize` |
| `opensearch` | `opensearch-py` | OpenSearch keyword index |
| `qdrant`, `chromadb`, `postgres`, `pinecone`, `weaviate`, `azure-ai-search`, `azure-cosmos` | One native client each (`qdrant-client`, `chromadb`, `psycopg[binary,pool]` and `pgvector`, `pinecone` 10, `weaviate-client`, `azure-search-documents` and `azure-identity`, `azure-cosmos`) | Vector store providers |
| `vectorstores` | All seven store extras | All vector stores |
| `openai-agents` | `openai-agents`, `openai` 3 | OpenAI and Azure OpenAI backend |
| `dashboard` | `dash`, `plotly`, `pandas` | `holodeck test view` |
| `temporal` | `temporalio` | Temporal worker and activities |
| `deploy`, `deploy-aws`, `deploy-gcp`, `deploy-azure`, `deploy-all` | `docker`, `boto3`, `google-cloud-run`, `azure-mgmt-appcontainers` and `azure-identity` | `holodeck deploy` and its targets |
| `s3`, `azure-blob`, `all-sources` | `boto3`, `azure-storage-blob` | Remote file sources |
| `otel-prometheus`, `otel-azmon`, `otel-all`, `claude-otel` | Exporters and the Claude Agent SDK instrumentation | Extra telemetry |
| `all` | Every runtime extra above | Everything except `dev` |
| `dev` | Test, lint and type tools, `mkdocs-material`, `mkdocstrings`, `twine`, audit tools | Development only |

A command that needs a missing extra stops early and prints the install command (`MissingExtraError`).

### Removed

- **Semantic Kernel** and what only it needed: `azure-ai-agents`, `azure-ai-projects`, `aiortc`, `av`, `scipy`, `openapi-core`, `prance` and others. The `faiss`, `sql-server` and `azure-cosmos-mongo` providers went with it.
- **Microsoft AI evaluation** (`azure-ai-evaluation`, `ruamel-yaml`) and **Hugging Face `evaluate`** (with `datasets`, `pyarrow`, `dill`, `multiprocess`, `xxhash`).
- **`markitdown[all]`**: ten packages, among them `azure-ai-documentintelligence`, `azure-ai-contentunderstanding`, `youtube-transcript-api`, `speechrecognition` and `pydub`.
- **Unused or duplicate direct declarations**: `requests` (HTTPX replaces it), `aiofiles`, `python-dateutil`, `absl-py`, `ml-dtypes`, the stale `sentry-sdk` constraint.
- **Moved out of the base install**: DeepEval, `anthropic`, `ollama`, NLTK, `rouge-score`, `sacrebleu`, `optuna`, `opensearch-py` (now extras); `mkdocs-material` and `twine` (now `dev` only).
- **Dependabot ignore for Pinecone majors**, and the `tool.uv.prerelease` setting.

### Constraints still in `[tool.uv]`

- `override-dependencies`: `pytz>=2024`. `bkflow-feel` pins `pytz<2024`, which would downgrade `pytz` and `optuna`; the FEEL conformance tests in `tests/unit/workflow/` cover the newer `pytz`.
- `constraint-dependencies` (23 entries) are security lower bounds on transitive packages, each with its advisory in a comment in `pyproject.toml`: `filelock`, `jaraco-context`, `nltk`, `pillow`, `protobuf`, `python-multipart`, `markdown`, `pygments`, `aiohttp`, `pyjwt`, `pip`, `msgpack`, `pydantic-settings`, `joserfc`, `soupsieve`, `setuptools`, `pymdown-extensions`, `anyio`, `certifi`, `fsspec`, `multidict`, `oauthlib`, `opentelemetry-api`. Remove one when the packages that need it require a fixed version themselves.
- No cap comes from Semantic Kernel any more. `mcp` allows 2.x (locked 2.3.0) and `pinecone` is 10.x. One cap comes from a client: `weaviate-client` needs `grpcio<1.80`, so the lock holds `grpcio` 1.78.2 for it.
- Known issue: the locked `pypdfium2` 5.15.1 is yanked on PyPI (reason given there: closing of remaining objects on interpreter exit is broken); 5.15.2 exists. See the [technical debt tracker](../../exec-plans/tech-debt-tracker.md).

## Audit baseline before the work

The rest of this file is the audit as it stood at `00d1ed4`.

## What changed since 2026-09-06

- `requires-python` is now `>=3.11` (was `>=3.10`). CI runs 3.11 and 3.13. FR-006 must be read against 3.11.
- Upgraded: openai-agents 0.17 → 0.24 with openai 3, LiteLLM 1.88 → 1.108, DeepEval 3.8 → 4.3, claude-agent-sdk 0.2.82 → 0.2.179, chromadb → 1.5, temporalio 1.32 → 1.35, opensearch-py 2 → 3, dash 3 → 4, twine 6 → 7, azure-mgmt-appcontainers 4 → 5, pypdf 6 → 7, optuna 4 → 5, MarkItDown 0.1.5b1 → 0.1.9 (stable), Semantic Kernel 1.43 → 1.47.
- SK is now the only package that holds back `mcp` 2 (it requires `mcp<2.0`) and newer Pinecone (`~=7.0`). Both native agent SDKs allow `mcp<3`.
- The CLI imports each command lazily (#430), so a heavy dependency no longer slows unrelated commands. Profile isolation (FR-005) still needs installed-wheel tests.
- The lock has 370 packages. The base install (core dependencies only) resolves 269; all extras resolve 368.

## Measured removal effect on the base install

Computed from `uv.lock` by removing a root and counting packages that nothing else in the base install requires. These are package counts, not bytes or import time (FR-008 still requires those).

| Root removed | Packages that leave the base install |
| --- | --- |
| `semantic-kernel` | 26 |
| `markitdown[all]` → selective extras | up to 26 (depends on the extras kept) |
| `deepeval` → evaluation profile | 17 |
| `mkdocs-material` → docs tooling | 14 |
| `twine` → release tooling | 14 |
| `evaluate` | 5 |
| `azure-ai-evaluation` | 3 |
| SK + azure-ai-evaluation + evaluate + mkdocs-material + twine together | 71 (269 → 198) |

## Direct runtime declarations

All **50** `[project].dependencies` declarations are accounted for. "No import" is evidence for review, not proof that a dependency is safe to remove: dynamic loading, upstream requirements and security constraints must be checked. Paths are relative to `src/holodeck/` unless stated otherwise.

| Dependency constraint | Locked version | Observed role | Proposed disposition |
| --- | --- | --- | --- |
| `requests>=2.32.5,<3.0.0` | `2.34.2` | `lib/file_processor.py`, `services/mcp_registry.py`. | Move both sites to HTTPX and drop the direct declaration. No blocker found; set `follow_redirects=True`, because HTTPX does not follow redirects by default. Requests stays in the graph through MarkItDown, tiktoken, DeepEval and OpenAI Agents. |
| `pydantic>=2.11.0` | `2.13.5` | 46 source files: models, config validation, backend output, workflow and serving. | Retain as HoloDeck schema/contract boundary. SK caps `pydantic<2.14`; the cap goes away with SK. |
| `pyyaml>=6.0.0` | `6.0.3` | Config loader/manager, templates, workflows, optimizer output. | Retain. |
| `click>=8.3.3,<9.0.0` | `8.3.3` | 13 CLI source files. Commands load lazily since #430. | Retain; preserve security lower bound. |
| `jinja2>=3.0.0,<4.0.0` | `3.1.6` | `lib/template_engine.py`, `deploy/dockerfile.py`. | Retain. |
| `python-dotenv>=1.0.0` | `1.2.4` | `cli/main.py`. | Retain. |
| `python-dateutil>=2.8.0` | `2.9.0.post0` | No import in `src/` or `tests/`. Required by bkflow-feel, botocore, opensearch-py and others. | Remove the direct declaration; it stays as a transitive dependency. |
| `urllib3>=2.8.0,<3.0.0` | `2.8.0` | No direct source import; security floor. | Move the floor to resolver constraints and published-wheel policy; do not lose it. |
| `cryptography>=50.0.0,<51.0.0` | `50.0.2` | No direct source import; security floor. | Same as urllib3: keep the floor, review where it is declared. |
| `mkdocs-material>=9.7.7,<10.0.0` | `9.8.1` | Documentation tooling; no runtime import. Brings 14 packages that nothing else needs. | Move out of runtime into docs tooling. |
| `twine>=6.2.0,<8.0.0` | `7.0.0` | Release tooling; no runtime import. Brings 14 packages that nothing else needs. 7.x reads metadata 2.5 wheels (#431). | Move out of runtime into release tooling. |
| `jsonschema[format-nongpl]>=4.18.0,<5.0.0` | `4.26.0` | `lib/workflow/edge.py`, native backend output validation. | Retain format-nongpl extra and >=4.18 registry semantics. |
| `referencing>=0.36.0,<1.0.0` | `0.36.2` | `lib/workflow/edge.py`. | Retain explicit direct dependency. |
| `lark>=1.1.0,<2.0.0` | `1.3.1` | `lib/workflow/feel.py`. | Retain explicit direct dependency. |
| `semantic-kernel>=1.39.4,<2.0.0` | `1.47.0` | Two importing files: `lib/vector_store.py` (record contracts, 11 dynamically imported connectors) and `lib/text_chunker.py`. Also telemetry helper names, logging namespaces and 10 test files. Brings 26 packages that nothing else needs (aiortc, av, azure-ai-agents, azure-ai-projects, scipy, openapi-core, prance, pybars4 and others). Its metadata caps `mcp<2.0`, `pydantic<2.14` and `pinecone~=7.0`. | Remove. Replace with HoloDeck-owned adapters on native store clients and an in-house tiktoken splitter; see the [selection evidence](../../design-docs/042-dependency-stack-revamp/selection-evidence.md). |
| `markitdown[all]>=0.1.4,<0.2.0` | `0.1.9` | `lib/file_processor.py`. Now a stable release (was 0.1.5b1). `[all]` brings 26 packages that nothing else needs, including audio/speech, YouTube, Azure Document Intelligence and `azure-ai-contentunderstanding>=1.2.0b1`, which has only prerelease versions. | Retain MarkItDown; replace `[all]` with `[pdf,docx,pptx,xlsx]` (add `xls`/`outlook` only on evidence). Base MarkItDown still requires magika, which pulls onnxruntime. Declare directly imported `openpyxl`, `pdfminer.six` and `python-pptx`. Verify with format fixtures (FR-021). |
| `pypdf>=6.21.4` | `7.0.0` | `lib/pdf_processor/page_extractor.py`. Now on 7.x. | Retain; preserve security floor. |
| `azure-ai-evaluation>=1.13.0,<2.0.0` | `1.18.11` | `lib/evaluators/azure_ai.py`. Own closure is 61 packages; only 3 are exclusive (msrest, oauthlib, requests-oauthlib). | Remove completely (FR-030–031, AC-13). Unchanged decision. |
| `evaluate>=0.4.6,<0.5.0` | `0.4.6` | `lib/evaluators/nlp_metrics.py`: only ROUGE and METEOR use `evaluate.load`, which fetches metric scripts from the Hugging Face Hub at first use. No release since 2025-09. Brings datasets, pyarrow, dill, multiprocess, xxhash. | Remove. Call `rouge_score` and NLTK METEOR directly; freeze fixtures first to prove equal scores (FR-025). Drops the `datasets` and `pyarrow` constraints if no other package needs them. |
| `aiofiles>=25.1.0,<26.0.0` | `25.1.0` | No import in `src/` or `tests/`; only HoloDeck requires it. | Remove. |
| `rouge-score>=0.1.2,<0.2.0` | `0.1.2` | ROUGE backend, reached through `evaluate` today. Last release 2022, sdist only; requires absl-py, nltk, numpy, six. | Retain and call directly. No maintained drop-in found. |
| `absl-py>=2.3.1,<3.0.0` | `2.5.1` | No direct import; rouge-score already requires it. | Remove the direct declaration. |
| `sacrebleu>=2.5.1,<3.0.0` | `2.6.0` | `lib/evaluators/nlp_metrics.py`, called directly for BLEU. | Retain. |
| `anthropic>=0.72.0,<2.7.0` | `2.6.0` | No import in `src/` or `tests/`; only HoloDeck requires it. DeepEval does not declare it. The `<2.7.0` cap excludes 2.7.0, which moves to `httpx2`. | Remove after a Claude backend and DeepEval-on-Anthropic test run without it. |
| `ollama>=0.4,<1.0` | `0.6.3` | No import in `src/` or `tests/`; only HoloDeck requires it. | Remove after Ollama runtime and embedding tests run without it. |
| `deepeval>=4.3.3,<5` | `4.3.3` | Eight evaluator modules under `lib/evaluators/deepeval/`. Now 4.x. Still requires pytest, pytest-asyncio, pytest-xdist, pytest-repeat, pytest-rerunfailures and posthog at runtime; no slim extra exists. No longer declares sentry-sdk. | Retain (FR-025, AC-09). Move to an evaluation profile so the base host does not install pytest and posthog. |
| `python-ulid>=3.1.0,<5.0.0` | `4.1.0` | Claude backend and serve sessions/protocols/models. Now 4.x. | Retain. |
| `ml-dtypes>=0.4.0,<0.7.0` | `0.6.0` | No direct import; only HoloDeck requires it. Cap widened to `<0.7` in #431. | Remove the direct declaration unless a reverse dependency appears after SK removal. |
| `mcp>=1.30.0,<2` | `1.30.0` | Consumed by native SDKs and tool bridges. Latest upstream is 2.3.0. SK requires `mcp<2.0`. claude-agent-sdk 0.2.179 and openai-agents 0.24.0 both allow `mcp<3`. | Retain. Lift the `<2` cap after SK removal, and only after the Claude backend passes on MCP 2: the manifest comment records a past `Server.list_tools` failure. |
| `inquirerpy>=0.3.4,<0.4.0` | `0.3.4` | `cli/utils/wizard.py`. | Retain. |
| `fastapi>=0.115.0,<1.0.0` | `0.162.2` | `serve/server.py`, REST protocol, middleware, tool-init routes. | Retain; consider explicit serving profile. |
| `starlette>=1.3.1` | `1.10.0` | `serve/middleware.py`; also FastAPI/MCP dependency. | Retain with security floor. |
| `uvicorn[standard]>=0.34.0,<1.0.0` | `0.54.0` | `cli/commands/serve.py`. | Retain. |
| `ag-ui-protocol>=0.1.18,<1.0.0` | `0.1.22` | Chat executor, Claude backend, AG-UI server/protocol. Upstream 1.0 needs a multimodal migration (Dependabot #413). | Retain; 1.0 migration is separate work. |
| `werkzeug>=3.1.9` | `3.1.9` | No direct import; security floor. Required by dash and flask (dashboard) and by openapi-core (SK). | Move the floor to the dashboard profile/constraints after SK removal. |
| `opentelemetry-sdk>=1.20.0,<2.0.0` | `1.45.2` | Observability providers, exporters, serving, optimizer, SDK tracing. | Retain. |
| `opentelemetry-exporter-otlp-proto-grpc>=1.20.0,<2.0.0` | `1.45.2` | `lib/observability/exporters/otlp.py`. | Retain. |
| `opentelemetry-exporter-otlp-proto-http>=1.20.0,<2.0.0` | `1.45.2` | `lib/observability/exporters/otlp.py`. | Retain. |
| `rank-bm25>=0.2.2,<0.3.0` | `0.2.2` | `lib/keyword_search.py` sparse fallback. Last release 2022; numpy only; works on Python 3.14. | Retain. `bm25s` 0.3.16 is active but has no mode that reproduces BM25Okapi's idf floor, so rankings would change. Fallback if maintenance fails: vendor BM25Okapi (about 60 lines). |
| `opensearch-py>=2.0.0,<4.0.0` | `3.2.0` | `lib/keyword_search.py` production keyword index. Now 3.x, verified against OpenSearch 2.19 (#431). | Optionalize as a provider capability. |
| `tiktoken>=0.12.0` | `0.14.0` | `lib/structured_chunker.py`, contextual generation paths. | Retain; also the token counter for the replacement plain splitter. |
| `aiohttp>=3.14.4` | `3.14.5` | No direct import; LiteLLM/DeepEval transport; security floor. | Keep the floor; review direct vs constraint scope. |
| `httpx>=0.27` | `0.28.1` | `lib/source_resolver.py`. | Retain; becomes the single HTTP client. |
| `authlib>=1.9.1` | `1.9.5` | No direct import; only HoloDeck requires it. | Review: either a security floor for a dependency that has since left the graph, or dead. Remove if no reverse dependency remains. |
| `azure-core>=1.38.0` | `1.41.0` | `deploy/deployers/azure_containerapps.py`, `lib/source_resolver.py`. | Move with the Azure deploy/source capabilities. |
| `claude-agent-sdk==0.2.179` | `0.2.179` | Claude backend, hooks, tool/MCP adapters, `lib/claude_context_generator.py`. | Retain native agent execution and exact pin. |
| `python-frontmatter>=1.1,<2.0` | `1.3.0` | `lib/prompt_version.py`. | Retain. |
| `optuna>=4.8.0` | `5.0.0` | `optimizer/proposers/numeric.py`. Now 5.x; brings sqlalchemy 2.1 and alembic. | Optionalize with the optimizer. |
| `litellm>=1.108.3,<1.109` | `1.108.3` | `lib/litellm_support.py`, `lib/llm_context_generator.py`, telemetry instrumentation. Pinned to one minor (`<1.109`). | Retain as the embedding/context inference boundary. |
| `bkflow-feel==1.2.0` | `1.2.0` | `lib/workflow/feel.py`. | Retain exact pin. |

## Optional capabilities

| Extra | Declared packages (locked) | Evidence and disposition |
| --- | --- | --- |
| `dev` | `pytest>=9.0.3` (`9.2.0`), `pytest-cov>=4.1.0` (`7.1.0`), `pytest-asyncio>=0.21.0` (`1.4.1.post1`), `pytest-mock>=3.11.0` (`3.17.0`), `pytest-xdist>=3.8.0` (`3.8.0`), `black>=26.10.1` (`26.10.1`), `ruff>=0.17.4` (`0.17.4`), `mypy>=2.4.1` (`2.4.1`), `pre-commit>=3.3.0` (`4.7.0`), `tox>=4.0.0` (`4.67.13`), `bandit[toml]>=1.8.0` (`1.9.4`), `pip-audit>=2.7.0` (`2.10.1`), `detect-secrets>=1.4.0` (`1.5.0`), `types-PyYAML>=6.0.12` (`6.0.12.20261103`), `mkdocstrings[python]>=0.25.0` (`1.0.8`), `mkdocs-llmstxt>=0.4` (`0.5.1`) | Tests, linters, type checking, security and docs tooling; keep separate from production profiles. |
| `pinecone` | `pinecone[asyncio,grpc]~=7.0` (`7.0.2.dev1`) | Held at 7.x only by SK (`7.0.2.dev1`, a dev build, is what the `~=7.0` pin resolves to; 7.1+ needs `packaging<25`). Latest client is 10.3.0. SK's connector fails to import with 8+. Dependabot ignores Pinecone majors until SK is removed. |
| `postgres` | `psycopg[binary,pool]~=3.2` (`3.3.8`) | SK connector plus native filtered reads in `lib/collection_filter.py`; preserve record layout and filters. |
| `qdrant` | `qdrant-client>=1.18,<2.0` (`1.20.0`) | Native async client in four files as well as the SK connector; live verify coverage. Preserve full-text hybrid/RRF and payload indexes. |
| `chromadb` | `chromadb>=0.5,<1.6` (`1.5.9`) | SK connector plus native client setup. Now 1.5.x with a live verify case (#425). SK caps `chromadb<1.6`. |
| `vectorstores` | `pinecone[asyncio,grpc]~=7.0` (`7.0.2.dev1`), `psycopg[binary,pool]~=3.2` (`3.3.8`), `qdrant-client>=1.18,<2.0` (`1.20.0`), `chromadb>=0.5,<1.6` (`1.5.9`) | Convenience union of four provider extras; eleven connector names exist. |
| `otel-prometheus` | `opentelemetry-exporter-prometheus>=0.60b0,<1.0.0` (`0.66b2`), `prometheus-client>=0.17.0,<1.0.0` (`0.26.0`) | Declared/configured; no exporter implementation module found. Must not claim installation yields a working exporter. |
| `otel-azmon` | `azure-monitor-opentelemetry-exporter>=1.0.0b24,<2.0.0` (`1.0.0b59`) | Declared/configured; no exporter implementation module found. Resolve supported status before packaging promise. |
| `otel-all` | `opentelemetry-exporter-prometheus>=0.60b0,<1.0.0` (`0.66b2`), `prometheus-client>=0.17.0,<1.0.0` (`0.26.0`), `azure-monitor-opentelemetry-exporter>=1.0.0b24,<2.0.0` (`1.0.0b59`) | Convenience group for the two exporters above; same implementation gap. |
| `claude-otel` | `otel-instrumentation-claude-agent-sdk>=0.0.6,<0.1.0` (`0.0.6`) | Lazily imported by the Claude backend; retain tracing/redaction behaviour. |
| `openai-agents` | `openai-agents==0.24.0` (`0.24.0`), `openai>=3,<4` (`3.48.0`) | Native OpenAI/Azure backend. Now on the openai 3 stack (#424); `openai` is declared here, which closes the earlier hidden-import gap. |
| `dashboard` | `dash>=3.0,<5.0` (`4.5.0`), `plotly>=5.20,<8.0` (`7.1.0`), `pandas>=2.0` (`2.3.3`) | Dash 4 (#431), Plotly 7, pandas. Retain as a separate profile. |
| `deploy` | `docker>=7.0.0` (`7.2.0`) | Docker SDK used by `deploy/builder.py`. |
| `deploy-aws` | `boto3>=1.42.0` (`1.43.126`) | AWS deployer is unimplemented; boto3 is actually used by the S3 source resolver. |
| `s3` | `boto3>=1.42.0` (`1.43.126`) | boto3 source resolution in `lib/source_resolver.py`. |
| `azure-blob` | `azure-storage-blob>=12.19` (`12.32.0b1`) | Azure Blob source resolution in `lib/source_resolver.py`. The lock resolves a beta (`12.32.0b1`) because prereleases are allowed. |
| `all-sources` | `boto3>=1.42.0` (`1.43.126`), `azure-storage-blob>=12.19` (`12.32.0b1`) | Convenience union of implemented source resolvers. |
| `deploy-gcp` | `google-cloud-run>=0.13.0` (`0.16.1`) | GCP deployer is unimplemented; no google-cloud-run import found. |
| `deploy-azure` | `azure-mgmt-appcontainers>=5.1.0,<6` (`5.1.0`), `azure-identity>=1.15.0` (`1.27.0`) | Implemented `deploy/deployers/azure_containerapps.py`; azure-mgmt-appcontainers moved to 5.x (#426). |
| `deploy-all` | `docker>=7.0.0` (`7.2.0`), `boto3>=1.42.0` (`1.43.126`), `google-cloud-run>=0.13.0` (`0.16.1`), `azure-mgmt-appcontainers>=5.1.0,<6` (`5.1.0`), `azure-identity>=1.15.0` (`1.27.0`) | Includes implemented Docker/Azure and unimplemented AWS/GCP declarations; resolve advertised support. |
| `temporal` | `temporalio==1.35.0` (`1.35.0`) | Durable activity/worker implementation; exact pin, moved to 1.35 with live tests (#431). |

## Vector store providers: what works today

Checked in the full development environment (all extras installed) at `00d1ed4` by importing each SK connector module. "Repo use" counts files under `sample/`, `docs/examples`, `docs/guides` and templates that set the provider.

| Provider | SK connector imports | HoloDeck extra for the client | Repo use | Live verify |
| --- | --- | --- | --- | --- |
| `qdrant` | Yes | `qdrant` | 10 | Yes |
| `chromadb` | Yes | `chromadb` | 22 | Yes |
| `postgres` | Yes | `postgres` | 8 | No |
| `pinecone` | Yes (7.x only) | `pinecone` | 4 | No |
| `in-memory` | Yes | none needed | 1 | No |
| `sql-server` | Yes (module only; needs `pyodbc`, not installed) | none | 0 | No |
| `azure-ai-search` | No: `azure.search` missing | none | 0 | No |
| `weaviate` | No: `weaviate` missing | none | 0 | No |
| `faiss` | No: `faiss` missing | none | 0 | No |
| `azure-cosmos-nosql` | No: `azure.cosmos` missing | none | 0 | No |
| `azure-cosmos-mongo` | No: `azure.cosmos` missing | none | 0 | No |

Six of the eleven provider names cannot run from any documented HoloDeck install. A user must install the client by hand. This is evidence for the provider-scope decision in the spec (FR-014); it is not a decision to retire them.

## Build and dependency groups

- Build: `hatchling`, `hatch-vcs`; retain VCS version/wheel behaviour.
- `[dependency-groups].dev`: `pyasn1>=0.6.5`, `types-requests>=2.32.4.20250913`, `virtualenv>=21.14.9`; distinct from the published `dev` extra. Reconcile contributor/CI install commands.
- `tool.uv.prerelease = "allow"`: the manifest comment names SK's `azure-ai-agents`. MarkItDown `[all]` also needs it (`azure-ai-contentunderstanding` has only prereleases). The setting lets other packages resolve prereleases: the lock holds `pinecone 7.0.2.dev1`, `azure-storage-blob 12.32.0b1`, `lxml 7.0.0b1` and `wrapt 1.17.4rc1`. Removing SK and MarkItDown `[all]` should allow `prerelease = "if-necessary-or-explicit"`; OTel 0.x beta packages are explicit pins.
- `tool.uv.override-dependencies`: `pytz>=2024` overrides bkflow-feel's old cap. Preserve FEEL conformance evidence.
- The published wheel carries no `readme`, so the PyPI page has no description (twine warns). Fix with the packaging changes.

## Hidden direct imports and transitive ownership

- Imported but not declared in core: `openpyxl`, `pdfminer`, `pptx` (all through MarkItDown `[all]`), `grpc` (through the OTLP gRPC exporter), `pandas` (dashboard extra; in the base install it arrives only through azure-ai-evaluation and `evaluate`), `weaviate` (no extra). Each needs an owning declaration before its current carrier is removed.
- `openai` is now declared in the `openai-agents` extra.
- `semantic_kernel` appears as imports in two source files, and as connector path strings, telemetry helper names (`enable_semantic_kernel_telemetry`), logging namespaces and default filter namespaces in six more. Import counting alone understates removal scope.

## High-impact locked transitive clusters

| Root (locked) | Selected dependencies and implication |
| --- | --- |
| `semantic-kernel 1.47.0` | Own closure 98 packages; 26 exclusive: aiortc, aioice, av, pylibsrtp, azure-ai-agents, azure-ai-projects, scipy, openapi-core, prance, pybars4, cloudevents and others. Caps `mcp<2.0`, `pydantic<2.14`. |
| `markitdown 0.1.9` with `all` | 26 exclusive: magika, onnxruntime, pdfminer-six, pdfplumber, python-pptx, openpyxl, mammoth, speechrecognition, pydub, youtube-transcript-api, azure-ai-documentintelligence, azure-ai-contentunderstanding (prerelease). |
| `azure-ai-evaluation 1.18.11` | Own closure 61; only msrest, oauthlib, requests-oauthlib are exclusive. It is one of two carriers of pandas and NLTK into the base install. |
| `evaluate 0.4.6` | datasets, pyarrow, dill, multiprocess, xxhash exclusive. Downloads metric scripts at runtime. |
| `deepeval 4.3.3` | 17 exclusive: pytest and four pytest plugins, posthog, questionary, pyfiglet, setuptools, wheel. |
| `litellm 1.108.3` | 9 exclusive: boto3/botocore/s3transfer, tokenizers, h2 stack. Keep one inference-owner boundary. |
| `optuna 5.0.0` | sqlalchemy 2.1.8, alembic, mako, colorlog. sqlalchemy 2.1 conflicts with the LlamaIndex Postgres connector (`sqlalchemy<2.1`). |
| `claude-agent-sdk 0.2.179` / `openai-agents 0.24.0` | Both allow `mcp<3`. Retaining both native runtimes preserves shared packages after SK removal. |

## Audit limits and acceptance implications

- This audit inspects declarations, AST import sites, dynamic connector paths, lock entries and connector imports in one developer environment. It does not certify that a provider is licensed, vulnerability-free or behaviour-compatible.
- Exclusive-package counts follow required dependencies and requested extras in `uv.lock`. Platform markers are ignored, so a count can differ by one or two per platform.
- Fresh isolated installs must exercise CLI, provider initialization, tool ingestion/retrieval, serving and evaluation profiles.

## Non-Python hosting and build prerequisites

Sources: [base image](../../../docker/Dockerfile), [local-wheel image](../../../docker/Dockerfile.local), [image generator](../../../src/holodeck/deploy/dockerfile.py), [runtime validation](../../../src/holodeck/lib/backends/validators.py), and [Claude backend](../../../src/holodeck/lib/backends/claude_backend.py).

| Dependency | Inspected behavior | Disposition / owner |
| --- | --- | --- |
| Python and Linux base image | Both Dockerfiles now use `python:3.13-slim`; published base defaults to latest PyPI HoloDeck when no version argument is supplied. | Packaging: record supported Python/OS/architecture and immutable build inputs; test actual images. |
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

These are current declarations (28; 20 at the first audit), not newly verified vulnerability claims. `uv` constraints do not constrain a downstream pip installation of a published wheel.

| Constraint | Proposed disposition / gate |
| --- | --- |
| `sentry-sdk>=2.0.0,<3.0.0` | Stale: sentry-sdk is no longer in the lock (DeepEval 4 dropped it). Remove after confirming no profile resolves it. |
| `filelock>=4.2.1` | Retain until AC-11 reviews the candidate graph. |
| `jaraco-context>=6.1.0` | Retain until AC-11 reviews the candidate graph. |
| `nltk>=3.10.4` | Still needed: rouge-score requires NLTK, and METEOR uses it. The scoped security exception stays. |
| `pyopenssl>=26.0.0` | Retain until AC-11 reviews the candidate graph. |
| `pillow>=13.0.0` | Retain until AC-11 reviews the candidate graph. |
| `protobuf>=5.29.6` | Retain until AC-11 reviews the candidate graph. |
| `python-multipart>=0.0.31` | Retain until AC-11 reviews the candidate graph. |
| `pyarrow>=23.0.1` | Needed only while `evaluate` is installed; remove with it if no reverse dependency remains. |
| `markdown>=3.10.2` | Retain until AC-11 reviews the candidate graph. |
| `pygments>=2.20.0` | Retain until AC-11 reviews the candidate graph. |
| `aiohttp>=3.14.4` | Retain until AC-11 reviews the candidate graph. |
| `pyjwt>=2.15.0` | Retain until AC-11 reviews the candidate graph. |
| `pip>=26.1.2` | Retain until AC-11 reviews the candidate graph. |
| `msgpack>=1.2.3` | Retain until AC-11 reviews the candidate graph. |
| `pydantic-settings>=2.14.2` | Retain until AC-11 reviews the candidate graph. |
| `joserfc>=1.7.6` | Retain until AC-11 reviews the candidate graph. |
| `soupsieve>=2.8.4` | Retain until AC-11 reviews the candidate graph. |
| `setuptools>=84.1.0` | Retain until AC-11 reviews the candidate graph. |
| `pymdown-extensions>=11.0.0` | Retain until AC-11 reviews the candidate graph. |
| `anyio>=4.14.2` | Retain until AC-11 reviews the candidate graph. |
| `certifi>=2026.10.18` | Retain until AC-11 reviews the candidate graph. |
| `dnspython>=2.9.0` | Retain until AC-11 reviews the candidate graph. |
| `fsspec>=2026.6.0` | Retain until AC-11 reviews the candidate graph. |
| `multidict>=6.9.1` | Retain until AC-11 reviews the candidate graph. |
| `oauthlib>=4.0.1` | Retain until AC-11 reviews the candidate graph. |
| `opentelemetry-api>=1.45.1` | Retain until AC-11 reviews the candidate graph. |
| `datasets>=5.0.1` | Needed only while `evaluate` is installed; remove with it if no reverse dependency remains. |

`pyproject.toml` and `uv.lock` remain the executable source of truth. Candidate changes must update this inventory with measured evidence.
