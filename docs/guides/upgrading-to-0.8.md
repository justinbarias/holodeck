# Upgrading from 0.7.x to 0.8

HoloDeck 0.8 removes Semantic Kernel, splits the install into feature extras, and moves vector stores onto HoloDeck's own adapters. This guide lists what to check, in the order to check it. Each step says what changed, how to find out if it affects you, and what to do.

The full list of changes is in the [changelog](../CHANGELOG.md).

## Quick start

For most projects, these three commands cover the upgrade. Run them in the project folder, then read the steps for anything they report.

```bash
pip install --upgrade "holodeck-ai[all]"      # everything installed as before
holodeck test run agent.yaml                  # a retired metric or provider fails here, before any model call
holodeck vectorstore inspect agent.yaml       # read-only: do your collections need a rebuild?
```

Expected: the test run either starts, or stops at once with a message that names the metric, provider or extra to change.

## Step 1: Check your install extras

`pip install holodeck-ai` now installs less. These features moved into extras:

| You use | Install |
| --- | --- |
| DeepEval metrics (`type: geval`, `rag`, and other LLM-judged metrics) or NLP metrics (BLEU, ROUGE, METEOR) | `pip install "holodeck-ai[eval,nlp]"` |
| `holodeck test optimize` | `pip install "holodeck-ai[optimizer]"` (add `eval,nlp` for its metrics) |
| An OpenSearch keyword index | `pip install "holodeck-ai[opensearch]"` |
| A vector store | `pip install "holodeck-ai[qdrant]"`, `chromadb`, `postgres`, `pinecone`, `weaviate`, `azure-ai-search` or `azure-cosmos`; `vectorstores` installs all of them |
| Everything as before | `pip install "holodeck-ai[all]"` |

`all` installs every runtime extra. MkDocs Material and Twine are now in the `dev` extra only.

A command that needs a missing extra stops early and prints the install command. For example, `holodeck test run` stops before any agent or model call when a metric needs `eval` and it is not installed. Commands that do not need the extra still start.

## Step 2: Update retired metric names

The Azure AI evaluation metrics are removed, together with the `azure-ai-evaluation` package. An `agent.yaml` that names one fails when the configuration loads, before any agent or model call. The error names the replacement. HoloDeck does not translate the old configuration, and the new scores are not equivalent to the old ones. Run the suite again and set new thresholds.

| Removed metric | Use instead (DeepEval) |
| --- | --- |
| `groundedness` | `type: rag`, `metric_type: faithfulness` |
| `relevance` | `type: rag`, `metric_type: answer_relevancy` |
| `coherence` | `type: geval` with coherence criteria |
| `fluency` | `type: geval` with fluency criteria |
| `similarity` | `type: geval` with `evaluation_params: [actual_output, expected_output]` |
| `safety` | `type: geval` with safety criteria |

Results stored by earlier runs that used these metrics still open in the dashboard. See [Removed Azure AI metrics](evaluations.md#removed-azure-ai-metrics) for a before and after example.

METEOR also changed. It no longer downloads NLTK data the first time it runs. If the data is missing, METEOR stops with an error that names the command. Run it once on each machine that scores METEOR:

```bash
python -m nltk.downloader wordnet punkt_tab omw-1.4
```

## Step 3: Update retired vector store providers

Three providers are retired. They had no install extra, sample or live test, and could not run from a documented install. An `agent.yaml` or `config.yaml` that names one fails at config load with a migration message, before any model call.

| Retired provider | Use instead |
| --- | --- |
| `faiss` | `chromadb` (embedded, local files) or `in-memory` |
| `sql-server` | `postgres` (pgvector) |
| `azure-cosmos-mongo` | `azure-cosmos-nosql` |

HoloDeck does not copy data between providers. Point the tool at the new provider and start the agent. It ingests the source again. See [Vector Stores](vector-stores.md) for each provider's settings.

Two more changes affect vector store tools:

- **A missing client library is an error.** Before, a `vectorstore` tool whose client was not installed logged a warning and stored data in memory. Now it fails and prints the install command (Step 1). A failure while the store adapter is created still falls back to memory. A client that connects on first use (Qdrant, for example) raises its connection error; it does not fall back.
- **Chroma scores changed.** Chroma returned a distance that HoloDeck used as the score, so results ranked worst-first. The score is now `1 - distance`, and the best match comes first. If you tuned `min_similarity_score` against the old behavior, check it.

## Step 4: Check your vector collections

HoloDeck now records the pipeline that wrote a collection: the embedding model and dimensions, the parser and splitter, the record layout and the retrieval mode. The record lives in a companion collection named `holodeck_manifest` in the same store. Your collection layout is unchanged.

Plain text chunking also changed. The splitter now counts real tokens (tiktoken `cl100k_base`), honors `chunk_size` as a hard limit and applies `chunk_overlap`. Chunk boundaries differ from 0.7, so a collection written by 0.7 holds different chunks than 0.8 would write.

Run the read-only check:

```bash
holodeck vectorstore inspect agent.yaml
```

It reports each collection, its record count, whether it matches the current pipeline, and what a rebuild would embed. It does not change anything and calls no model.

- A collection created before 0.8 is marked **legacy**. It stays readable and writable, and keeps its old chunks until you re-ingest.
- A collection written by a different pipeline stays searchable, but a write to it fails and lists the inputs that differ. This protects you from mixing old and new chunks in one collection.

To rebuild a collection with the current pipeline:

```bash
holodeck vectorstore migrate agent.yaml --tool NAME   # builds a new collection, verifies it, then switches; resumable
holodeck vectorstore rollback agent.yaml --tool NAME  # switch back to the old collection
holodeck vectorstore prune agent.yaml --tool NAME --yes  # delete the old collection when you are sure
```

`migrate` calls the embedding model and keeps the old collection until you run `prune`. Details are in [Pipeline fingerprint and migration](vector-stores.md#pipeline-fingerprint-and-migration).

## Step 5: Rebuild deployment images

Images are now pinned. `holodeck deploy build` and the base image install one exact HoloDeck version, with dependency versions taken from that release's lock file. They no longer use `--prerelease`. Each build writes a manifest to `.holodeck/builds/<tag>.json` with the version, extras, base image digest and package list.

- Rebuild your images with the 0.8 CLI. The build uses `holodeck-base:<version>` of the same release.
- A development version that is not on PyPI needs `--wheel dist/<wheel>`.
- Serving images carry only the extras the agent needs. They never carry `eval`, `nlp` or `optimizer`.

See [Reproducible images](deployment.md#reproducible-images).

## For Python code that imports HoloDeck

If your code imports HoloDeck modules, note these removals:

- `semantic-kernel` is no longer installed by any profile. Remove any import of it that relied on HoloDeck bringing it in.
- The `holodeck.lib.vector_store` module is gone. `QueryResult` and `StructuredQueryResult` now live in `holodeck.lib.retrieval`.
- `holodeck.lib.observability.enable_semantic_kernel_telemetry` and the `SK_OTEL_*` constants are gone.
- The default of `observability.logs.filter_namespaces` is now an empty list. It was `["semantic_kernel"]`.
- `mcp` 2 is allowed (`>=1.30.0,<3`).

## Other breaking changes in 0.8

These do not come from the dependency work. See the [changelog](../CHANGELOG.md) for each one:

- Python 3.11 or later is required.
- The `tool_filtering` agent field is removed. Delete the block.
- The Claude backend isolates settings by default. Add `setting_sources` to the `claude:` block if your agent relied on host Claude configuration.
- `NotebookEdit` and `WebSearch` need a declaration on the Claude backend.
