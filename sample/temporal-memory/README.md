# Temporal memory sample: Family Assistance law over time

This sample shows a HoloDeck `temporal_memory` tool on real legislation. The agent answers questions about two Australian Acts as they stood on a given date:

- *A New Tax System (Family Assistance) Act 1999* (series C2004A00490).
- *A New Tax System (Family Assistance) (Administration) Act 1999* (series C2004A00491), volume 1 only: Parts 1 to 5, sections 1 to 152D.

For each Act, the sample uses two compilations: the one in force on 1 July 2020 and the one in force on 1 July 2026. Each compilation becomes one markdown file with its start date as `effective_date`. The memory reads both, so the agent can say what a provision was on a date, how it changed, and which compilation a fact comes from.

The Act text is not in the repository. `fetch_corpus.py` downloads it from the Federal Register of Legislation and converts it.

## Licence and attribution

Based on content from the Federal Register of Legislation at 11 November 2026. For the latest information on Australian Government legislation please go to https://www.legislation.gov.au.

The Register content is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([terms of use](https://www.legislation.gov.au/terms-of-use)). The Commonwealth Coat of Arms is not covered by the licence; the converted files do not contain it.

The material is changed: `fetch_corpus.py` converts it from Word to markdown, removes the cover, the "About this compilation" pages, the table of contents and the endnotes, and reformats the headings. Each converted file repeats this statement in its front matter (`attribution`), with a link to its source.

## What the sample contains

| File | Purpose |
| --- | --- |
| `manifest.json` | The four pinned compilations: series id, register id, compilation number, start date, volume, download URL, size and SHA-256. |
| `fetch_corpus.py` | Downloads each compilation once, checks its hash, converts it, and prints chunk counts. |
| `agent.yaml` | Two `hierarchical_document` tools on Qdrant, the `benefit_memory` tool on Neo4j, instructions and five test cases, each graded against its ground truth by a `geval` judge. |
| `compose.yaml` | Qdrant 1.20.0 and Neo4j 5.26 with named volumes and health checks. |
| `.env.example` | Every environment variable the sample reads. |

`fetch_corpus.py` writes `.cache/` (the Word files) and `docs/` (the markdown). Both are git-ignored.

## Prerequisites

- HoloDeck with the Graphiti and Qdrant extras: `uv sync --extra graphiti --extra qdrant`.
- Docker with Compose.
- An OpenAI API key (`OPENAI_API_KEY`). The sample uses it for the agent model (`gpt-6-luna`), for the memory extraction calls and for embeddings (`text-embedding-3-small`).

Copy `.env.example` to `.env` in this directory and fill it in. Run every command below from this directory.

## Commands, in order

```bash
cd sample/temporal-memory
cp .env.example .env                              # then edit .env

docker compose up -d --wait                       # Qdrant and Neo4j
uv run python fetch_corpus.py                     # download, convert, report
uv run python fetch_corpus.py --check             # verify the cache only

uv run holodeck memory ingest agent.yaml --dry-run   # count; no model call
uv run holodeck memory ingest agent.yaml             # see the warning below
uv run holodeck memory status agent.yaml

uv run holodeck test run agent.yaml
uv run holodeck serve agent.yaml                  # AG-UI on http://127.0.0.1:8000/awp
```

`fetch_corpus.py` downloads each file once, one request at a time with an 11-second pause, as the Register's `robots.txt` asks. The first download records the SHA-256 in `manifest.json`. Later runs use the cached file and stop with an error that names the compilation if a hash does not match.

To use other compilations, edit `manifest.json`: set `url`, `start_date`, `compilation_number` and `register_id`, and set `retrieved`, `size_bytes` and `sha256` to `null`. The Register API lists the versions of a title at `https://api.prod.legislation.gov.au/v1/Versions` (filter on `titleId`).

### CopilotKit front end

To use the sample in a browser, run `holodeck serve agent.yaml`, then start `sample/copilotkit` with these values in its `.env.local`:

```bash
AGENT_ID=family-assistance-memory
AGENT_TITLE=Family Assistance memory
AGENT_DESCRIPTION=Family Assistance law as it stood on a date
HOLODECK_BACKEND_URL=http://127.0.0.1:8000/awp
```

`memory_timeline` results show as a fact timeline. The page http://localhost:3000/memory-demo shows the memory views from fixtures and needs no backend.

## What to expect

`fetch_corpus.py` printed these numbers on 11 November 2026. Chunks are counted with the sample's own tool settings, the same way `holodeck memory ingest --dry-run` counts them: no embedding and no store.

| Compilation | Start date | Sections | Words | Chunks |
| --- | --- | --- | --- | --- |
| C2004A00490 compilation 101 | 2020-07-01 | 303 | 79,435 | 325 |
| C2004A00490 compilation 124 | 2026-07-01 | 313 | 83,621 | 336 |
| C2004A00491 compilation 104, volume 1 | 2020-07-01 | 320 | 79,934 | 336 |
| C2004A00491 compilation 131, volume 1 | 2026-07-01 | 317 | 81,895 | 336 |

"Sections" counts sections of the Act and clauses of its Schedules.

A memory episode is identified by its section and its text. A section that is the same in both compilations is one episode, so the second compilation costs model calls only for the sections that changed:

- Family Assistance Act: 243 of the 336 chunks of 2026 are identical to 2020.
- Administration Act: 238 of the 336 chunks of 2026 are identical to 2020.
- A first ingest of all four files writes 851 episodes.

**The full ingest is expensive.** Each episode takes several model calls. A measured ingest of the Family Assistance Act (both compilations, 2020-07-01 and 2026-07-01) with `gpt-6-luna`, run sequentially:

| Measure | Value |
| --- | --- |
| Episodes written | 418 |
| Episodes skipped (identical across compilations) | 243 |
| Model calls | 3,111 |
| Input tokens | about 6.35 million |
| Output tokens | about 0.30 million |
| Time | 74 minutes |
| Result | 1,463 facts, 1,015 entities |

Expect a similar order of cost for the Administration Act. Run the dry run first. Use `--tool family_assistance_admin_act` to ingest one Act at a time. A repeated run skips the episodes that are already in the graph.

Vector ingestion alone is quick: it needs only embedding calls and no extraction.

### Test cases

Each ground truth comes from the converted text. Each case also has a `geval` evaluation (`ground truth agreement`, judged by `gpt-6-luna`, threshold 0.7), so a case fails when the answer contradicts the ground truth or says nothing was found, not only when a tool is missing. Section numbers are those of the Act named.

| Test case | Source | Expected tools |
| --- | --- | --- |
| Review deadline in 2020 | Administration Act s 109D(1)(a): 13 weeks in 2020, 90 days in 2026 | `memory_search` |
| Child care subsidy percentage in 2026 | Family Assistance Act Sch 2 cl 3: 90% up to $80,000 in 2026 | `memory_search` |
| Child care subsidy percentage over time | Same clause: 85% up to $65,710 in 2020 | `memory_timeline` |
| Shared care percentage stable | Family Assistance Act s 59(2) table, the same in both | `memory_search` |
| Tribunal review source | Administration Act s 111(1) and s 3: AAT in 2020, ART in 2026 | `memory_search`, `memory_provenance` |

## Settings and why

- **Headings.** The converter reads the Word paragraph styles of the Office of Parliamentary Counsel: `ActHead 2` Part (`#`), `ActHead 3` Division (`##`), `ActHead 4` Subdivision (`###`), `ActHead 5` section (`####`, for example `#### 3 Definitions`). A Schedule (`ActHead 1`) is `#`, and the headings inside it move one level down, so a Schedule clause is `#####`. Each section is one chunk, and its heading text stays the same across compilations.
- **`document_domain: none`.** The headings are already markdown. The `au_legislative` patterns would also read body lines that start with a number, such as table rows, as headings.
- **`max_chunk_tokens: 1000`.** The median section is about 250 tokens and 95% are under 920. About 14 long sections per file (the definitions in section 3, some rate provisions) are split at sentence boundaries.
- **`contextual_embeddings: false`.** It makes one model call per batch of chunks during vector ingestion. Each chunk already has its Part, Division and section heading.
- **Embeddings.** One model, `text-embedding-3-small` from OpenAI, for both source tools and the memory graph. A change of embedding model needs a new vector collection and a new graph.

### Alternatives

The agent model and the embedding model are set separately.

**Anthropic agent model.** Replace the `model` block in `agent.yaml`:

```yaml
model:
  provider: anthropic
  name: <an Anthropic model name>
  api_key: ${ANTHROPIC_API_KEY}
  max_tokens: 4000
```

and add `ANTHROPIC_API_KEY` to `.env`. Anthropic has no embedding model, so keep `embedding_provider` on an embedding provider such as OpenAI (as now) or Ollama. The `geval` judges in `agent.yaml` set their own OpenAI model (anchored as `judge` on the first test case); change it too, or the judges keep using OpenAI.

**Local embeddings.** Change `embedding_provider` to, for example, `provider: ollama` and `name: nomic-embed-text` (run `ollama pull nomic-embed-text` first). The vectors have a different size, so a change of embedding model needs a fresh vector collection and a fresh graph: run `docker compose down -v`, then ingest again.

**Other agent models.** Azure OpenAI and Ollama agent models are also supported.

## Known limits

- Closing is imperfect after the two-compilation ingest. Some facts are closed by the 2026 compilation although their text did not change (over-closing), and some facts have no start date. This is under investigation. Check `memory_timeline` output against the text.
- Closing a superseded provision is a model decision inside Graphiti. A changed rate can stay open next to the new one, and a fact that should stay open can be closed. Check `memory_timeline` output against the text.
- `memory_timeline` needs the entity's name as the Act states it, for example "lower income (base rate) threshold", not a paraphrase.
- Only Neo4j has been run. The FalkorDB, Neptune and Kuzu drivers are configured the same way but are untested.
- The Administration Act is volume 1 only (sections 1 to 152D). Volume 2 (sections 153 to 235 and the endnotes) is not fetched.
- The 2020 compilations print formulas as pictures with no text description. The converter writes `[Formula: picture with no text description in the source]` in their place. The 2026 compilations describe each formula, and the converter keeps the description.
- Tables become markdown tables. A table's first row is its caption, so the caption is the table header.
- The amounts in the Act are base amounts; many are indexed under Schedule 4. The agent reports the text of the Act, not the indexed rate paid on a date.
