# company-rag-agent

A small, **hand-rolled** Retrieval-Augmented Generation agent for company
knowledge — documents, emails, and chats — implemented from scratch without
any integrated RAG framework. The agent calls two tools, `search` and
`open_document`, on a hybrid BM25 + dense index over ~35,000 chunks built
from a 10,000-document corpus spanning Slack, Gmail, Confluence, Linear,
Jira, HubSpot, GitHub, Google Drive, and Fireflies.

Built for the **UZH FS2026 RAG** course. Every retrieval primitive — BM25,
cosine similarity, weighted fusion, sliding-window and metadata-aware
chunking — is implemented in this repository; only Ollama is used to serve
the open-source models locally.

---

## Table of contents

1. [What this is](#what-this-is)
2. [Quick start](#quick-start)
3. [System design](#system-design)
4. [Retrieval pipeline](#retrieval-pipeline)
5. [Chunking strategy](#chunking-strategy)
6. [Tools exposed to the agent](#tools-exposed-to-the-agent)
7. [The dataset](#the-dataset)
8. [Evaluation](#evaluation)
9. [Design decisions and trade-offs](#design-decisions-and-trade-offs)
10. [Course-pillar mapping](#course-pillar-mapping)
11. [Project layout](#project-layout)
12. [Known limitations and future work](#known-limitations-and-future-work)

---

## What this is

The agent answers natural-language questions about a fictional company's
internal communication, e.g.:

> "Who complained about the November invoice spike from HybridAI?"
> "What did the engineering team change in the linter config to get CI
> passing when the gocritic rule complained about context parameter types
> in tests?"

It does this by:

1. Embedding the question with `nomic-embed-text`.
2. Running BM25 (SQLite FTS5) and dense cosine search in parallel against a
   single SQLite index.
3. Fusing both rankings with a weighted-sum scheme.
4. Returning the top hits — each with a `doc_id`, a small preview, and a
   score — to the LLM (Qwen 3.5 9B via Ollama).
5. Letting the LLM call `open_document` to read the full text before it
   answers, and citing the `doc_id` it used.

Optionally, the `search` tool accepts structured pre-filters
(`source_types`, `date_from`, `date_to`, `participant`) so the agent can
narrow down by who, when, or where before either retrieval branch runs.
This is what makes emails and chats searchable by sender or date, not just
by topic.

---

## Quick start

### Prerequisites

- macOS or Linux
- [Ollama](https://ollama.ai) installed and running locally (default port
  11434)
- Python 3.11+ (for the indexer and eval harness)
- Node.js 22+ and `npm` (for the agent)

### One-time setup

```bash
# Pull the open-source models.
ollama pull nomic-embed-text          # 274 MB, 137M params, 768-dim, 8192-context
ollama pull qwen3:8b                  # 5.2 GB — the model used on Nuvolos
                                      # change via LLM_MODEL env var if you prefer another

# Python deps (the data/.venv is gitignored — make your own).
python -m venv data/.venv
source data/.venv/bin/activate
pip install pyarrow numpy httpx pandas sentence-transformers fastapi uvicorn

# Node deps.
npm install
```

### Build the index

```bash
source data/.venv/bin/activate
python indexing/build_index.py \
    --input data/raw/documents_subset.parquet \
    --out   data/index/rag.db
```

Expect ~22 minutes on Apple Silicon (~25 chunks/sec from the embedder).
The indexer is **resumable**: it commits embeddings every 200 chunks, so if
Ollama 500s mid-run, rerunning the same command picks up exactly where it
left off. The final `rag.db` is ~337 MB (33,827 vectors + FTS index + raw
chunk text).

### Try it from the CLI without the TUI

```bash
npx tsx src/rag/smoke.ts "who complained about the November invoice spike from HybridAI?"
```

You should see the relevant Gmail thread as the top hit with `score ≈ 2.6`,
followed by related invoice-spike threads from different companies.

### Run the agent

Two processes are needed: the reranker service and the agent itself.

```bash
# Terminal 1 — cross-encoder reranker (keep this running)
source data/.venv/bin/activate
python -m uvicorn indexing.reranker:app --port 8001
# First run downloads cross-encoder/ms-marco-MiniLM-L-6-v2 (~85 MB).
# Subsequent runs start in < 5 s.

# Terminal 2 — the agent
npm start
```

A TUI opens. Ask anything. The agent calls `search`, optionally
`open_document`, and cites the `doc_id`. Type `/quit` to exit.

If the reranker is not running the agent degrades gracefully — search
still works, results are ordered by hybrid fusion score instead.

### Run the retrieval eval

```bash
python indexing/eval_retrieval.py \
    --db        data/index/rag.db \
    --questions data/raw/questions_test.parquet \
    --top-k     10
```

See [Evaluation](#evaluation) for what to expect.

---

## System design

```
                ┌────────────────────────────────────────────────────────────┐
                │              TUI (pi-agent-core + pi-tui)                  │
                │                  src/main.ts                               │
                └─────────────────────────────┬──────────────────────────────┘
                                              │
                          ┌───────────────────┴───────────────────┐
                          │  Qwen 3.5 9B (Ollama HTTP, /v1)        │
                          │  system prompt: src/prompt.ts          │
                          └───┬──────────────────────────────┬─────┘
                              │ tool: search                 │ tool: open_document
              ┌───────────────▼───────────────┐   ┌──────────▼──────────┐
              │  src/tools/search.ts          │   │ src/tools/open_     │
              │  → src/rag/fusion.ts          │   │   document.ts       │
              └───┬──────────────────┬────────┘   └──────────┬──────────┘
                  │ keyword          │ vector                │
       ┌──────────▼─────────┐   ┌────▼────────────────┐      │
       │ FTS5 MATCH +       │   │ in-memory matmul    │      │
       │ bm25() (SQLite)    │   │ Float32Array (Node) │      │
       └──────────┬─────────┘   └────┬────────────────┘      │
                  │                  │ embed query via       │
                  │                  │ Ollama /api/embeddings │
                  │                  │ (nomic-embed-text)    │
                  └─────────┬────────┘                       │
                            │ weighted fusion (≤16 candidates)│
                            ▼                                │
              ┌─────────────────────────────┐               │
              │  cross-encoder reranker      │               │
              │  indexing/reranker.py        │               │
              │  ms-marco-MiniLM-L-6-v2      │               │
              │  HTTP POST :8001/rerank      │               │
              └─────────────┬───────────────┘               │
                            │ top-N reranked                │
                            ▼                                ▼
                     ┌─────────────────────────────────────────────────┐
                     │            data/index/rag.db (SQLite)            │
                     │  documents | chunks | chunks_fts (FTS5) | meta   │
                     │  vectors: float32 BLOBs (768-dim, L2 normalized)│
                     └─────────────────────────────────────────────────┘
                                          ▲
                                          │ python indexing/build_index.py
                                          │
                                ┌─────────┴─────────┐
                                │  data/raw/        │
                                │  *.parquet        │
                                └───────────────────┘
```

A deliberate split:

- **Python** owns ingestion. `pyarrow` is good at Parquet; SQLite + the
  Python `sqlite3` module is the simplest possible store; the indexer
  benefits from being a one-shot batch job that can crash and resume.
- **TypeScript** owns the agent. `better-sqlite3` is a tiny dependency,
  `fetch` is built in, and the pi-agent harness is already TS. The query
  hot path is in-process: every chunk vector is loaded once into a single
  `Float32Array`, so each search is a single matmul against ~35k vectors
  (~50 ms in practice).

Communication between the two halves is the SQLite file itself — no API,
no message broker, no daemon.

---

## Retrieval pipeline

For each user query `q`:

### 1. Embed the query

```
q_vec = nomic-embed-text(q)      # 768-dim float32, L2-normalized
```

Normalization matters because cosine similarity then reduces to a plain
dot product, which is a single contiguous SIMD-friendly multiply.

### 2. Keyword branch — sparse retrieval (course pillar W1/W2)

SQLite's FTS5 virtual table with the `porter unicode61` tokenizer indexes
every chunk's text (including the metadata header — see
[Chunking](#chunking-strategy)). FTS5 ships with the `bm25()` ranking
function out of the box, so this is *literally* hand-rolled BM25 from the
course pillar: no `rank_bm25`, no external library.

```sql
SELECT chunks.chunk_id
FROM chunks_fts JOIN chunks ON chunks_fts.rowid = chunks.chunk_id
WHERE chunks_fts MATCH ?
ORDER BY bm25(chunks_fts)
LIMIT 8;
```

Each hit gets a score:

```
kw_score = (1 / (1 + rank)) * 4    # rank = 1..8
```

### 3. Vector branch — dense retrieval (course pillar W3)

All `embedding` BLOBs are loaded once at startup into one
`Float32Array` of shape `[N, 768]`. A query is then one matmul:

```
sims = matrix @ q_vec              # cosine = dot product (both normalized)
top_k_idx = argsort(-sims)[:8]
vec_score = (1 - cosine_distance) * 4    # equivalent to (1 + sim) * 2
```

No FAISS, no `sqlite-vec`. At 35k vectors this fits in ~100 MB of RAM and
runs in milliseconds — adding an ANN index would just hide the math the
course wants us to demonstrate.

### 4. Weighted fusion

The two candidate lists are unioned (≤ 16 unique chunks). Each chunk gets a
final score:

```
final = 0.7 * vec_score + 0.3 * kw_score
```

The 0.7/0.3 split is deliberate: BM25 is excellent at exact-term matches
(e.g. error codes, model IDs, ticket numbers — which appear in many real
questions), but dense embeddings handle paraphrasing. The vector branch
generally wins on this dataset, so it carries more weight, while BM25 acts
as a precision booster for term-heavy queries.

### 5. Threshold + candidate pool

Anything below `final ≥ 0.35` is dropped. The surviving candidates
(at most 16: 2 branches × 8 each) are passed to the cross-encoder.

### 6. Cross-encoder reranking (course pillar W9)

The full candidate pool (≤ 16 chunks) is sent to a local Python
service (`indexing/reranker.py`) running `cross-encoder/ms-marco-MiniLM-L-6-v2`
(22 M params, CPU-pinned). Unlike the bi-encoder, which embeds query and
chunk independently, the cross-encoder reads `(query, chunk_text)` as a
single input and scores their interaction directly — catching relevance
signals the dot-product step misses (negation, context, paraphrasing).

The reranker runs in ~150–250 ms on CPU, negligible against the LLM
generation time. After reranking, results are **deduplicated by `doc_id`**
so the agent always sees up to N distinct documents rather than multiple
chunks from the same source. Top N (default 6) is returned in reranker
order with `doc_id`, `source_type`, `title`, `score`, `vec_score`,
`kw_score`, and a ~300-char `preview`. If the service is unreachable the
pipeline falls back to fusion order silently.

### 7. Optional structured pre-filter

When the user is explicit about who, when, or where, the agent can pass
filters on the `search` tool:

| filter | column queried | example use |
|---|---|---|
| `source_types: ["slack"]`  | `source_type IN (?)` | "what did the eng-platform channel say…" |
| `date_from: "2026-11-01"`  | `ts_to >= ?`         | "emails from November onward" |
| `date_to:   "2026-11-30"`  | `ts_from <= ?`       | "before December" |
| `participant: "alex@…"`    | `participants_json LIKE %?%` | "what did Alex tell us about…" |

Filters are applied *before* both retrieval branches run, so the candidate
pool is identical for BM25 and vector — fusion math is unchanged.

---

## Chunking strategy

Three chunkers, all targeting ~500 tokens per chunk with ~50-token
overlap. Token count is approximated as `len(text) // 4` to avoid pulling a
tokenizer into the pipeline.

**Every chunker prepends a metadata header to its chunk text**, so both
BM25 (which sees the tokens) and the embedder (which sees the semantics)
have access to source, title, participants, and dates.

### Document-like sources

`confluence`, `google_drive`, `jira`, `linear`, `hubspot`, `github`,
`fireflies` — straight sliding window over the raw content.

```
[source: confluence] [title: Inference cost optimizer rollout]

<~500 tokens of content>
```

### Gmail

`content` is a Python-repr'd list of message strings with double-escaped
newlines (e.g., `\\n` instead of `\n` in the bytes). The chunker:

1. Tries `json.loads` first.
2. Falls back to a deterministic forward scan that splits on `From:`
   occurrences — avoids catastrophic backtracking that a quoted-string
   regex would hit on long, escape-heavy bodies.
3. Unescapes both double- and single-escaped newlines/tabs/quotes.
4. Extracts `From / To / Date / Subject` per message via a regex.
5. Groups consecutive messages with the same normalized subject (stripping
   `Re:` / `Fwd:`) into one thread.
6. Slides a window of 4 messages with 1-message overlap inside each thread.

The resulting chunk header looks like:

```
[source: gmail] [thread: Unexpected spike on November invoice - INV-2026-11-331]
[participants: alex@hybridai.io, ben_carter@redwood.com, kimberly_park@redwood.com]
[dates: 2026-11-01T09:12:00-07:00 -> 2026-11-01T10:03:00-07:00]
```

These are also stored as **typed columns** (`ts_from`, `ts_to`,
`participants_json`) so the structured pre-filter on `search` is exact —
not a string match against the header.

### Slack

`title` is the channel name. Each row's `content` is a sequence of
`speaker: text` lines split by blank lines. The chunker:

1. Splits on blank lines into conversational blocks.
2. Greedy-packs blocks into chunks until ~500 tokens.
3. Extracts speakers via the `^([\w.-]+):` pattern per chunk.
4. Re-uses the previous chunk's last block as overlap.

Header:

```
[source: slack] [channel: eng-platform]
[participants: alyssa, console-team-bot, liz, mohit, oncall-joe]
```

Slack rows in the source corpus don't carry per-message timestamps, so
`ts_from`/`ts_to` are left null.

---

## Tools exposed to the agent

### `search`

Parameters:

| name | type | required | description |
|---|---|---|---|
| `query` | string | yes | Natural-language query. |
| `source_types` | string[] | no | Restrict to one or more of: `slack`, `gmail`, `linear`, `jira`, `confluence`, `google_drive`, `hubspot`, `github`, `fireflies`. |
| `date_from` | string | no | ISO-8601 lower bound (matters for gmail). |
| `date_to` | string | no | ISO-8601 upper bound (matters for gmail). |
| `participant` | string | no | Substring match against participants — email or Slack handle. |
| `top_n` | number | no | How many fused hits to return (default 6, max 20). |

Returns: text summary + a `details.results` array of `{chunk_id, doc_id,
source_type, title, score, vec_score, kw_score, preview, ts_from, ts_to}`.

### `open_document`

Parameters:

| name | type | required | description |
|---|---|---|---|
| `doc_id` | string | yes | The `doc_id` from a `search` result, e.g. `dsid_c37d9b…` |

Returns the full document content prefixed with its `doc_id`, `source`,
and `title` so the LLM can cite cleanly.

Both tools are auto-allowed (`AUTO_ALLOWED` in `src/main.ts`) so the agent
doesn't prompt the user before reading from the knowledge base. The
filesystem tools (`read`, `write`, `edit`, `bash`) remain gated.

---

## The dataset

This project uses a 10,000-document subset of [**EnterpriseRAG-Bench**](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
by [Onyx](https://onyx.app/enterpriserag-bench) — an open, MIT-licensed
benchmark of company-internal knowledge spanning 500,000+ synthetic
documents and 500 gold-labeled questions across nine source types
(Slack, Gmail, Linear, Jira, Confluence, Google Drive, HubSpot, GitHub,
Fireflies). Paper: [arXiv:2605.05253](https://arxiv.org/abs/2605.05253).

Stored under `data/raw/`:

| file | rows | size | purpose |
|---|---|---|---|
| `documents_subset.parquet` | 10,000 | 28 MB | The corpus the index is built from. **Committed.** |
| `questions_test.parquet` | 500 | 556 KB | Gold questions with `expected_doc_ids`, `gold_answer`, `answer_facts`. **Committed.** |
| `subset_manifest.json` | — | 68 KB | Provenance for the subset (seed, gold IDs). **Committed.** |
| `documents_test.parquet` | — | 1.4 GB | Full upstream dump. **Not committed** (gitignored). |

Schema of `documents`:

| column | type | notes |
|---|---|---|
| `doc_id` | string | Stable ID, e.g. `dsid_c37d…` |
| `source_type` | string | One of the 9 sources above |
| `title` | string | Slack channel, email subject, doc title, ticket name… |
| `content` | string | Free-form; gmail is Python-repr'd lists; slack is `name: text` lines |

Distribution by source (chunks after indexing the 10k subset, gmail
re-chunked with the deterministic parser):

| source | docs | chunks |
|---|---|---|
| slack | 5,260 | 14,150 |
| gmail | 2,260 | 11,551 |
| linear | 700 | 2,302 |
| google_drive | 517 | 2,245 |
| confluence | 304 | 1,574 |
| fireflies | 215 | 1,461 |
| jira | 228 | 758 |
| hubspot | 306 | 653 |
| github | 210 | 650 |
| **total** | **~10k** | **35,344** |

---

## Evaluation

`indexing/eval_retrieval.py` re-implements the exact fusion + reranking
pipeline used by the TS tool and computes Recall@k, MRR@k, and nDCG@k
against `expected_doc_ids`.

Run without `--rerank` for the fusion-only baseline, or with `--rerank`
(requires the reranker service running) to measure the full live pipeline:

```bash
# Fusion only
python indexing/eval_retrieval.py --db data/index/rag.db \
    --questions data/raw/questions_test.parquet --top-k 10

# Full pipeline (fusion + cross-encoder)
python indexing/eval_retrieval.py --db data/index/rag.db \
    --questions data/raw/questions_test.parquet --top-k 10 --rerank
```

On the first 100 questions of the gold set — **fusion only, pre-reranker
baseline:**

| k | Recall | MRR | nDCG |
|---|--------|------|------|
| 1 | 0.670 | 0.670 | 0.670 |
| 3 | 0.710 | 0.690 | 0.695 |
| 5 | 0.740 | 0.696 | 0.707 |
| 10 | **0.930** | 0.720 | 0.767 |

Recall@10 of 0.93 means the right document is in the agent's candidate
pool on 93 % of queries. The cross-encoder reranker is expected to push
Recall@1 substantially above 0.67; run with `--rerank` to get current
numbers.

Two questions we sanity-checked end-to-end through the live agent:

- **qst_0010** (GitHub, "How does the new alerting approach group
  model-serving requests…"): agent returned the gold doc on the first
  search, hit all 5 gold facts, cited the `doc_id` verbatim.
- **qst_0153** (Slack, "what did they change in the linter config…"):
  agent returned the gold doc, hit both gold facts.

### End-to-end accuracy eval (`eval_e2e.py`)

`indexing/eval_e2e.py` runs the **full pipeline** — retrieval → LLM answer
generation → fact scoring — against a sample of gold questions. It uses the
**same LLM model as the live agent** via the `LLM_MODEL` environment variable
(on Nuvolos: `qwen3:8b`). No separate model or service is involved — the eval
and the agent always measure the same thing.

The script automatically filters to questions whose answer documents are
present in the 10k-doc subset (roughly half the 500-question gold set),
then samples N of those for evaluation.

```bash
# Requires: Ollama running with the LLM model, and the index built.
# Optional: start the reranker service before passing --rerank.

# On Nuvolos (qwen3:8b, thinking disabled):
LLM_MODEL=qwen3:8b LLM_DISABLE_THINKING=1 \
python indexing/eval_e2e.py \
    --db        data/index/rag.db \
    --questions data/raw/questions_test.parquet \
    --n         25

# With cross-encoder reranking:
LLM_MODEL=qwen3:8b LLM_DISABLE_THINKING=1 \
python indexing/eval_e2e.py \
    --db        data/index/rag.db \
    --questions data/raw/questions_test.parquet \
    --n         25 \
    --rerank
```

Three metrics are reported:

| metric | what it measures |
|---|---|
| **Retrieval Hit@1** | The correct document was ranked #1 — the LLM had the best possible context |
| **Retrieval Hit@6** | The correct document was somewhere in the top 6 — retrieval succeeded but may have ranked it low |
| **Avg Fact Recall** | Fraction of gold `answer_facts` that appear in the LLM's response — the true end-to-end accuracy |

A `~` in the per-question output means the right document was retrieved but
not ranked first; the LLM answered from the wrong top document. A `✗` means
retrieval missed entirely. This lets you distinguish retrieval failures from
LLM failures.

---

## Design decisions and trade-offs

### Why hand-rolled retrieval

The course brief explicitly forbids integrated RAG frameworks (RAGFlow,
LightRAG, etc.) and expects every retrieval primitive to be demonstrable.
We use only the underlying engines: SQLite's FTS5 (BM25 is in the engine),
Ollama for model serving, and `numpy`/`Float32Array` for the dense math.
No FAISS, no `sqlite-vec`, no LangChain.

### Why `nomic-embed-text`

Of the open small embedders available via Ollama:

| model | params | dim | context | reason ruled in/out |
|---|---|---|---|---|
| `all-minilm` | 22M | 384 | 512 | Too weak on MTEB English. |
| `nomic-embed-text` | 137M | 768 | **8192** | **Chosen.** Largest context window of the small models — covers the longest emails without truncation, MIT-license-compatible, fast (~25/sec on M-series). |
| `mxbai-embed-large` | 335M | 1024 | 512 | Higher quality but 512 context would truncate long emails. |
| `bge-m3` | 560M | 1024 | 8192 | Multilingual; overkill for an English-only business corpus and 4× slower. |

### Why pre-filter rather than a third score channel

The conversation that produced this design considered making participant /
date / source matches a *third weighted score channel* alongside BM25 and
vector. We rejected that because:

- It adds a hyperparameter (a third weight) that needs tuning.
- It degrades gracefully *wrong* when the user doesn't mention a person
  or date — you'd still be mixing in a constant zero or near-zero signal.
- A hard `WHERE` clause is exact, cheap, and cooperative with the existing
  fusion math: both branches operate on the same narrowed set.

### Why a per-chunk metadata header instead of just typed columns

We also considered keeping participants/dates only as structured columns.
Putting them in the chunk *text* as well means:

- BM25 finds "alex@hybridai.io" in keyword search even when the user
  doesn't engage the filter.
- The embedder sees the participant list, which improves semantic recall
  on questions like "what did Alex complain about?"

The cost is a few extra tokens of chunk overhead — negligible at 6000-char
chunks.

### Why commit `documents_subset.parquet`

It's 28 MB — under GitHub's 50 MB soft limit and 100 MB hard limit, and
without it the indexer can't reproduce the index. The 1.4 GB full dump is
gitignored.

### Why not commit `rag.db`

It's 337 MB (~10× the soft limit) and contains nothing that isn't
deterministically reproducible from the subset parquet. The README's
quickstart rebuilds it in ~22 minutes.

---

## Course-pillar mapping

The course evaluates against named pillars. Where each one lives in this
repo:

| Pillar | Slide source | Implementation | File |
|---|---|---|---|
| Sparse retrieval | W1, W2 | SQLite FTS5 + `bm25()` | `indexing/schema.sql`, `src/rag/fusion.ts` |
| Dense retrieval | W3 | `nomic-embed-text` via Ollama + cosine | `indexing/embed.py`, `src/rag/{db,embed,fusion}.ts` |
| Hybrid retrieval | W3 hint | Weighted fusion `0.7·vec + 0.3·kw`, threshold 0.35 | `src/rag/fusion.ts` |
| Chunking | W9 | Sliding window 500/50 + semantic per-source headers | `indexing/chunkers.py` |
| LM decoding | W4 | Qwen 3.5 9B via Ollama OpenAI-compatible endpoint | `src/model.ts` |
| LM prompting | W5 | System prompt with citation rules + ≤3-search cap | `src/prompt.ts` |
| Open foundation models | W6 | Qwen 3.5 9B + nomic-embed-text (both open) | `src/model.ts`, `indexing/embed.py` |
| Production engineering | W9 | Resumable indexer, retries, payload-shrinking on 500s, ANALYZE/optimize | `indexing/build_index.py`, `indexing/embed.py` |
| IR evaluation | W1, W2 | Recall@k, MRR@k, nDCG@k for k ∈ {1, 3, 5, 10} | `indexing/eval_retrieval.py` |
| Frontend (W10) | W10 | pi-tui TUI with visible tool-call traces and citations | `src/main.ts` |
| Re-ranking | W9 | Cross-encoder rerank over ≤16 fused candidates (ms-marco-MiniLM-L-6-v2, CPU) | `indexing/reranker.py`, `src/rag/rerank.ts`, `src/rag/fusion.ts` |

---

## Project layout

```
.
├── README.md                  this file
├── package.json               TS deps: pi-agent-core, pi-tui, better-sqlite3
├── package-lock.json
├── tsconfig.json
├── Modelfile                  optional Ollama Modelfile for a custom qwen build
│
├── data/
│   ├── raw/
│   │   ├── documents_subset.parquet   (28 MB, committed)
│   │   ├── questions_test.parquet     (556 KB, committed)
│   │   ├── subset_manifest.json       (68 KB, committed)
│   │   └── documents_test.parquet     (1.4 GB, gitignored)
│   ├── .venv/                          (gitignored)
│   └── index/
│       └── rag.db                      (337 MB, gitignored)
│
├── indexing/                  Python — one-shot, resumable
│   ├── schema.sql             documents | chunks | chunks_fts (FTS5) | meta
│   ├── chunkers.py            per-source: gmail / slack / document-like
│   ├── embed.py               Ollama embedding client, retry + payload shrink
│   ├── build_index.py         end-to-end (commits every 200 chunks)
│   ├── rechunk_source.py      re-do one source_type after a chunker change
│   └── eval_retrieval.py      Recall@k / MRR@k / nDCG@k against the gold set
│
└── src/                       TypeScript — the agent
    ├── main.ts                pi-agent-core entrypoint + TUI
    ├── prompt.ts              system prompt (workflow, ≤3 searches/q,
    │                          citation format)
    ├── model.ts               Ollama LLM config (Qwen 3.5 9B, openai-completions)
    ├── permissions.ts         interactive tool-call gating (read/write/edit/bash)
    ├── rag/
    │   ├── db.ts              better-sqlite3 read-only + Float32Array vectors
    │   ├── embed.ts           Ollama /api/embeddings client (L2-normalized)
    │   ├── fusion.ts          hybrid search: BM25 + vector + fusion + reranking + dedup
    │   ├── rerank.ts          HTTP client for the Python cross-encoder service
    │   ├── smoke.ts           one-shot CLI search for development
    │   └── filter_smoke.ts    sanity tests for the structured filters
    └── tools/
        ├── index.ts           tool exports
        ├── search.ts          → src/rag/fusion.ts
        ├── open_document.ts   → src/rag/db.ts (fetchDocument)
        ├── read.ts            local FS — read
        ├── write.ts           local FS — write
        ├── edit.ts            local FS — edit
        └── bash.ts            local FS — bash
```

---

## Known limitations and future work

- **Slack chunks have no timestamps.** The source rows don't carry
  per-message dates, so `date_from`/`date_to` filters silently exclude
  Slack. If we get a date-stamped Slack dump later, the chunker just needs
  to populate `ts_from`/`ts_to`.
- **Embedding model is loaded at agent startup.** ~100 MB in memory.
  Fine for local dev; if we host this we'd lazy-load or use a memory-mapped
  format.
- **LLM over-searches on terse questions.** The system prompt caps it at
  three `search` calls per question, but a stricter agent harness with a
  hard `maxToolIterations` would be cleaner. Tracked as a Qwen-side
  prompting issue rather than a retrieval one.
- **No query rewriting.** Some "high-level" questions in the gold set
  would benefit from a HyDE-style step (embed a hypothetical answer
  instead of the question). Left out to keep the baseline honest.
- **Eval harness re-embeds queries.** Adding a 500-entry query embedding
  cache would cut eval time roughly in half. Not on the critical path.

---

## Credits and license

- Models: `nomic-embed-text` (Apache-2.0, Nomic AI), Qwen 3.5 9B
  (Apache-2.0, Alibaba), both served via Ollama.
- Agent harness: [`@mariozechner/pi-agent-core`](https://www.npmjs.com/package/@mariozechner/pi-agent-core)
  and [`@mariozechner/pi-tui`](https://www.npmjs.com/package/@mariozechner/pi-tui)
  — minimal TypeScript primitives, not an integrated RAG framework.
- Storage: SQLite (FTS5 + bm25 are built in), `better-sqlite3` for the
  Node side.
- Dataset: [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
  by Onyx (MIT). 10k-doc subset committed; full 500k-doc dump is upstream
  on HuggingFace.

Coursework artifact for the UZH FS2026 RAG course — not for production use
without auditing the synthetic-data provenance of the gold set.
