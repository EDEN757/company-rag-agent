# company-rag-agent

A **hand-rolled** Retrieval-Augmented Generation agent for company knowledge — documents,
emails, and chats — built from scratch without any integrated RAG framework. The system
runs a multi-stage hybrid retrieval pipeline (BM25 + dense vector + HyDE query expansion
+ cross-encoder reranking) over ~35,000 chunks from a 10,000-document corpus spanning
nine source types, and exposes the results through a multi-turn LLM agent with
tool-calling and a streaming Gradio UI.

Built for the **UZH FS2026 RAG** course. Every retrieval primitive is implemented in
this repository; only Ollama is used to serve open-source models locally.

---

## Table of contents

1. [What this is](#what-this-is)
2. [Running on Nuvolos](#running-on-nuvolos)
3. [System architecture](#system-architecture)
4. [Retrieval pipeline](#retrieval-pipeline)
5. [Knowledge base write path](#knowledge-base-write-path)
6. [LLM agent loop](#llm-agent-loop)
7. [Chunking strategy](#chunking-strategy)
8. [Tools exposed to the agent](#tools-exposed-to-the-agent)
9. [The dataset](#the-dataset)
10. [Evaluation](#evaluation)
11. [Design decisions and trade-offs](#design-decisions-and-trade-offs)
12. [Course-pillar mapping](#course-pillar-mapping)
13. [Project layout](#project-layout)
14. [Known limitations and future work](#known-limitations-and-future-work)

---

## What this is

The agent answers natural-language questions about a fictional company's internal
knowledge base, for example:

> *"What is Redwood Inference's mission statement?"*
> *"What caused the EU-West activation funnel and onboarding email issues in late January,
> and which code/config changes fixed it?"*
> *"In the SOC2 readiness notes, what log retention duration is mentioned as a risk?"*

It does this by running a five-stage retrieval pipeline — keyword search, vector search,
RRF fusion, deduplication, and cross-encoder reranking — and then handing the top results
to a Qwen 3 8B agent that can call `search`, `open_document`, `add_document`,
`edit_document`, `read`, `write`, `edit`, and `bash` tools over up to six turns before
producing a cited answer.

New documents can be added or edited through the same interface; the knowledge base
updates atomically and BM25 corpus statistics are refreshed incrementally in memory
without a restart.

---

## Running on Nuvolos

The project runs as three separate Nuvolos apps that share an instance-wide network.

### App roles

| Nuvolos app | What runs there | Port |
|---|---|---|
| **Database** | PostgreSQL + pgvector extension | 5432 |
| **Backend** VS Code | Ollama + `uvicorn main:app` + indexer | 8500 |
| **Frontend** VS Code | `python app.py` (Gradio) | 7860 |

> The **Editor** app is network-isolated and cannot reach the Database.
> Always run the indexer and uvicorn from the **Backend** app.

### Network hostnames

| Service | Hardcoded default |
|---|---|
| Database (pgvector) | `nv-service-b01d63337fab32ac94f65eb2dc8a62ba` |
| Backend API | `nv-service-e4bb2876d3e69f18fd98d56e852aa814` |

These are set as code defaults — no environment variable configuration is needed unless
they change.

---

### First-time setup

#### Step 1 — Start the Database app

Start it in Nuvolos. No further configuration needed; pgvector is pre-installed and the
indexer creates all tables.

#### Step 2 — Install Ollama and pull models (Backend app)

```bash
# Download Ollama to /files/bin (persists across Nuvolos resets)
OLLAMA_VERSION=$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
    | grep '"tag_name"' | cut -d'"' -f4)
mkdir -p /files/bin /files/lib
curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
     -o /tmp/ollama.tar.zst
tar -x --zstd -f /tmp/ollama.tar.zst -C /files

# Model storage (shared, persistent)
export OLLAMA_MODELS=/space_mounts/pars/ollama_models
echo 'export OLLAMA_MODELS=/space_mounts/pars/ollama_models' >> ~/.bashrc

# Start the server and pull models (~5.5 GB total)
export PATH="/files/bin:$PATH"
ollama serve &
sleep 2
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

#### Step 3 — Clone the repo and index the data (Backend app)

```bash
cd /files
git clone https://github.com/EDEN757/company-rag-agent.git
```

Upload `data/raw/documents_subset.parquet` to `/files/company-rag-agent/data/raw/`, then:

```bash
cd /files/company-rag-agent
pip install -r backend/requirements.txt pyarrow pandas
python indexing/build_index_pg.py --input data/raw/documents_subset.parquet
```

This chunks, embeds, and stores ~35k vectors in pgvector. Takes ~15–30 minutes.
**Resumable** — re-run the same command to continue after a crash or timeout.

#### Step 4 — Install frontend dependencies (Frontend app)

```bash
cd /files/company-rag-agent/frontend
pip install -r requirements.txt
```

---

### Starting up

Run these every time. Start the Database app in Nuvolos first.

> **Tip:** Nuvolos conda environments do not reliably source `~/.bashrc` on new terminals.
> Always use explicit `export` commands — do not rely on `source ~/.bashrc`.

**Terminal 1 — Ollama (GPU Backend):**
```bash
export PATH="/files/bin:$PATH"
export OLLAMA_MODELS=/space_mounts/pars/ollama_models
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_FLASH_ATTENTION=1
ollama serve
```

For a CPU Backend, omit the last two `export` lines. Responses take ~2–5 minutes instead
of seconds.

**Terminal 2 — Backend API:**
```bash
cd /files/company-rag-agent/backend
uvicorn main:app --host 0.0.0.0 --port 8500
```

Expected startup output:
```
INFO: Connecting to pgvector @ nv-service-b01d63337fab32ac94f65eb2dc8a62ba:5432/nuvolos
INFO: pgvector connected — 35344 chunks indexed.
INFO: BM25 corpus stats: N=35344, avgdl=247.3 tokens, vocab=89412 terms
INFO: LLM: http://localhost:11434  model=qwen3:8b  embed=nomic-embed-text
INFO: Application startup complete.
```

**Frontend app:**
```bash
cd /files/company-rag-agent/frontend
python app.py
```

UI available at: `https://<hash>.proxy-eu1.nuvolos.cloud/proxy/7860/`

---

### Environment variables

Override in `~/.bashrc` on the relevant app only if defaults need to change.

**Backend app:**

| Variable | Default | Purpose |
|---|---|---|
| `PGHOST` | `nv-service-b01d63337fab32ac94f65eb2dc8a62ba` | pgvector hostname |
| `PGPORT` | `5432` | pgvector port |
| `PGUSER` | `nuvolos` | DB user |
| `PGPASSWORD` | `nuvolos` | DB password |
| `PGDATABASE` | `nuvolos` | DB name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `LLM_MODEL` | `qwen3:8b` | LLM served via Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder (set empty to disable) |
| `HYDE_ENABLED` | `true` | HyDE query expansion (set `false` to skip) |
| `USE_RRF` | `true` | Sort by RRF rank (`false` = sort by fusion score) |

**Frontend app:**

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_URL` | `http://nv-service-e4bb2876d3e69f18fd98d56e852aa814:8500` | Backend API URL |

---

### Gradio UI features

| Panel / Control | What it does |
|---|---|
| **Conversation** | Multi-turn chat; doc IDs in answers are clickable links that open the document viewer |
| **Send / Stop** | Stop cancels an in-progress search mid-stream |
| **Show retrieved sources** | Checkbox — shows retrieved chunks and opened documents in the agent steps panel |
| **Reasoning mode** | Activates Qwen3's extended thinking (`/think` token + `think: true`). Model emits `<think>…</think>` blocks visible in the Model reasoning accordion. Slower but more thorough on complex queries |
| **Agent steps & sources** | Every tool call trace (e.g. `[search] "invoice spike" → 3 result(s)`) plus sources |
| **Document viewer** | Opens at `/doc/{doc_id}?q={query}`. Highlights query terms ranked by term-frequency in the document |
| **Query history** | Last 5 queries with their agent steps and sources — resets on page refresh |
| **System info** | Backend health + chunk count by source with a Refresh button |

---

## System architecture

```
  Browser
    │  HTTPS /proxy/7860/
    ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Frontend  (Gradio + FastAPI, port 7860)   frontend/app.py     │
  │  · Streaming SSE consumer (real-time agent step display)        │
  │  · Document viewer with BM25-term highlighting                  │
  └───────────────────────────┬─────────────────────────────────────┘
                              │  HTTP POST /query/stream (SSE)
                              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Backend  (FastAPI + uvicorn, port 8500)   backend/main.py     │
  │                                                                  │
  │  ┌──────────────┐   ┌────────────┐   ┌────────────────────┐    │
  │  │  agent.py    │   │ fusion.py  │   │     kb.py          │    │
  │  │  LLM loop    │──▶│  Retrieval │   │  add / edit docs   │    │
  │  │  ≤6 turns    │   │  pipeline  │   │  atomic writes     │    │
  │  └──────┬───────┘   └─────┬──────┘   └────────────────────┘    │
  │         │                 │                                      │
  │         │   ┌─────────────┼──────────────────────┐              │
  │         │   │             │  Ollama (port 11434)  │              │
  │         │   │  ┌──────────▼────────┐              │              │
  │         │   │  │  qwen3:8b         │              │              │
  │         │   │  │  num_ctx=12288    │              │              │
  │         │   │  │  max_tokens=1000  │              │              │
  │         │   │  └───────────────────┘              │              │
  │         │   │  ┌────────────────────┐             │              │
  │         │   │  │  nomic-embed-text  │             │              │
  │         │   │  │  768-dim, /api/embed│            │              │
  │         │   │  └────────────────────┘             │              │
  │         │   └─────────────────────────────────────┘              │
  │         │                                                         │
  │         │   psycopg2 + pgvector                                   │
  └─────────┼───────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Database  (PostgreSQL 15 + pgvector, port 5432)                │
  │                                                                  │
  │  rag_documents  — full document content                          │
  │  rag_chunks     — chunk text, embedding (768-dim), text_tsv,    │
  │                   ts_from, ts_to, participants_json              │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Retrieval pipeline

Every call to the `search` tool runs these stages in order:

```
  User query
      │
      ├─────────────────────────────────────────────────────────┐
      │  [BM25 branch]  caller thread                           │
      │                                                          │
      │  fts_or_query(q) → OR-joined to_tsquery                 │
      │  → fetch TOP_K×3 = 24 FTS candidates from pgvector      │
      │  → bm25_scores(query_terms, candidates)                  │
      │    (k₁=1.5, b=0.75, corpus IDF from startup scan)       │
      │  → top-8 by BM25 score → kw_ranks, kw_scores            │
      │                                                          │
      │  [HyDE branch]  thread pool (concurrent with BM25)      │
      │                                                          │
      │  embed(query) ──────────────────────────────┐           │
      │  (hyde pool thread)                          │ parallel  │
      │                                              │           │
      │  LLM("Write a passage answering: {query}")   │           │
      │  max_tokens=80, num_ctx=512                  │           │
      │  → hypothetical doc text                     │           │
      │  → embed(hypothetical)                       │           │
      │  → average + re-normalise ←──────────────────┘           │
      │  → q_vec (HyDE-augmented)                               │
      │                                                          │
      └──────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
              Vector search (pgvector cosine <=>)
              SELECT … FROM rag_chunks c
              LEFT JOIN rag_documents d ON c.doc_id = d.doc_id
              ORDER BY embedding <=> q_vec LIMIT 8
              → vec_ranks, vec_scores, titles (one round-trip)
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │  Fuse  (all_chunk_ids = vec ∪ kw)    │
              │                                      │
              │  fusion_score = 0.7·vec + 0.3·kw     │
              │  rrf_score = 1/(60+vec_rank)          │
              │            + 1/(60+kw_rank)           │
              │                                      │
              │  Sort by rrf_score  (USE_RRF=true)   │
              │  or fusion_score    (USE_RRF=false)   │
              │                                      │
              │  [Threshold: fusion_score ≥ 0.5      │
              │   applied only when USE_RRF=false]   │
              └──────────────────────────────────────┘
                                 │
                                 ▼
              Deduplicate — keep best chunk per doc_id
                                 │
                                 ▼
              Cross-encoder rerank
              CrossEncoder("ms-marco-MiniLM-L-6-v2")
              scores (query, full_chunk_text) pairs jointly
              → sorted by rerank_score → top_n
                                 │
                                 ▼
              Return to LLM agent:
              chunk_id, doc_id, source_type, title,
              score, vec_score, kw_score, rerank_score,
              preview (600 chars), ts_from, ts_to
```

### Key parameters

| Parameter | Value | Effect |
|---|---|---|
| `TOP_K_PER_BRANCH` | 8 | Candidates per branch before fusion |
| `RRF_K` | 60 | RRF smoothing constant |
| `VEC_WEIGHT` | 0.7 | Vector weight in fusion score |
| `KW_WEIGHT` | 0.3 | BM25 weight in fusion score |
| `SCORE_THRESHOLD` | 0.5 | Noise filter (only active when `USE_RRF=false`) |
| `BM25_K1 / B` | 1.5 / 0.75 | BM25 term-frequency saturation / length normalisation |
| `MAX_EMBED_CHARS` | 6000 | Input truncation before embedding |

### BM25 corpus statistics

At startup, `bm25.init_corpus_stats()` scans all chunk texts using a server-side
streaming cursor (rows fetched in 2000-row batches, never fully loaded into memory)
and builds a corpus-level vocabulary:

```
N     — total chunk count
avgdl — average token count per chunk
df    — document frequency per token (keyed to our tokenizer output)
```

The tokenizer (`[A-Za-z0-9]\w*`, length > 1) is used identically at index time and
query time, so `corpus_df.get(term)` always hits — no stemming mismatch. When documents
are added or edited through the agent, corpus stats are updated incrementally in memory
without a restart.

### Structured pre-filters

When the user specifies who, when, or where, the agent passes filters to `search`:

| Filter | Column queried | Example |
|---|---|---|
| `source_types: ["slack"]` | `source_type IN (…)` | "What did the eng-platform channel say…" |
| `date_from: "2026-11-01"` | `ts_to >= ?` | "Emails from November onward" |
| `date_to: "2026-11-30"` | `ts_from <= ?` | "Before December" |
| `participant: "alex@…"` | `participants_json LIKE %?%` | "What did Alex tell us about…" |

Filters are applied as SQL `WHERE` clauses before both retrieval branches, so the
candidate pool is identical for BM25 and vector — fusion math is unchanged.

---

## Knowledge base write path

Documents can be added or edited through the agent's `add_document` and `edit_document`
tools. The write path is designed for atomicity and correctness:

```
  add_document(source_type, title, content, participants?, date?)
      │
      ├─ smart_chunk(content)
      │   Split on paragraph boundaries (\n\n).
      │   Oversized paragraphs (> 2000 chars) are hard-split with 200-char overlap.
      │   Last paragraph of a full chunk carries over as overlap into the next.
      │
      ├─ Prepend metadata header to each chunk:
      │   "[source: X] [title: Y] [participants: Z] [dates: A → B]"
      │
      ├─ batch_embed(all_chunk_texts)        ← single Ollama /api/embed round-trip
      │   All embeddings computed before any DB write.
      │   If Ollama fails here, no DB state is touched.
      │
      └─ db.transaction() {                  ← atomic: all-or-nothing
             INSERT INTO rag_documents
             INSERT INTO rag_chunks × N
         }
      │
      └─ bm25.corpus_add_chunks(new_texts)   ← incremental in-memory update
         Updates N, avgdl, and df counts without a full rescan.
```

`edit_document` follows the same pattern but additionally:
1. Fetches old chunk texts **before** the transaction (to decrement corpus stats after commit)
2. Wraps `UPDATE rag_documents` + `DELETE rag_chunks` + `INSERT rag_chunks` in one transaction
3. Calls `corpus_remove_chunks(old_texts)` then `corpus_add_chunks(new_texts)` after commit

---

## LLM agent loop

```
  User question
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │  agent.run_agent_streaming()  (≤ MAX_AGENT_TURNS=6)  │
  │                                                       │
  │  System prompt (intent routing table, citation rules) │
  │  + history[-MAX_HISTORY_TURNS×2:] (last 5 turns)     │
  │  + user question                                      │
  │                                                       │
  │  Tool subset:                                         │
  │  · Always: search, open_document,                     │
  │            add_document, edit_document                │
  │  · If filesystem markers in query:                    │
  │            read, write, edit, bash                    │
  │                                                       │
  │  Each turn:                                           │
  │  1. LLM call → tool_calls or final answer             │
  │  2. execute_tool() dispatches to fusion/kb/fs         │
  │  3. Tool result appended to message list              │
  │  4. Repeat until no tool call or turn limit hit       │
  │                                                       │
  │  Streaming: SSE events emitted per token/trace        │
  │  (trace_start → trace_done → token → done)            │
  └──────────────────────────────────────────────────────┘
       │
       ▼
  Answer + cited doc_ids + source list + latency_ms
```

**Intent routing (system prompt):** The prompt classifies every query before the first
tool call:

| Intent | Trigger | First tool call |
|---|---|---|
| Find / retrieve | find, search, show, what, who, when… | `search` |
| Create / draft | create, write, draft, save, record, log… | `add_document` directly |
| Update existing | update, edit, modify, revise, append… | `search` (to get doc_id), then `edit_document` |
| Ambiguous write | "write the notes", "document this" | `add_document` (default to create) |

The LLM is instructed never to narrate its plan before calling a tool.

**Reranker score guidance:** Search results include a `rerank` field when the
cross-encoder has run. The prompt instructs the LLM to use `rerank` as the primary
relevance signal (higher = more relevant; a positive value is a good match) and to
fall back to `score ≥ 2.0` when only fusion scores are available.

---

## Chunking strategy

All chunking happens at index time in `indexing/chunkers.py`. At query time, the backend
uses `kb.smart_chunk()` for agent-created documents.

**Every chunk carries a metadata header** prepended to the raw text, so both BM25
(token overlap) and the embedder (semantic context) see participant names, source type,
and dates:

```
[source: confluence] [title: Inference cost optimizer rollout] [dates: 2026-03-01]

<paragraph-aware chunk content, targeting ~2000 chars>
```

### Document-like sources

`confluence`, `google_drive`, `jira`, `linear`, `hubspot`, `github`, `fireflies` —
paragraph-aware sliding window. Split on `\n\n`; only hard-split when a single paragraph
exceeds the chunk size. The last paragraph of a full chunk carries over as overlap.

### Gmail

`content` is a Python-repr'd list of message strings. The chunker:

1. Tries `json.loads` first.
2. Falls back to a deterministic forward scan splitting on `From:` occurrences.
3. Unescapes double- and single-escaped newlines/tabs/quotes.
4. Extracts `From / To / Date / Subject` per message via regex.
5. Groups messages by normalized subject (strips `Re:` / `Fwd:`) into threads.
6. Slides a window of 4 messages with 1-message overlap per thread.

Participant names and date ranges are stored as typed columns (`ts_from`, `ts_to`,
`participants_json`) enabling exact structured filters on `search`.

### Slack

Each row's `content` is `speaker: text` lines separated by blank lines. The chunker
greedily packs conversational blocks to ~500 tokens, extracts speakers per chunk, and
carries the last block as overlap. Slack rows have no per-message timestamps, so
`ts_from`/`ts_to` are null.

---

## Tools exposed to the agent

### `search`

Runs the full hybrid retrieval pipeline described above.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Natural-language search query |
| `source_types` | string[] | no | Filter to one or more of: `slack`, `gmail`, `linear`, `jira`, `confluence`, `google_drive`, `hubspot`, `github`, `fireflies` |
| `date_from` | string | no | ISO-8601 lower bound |
| `date_to` | string | no | ISO-8601 upper bound |
| `participant` | string | no | Substring match against participants |
| `top_n` | integer | no | How many results to return (default 6, max 20) |

Returns ranked chunks with `chunk_id`, `doc_id`, `source_type`, `title`, `score`,
`vec_score`, `kw_score`, `rerank_score` (when reranker is active), and a 600-char
`preview`.

### `open_document`

Fetches the full text of a document by `doc_id`. The agent opens a document when the
preview is insufficient — truncated lists, exact figures, or cut-off details.

### `add_document`

Creates a new document in the knowledge base, chunks and embeds it atomically, and
returns the new `doc_id`. Default intent for write-like queries.

| Parameter | Type | Required |
|---|---|---|
| `source_type` | string | yes |
| `title` | string | yes |
| `content` | string | yes |
| `participants` | string | no |
| `date` | string (ISO-8601) | no |

### `edit_document`

Updates an existing document by `doc_id` and re-indexes it. Requires a real `doc_id`
from a prior `search` result — the agent must search first if it doesn't already have one.

| Parameter | Type | Notes |
|---|---|---|
| `doc_id` | string | required |
| `new_content` | string | Replace entire document |
| `old_string` | string | Exact text to find and replace |
| `new_string` | string | Replacement text (used with `old_string`) |

### `read` / `write` / `edit` / `bash`

Filesystem and shell tools operating on the Backend container (`/files/`). Only exposed
when the user's query explicitly references file paths, scripts, or terminal commands.
`bash` blocks dangerous patterns (`rm -rf`, `sudo`, `dd if=`, etc.) server-side.

---

## The dataset

[**EnterpriseRAG-Bench**](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
by [Onyx](https://onyx.app/enterpriserag-bench) — an open, MIT-licensed benchmark of
company-internal knowledge. Paper: [arXiv:2605.05253](https://arxiv.org/abs/2605.05253).

This project uses a 10,000-document subset spanning nine source types with 500
gold-labeled questions.

Files under `data/raw/`:

| File | Rows | Size | Purpose |
|---|---|---|---|
| `documents_subset.parquet` | 10,000 | 28 MB | Corpus the index is built from. Committed. |
| `questions_test.parquet` | 500 | 556 KB | Gold questions with `expected_doc_ids`, `gold_answer`, `answer_facts`. Committed. |
| `subset_manifest.json` | — | 68 KB | Provenance (seed, gold IDs). Committed. |

Document schema:

| Column | Type | Notes |
|---|---|---|
| `doc_id` | string | Stable ID, e.g. `dsid_c37d…` |
| `source_type` | string | One of the 9 source types |
| `title` | string | Slack channel, email subject, doc title, ticket name… |
| `content` | string | Free-form; gmail is Python-repr'd lists; slack is `name: text` lines |

Corpus distribution (chunks after indexing):

| Source | Docs | Chunks |
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

### Retrieval quality — `backend/eval.py`

Measures retrieval-only quality without starting the LLM agent. Metrics: Recall@k,
MRR@k, nDCG@k for k ∈ {1, 3, 5, 10}.

```bash
cd /files/company-rag-agent/backend
python eval.py --questions ../data/raw/questions_test.parquet --top-k 10
python eval.py --questions ../data/raw/questions_test.parquet --top-k 10 --no-hyde --limit 100
```

Output format:
```
   k    Recall       MRR      nDCG
--------------------------------------
   1     0.XXX     0.XXX     0.XXX
   3     0.XXX     0.XXX     0.XXX
   5     0.XXX     0.XXX     0.XXX
  10     0.XXX     0.XXX     0.XXX
```

**Baseline** (weighted fusion only, no reranking, measured on first 100 questions):

| k | Recall | MRR | nDCG |
|---|---|---|---|
| 1 | 0.670 | 0.670 | 0.670 |
| 3 | 0.710 | 0.690 | 0.695 |
| 5 | 0.740 | 0.696 | 0.707 |
| 10 | 0.930 | 0.720 | 0.767 |

Recall@10 = 0.93 means the right document is in the top-10 working set on 93% of
queries. Recall@1 = 0.67 is the target for RRF + HyDE + cross-encoder reranking to
improve. Updated numbers pending a full pipeline eval run.

### End-to-end agent quality — `indexing/eval_agent.py`

Sends questions through the live `/query` endpoint (uvicorn must be running) and measures:

- **Source hit rate** — did the agent cite an `expected_doc_id` in its returned sources?
- **Fact coverage** — what fraction of the gold `answer_facts` keywords appear in the
  LLM's response? (token-overlap, no extra LLM call)

```bash
python indexing/eval_agent.py --limit 20     # quick smoke test
python indexing/eval_agent.py                # full 500 questions
```

Each question goes through the full agent loop. On CPU (~2–5 min/question), use
`--limit 50` for a representative sample.

---

## Design decisions and trade-offs

### Why hand-rolled retrieval

The course brief forbids integrated RAG frameworks and expects every retrieval primitive
to be demonstrable. Only the underlying engines are used: PostgreSQL (FTS), pgvector
(ANN), Ollama (model serving), and Python for all the math. No LangChain, no RAGFlow,
no FAISS.

### Why `nomic-embed-text`

| Model | Params | Dim | Context | Decision |
|---|---|---|---|---|
| `all-minilm` | 22M | 384 | 512 | Too weak on MTEB English |
| `nomic-embed-text` | 137M | 768 | **8192** | **Chosen** — largest context of the small models, covers long emails without truncation, MIT-compatible, fast |
| `mxbai-embed-large` | 335M | 1024 | 512 | Higher quality but 512-token context truncates long emails |
| `bge-m3` | 560M | 1024 | 8192 | Multilingual; overkill for English-only corpus, 4× slower |

### Why RRF instead of pure weighted fusion

RRF (`1/(60+rank_vec) + 1/(60+rank_kw)`) is more robust than a fixed `0.7·vec + 0.3·kw`
for two reasons:

- **No magnitude assumption.** BM25 and cosine similarity live on different scales; a
  linear combination requires a hand-tuned `SCORE_SCALE` constant to make them
  commensurable. RRF ignores magnitudes and uses only rank positions.
- **No hyperparameter to tune.** The k=60 default is well-studied and performs well
  across benchmarks without dataset-specific tuning.

The weighted fusion score is still computed and shown to the LLM alongside the reranker
score. Set `USE_RRF=false` to sort by fusion score instead.

### Why HyDE for query expansion

Questions like "who raised concerns about latency?" don't match the vocabulary of a Slack
message that says "this is unacceptably slow." HyDE bridges that gap by asking the LLM
to write a short hypothetical passage that *would* answer the question, then averaging
its embedding with the raw query embedding. The averaged vector sits closer to
document-space language.

Implementation detail: `embed(query)` is submitted to a background thread pool
immediately when `hyde_embed()` is called, running concurrently with the LLM call for
the hypothetical document (~3–5 s on GPU). By the time the LLM returns, the query
embedding is already done — saving one sequential embed call per search.

Trade-off: each `search` call adds one small LLM call (`num_ctx=512`, `max_tokens=80`).
Set `HYDE_ENABLED=false` to skip.

### Why cross-encoder reranking

Bi-encoder retrieval (BM25 + dense) scores query and document independently and cannot
capture cross-attention between them. A cross-encoder sees the full `(query, passage)`
pair and learns fine-grained relevance signals — negation, entity matching, causal
language — that bi-encoders miss.

The gap between Recall@1 (0.67) and Recall@10 (0.93) is precisely what the cross-encoder
addresses: the right document is usually retrieved, just not ranked first.

`ms-marco-MiniLM-L-6-v2` (22M params) was chosen because it is small enough to load at
startup with negligible VRAM, fast enough (~50 ms for 16 pairs on GPU), and trained on
MS MARCO passage ranking which closely matches the Q→passage setting here.

The reranker score is surfaced to the LLM alongside the fusion score, enabling it to
make better decisions about which documents are worth opening in full.

### Why structured pre-filters instead of a third score channel

We considered making participant/date/source matches a third weighted score channel
alongside BM25 and vector. We rejected this because:

- It adds a hyperparameter (a third weight) that needs tuning.
- It degrades gracefully *wrong* when the user doesn't mention a person or date —
  mixing in a constant zero signal.
- A hard `WHERE` clause is exact, cheap, and cooperative with the existing fusion math:
  both branches operate on the same narrowed candidate pool.

### Why per-chunk metadata headers

Storing participants/dates in the chunk *text* in addition to typed columns means:

- BM25 finds "alex@hybridai.io" in keyword search even without an explicit filter.
- The embedder sees the participant list, improving semantic recall on queries like
  "what did Alex complain about?"

The cost is a few dozen extra tokens of header padding per chunk — negligible.

### Why atomic transactions for KB writes

With PostgreSQL's `autocommit=True`, multi-statement writes commit each statement
immediately. An interrupted `edit_document` (DELETE old chunks → INSERT new chunks)
could leave a document in the DB with zero chunks — permanently invisible to search but
still retrievable by `open_document`. All write operations use an explicit
`db.transaction()` context manager that wraps the entire write in a single atomic
commit.

### Why batch embedding for document ingestion

`_prepare_chunks()` previously called `embed()` once per chunk (one Ollama HTTP
round-trip each). For a 10-chunk document, that was 10 sequential calls. Now it calls
`batch_embed()` once via Ollama's `/api/embed` endpoint, which embeds all chunks in a
single round-trip regardless of document size.

---

## Course-pillar mapping

| Pillar | Slide source | Implementation | File |
|---|---|---|---|
| Sparse retrieval | W1, W2 | OR-joined `to_tsquery` candidates reranked by Python BM25 (k₁=1.5, b=0.75, corpus IDF) | `backend/bm25.py`, `backend/fusion.py` |
| Dense retrieval | W3 | `nomic-embed-text` via Ollama + pgvector cosine (`<=>`) | `backend/embed.py`, `backend/fusion.py` |
| Hybrid retrieval | W3 | RRF fusion + weighted score (0.7·vec + 0.3·kw) | `backend/fusion.py`, `backend/config.py` |
| Chunking | W9 | Paragraph-aware smart chunking + per-source semantic headers | `indexing/chunkers.py`, `backend/kb.py` |
| LM decoding | W4 | Qwen 3 8B via Ollama OpenAI-compatible endpoint (`num_ctx=12288`) | `backend/config.py`, `backend/agent.py` |
| LM prompting | W5 | Intent-routing system prompt, citation rules, ≤3-search cap, no-narration rule | `backend/prompt.py` |
| Open foundation models | W6 | Qwen 3 8B + nomic-embed-text (Apache-2.0, served via Ollama) | `backend/config.py` |
| Query expansion | W9 | HyDE: LLM-generated hypothetical passage averaged with raw query embedding | `backend/hyde.py` |
| Re-ranking | W9 | Cross-encoder `ms-marco-MiniLM-L-6-v2` scores `(query, chunk)` pairs jointly | `backend/reranker.py` |
| IR evaluation | W1, W2 | Recall@k, MRR@k, nDCG@k for k ∈ {1, 3, 5, 10} | `backend/eval.py` |
| End-to-end evaluation | W1, W2 | Source hit rate + fact coverage via live `/query` endpoint | `indexing/eval_agent.py` |
| Frontend | W10 | Gradio web UI: reasoning mode, stop button, agent step traces, sourced citations, document viewer with BM25-term highlighting, query history | `frontend/app.py` |
| Production engineering | W9 | Thread-local DB connections, atomic writes, incremental corpus stats, resumable indexer | `backend/db.py`, `backend/kb.py`, `backend/bm25.py` |

---

## Project layout

```
.
├── README.md
│
├── backend/                   Python — FastAPI backend (primary solution)
│   ├── main.py                FastAPI entry point (lifespan, endpoints)
│   ├── agent.py               LLM agent loop, intent-based tool selection, streaming
│   ├── config.py              All env vars and tuning constants
│   ├── prompt.py              System prompt (intent routing, citation rules) + tool schemas
│   ├── db.py                  Thread-local psycopg2 connections, transaction() context manager
│   ├── embed.py               Ollama embed() + batch_embed() clients
│   ├── bm25.py                BM25 tokenizer, corpus stats, scoring, incremental updates
│   ├── hyde.py                HyDE query expansion (parallel embed + LLM hypothetical)
│   ├── fusion.py              Hybrid search: BM25 + vector + RRF fusion + deduplication
│   ├── reranker.py            Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   ├── kb.py                  Knowledge base writes: smart_chunk, batch_embed, transactions
│   ├── tools.py               Tool dispatch (search → fusion, add/edit → kb, fs tools)
│   ├── eval.py                Retrieval eval harness: Recall@k, MRR@k, nDCG@k
│   └── requirements.txt
│
├── frontend/                  Python — Gradio web UI
│   ├── app.py                 Streaming SSE consumer, document viewer, query history
│   └── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── documents_subset.parquet   (28 MB, committed)
│   │   ├── questions_test.parquet     (500 gold questions, 556 KB, committed)
│   │   ├── subset_manifest.json       (68 KB, committed)
│   │   └── documents_test.parquet     (1.4 GB, gitignored)
│   └── .venv/                         (gitignored)
│
├── indexing/                  Python — one-time indexing and end-to-end eval
│   ├── schema_pg.sql          pgvector schema: rag_documents | rag_chunks
│   ├── chunkers.py            Per-source chunkers: gmail / slack / document-like
│   ├── embed.py               Ollama embedding client with retry + payload shrink
│   ├── build_index_pg.py      Parquet → pgvector indexer (resumable)
│   ├── rechunk_source_pg.py   Re-index one source_type after a chunker change
│   └── eval_agent.py          End-to-end eval: source hit rate + fact coverage
│
└── typescript_deprecated/     Legacy TypeScript TUI + SQLite (local only, not maintained)
    ├── src/                   TypeScript agent source
    └── indexing/              SQLite-only indexing scripts
```

---

## Known limitations and future work

- **Eval numbers need a fresh run.** The baseline Recall@1 = 0.67 was measured before
  RRF, HyDE, and cross-encoder reranking were added. A full run of `backend/eval.py`
  is needed to quantify the improvement from the complete pipeline.

- **IVFFlat → HNSW migration pending.** The vector index uses IVFFlat with
  `ivfflat.probes=10`. pgvector's HNSW index consistently achieves better recall@k
  at comparable query speed with no probe tuning required. Migration is a one-time
  `CREATE INDEX USING hnsw` + `DROP` of the old index.

- **Slack chunks have no timestamps.** The source rows don't carry per-message dates,
  so `date_from`/`date_to` filters silently exclude Slack. If a date-stamped Slack dump
  becomes available, the chunker only needs to populate `ts_from`/`ts_to`.

- **HyDE adds latency per search.** Each `search` tool call includes one small LLM
  inference (~3–5 s on GPU). For multi-hop questions that call `search` twice, this
  doubles the HyDE overhead. Set `HYDE_ENABLED=false` if latency is critical.

- **`participants_json LIKE` is a full table scan.** Participant filtering uses a
  `LIKE %substring%` pattern with no index. For large corpora or frequent participant
  filters, a GIN index or a normalized participants table would improve performance.

- **No embedding cache.** `embed()` is called fresh for every query. Adding a
  bounded LRU cache keyed on the input text would reduce latency on repeated queries
  and cut eval runtime roughly in half.

- **`batch_embed` requires Ollama ≥ 0.1.25.** The `/api/embed` endpoint that accepts
  a list of inputs was introduced in that release. Older deployments will fail on
  multi-chunk document ingestion. A fallback to sequential `embed()` calls would make
  the system resilient across Ollama versions.

---

## Credits and license

- **Models:** `nomic-embed-text` (Apache-2.0, Nomic AI), Qwen 3 8B (Apache-2.0,
  Alibaba Cloud), `ms-marco-MiniLM-L-6-v2` (Apache-2.0, Microsoft) — all served via
  [Ollama](https://ollama.ai).
- **Dataset:** [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
  by Onyx (MIT). 10k-doc subset committed; full 500k-doc dump is upstream on HuggingFace.
- **Legacy agent harness:** [`@mariozechner/pi-agent-core`](https://www.npmjs.com/package/@mariozechner/pi-agent-core)
  (TypeScript TUI, deprecated).

Coursework artifact for the UZH FS2026 RAG course.
