# Demo skills — what we built and how to demo them

This document is the **human walkthrough** of the three skills the
agent supports during a live demo. The companion file
[`data/demo_docs/QA.md`](../data/demo_docs/QA.md) is the
**machine-friendly ground truth** (exact questions, expected `doc_id`s,
expected answer shape). Edit them together.

---

## 1. Why three skills?

A retrieval-focused company assistant fails in three distinct ways
that aren't fixed by better retrieval alone:

1. **Generic Q&A loses chronology.** Even with the right docs in hand,
   a default agent will summarize whichever doc scored highest and
   skip the time dimension. → **`trace`** skill.
2. **Generic Q&A loses outcomes.** When asked "what did we decide",
   default agents recap the discussion. → **`decide`** skill.
3. **Generic Q&A is single-angle.** "Onboard me onto X" needs several
   searches with different framings (status, team, customers); a
   default agent does one. → **`onboard`** skill.

Each skill is a short prompt (in [`/skills/`](../skills/)) that the
future frontend will prepend to the user's question when they press
the corresponding button.

| Skill     | When to pick it | Prompt file |
|-----------|------------------|-------------|
| `trace`   | "Trace / walk me through / what happened with X" | [`skills/trace.md`](../skills/trace.md) |
| `decide`  | "What did we decide / what was agreed on X" | [`skills/decide.md`](../skills/decide.md) |
| `onboard` | "Onboard me / give me the X / TL;DR on Y" | [`skills/onboard.md`](../skills/onboard.md) |

---

## 2. The demo corpus

We authored a small set of demo source documents under
[`data/demo_docs/`](../data/demo_docs/). They are **not** for gaming
retrieval against the real eval set (`data/raw/questions_test.parquet`)
— they exist purely to give the live demo a reliable, easy-to-narrate
example for each skill.

### Why a separate folder, and what tags them
- Source files live as plain `.md` with YAML frontmatter at
  `data/demo_docs/<skill>/<doc>.md`. This folder is the canonical
  copy — version-controlled, editable.
- After ingestion (see §3), they also exist as rows in `rag.db`,
  reachable by the same hybrid retriever as the rest of the corpus.
- Every demo doc is tagged in `metadata_json`:
  ```json
  {"synthetic": true, "skill": "<name>", "date": "...", "participants": [...]}
  ```
  Enumerate them at any time:
  ```sql
  SELECT doc_id, json_extract(metadata_json,'$.skill') AS skill
  FROM   documents
  WHERE  json_extract(metadata_json,'$.synthetic') = 1
  ORDER  BY skill, doc_id;
  ```

### What's in the corpus

| Skill   | doc_id                          | Source flavor | Date       |
|---------|---------------------------------|---------------|------------|
| trace   | `demo_aquarium_kickoff`         | confluence    | 2025-08-12 |
| trace   | `demo_aquarium_leak`            | jira          | 2025-09-15 |
| trace   | `demo_aquarium_resolution`      | confluence    | 2025-09-28 |
| decide  | `demo_gocritic_thread`          | confluence    | 2025-09-18 |
| decide  | `demo_gocritic_decision`        | confluence    | 2025-09-24 |
| onboard | `demo_verbier_overview`         | confluence    | 2025-07-02 |
| onboard | `demo_verbier_status`           | jira          | 2025-09-30 |
| onboard | `demo_verbier_committee`        | confluence    | 2025-08-05 |
| onboard | `demo_verbier_vendors`          | hubspot       | 2025-09-10 |

### Why these specific topics

Each topic was chosen for **zero or near-zero collision with the real
corpus vocabulary** so the demo docs win on both branches of the
hybrid retriever. We learned this the hard way — an earlier draft used
"Project Bluefin" (invoice spike) and "Project Atlas" (auth migration),
and verification showed those demo docs never made the top 8: a search
for the corpus had 12 hits for `bluefin` and 203 hits for `atlas` even
before our docs were added, and the vec branch was full of real-corpus
invoice/auth content that outscored the small demos.

The current topics — office aquarium, Verbier ski retreat, gocritic
linter rule — each use vocabulary with 0–2 hits outside our demo
files, so retrieval lands them at the top reliably. See
[`data/demo_docs/QA.md`](../data/demo_docs/QA.md) for the verified
retrieval ranks and scores.

### Note on `source_type`

All demo docs go through the `confluence`/`jira`/`hubspot` chunkers
(i.e. the simple sliding-window `chunk_document_like` path),
regardless of what the document is *about*. We don't masquerade as
`gmail` or `slack` because their chunkers expect specific encodings
(`From:/To:/Date:/Subject:` blocks for gmail; `name: message` lines
for slack) that are too easy to get wrong for short demo files.
Source flavor like "Slack thread" or "email exchange" is communicated
in the prose where relevant.

---

## 3. Ingesting the demo corpus

Once authored, push them into the index:

```bash
# Activate the env (sets RAG_DB_PATH, OLLAMA_HOST, etc.)
set -a; source .env; set +a

# Idempotent — safe to re-run after edits
python indexing/add_demo_docs.py
```

Behavior:
- Walks `data/demo_docs/**/*.md` (skipping `QA.md`).
- Skips any file whose `doc_id` already exists in `documents`.
- INSERTs the doc, chunks it via the same dispatcher used by the main
  indexer, then embeds only the new chunks via Ollama
  (`nomic-embed-text`).
- Runs FTS `optimize` + `ANALYZE` only if anything new was added.

If you edit a doc body and want the new content in the index, change
its `doc_id` (e.g. `demo_atlas_status_v2`) or delete its existing rows
from `documents` and `chunks` before re-running. We intentionally
didn't add an `--update` flag — silently overwriting indexed content
is the kind of thing that masks bugs.

---

## 4. Demo storyboard (live presentation)

This is the suggested order for the live demo. The exact questions
and expected answers live in `data/demo_docs/QA.md`; the storyboard
below is the narrative around them.

### Take 1 — Baseline (no skill button pressed)

1. Open the chat with **no skill selected**.
2. Ask: *"Trace the office aquarium leak incident."*
3. Show what happens: the agent does one search, opens probably one
   document, and produces a single paragraph that summarizes the
   resolution doc but loses the August kickoff and the September
   leak report.

This take is the strawman — it sets up why a skill matters.

### Take 2 — `trace` skill

1. Press the **Trace** button. The chat input is now backed by
   `skills/trace.md`.
2. Ask the same question.
3. Expected output: a three-line dated timeline citing all three
   `demo_aquarium_*` doc_ids, plus a one-sentence status. Compare
   directly with Take 1.

### Take 3 — `decide` skill

1. Press the **Decide** button.
2. Ask: *"What did we decide about the gocritic paramTypeCombine
   linter rule?"*
3. Expected output: the four-line structured block
   (Decision/Made by/When/Source) with the decision quoted verbatim
   from `demo_gocritic_decision`. The model should NOT recap the
   debate from `demo_gocritic_thread`.

### Take 4 — `onboard` skill

1. Press the **Onboard** button.
2. Ask: *"Onboard me onto the Verbier Q4 ski retreat."*
3. Expected output: a five-section brief (What it is / Goal / Status
   / Key people / Open issues) citing all four `demo_verbier_*`
   doc_ids across the sections. The "Open issues" section is the
   litmus test — that's the one that requires the multi-angle
   retrieval the onboard prompt forces.

---

## 5. Pitfalls and tips

- **Small model, short prompts.** qwen3:8b on Nuvolos is slow and
  loses focus with long system prompts. Keep each `skills/*.md` body
  under ~20 lines.
- **qwen3 thinking tokens.** Always pass `"think": false` in the
  generate/chat call when the production model is qwen3:8b. The
  emitted `<think>` blocks break tool-call parsing and triple
  latency. See `docs/date-filter.md` for the analogous "default
  behavior we always override" pattern.
- **BM25-friendly names.** The unique strings (`Bluefin`, `gocritic`,
  `Atlas`) are deliberate. If you rename a demo project, make sure
  the new name is rare enough in the rest of the corpus that BM25
  alone surfaces the demo docs reliably.
- **Adding a skill.** Add a `skills/<name>.md` with frontmatter,
  add demo docs to `data/demo_docs/<name>/`, append a section to
  `data/demo_docs/QA.md`, run the indexer. No code change required
  in the agent.

---

## 6. Where the relevant code lives

- Skill prompts (frontend reads these): [`skills/`](../skills/)
- Demo source docs: [`data/demo_docs/`](../data/demo_docs/)
- Demo ground truth: [`data/demo_docs/QA.md`](../data/demo_docs/QA.md)
- Ingester: [`indexing/add_demo_docs.py`](../indexing/add_demo_docs.py)
- Chunker dispatcher (reused, not rewritten):
  [`indexing/chunkers.py`](../indexing/chunkers.py)
- Embedder (reused): [`indexing/embed.py`](../indexing/embed.py)
- Hybrid retriever the agent uses at query time:
  [`src/rag/fusion.ts`](../src/rag/fusion.ts)
- Where to set `RAG_DB_PATH`, `OLLAMA_HOST`, etc.:
  [`.env.example`](../.env.example)
