# Demo QA — ground truth

This file is the authored ground-truth for the three skill demos. Each
section lists the demo question, the expected one-paragraph answer, the
exact `doc_id`s that must appear in the retrieved set and citations, and
a note on how a generic agent (no skill prompt prepended) would likely
fail the question.

All listed `doc_id`s are tagged `synthetic: true, skill: <name>` in
`metadata_json` once `indexing/add_demo_docs.py` has been run. They can
be enumerated at any time with:

```sql
SELECT doc_id, json_extract(metadata_json, '$.skill') AS skill
FROM   documents
WHERE  json_extract(metadata_json, '$.synthetic') = 1
ORDER BY skill, doc_id;
```

The demo topics were chosen so that their core vocabulary (`aquarium`,
`saltwater`, `verbier`, `gocritic`, `paramTypeCombine`) has zero or
near-zero collisions with the real corpus. This is what lets the
demo docs win on both the keyword (BM25) and vector branches of the
hybrid retriever, so they land in the top hits when their demo
question is asked.

---

## Skill: trace

### Question
> Trace the office aquarium leak incident.

### Expected answer (shape)
A chronological timeline like:
- `2025-08-12 — [confluence] Saltwater aquarium installation approved for the Zurich office lobby by morgan@acmeco.io; vendor Aqua-Marine Zürich; jamie@acmeco.io owns the project (demo_aquarium_kickoff)`
- `2025-09-15 — [jira] AQUA-7 filed after overnight leak from the aquarium sump soaked ~3 sq m of carpet; Aqua-Marine on-site by morning (demo_aquarium_leak)`
- `2025-09-28 — [confluence] Faulty sump replaced and cabinet reinforced; Aqua-Marine covered carpet replacement under warranty; tank back online; AQUA-7 closed (demo_aquarium_resolution)`

Followed by a one-sentence status: closed on 2025-09-28, owned by jamie@acmeco.io.

### Must-cite doc_ids
- `demo_aquarium_kickoff`
- `demo_aquarium_leak`
- `demo_aquarium_resolution`

### Verified retrieval (against /tmp/test-rag.db on 2026-05-23)
With the query `"office aquarium leak"`, all three demo docs land at
positions #1, #2, #3 (scores 2.786, 2.365, 2.192) — the entire top of
the result set is the demo corpus.

### Without the trace skill
A generic agent typically answers with a single paragraph paraphrasing
whichever doc scored highest, without ordering events, without
distinguishing causes from outcomes, and often without citing more than
one source. The chronology — which is the whole point of "trace" — is
lost.

---

## Skill: decide

### Question
> What did we decide about the gocritic paramTypeCombine linter rule?

### Expected answer (shape)
Strictly the four labeled fields, with the decision quoted verbatim:

- **Decision:** "Agreed: we'll disable paramTypeCombine in `_test.go` files and keep it as an error on production code. This lands in CI on 2025-09-26."
- **Made by:** priya@acmeco.io, with approval the same day from sam@acmeco.io; closed out by dev-platform@acmeco.io.
- **When:** 2025-09-24 (rolled out 2025-09-26).
- **Source:** `demo_gocritic_decision`

### Must-cite doc_ids
- `demo_gocritic_decision` (required)
- `demo_gocritic_thread` may also be opened for context, but the decision itself comes from `demo_gocritic_decision`.

### Verified retrieval (against /tmp/test-rag.db on 2026-05-23)
With the query `"gocritic paramTypeCombine decision"`, both demo docs
land at positions #1 and #2 (scores 2.719 and 2.199).

### Without the decide skill
A generic agent tends to summarize *both* documents and present "the
team debated whether to keep the rule, with views on both sides" — it
explains the discussion but never extracts the resolution. The user
asked "what did we decide", not "what was discussed".

---

## Skill: onboard

### Question
> Onboard me onto the Verbier Q4 ski retreat.

### Expected answer (shape)
Exactly five sections, one short paragraph each, each citing ≥1 doc_id:

- **What it is.** The company's Q4 2025 offsite at Hotel Sundial in
  Verbier, 2025-12-08 to 2025-12-11 (`demo_verbier_overview`).
- **Goal.** Cross-team bonding and on-snow ski lessons for new hires;
  roughly 52 employees invited (`demo_verbier_overview`).
- **Status.** As of 2025-09-30: 47/52 confirmed, ski instructor
  contract signed with Verbier Sport Academy, lift passes
  pre-purchased through Téléverbier, dietary survey closed
  (`demo_verbier_status`).
- **Key people.** morgan@acmeco.io (program owner / hotel liaison),
  kai@acmeco.io (transport), taylor@acmeco.io (off-snow activities),
  jules@acmeco.io (new-hire onboarding & vendor escalation contact)
  (`demo_verbier_committee`, `demo_verbier_vendors`).
- **Open issues.** Cancellation deadline 2025-10-15 for the venue; 2
  attendees still pending confirmation; jules@acmeco.io is single
  point of contact for vendor escalations during the offsite
  (`demo_verbier_overview`, `demo_verbier_status`,
  `demo_verbier_vendors`).

### Must-cite doc_ids
At least one of each per section as listed above. Minimum set across the whole answer:
- `demo_verbier_overview`
- `demo_verbier_status`
- `demo_verbier_committee`
- `demo_verbier_vendors`

### Verified retrieval (against /tmp/test-rag.db on 2026-05-23)
With the query `"Verbier ski retreat"`, all four demo docs land at
positions #1, #2, #3, #4 (scores 2.432, 2.430, 2.251, 2.193) — the
entire top of the result set is the demo corpus.

### Without the onboard skill
A generic agent usually opens one document, then either produces a
single dense paragraph or a generic bullet list that mixes status,
team, and goals into one blob. The "Open issues" and "Key people"
sections are typically missing because the agent never thinks to do
a second search with a different angle — which is what the onboard
skill forces.

---

## Notes for evaluators

- Retrieval ordering is not strictly checked — what matters is that
  every must-cite `doc_id` appears in the top-K returned by `search`,
  and that the answer cites them. If a `doc_id` is missing from
  retrieval, the failure is retrieval-side; if it is retrieved but
  not cited in the answer, the failure is skill-prompt-side.
- The demo dates are all in 2025, matching the date range of the
  real corpus (~2025-03 through ~2025-10).
- These three demos are intentionally small and unambiguous. They
  are not a substitute for the real eval against
  `data/raw/questions_test.parquet`; they exist to make the skill
  behaviors easy to demonstrate live.
- The earlier "Project Bluefin" (invoice) and "Project Atlas"
  (auth migration) drafts were discarded because their vocabulary
  collided heavily with the real corpus (`bluefin` had 12 hits,
  `atlas` had 203 hits before our docs were added), so the demo
  docs lost the vec-similarity contest against real corpus content.
  The current topics were chosen for zero corpus collision —
  `aquarium`, `saltwater`, `verbier`, `gocritic` all have 0–2 hits
  outside the demo set.
