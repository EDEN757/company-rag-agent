---
name: decide
description: Extract the final decision on a question from a discussion thread. Use when the user wants to know what was actually decided about something — not the full conversation, just the outcome.
suggested_question: What did we decide about the gocritic paramTypeCombine linter rule?
---

# Decide skill

Workflow for the user's question:

1. Call `search` for the topic.
2. Open the top 2 or 3 hits with `open_document`. Read them in full.
3. Identify the most recent statement that uses decision language. Examples to look for:
   - "we'll go with"
   - "agreed"
   - "final decision is"
   - "approved"
   - "we decided"
   - "let's move forward with"

4. Output exactly this structure:

   - **Decision:** quote the deciding sentence verbatim.
   - **Made by:** the person or role attributed to the decision.
   - **When:** the date or timestamp closest to the decision.
   - **Source:** the doc_id of the document the decision was extracted from.

5. If no decision language appears in any opened document, say so plainly:
   "No final decision found — the discussion is still open." Cite the most recent doc_id you opened so the user can read the latest state.

Do not paraphrase the decision. Quote it.
