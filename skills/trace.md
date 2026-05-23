---
name: trace
description: Trace an event, project, or issue across sources in chronological order. Use when the user wants to follow a topic through emails, chats, tickets, and meetings as it evolved over time.
suggested_question: Trace the office aquarium leak incident.
---

# Trace skill

Workflow for the user's topic:

1. Call `search` once with the topic name and the broadest natural-language description. Do not pre-filter by `source_types`; we want hits from anywhere.
2. Read every preview. For each distinct `doc_id` that looks topical, call `open_document`.
3. From each opened document, extract the date or dates discussed, and a one-sentence summary of what happened.
4. Produce a chronological timeline, one line per event, formatted exactly like:

   `YYYY-MM-DD — [source_type] one-sentence summary (doc_id)`

5. End with a short paragraph stating the current status and who is owning the issue.
6. Cite every doc_id you opened. Never invent a doc_id or a date.

If two documents disagree about a date, prefer the date written inside the body over a metadata field, and note the disagreement in parentheses.
