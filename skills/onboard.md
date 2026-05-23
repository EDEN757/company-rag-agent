---
name: onboard
description: Produce a five-minute onboarding brief for a project, team, or topic. Use when the user is new to something and wants a structured summary, not raw search results.
suggested_question: Onboard me onto the Verbier Q4 ski retreat.
---

# Onboard skill

Workflow for the user's project, team, or topic name:

1. Call `search` 2–3 times with **different angles** of the same topic, for example:
   - the topic name alone (overview)
   - the topic plus "status" or "current"
   - the topic plus "team" or "owner" or "contact"

2. For each search, open the single top hit with `open_document`.

3. Produce the brief in exactly five sections, one short paragraph each — no bullet lists inside sections:

   - **What it is.** A one-sentence definition.
   - **Goal.** Why it exists, the business outcome it targets.
   - **Status.** Where it currently stands, with the most recent date mentioned in any opened document.
   - **Key people.** Names, roles, and contact handles, drawn from participants and bylines.
   - **Open issues.** Anything described as a blocker, risk, complaint, or in-flight work.

4. Each section must cite at least one `doc_id` from an opened document — inline, in parentheses.

If a section truly has no supporting content in the retrieved documents, write "Not found in the indexed corpus." rather than guessing.
