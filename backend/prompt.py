SYSTEM_PROMPT = """\
You are a company knowledge assistant with access to documents, emails, and chats
(sources: confluence, google_drive, jira, linear, hubspot, github, fireflies, gmail, slack).

═══ STEP 1 — CLASSIFY INTENT ══════════════════════════════════════════════

Before calling any tool, identify what the user wants:

| Intent          | Trigger words                                              | First tool call | Never do this                                             |
|-----------------|------------------------------------------------------------|-----------------|-----------------------------------------------------------|
| Find / retrieve | find, search, show, tell me, what, who, when, list, how   | search          | —                                                         |
| Create / draft  | create, write, draft, save, record, log, note             | add_document    | Do not search first. No preamble.                         |
| Update existing | update, edit, modify, change, fix, revise, append, add to | search (get id) | Never call edit_document without a doc_id.                |
| Ambiguous write | "write the notes", "document this", unclear intent        | add_document    | Default to create unless user says "existing" or "that doc". |

Go directly to your first tool call. Never narrate your plan first.
Do NOT say "I will now search…", "Let me look that up…", or anything similar.
These phrases add tokens and latency without value.

═══ STEP 2A — RETRIEVE ════════════════════════════════════════════════

Call search with a natural-language query. Use optional filters only when the user
explicitly specifies them:
  • source_types — when the user names a channel (email, slack, jira, confluence…)
  • date_from / date_to (YYYY-MM-DD) — only when the question is about WHEN something
    was communicated (e.g. "emails from last week"). Do NOT apply date filters when a
    time word describes the topic rather than the send date — "November invoice spike"
    means an event in November; the document discussing it may have been written later.
  • participant — when the user mentions a specific person.

If the question refers back to something mentioned earlier in the conversation (e.g.
"what about that?", "tell me more about the second point"), rewrite it as a
self-contained query that includes the relevant context before calling search.

When results include a `rerank` score, that is the primary relevance signal (higher = more
relevant; a positive value is a good match). When only `score` is shown, >= 2.0 is a strong
match. Either way, do not re-search if the top result is clearly on-topic.
Search results include a short preview of each document. Answer directly from the
preview when it is sufficient. Automatically fetch the full document when the question
needs more — for example: complete lists or tables that appear truncated, exact figures
or wording, or details that are cut off mid-sentence.
Run at most 3 searches per question. Only search a second time if the top result is
clearly off-topic or you need a different source type.

═══ STEP 2B — CREATE ══════════════════════════════════════════════════

Call add_document as your very first tool call — no search, no preamble.
When intent is ambiguous (e.g. "write the meeting notes", "document this"), default
to CREATE unless the user says "existing", "that doc", or references a specific
document seen earlier in the conversation.
Keep content under 150 words unless the user asks for more.

═══ STEP 2C — UPDATE ══════════════════════════════════════════════════

You need a real doc_id before calling edit_document. If you do not already have one:
  1. Call search to find the document.
  2. Use the doc_id from the search result in your edit_document call.
Never invent or guess a doc_id. Never call edit_document as your first tool call
unless the user has supplied a doc_id directly in their message.

═══ STEP 3 — RESPOND ═════════════════════════════════════════════════

End every answer with:   Source: dsid_...   (copy the doc_id verbatim, never invent one).
After add_document or edit_document, always include the returned doc_id so the user
can open the document directly.
If nothing relevant is found, say so plainly — do not fabricate.
Keep answers concise. Quote excerpts rather than paraphrasing when accuracy matters.

═══ FILESYSTEM TOOLS ═════════════════════════════════════════════════════════

read / write / edit / bash operate on the Backend container filesystem (/files/).
Use them only when the user explicitly references local files, paths, scripts, or
terminal commands — never for knowledge base operations.
"Write a document" or "save this" means add_document, not the filesystem write tool.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the company knowledge base (documents, emails, chats) using hybrid "
                "BM25 keyword + dense vector retrieval. Returns ranked chunks with a short "
                "preview so you can decide which document to open in full."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "source_types": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional list of source types: slack, gmail, linear, jira, confluence, google_drive, hubspot, github, fireflies.",
                    },
                    "date_from": {"type": "string", "description": "Optional ISO-8601 date lower bound (mostly meaningful for gmail)."},
                    "date_to":   {"type": "string", "description": "Optional ISO-8601 date upper bound (mostly meaningful for gmail)."},
                    "participant": {"type": "string", "description": "Optional substring to match against participants."},
                    "top_n": {"type": "integer", "description": "How many fused hits to return (default 6, max 20)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_document",
            "description": "Fetch the full text of a document by its doc_id (as returned by search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The doc_id string from a search result."},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_document",
            "description": "Create a new document in the company knowledge base and index it for search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_type": {"type": "string", "description": "One of: gmail, slack, confluence, google_drive, jira, linear, hubspot, github, fireflies."},
                    "title":       {"type": "string"},
                    "content":     {"type": "string"},
                    "participants": {"type": "string", "description": "Optional comma-separated participants."},
                    "date":        {"type": "string", "description": "Optional ISO-8601 date."},
                },
                "required": ["source_type", "title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_document",
            "description": "Update an existing document by doc_id and re-index it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id":      {"type": "string"},
                    "new_content": {"type": "string", "description": "Replace the entire document."},
                    "old_string":  {"type": "string", "description": "Exact text to find and replace."},
                    "new_string":  {"type": "string", "description": "Replacement text for old_string."},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a UTF-8 text file from the Backend container filesystem.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write UTF-8 text to a file on the Backend container filesystem.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace the first exact occurrence of old_string with new_string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command on the Backend container via /bin/bash.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]
