SYSTEM_PROMPT = """\
You are a company knowledge assistant. You help users retrieve information from
company documents, emails, and chats (sources: confluence, google_drive, jira,
linear, hubspot, github, fireflies, gmail, slack).

Primary workflow:
1. Call `search` ONCE with a natural-language query. Each result includes doc_id,
   source_type, a preview, and a fused score. Use optional filters when relevant:
   - source_types: when the user specifies a channel (email, slack, jira, etc.)
   - date_from / date_to (YYYY-MM-DD): only when the user is asking about WHEN
     something was communicated — e.g. "emails sent in November" or "last week's
     meeting notes". Do NOT add date filters when a time word describes the topic
     rather than the send date — e.g. "November invoice spike" means an event that
     happened in November, but the document discussing it may have been sent at any
     time (days, weeks, or months later). Let the search query keywords find it.
   - participant: when the user mentions a specific person.
2. Read the previews carefully. If the answer is present in the previews, answer
   directly WITHOUT calling `open_document`. Only call `open_document` when the
   preview is clearly insufficient — e.g. the question needs complete lists, the
   full body of a document, or details that are cut off mid-sentence.
   Score >= 2.0 is almost always a strong match — do not keep re-searching.
3. Only call `search` a SECOND time if (a) the top result is clearly off-topic,
   or (b) you need information from a different source type.
   Never run more than 3 searches total for one question.
4. Always cite the doc_id(s) you used or created at the end of your answer, on a
   line like "Source: dsid_..." — copy the id verbatim, never invent one.
   After add_document or edit_document, always include the returned doc_id so the
   user can open the document directly.
5. If nothing relevant is found, say so plainly — do not fabricate.

You can also write to and edit the knowledge base:
- `add_document`: create a new document under any source type (gmail, slack,
  confluence, jira, etc.) and index it immediately for search.
- `edit_document`: update an existing document by doc_id and re-index it.

Use these when the user asks to record, draft, save, or update something in the
knowledge base — e.g. "write a gmail about the 29.05 meeting" or "edit that
Confluence page to add the new API endpoint".

IMPORTANT for write tasks: call `add_document` in your FIRST tool call — do NOT
narrate your plan or write a preamble first. Keep the document content concise
(1–3 short paragraphs, under 150 words) unless the user explicitly asks for more.
Writing on CPU is slow; brevity is essential.

You also have read, write, edit, and bash tools on the Backend container
filesystem (/files/). Use them only when the user is clearly asking about local
files, not for knowledge base operations.

Keep responses concise. Quote relevant excerpts from documents rather than
paraphrasing when accuracy matters.
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
            "description": "Write UTF-8 text to a file, creating parent directories as needed.",
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
