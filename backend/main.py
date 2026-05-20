"""
backend/main.py — Company RAG Agent API (Nuvolos)
==================================================
FastAPI agentic backend running on the Backend VS Code app.

Architecture:
  Frontend (Gradio, port 7860)
      │  HTTP POST /query
      ▼
  Backend (FastAPI, port 8500)     ← this file
      │  psycopg2 + pgvector       (hybrid retrieval)
      │  Ollama /api/embeddings    (nomic-embed-text — same model as local)
      │  Ollama /v1                (Qwen 3 8B — same model as local)
      ▼
  Database (PostgreSQL + pgvector, port 5432)

Run:
    # Terminal 1 — keep Ollama running
    OLLAMA_MODELS=/space_mounts/pars/ollama_models ollama serve

    # Terminal 2 — start the API
    cd /files/backend
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8500

Environment variables (set in Nuvolos Backend app CONFIGURE):
    OLLAMA_HOST     http://localhost:11434      (default)
    LLM_MODEL       qwen3-8b-32k               (default — matches Modelfile)
    EMBED_MODEL     nomic-embed-text            (default — same as local pipeline)
    PGHOST          nv-service-b01d63337fab32ac94f65eb2dc8a62ba  (default)
    PGPORT          5432
    PGUSER          nuvolos
    PGPASSWORD      nuvolos
    PGDATABASE      nuvolos
"""

import os
import re
import json
import logging
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL   = os.environ.get("LLM_MODEL",   "qwen3:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

DB_HOST     = os.environ.get("PGHOST",     "nv-service-b01d63337fab32ac94f65eb2dc8a62ba")
DB_PORT     = int(os.environ.get("PGPORT", "5432"))
DB_USER     = os.environ.get("PGUSER",     "nuvolos")
DB_PASSWORD = os.environ.get("PGPASSWORD", "nuvolos")
DB_NAME     = os.environ.get("PGDATABASE", "nuvolos")

TOP_K_PER_BRANCH = 8
VEC_WEIGHT       = 0.7
KW_WEIGHT        = 0.3
SCORE_THRESHOLD  = 0.35
MAX_AGENT_TURNS  = 6
MAX_HISTORY_TURNS = 10   # keep last N conversation turns; bounds context growth
MAX_NEW_TOKENS   = 1024
TEMPERATURE      = 0.1
MAX_EMBED_CHARS  = 6000   # mirrors embed.py — prevents Ollama 500s on long inputs
TABLE_DOCS       = "rag_documents"
TABLE_CHUNKS     = "rag_chunks"

BASH_DENY = [
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\b>\s*/dev/",
    r"\bshutdown\b",
    r"\breboot\b",
]

# Matches Qwen3 <think>…</think> blocks (enabled when the model reasons aloud)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ── Global singletons ──────────────────────────────────────────────────────────
llm_client: Optional[OpenAI] = None
db_conn = None
db_cur  = None

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a company knowledge assistant. You help users retrieve information from
company documents, emails, and chats (sources: confluence, google_drive, jira,
linear, hubspot, github, fireflies, gmail, slack).

Primary workflow:
1. Call `search` ONCE with a natural-language query. Each result includes doc_id,
   source_type, a preview, and a fused score. Use optional filters (source_types,
   date_from, date_to, participant) only when the user is explicit about who, when,
   or where.
2. Look at the top results. If the highest-scoring hit's preview clearly addresses
   the question, call `open_document` on its doc_id and answer from the full text.
   Score >= 2.0 is almost always a strong match — do not keep re-searching.
3. Only call `search` a SECOND time if (a) the opened document is clearly off-topic,
   or (b) you need a different piece of information from a different source.
   Never run more than 3 searches total for one question.
4. Always cite the doc_id(s) you used at the end of your answer, on a line like
   "Source: dsid_..." — copy the id verbatim, never invent one.
5. If nothing relevant is found, say so plainly — do not fabricate.

You also have read, write, edit, and bash tools on the Backend container
filesystem (/files/). Use them only when the user is clearly asking about local
files or wants to create/modify something, not for knowledge queries.

Keep responses concise. Quote relevant excerpts from documents rather than
paraphrasing when accuracy matters.
"""

# ── Tool definitions ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the company knowledge base using hybrid keyword + vector retrieval. "
                "Returns ranked chunks with a short preview and doc_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "source_types": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional filter: slack, gmail, linear, jira, confluence, google_drive, hubspot, github, fireflies.",
                    },
                    "date_from": {"type": "string", "description": "Optional ISO-8601 date lower bound."},
                    "date_to":   {"type": "string", "description": "Optional ISO-8601 date upper bound."},
                    "participant": {"type": "string", "description": "Optional email or Slack handle to filter by."},
                    "top_n": {"type": "integer", "description": "Number of results (default 6, max 20)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_document",
            "description": "Fetch the full text of a document by its doc_id (as returned by `search`).",
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
            "name": "read",
            "description": "Read a UTF-8 text file from the Backend container filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write UTF-8 text to a file, overwriting any existing content. Creates parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace the first exact occurrence of old_string with new_string in a file. Fails if old_string is not found or appears more than once.",
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
            "description": "Run a shell command on the Backend container via /bin/bash. Returns stdout, stderr, and exit code. Refuses dangerous patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."},
                },
                "required": ["command"],
            },
        },
    },
]


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, db_conn, db_cur

    log.info(f"Connecting to pgvector @ {DB_HOST}:{DB_PORT}/{DB_NAME}")
    kwargs: dict = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, dbname=DB_NAME)
    if DB_PASSWORD:
        kwargs["password"] = DB_PASSWORD
    db_conn = psycopg2.connect(**kwargs)
    db_conn.autocommit = True
    register_vector(db_conn)
    db_cur = db_conn.cursor()
    db_cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS};")
    n_chunks = db_cur.fetchone()[0]
    log.info(f"pgvector connected — {n_chunks} chunks indexed.")

    log.info(f"LLM: {OLLAMA_HOST}  model={LLM_MODEL}  embed={EMBED_MODEL}")
    llm_client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")
    log.info("Application startup complete.")

    yield

    if db_conn:
        db_conn.close()


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Company RAG Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class HistoryMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []

class Source(BaseModel):
    doc_id: str
    source_type: str
    title: Optional[str]
    score: float
    preview: str
    opened: bool = False   # True when the agent read the full document text

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float
    tool_traces: list[str] = []
    thinking_steps: list[str] = []


# ── Embedding (Ollama, same model as local pipeline) ───────────────────────────
def _embed(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text[:MAX_EMBED_CHARS]},
        timeout=60.0,
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


# ── Retrieval ──────────────────────────────────────────────────────────────────
def _build_filters(
    source_types: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    participant: str | None,
) -> tuple[list[str], list]:
    clauses: list[str] = []
    params: list = []
    if source_types:
        ph = ",".join(["%s"] * len(source_types))
        clauses.append(f"source_type IN ({ph})")
        params.extend(source_types)
    if date_from:
        clauses.append("(ts_to IS NULL OR ts_to >= %s)")
        params.append(date_from)
    if date_to:
        clauses.append("(ts_from IS NULL OR ts_from <= %s)")
        params.append(date_to)
    if participant:
        clauses.append("participants_json LIKE %s")
        params.append(f"%{participant}%")
    return clauses, params


def rag_search(
    query: str,
    source_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    participant: str | None = None,
    top_n: int = 6,
) -> list[dict]:
    top_n = max(1, min(20, top_n))
    qvec = _embed(query)
    filter_clauses, filter_params = _build_filters(source_types, date_from, date_to, participant)
    base_where = ("WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""

    # Vector branch
    db_cur.execute(
        f"""SELECT chunk_id, doc_id, source_type, text, ts_from, ts_to,
                   (embedding <=> %s::vector) AS dist
            FROM   {TABLE_CHUNKS} {base_where}
            ORDER  BY dist ASC LIMIT %s""",
        [qvec] + filter_params + [TOP_K_PER_BRANCH],
    )
    vec_scores: dict[int, tuple] = {}
    for chunk_id, doc_id, source_type, text, ts_from, ts_to, dist in db_cur.fetchall():
        vec_scores[chunk_id] = (doc_id, source_type, text, ts_from, ts_to, (1.0 - float(dist)) * 4)

    # Keyword branch (PostgreSQL FTS — ts_rank, not BM25, but best available on pgvector)
    kw_scores: dict[int, tuple] = {}
    try:
        kw_conditions = filter_clauses + ["text_tsv @@ plainto_tsquery('english', %s)"]
        kw_where = "WHERE " + " AND ".join(kw_conditions)
        db_cur.execute(
            f"""SELECT chunk_id, doc_id, source_type, text, ts_from, ts_to
                FROM   {TABLE_CHUNKS} {kw_where}
                ORDER  BY ts_rank(text_tsv, plainto_tsquery('english', %s)) DESC
                LIMIT  %s""",
            [query] + filter_params + [query, TOP_K_PER_BRANCH],
        )
        for i, (chunk_id, doc_id, source_type, text, ts_from, ts_to) in enumerate(db_cur.fetchall()):
            kw_scores[chunk_id] = (doc_id, source_type, text, ts_from, ts_to, (1 / (i + 2)) * 4)
    except Exception as e:
        log.warning(f"Keyword branch failed (vector-only fallback): {e}")
        db_conn.autocommit = True

    all_chunk_ids = set(vec_scores) | set(kw_scores)
    if not all_chunk_ids:
        return []

    all_doc_ids = list({d[0] for d in list(vec_scores.values()) + list(kw_scores.values())})
    ph = ",".join(["%s"] * len(all_doc_ids))
    db_cur.execute(f"SELECT doc_id, title FROM {TABLE_DOCS} WHERE doc_id IN ({ph})", all_doc_ids)
    titles: dict[str, str | None] = {row[0]: row[1] for row in db_cur.fetchall()}

    fused: list[dict] = []
    for cid in all_chunk_ids:
        vec_d = vec_scores.get(cid)
        kw_d  = kw_scores.get(cid)
        data  = vec_d or kw_d
        vec   = vec_d[5] if vec_d else 0.0
        kw    = kw_d[5]  if kw_d  else 0.0
        final = VEC_WEIGHT * vec + KW_WEIGHT * kw
        if final < SCORE_THRESHOLD:
            continue
        fused.append({
            "chunk_id":    cid,
            "doc_id":      data[0],
            "source_type": data[1],
            "title":       titles.get(data[0]),
            "score":       round(final, 3),
            "vec_score":   round(vec, 3),
            "kw_score":    round(kw, 3),
            "preview":     data[2][:320].replace("\n", " ").strip(),
            "ts_from":     data[3],
            "ts_to":       data[4],
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_n]


def fetch_document(doc_id: str) -> dict | None:
    db_cur.execute(
        f"SELECT doc_id, source_type, title, content FROM {TABLE_DOCS} WHERE doc_id = %s",
        (doc_id,),
    )
    row = db_cur.fetchone()
    if not row:
        return None
    return {"doc_id": row[0], "source_type": row[1], "title": row[2], "content": row[3]}


# ── Tool execution ─────────────────────────────────────────────────────────────
def execute_tool(name: str, args: dict) -> tuple[str, list[dict]]:
    """Returns (text_for_llm, search_hits_for_sources)."""

    if name == "search":
        hits = rag_search(
            query=args["query"],
            source_types=args.get("source_types"),
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            participant=args.get("participant"),
            top_n=args.get("top_n", 6),
        )
        if not hits:
            return "No results above threshold. Try broadening the query or removing filters.", []
        lines = []
        for h in hits:
            ts = f" [{h['ts_from']}]" if h.get("ts_from") else ""
            lines.append(
                f"chunk_id={h['chunk_id']}  doc_id={h['doc_id']}  "
                f"source={h['source_type']}{ts}  score={h['score']}\n"
                f"title: {h['title'] or ''}\npreview: {h['preview']}"
            )
        return "\n\n".join(lines), hits

    if name == "open_document":
        doc = fetch_document(args["doc_id"])
        if not doc:
            return f"No document found with doc_id={args['doc_id']}", []
        source_hit = {
            "chunk_id": -1,
            "doc_id": doc["doc_id"],
            "source_type": doc["source_type"],
            "title": doc["title"],
            "score": 0.0,
            "vec_score": 0.0,
            "kw_score": 0.0,
            "preview": doc["content"][:320].replace("\n", " ").strip(),
            "ts_from": None,
            "ts_to": None,
            "opened": True,
        }
        return (
            f"doc_id: {doc['doc_id']}\nsource: {doc['source_type']}\n"
            f"title: {doc['title'] or ''}\n\n{doc['content']}"
        ), [source_hit]

    if name == "read":
        path = args["path"]
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), []

    if name == "write":
        path = args["path"]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return f"Wrote {len(args['content'])} bytes to {path}", []

    if name == "edit":
        path = args["path"]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        old_s, new_s = args["old_string"], args["new_string"]
        count = text.count(old_s)
        if count == 0:
            return f"Error: old_string not found in {path}", []
        if count > 1:
            return f"Error: old_string appears {count} times in {path} — make it more specific", []
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old_s, new_s, 1))
        return f"Edited {path}", []

    if name == "bash":
        command = args["command"]
        for pattern in BASH_DENY:
            if re.search(pattern, command):
                return f"Refused: command matches deny pattern '{pattern}'. Run it manually if needed.", []
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, executable="/bin/bash",
        )
        output = f"exit={result.returncode}\n--- stdout ---\n{result.stdout}"
        if result.stderr:
            output += f"\n--- stderr ---\n{result.stderr}"
        return output, []

    return f"Unknown tool: {name}", []


# ── Thinking helpers ───────────────────────────────────────────────────────────
def _extract_thinking(content: str) -> tuple[str, str]:
    """Split Qwen3 <think>…</think> blocks from visible content.
    Returns (thinking_text, clean_content). If no <think> blocks, thinking_text is ''."""
    thoughts: list[str] = []
    clean = THINK_RE.sub(lambda m: thoughts.append(m.group(1).strip()) or "", content)
    return "\n\n".join(thoughts), clean.strip()


# ── Trace helpers ──────────────────────────────────────────────────────────────
def _trace_args(name: str, args: dict) -> str:
    if name == "search":
        q = args.get("query", "")[:60]
        extras: list[str] = []
        if args.get("source_types"):
            extras.append(f"source={args['source_types']}")
        if args.get("participant"):
            extras.append(f"participant={args['participant']}")
        if args.get("date_from") or args.get("date_to"):
            extras.append(f"date={args.get('date_from', '')}..{args.get('date_to', '')}")
        return f'"{q}"' + (f" ({', '.join(extras)})" if extras else "")
    if name == "open_document":
        return args.get("doc_id", "")
    if name in ("read", "write", "edit"):
        return args.get("path", "")
    if name == "bash":
        return args.get("command", "")[:80]
    return ""


def _trace_result(name: str, result_text: str, hits: list) -> str:
    if name == "search":
        return f"{len(hits)} result(s)" if hits else "no results"
    if name == "open_document":
        return "not found" if result_text.startswith("No document") else f"{len(result_text):,} chars"
    if name == "bash":
        m = re.match(r"exit=(\d+)", result_text)
        return f"exit={m.group(1)}" if m else "done"
    if result_text.startswith("Error"):
        return result_text[:60]
    return "done"


# ── Agent loop ─────────────────────────────────────────────────────────────────
def run_agent(
    question: str, history: list[HistoryMessage]
) -> tuple[str, list[dict], list[str], list[str]]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-MAX_HISTORY_TURNS * 2:]:   # each turn = 2 messages (user + assistant)
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": question})

    all_sources: list[dict] = []
    traces: list[str] = []
    thinking_steps: list[str] = []

    for _ in range(MAX_AGENT_TURNS):
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            extra_body={"options": {"num_ctx": 32768}},
        )
        msg = resp.choices[0].message
        raw_content = msg.content or ""

        # Extract <think>…</think> blocks — present when Qwen3 reasoning is active
        think_text, visible_content = _extract_thinking(raw_content)
        if think_text:
            thinking_steps.append(think_text)

        if not msg.tool_calls:
            # Any remaining visible content before the answer counts as reasoning
            return visible_content.strip(), all_sources, traces, thinking_steps

        # Capture inter-turn reasoning text (what the model says before calling tools)
        if visible_content.strip():
            thinking_steps.append(visible_content.strip())

        messages.append({
            "role":       "assistant",
            "content":    raw_content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args_parsed = json.loads(tc.function.arguments)
            try:
                result_text, hits = execute_tool(tc.function.name, args_parsed)
                all_sources.extend(hits)
                summary = _trace_result(tc.function.name, result_text, hits)
            except Exception as e:
                result_text = f"Tool error: {e}"
                summary = f"error: {str(e)[:50]}"
            traces.append(
                f"[{tc.function.name}] {_trace_args(tc.function.name, args_parsed)} → {summary}"
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

    return "Agent reached the maximum number of turns without a final answer.", all_sources, traces, thinking_steps


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL, "embed": EMBED_MODEL, "ollama": OLLAMA_HOST}


@app.get("/stats")
def stats():
    db_cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS};")
    n_chunks = db_cur.fetchone()[0]
    db_cur.execute(f"SELECT COUNT(*) FROM {TABLE_DOCS};")
    n_docs = db_cur.fetchone()[0]
    db_cur.execute(
        f"SELECT source_type, COUNT(*) FROM {TABLE_CHUNKS} "
        f"GROUP BY source_type ORDER BY COUNT(*) DESC;"
    )
    return {"chunks": n_chunks, "documents": n_docs,
            "by_source": {r[0]: r[1] for r in db_cur.fetchall()}}


@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    doc = fetch_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return doc


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    t0 = time.perf_counter()
    answer, raw_sources, traces, thinking = run_agent(req.question, req.history)
    latency = round((time.perf_counter() - t0) * 1000, 1)

    # Keep best search hit per doc_id; opened documents are always included
    best: dict[str, dict] = {}
    for s in raw_sources:
        if s.get("opened"):
            best[s["doc_id"]] = s          # opened always wins — it was actually read
        elif s["doc_id"] not in best or s["score"] > best[s["doc_id"]]["score"]:
            best[s["doc_id"]] = s

    # Opened docs first, then search hits sorted by score descending
    opened  = [s for s in best.values() if s.get("opened")]
    found   = sorted([s for s in best.values() if not s.get("opened")],
                     key=lambda x: x["score"], reverse=True)

    return QueryResponse(
        answer=answer,
        sources=[
            Source(doc_id=s["doc_id"], source_type=s["source_type"],
                   title=s.get("title"), score=s["score"], preview=s["preview"],
                   opened=s.get("opened", False))
            for s in opened + found
        ],
        latency_ms=latency,
        tool_traces=traces,
        thinking_steps=thinking,
    )
