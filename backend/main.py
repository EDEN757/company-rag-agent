"""
backend/main.py — Company RAG Agent API (Nuvolos)
==================================================
Thin FastAPI entry point. Business logic lives in:
  config.py   — env vars and constants
  prompt.py   — system prompt and tool schemas
  db.py       — PostgreSQL connection helpers
  embed.py    — Ollama embedding
  bm25.py     — BM25 tokenize / score helpers
  fusion.py   — hybrid vector + keyword search
  kb.py       — knowledge-base write helpers
  tools.py    — tool dispatch
  agent.py    — LLM agent loop

Run:
    cd /files/company-rag-agent/backend
    uvicorn main:app --host 0.0.0.0 --port 8500
"""

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import agent
import db
import hyde
import reranker
from config import OLLAMA_HOST, LLM_MODEL, EMBED_MODEL, TABLE_CHUNKS, TABLE_DOCS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.connect()
    agent.init_client()
    hyde.init_hyde()
    reranker.init_reranker()
    log.info(f"LLM: {OLLAMA_HOST}  model={LLM_MODEL}  embed={EMBED_MODEL}")
    log.info("Application startup complete.")
    yield
    db.disconnect()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Company RAG Agent API", version="1.0.0", lifespan=lifespan)

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
    thinking_mode: bool = False


class Source(BaseModel):
    doc_id: str
    source_type: str
    title: Optional[str]
    score: float
    preview: str
    opened: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float
    tool_traces: list[str] = []
    thinking_steps: list[str] = []


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL, "embed": EMBED_MODEL, "ollama": OLLAMA_HOST}


@app.get("/stats")
def stats():
    db.cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS};")
    n_chunks = db.cur.fetchone()[0]
    db.cur.execute(f"SELECT COUNT(*) FROM {TABLE_DOCS};")
    n_docs = db.cur.fetchone()[0]
    db.cur.execute(
        f"SELECT source_type, COUNT(*) FROM {TABLE_CHUNKS} "
        f"GROUP BY source_type ORDER BY COUNT(*) DESC;"
    )
    return {
        "chunks":    n_chunks,
        "documents": n_docs,
        "by_source": {r[0]: r[1] for r in db.cur.fetchall()},
    }


@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    doc = db.fetch_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return doc


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """SSE endpoint — yields events as the agent runs so the UI can update in real time."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    history = [h.model_dump() for h in req.history]

    def event_gen():
        for event in agent.run_agent_streaming(req.question, history, req.thinking_mode):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    t0 = time.perf_counter()
    history = [h.model_dump() for h in req.history]
    answer, raw_sources, traces, thinking = agent.run_agent(
        req.question, history, req.thinking_mode
    )
    latency = round((time.perf_counter() - t0) * 1000, 1)

    best: dict[str, dict] = {}
    for s in raw_sources:
        if s.get("opened"):
            best[s["doc_id"]] = s
        elif s["doc_id"] not in best or s["score"] > best[s["doc_id"]]["score"]:
            best[s["doc_id"]] = s

    opened = [s for s in best.values() if s.get("opened")]
    found  = sorted(
        [s for s in best.values() if not s.get("opened")],
        key=lambda x: x["score"], reverse=True,
    )

    return QueryResponse(
        answer=answer,
        sources=[
            Source(
                doc_id=s["doc_id"], source_type=s["source_type"],
                title=s.get("title"), score=s["score"], preview=s["preview"],
                opened=s.get("opened", False),
            )
            for s in opened + found
        ],
        latency_ms=latency,
        tool_traces=traces,
        thinking_steps=thinking,
    )
