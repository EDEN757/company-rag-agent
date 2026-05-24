"""Cross-encoder reranking service.

Loads cross-encoder/ms-marco-MiniLM-L-6-v2 once on startup (CPU-pinned to
avoid fighting Ollama for GPU memory) and exposes a single /rerank endpoint.

Run with:
    uvicorn indexing.reranker:app --port 8001

Or from the repo root:
    python -m uvicorn indexing.reranker:app --port 8001

Dependencies (add to the data/.venv):
    pip install sentence-transformers fastapi uvicorn
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

MODEL_NAME = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_model: CrossEncoder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print(f"[reranker] loading {MODEL_NAME} on CPU …")
    _model = CrossEncoder(MODEL_NAME, device="cpu", max_length=512)
    print("[reranker] ready")
    yield


app = FastAPI(lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    passages: list[str]


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict:
    if not req.passages or _model is None:
        return {"scores": []}
    pairs = [(req.query, p) for p in req.passages]
    scores = _model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
    return {"scores": scores.tolist()}


@app.get("/health")
def health() -> dict:
    return {"ok": _model is not None}
