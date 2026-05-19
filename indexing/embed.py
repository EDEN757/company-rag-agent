"""Ollama embedding client. Returns L2-normalized float32 vectors so that
cosine similarity reduces to a plain dot product at search time."""

from __future__ import annotations

import os
import time
from typing import Callable, Iterable, Optional

import httpx
import numpy as np

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
DIM = 768

# nomic-embed-text has an 8192-token context. We pass ~6000 chars (~1500
# tokens) max to stay well below it and to give Ollama some headroom — at
# very long inputs it has been observed to 500.
MAX_CHARS = 6000


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def embed_one(client: httpx.Client, text: str, retries: int = 5) -> np.ndarray:
    payload_text = text if len(text) <= MAX_CHARS else text[:MAX_CHARS]
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = client.post(
                f"{OLLAMA_BASE}/api/embeddings",
                json={"model": MODEL, "prompt": payload_text},
                timeout=120.0,
            )
            r.raise_for_status()
            vec = np.asarray(r.json()["embedding"], dtype=np.float32)
            if vec.shape[0] != DIM:
                raise RuntimeError(f"Unexpected embedding dim {vec.shape[0]} (expected {DIM})")
            return _normalize(vec)
        except (httpx.HTTPStatusError, httpx.HTTPError, RuntimeError) as e:
            last_err = e
            # On 500, halve the payload and back off — usually a transient
            # server issue or an oversize prompt the model rejected.
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (500, 503):
                payload_text = payload_text[: max(500, len(payload_text) // 2)]
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"embed_one failed after {retries} retries: {last_err}")


def stream_embeddings(
    items: Iterable[tuple],
    text_index: int,
    on_result: Callable[[tuple, np.ndarray], None],
    progress_every: int = 200,
) -> None:
    """Iterate `items` (arbitrary tuples). Embed the element at `text_index`.
    Call `on_result(item, vec)` for each. Caller is responsible for
    committing to DB so we get checkpointing."""
    with httpx.Client(http2=False, timeout=120.0) as client:
        t0 = time.time()
        n = 0
        for item in items:
            text = item[text_index]
            vec = embed_one(client, text)
            on_result(item, vec)
            n += 1
            if n % progress_every == 0:
                dt = max(time.time() - t0, 0.001)
                print(f"        embedded {n}  ({n/dt:.1f} chunks/s)")
