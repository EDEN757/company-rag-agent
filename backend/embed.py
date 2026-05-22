import httpx
import numpy as np
from config import OLLAMA_HOST, EMBED_MODEL, MAX_EMBED_CHARS


def _normalize(vec: list) -> list[float]:
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v.tolist()


def embed(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text[:MAX_EMBED_CHARS]},
        timeout=60.0,
    )
    resp.raise_for_status()
    return _normalize(resp.json()["embedding"])


def batch_embed(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one Ollama round-trip via /api/embed."""
    if not texts:
        return []
    if len(texts) == 1:
        return [embed(texts[0])]
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": [t[:MAX_EMBED_CHARS] for t in texts]},
        timeout=max(60.0, 30.0 * len(texts)),
    )
    resp.raise_for_status()
    return [_normalize(v) for v in resp.json()["embeddings"]]
