import httpx
import numpy as np
from config import OLLAMA_HOST, EMBED_MODEL, MAX_EMBED_CHARS


def embed(text: str) -> list[float]:
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
