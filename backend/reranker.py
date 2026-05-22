import logging
from config import RERANKER_MODEL

log = logging.getLogger(__name__)
_model = None


def init_reranker():
    global _model
    if not RERANKER_MODEL:
        log.info("Reranker disabled (RERANKER_MODEL is empty).")
        return
    try:
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading cross-encoder reranker: {RERANKER_MODEL} (device={device})")
        _model = CrossEncoder(RERANKER_MODEL, max_length=512, device=device)
        log.info("Cross-encoder reranker ready.")
    except Exception as e:
        log.warning(f"Could not load reranker ({e}) — falling back to fusion order.")


def rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    """Score (query, chunk_text) pairs and return top_n by cross-encoder score.

    Falls back to fusion order if the model is not loaded.
    Each hit must carry a '_text' key with the full chunk text.
    """
    if _model is None or not hits:
        return hits[:top_n]
    pairs = [(query, h.get("_text") or h["preview"]) for h in hits]
    scores = _model.predict(pairs, show_progress_bar=False)
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    return sorted(hits, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
