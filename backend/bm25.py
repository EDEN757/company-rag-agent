import logging
import math
import re

from config import BM25_K1, BM25_B, TABLE_CHUNKS

log = logging.getLogger(__name__)

# Corpus-level stats for correct IDF computation.
# Populated by init_corpus_stats() at startup; falls back to batch estimates if not called.
_corpus: dict = {"N": 1, "avgdl": 300.0, "df": {}}


def init_corpus_stats() -> None:
    """Cache N, avgdl, and per-term df by scanning the corpus with our tokenizer.

    Uses a server-side cursor to stream rows in batches so the full text column
    is never loaded into Python memory at once.
    Corpus df keys match tokenize() output exactly, so IDF lookups always hit.
    Call once at startup (after db.connect()) and again after large corpus changes.
    """
    import db  # local import avoids circular-import at module load time
    global _corpus
    df: dict[str, int] = {}
    total_dl = 0
    N = 0
    # Named cursor keeps results on the server, streaming itersize rows at a time.
    # Must run inside a transaction (autocommit=False).
    with db.transaction() as conn:
        with conn.cursor("_bm25_init") as cur:
            cur.itersize = 2000
            cur.execute(f"SELECT text FROM {TABLE_CHUNKS}")
            for (text,) in cur:
                tokens = tokenize(text)
                total_dl += len(tokens)
                N += 1
                for t in set(tokens):
                    df[t] = df.get(t, 0) + 1

    avgdl = (total_dl / N) if N > 0 else 300.0
    _corpus = {"N": max(N, 1), "avgdl": avgdl, "df": df}
    log.info(f"BM25 corpus stats: N={N}, avgdl={avgdl:.1f} tokens, vocab={len(df)} terms")


def corpus_add_chunks(texts: list[str]) -> None:
    """Incrementally update corpus stats after new chunks are successfully written."""
    global _corpus
    if not texts:
        return
    old_N   = _corpus["N"]
    avgdl   = _corpus["avgdl"]
    df      = _corpus["df"]
    total_dl = 0
    for text in texts:
        tokens = tokenize(text)
        total_dl += len(tokens)
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    new_N = old_N + len(texts)
    _corpus["N"]     = new_N
    _corpus["avgdl"] = (avgdl * old_N + total_dl) / new_N


def corpus_remove_chunks(texts: list[str]) -> None:
    """Incrementally update corpus stats when chunks are about to be deleted."""
    global _corpus
    if not texts:
        return
    old_N   = _corpus["N"]
    avgdl   = _corpus["avgdl"]
    df      = _corpus["df"]
    total_dl = 0
    for text in texts:
        tokens = tokenize(text)
        total_dl += len(tokens)
        for t in set(tokens):
            new_count = df.get(t, 1) - 1
            if new_count <= 0:
                df.pop(t, None)
            else:
                df[t] = new_count
    remaining = max(1, old_N - len(texts))
    _corpus["N"]     = remaining
    if old_N > len(texts):
        _corpus["avgdl"] = max(1.0, (avgdl * old_N - total_dl) / remaining)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9]\w*", text.lower()) if len(w) > 1]


def fts_or_query(q: str) -> str | None:
    """OR-join words for a PostgreSQL to_tsquery — any-word matching for maximum recall."""
    words = tokenize(q)
    return " | ".join(words) if words else None


def bm25_scores(query_terms: list[str], candidates: list[tuple[int, str]]) -> dict[int, float]:
    """Score (chunk_id, text) pairs using corpus-level IDF where available."""
    if not candidates or not query_terms:
        return {}

    tokenized = [(cid, tokenize(text)) for cid, text in candidates]
    lengths = [len(toks) for _, toks in tokenized]

    # Use corpus stats if populated; fall back to batch estimates
    N = max(_corpus["N"], len(tokenized))
    avgdl = _corpus["avgdl"] if _corpus["avgdl"] > 0 else (
        sum(lengths) / len(lengths) if lengths else 1.0
    )
    corpus_df = _corpus["df"]

    # Batch df as fallback for terms absent from corpus_df (e.g. before init_corpus_stats)
    batch_df: dict[str, int] = {}
    for _, toks in tokenized:
        for term in set(toks):
            batch_df[term] = batch_df.get(term, 0) + 1

    result: dict[int, float] = {}
    for cid, toks in tokenized:
        dl = len(toks)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            df_val = corpus_df.get(term) or batch_df.get(term)
            if not df_val:
                continue
            idf = math.log((N - df_val + 0.5) / (df_val + 0.5) + 1)
            score += idf * (f * (BM25_K1 + 1)) / (
                f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
            )
        result[cid] = score
    return result
