import math
import re
from config import BM25_K1, BM25_B


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z]\w*", text.lower()) if len(w) > 1]


def fts_or_query(q: str) -> str | None:
    """OR-join words for a PostgreSQL to_tsquery — any-word matching for maximum recall."""
    words = tokenize(q)
    return " | ".join(words) if words else None


def bm25_scores(query_terms: list[str], candidates: list[tuple[int, str]]) -> dict[int, float]:
    if not candidates or not query_terms:
        return {}
    tokenized = [(cid, tokenize(text)) for cid, text in candidates]
    lengths   = [len(toks) for _, toks in tokenized]
    avgdl     = sum(lengths) / len(lengths) if lengths else 1
    N         = len(tokenized)

    df: dict[str, int] = {}
    for _, toks in tokenized:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1

    result: dict[int, float] = {}
    for cid, toks in tokenized:
        dl = len(toks)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term in query_terms:
            if term not in df:
                continue
            f   = tf.get(term, 0)
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            score += idf * (f * (BM25_K1 + 1)) / (
                f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
            )
        result[cid] = score
    return result
