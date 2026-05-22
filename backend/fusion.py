import concurrent.futures
import logging

import db
import hyde
import reranker
from embed import embed
from bm25 import fts_or_query, bm25_scores, tokenize
from config import (
    TOP_K_PER_BRANCH, VEC_WEIGHT, KW_WEIGHT,
    SCORE_THRESHOLD, SCORE_SCALE, TABLE_CHUNKS, TABLE_DOCS,
    USE_RRF, RRF_K, HYDE_ENABLED,
)

log = logging.getLogger(__name__)

# Module-level thread pool reused across requests (HyDE runs here, BM25 on caller thread)
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-io")


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


def search(
    query: str,
    source_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    participant: str | None = None,
    top_n: int = 6,
) -> list[dict]:
    top_n = max(1, min(20, top_n))
    filter_clauses, filter_params = _build_filters(source_types, date_from, date_to, participant)
    base_where = ("WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""

    # Submit HyDE to the thread pool immediately — it only calls Ollama HTTP, no DB access.
    # BM25 runs on the caller thread concurrently while HyDE generates the hypothetical doc.
    embed_fn = hyde.hyde_embed if HYDE_ENABLED else embed
    hyde_future = _pool.submit(embed_fn, query)

    kw_scores: dict[int, tuple] = {}
    kw_ranks:  dict[int, int]   = {}
    fts_q = fts_or_query(query)

    with db.cursor() as cur:
        # ── Keyword branch (BM25) ────────────────────────────────────────────
        # Runs on the caller thread while HyDE computes in the background.
        if fts_q:
            try:
                kw_conditions = filter_clauses + ["text_tsv @@ to_tsquery('english', %s)"]
                kw_where = "WHERE " + " AND ".join(kw_conditions)
                cur.execute(
                    f"""SELECT chunk_id, doc_id, source_type, text, ts_from, ts_to
                        FROM   {TABLE_CHUNKS} {kw_where}
                        LIMIT  %s""",
                    filter_params + [fts_q, TOP_K_PER_BRANCH * 3],
                )
                rows = cur.fetchall()
                query_terms = tokenize(query)
                bm25 = bm25_scores(query_terms, [(r[0], r[3]) for r in rows])
                ranked = sorted(rows, key=lambda r: bm25.get(r[0], 0.0), reverse=True)[:TOP_K_PER_BRANCH]
                for rank, (cid, doc_id, st, text, ts_from, ts_to) in enumerate(ranked, 1):
                    kw_ranks[cid]  = rank
                    kw_scores[cid] = (doc_id, st, text, ts_from, ts_to,
                                      (1 / (1 + rank)) * SCORE_SCALE)
            except Exception as e:
                log.warning(f"Keyword branch failed (vector-only fallback): {e}")

        # ── Wait for HyDE (likely already done while BM25 was running) ───────
        try:
            qvec = hyde_future.result()
        except Exception as e:
            log.warning(f"HyDE/embed failed ({e}) — falling back to plain query embed.")
            qvec = embed(query)

        # ── Vector branch (title fetched via JOIN — saves a round-trip) ────────
        vec_scores: dict[int, tuple] = {}
        vec_ranks:  dict[int, int]   = {}
        titles: dict[str, str | None] = {}
        # All filter columns (source_type, ts_from, ts_to, participants_json) exist
        # only in TABLE_CHUNKS so no table alias is needed in the WHERE clause.
        cur.execute(
            f"""SELECT c.chunk_id, c.doc_id, c.source_type, c.text, c.ts_from, c.ts_to,
                       (c.embedding <=> %s::vector) AS dist, d.title
                FROM   {TABLE_CHUNKS} c
                LEFT JOIN {TABLE_DOCS} d ON c.doc_id = d.doc_id
                {base_where}
                ORDER  BY dist ASC LIMIT %s""",
            [qvec] + filter_params + [TOP_K_PER_BRANCH],
        )
        for rank, (cid, doc_id, st, text, ts_from, ts_to, dist, title) in \
                enumerate(cur.fetchall(), 1):
            vec_ranks[cid]  = rank
            vec_scores[cid] = (doc_id, st, text, ts_from, ts_to,
                               (1.0 - float(dist)) * SCORE_SCALE)
            titles[doc_id]  = title

        # ── Title lookup — BM25-only doc_ids not covered by the vector JOIN ───
        all_chunk_ids = set(vec_scores) | set(kw_scores)
        if not all_chunk_ids:
            return []

        vec_doc_ids = {d[0] for d in vec_scores.values()}
        bm25_only_doc_ids = list({d[0] for d in kw_scores.values()} - vec_doc_ids)
        if bm25_only_doc_ids:
            ph = ",".join(["%s"] * len(bm25_only_doc_ids))
            cur.execute(
                f"SELECT doc_id, title FROM {TABLE_DOCS} WHERE doc_id IN ({ph})",
                bm25_only_doc_ids,
            )
            for row in cur.fetchall():
                titles[row[0]] = row[1]

    # ── Fuse ─────────────────────────────────────────────────────────────────
    fused: list[dict] = []
    for cid in all_chunk_ids:
        vec_d = vec_scores.get(cid)
        kw_d  = kw_scores.get(cid)
        data  = vec_d or kw_d
        vec   = vec_d[5] if vec_d else 0.0
        kw    = kw_d[5]  if kw_d  else 0.0
        fusion_score = VEC_WEIGHT * vec + KW_WEIGHT * kw
        # When RRF is sorting, ranking handles relevance — threshold would cut valid results.
        # When weighted fusion is sorting, threshold trims noise.
        if not USE_RRF and fusion_score < SCORE_THRESHOLD:
            continue
        rrf_score = (
            (1.0 / (RRF_K + vec_ranks[cid]) if cid in vec_ranks else 0.0) +
            (1.0 / (RRF_K + kw_ranks[cid])  if cid in kw_ranks  else 0.0)
        )
        fused.append({
            "chunk_id":    cid,
            "doc_id":      data[0],
            "source_type": data[1],
            "title":       titles.get(data[0]),
            "score":       round(fusion_score, 3),
            "vec_score":   round(vec, 3),
            "kw_score":    round(kw, 3),
            "preview":     data[2][:600].replace("\n", " ").strip(),
            "ts_from":     data[3],
            "ts_to":       data[4],
            "_rrf":        rrf_score,
            "_text":       data[2],
        })

    sort_key = (lambda x: x["_rrf"]) if USE_RRF else (lambda x: x["score"])
    fused.sort(key=sort_key, reverse=True)

    # ── Deduplicate: keep the best-scoring chunk per doc_id ──────────────────
    seen: set[str] = set()
    deduped: list[dict] = []
    for chunk in fused:
        if chunk["doc_id"] not in seen:
            seen.add(chunk["doc_id"])
            deduped.append(chunk)

    reranked = reranker.rerank(query, deduped, top_n)
    for h in reranked:
        h.pop("_rrf",  None)
        h.pop("_text", None)
    return reranked
