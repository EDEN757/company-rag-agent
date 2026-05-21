import logging
import db
from embed import embed
from bm25 import fts_or_query, bm25_scores, tokenize
from config import (
    TOP_K_PER_BRANCH, VEC_WEIGHT, KW_WEIGHT,
    SCORE_THRESHOLD, SCORE_SCALE, TABLE_CHUNKS, TABLE_DOCS,
)

log = logging.getLogger(__name__)


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
    qvec = embed(query)
    filter_clauses, filter_params = _build_filters(source_types, date_from, date_to, participant)
    base_where = ("WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""

    # Vector branch
    db.cur.execute(
        f"""SELECT chunk_id, doc_id, source_type, text, ts_from, ts_to,
                   (embedding <=> %s::vector) AS dist
            FROM   {TABLE_CHUNKS} {base_where}
            ORDER  BY dist ASC LIMIT %s""",
        [qvec] + filter_params + [TOP_K_PER_BRANCH],
    )
    vec_scores: dict[int, tuple] = {}
    for chunk_id, doc_id, source_type, text, ts_from, ts_to, dist in db.cur.fetchall():
        vec_scores[chunk_id] = (doc_id, source_type, text, ts_from, ts_to,
                                (1.0 - float(dist)) * SCORE_SCALE)

    # Keyword branch — OR-joined FTS candidates, reranked by BM25
    kw_scores: dict[int, tuple] = {}
    fts_q = fts_or_query(query)
    if fts_q:
        try:
            kw_conditions = filter_clauses + ["text_tsv @@ to_tsquery('english', %s)"]
            kw_where = "WHERE " + " AND ".join(kw_conditions)
            db.cur.execute(
                f"""SELECT chunk_id, doc_id, source_type, text, ts_from, ts_to
                    FROM   {TABLE_CHUNKS} {kw_where}
                    LIMIT  %s""",
                filter_params + [fts_q, TOP_K_PER_BRANCH * 3],
            )
            rows = db.cur.fetchall()
            query_terms = tokenize(query)
            bm25 = bm25_scores(query_terms, [(r[0], r[3]) for r in rows])
            ranked = sorted(rows, key=lambda r: bm25.get(r[0], 0.0), reverse=True)[:TOP_K_PER_BRANCH]
            for rank, (chunk_id, doc_id, source_type, text, ts_from, ts_to) in enumerate(ranked, 1):
                kw_scores[chunk_id] = (doc_id, source_type, text, ts_from, ts_to,
                                       (1 / (1 + rank)) * SCORE_SCALE)
        except Exception as e:
            log.warning(f"Keyword branch failed (vector-only fallback): {e}")
            db.conn.autocommit = True

    all_chunk_ids = set(vec_scores) | set(kw_scores)
    if not all_chunk_ids:
        return []

    all_doc_ids = list({d[0] for d in list(vec_scores.values()) + list(kw_scores.values())})
    ph = ",".join(["%s"] * len(all_doc_ids))
    db.cur.execute(f"SELECT doc_id, title FROM {TABLE_DOCS} WHERE doc_id IN ({ph})", all_doc_ids)
    titles: dict[str, str | None] = {row[0]: row[1] for row in db.cur.fetchall()}

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
            "preview":     data[2][:600].replace("\n", " ").strip(),
            "ts_from":     data[3],
            "ts_to":       data[4],
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_n]
