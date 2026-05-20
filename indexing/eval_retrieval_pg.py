"""Evaluate the hybrid retriever against questions_test.parquet (pgvector version).

Nuvolos counterpart to eval_retrieval.py — same fusion math and metrics, but
queries PostgreSQL/pgvector instead of SQLite.

Run from the Backend VS Code app on Nuvolos:
    cd /files
    python indexing/eval_retrieval_pg.py \
        --questions data/raw/questions_test.parquet \
        --top-k     10

For a quick sanity check on the first 100 questions:
    python indexing/eval_retrieval_pg.py \
        --questions data/raw/questions_test.parquet \
        --limit 100

Environment variables (same defaults as backend):
    PGHOST / PGPORT / PGUSER / PGPASSWORD / PGDATABASE
    OLLAMA_HOST        http://localhost:11434
    RAG_EMBED_MODEL    nomic-embed-text
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
import psycopg2
from pgvector.psycopg2 import register_vector
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embed import DIM, MODEL, embed_one  # noqa: E402

# ── Configuration (mirrors backend/main.py) ────────────────────────────────────
DB_HOST     = os.environ.get("PGHOST",     "nv-service-b01d63337fab32ac94f65eb2dc8a62ba")
DB_PORT     = int(os.environ.get("PGPORT", "5432"))
DB_USER     = os.environ.get("PGUSER",     "nuvolos")
DB_PASSWORD = os.environ.get("PGPASSWORD", "nuvolos")
DB_NAME     = os.environ.get("PGDATABASE", "nuvolos")

TABLE_CHUNKS = "rag_chunks"

TOP_K_PER_BRANCH = 8
KW_W      = 0.3
VEC_W     = 0.7
SCALE     = 4
THRESHOLD = 0.35


# ── DB ─────────────────────────────────────────────────────────────────────────
def connect():
    kwargs: dict = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, dbname=DB_NAME)
    if DB_PASSWORD:
        kwargs["password"] = DB_PASSWORD
    conn = psycopg2.connect(**kwargs)
    conn.autocommit = True
    register_vector(conn)
    return conn, conn.cursor()


# ── Search (same fusion as backend/main.py and eval_retrieval.py) ──────────────
def search(cur, client: httpx.Client, query: str, top_n: int = 10) -> list[str]:
    qvec = embed_one(client, query)

    # Vector branch — cosine distance via pgvector <=> operator
    cur.execute(
        f"""SELECT chunk_id, doc_id, (embedding <=> %s::vector) AS dist
            FROM   {TABLE_CHUNKS}
            WHERE  embedding IS NOT NULL
            ORDER  BY dist ASC
            LIMIT  %s""",
        (qvec.tolist(), TOP_K_PER_BRANCH),
    )
    vec_scores: dict[int, tuple[str, float]] = {}
    for chunk_id, doc_id, dist in cur.fetchall():
        vec_scores[chunk_id] = (doc_id, (1.0 - float(dist)) * SCALE)

    # Keyword branch — tsvector + ts_rank (PostgreSQL equivalent of FTS5 bm25)
    # Rank starts at 1 to match eval_retrieval.py: (1 / (1 + rank)) * SCALE
    kw_scores: dict[int, tuple[str, float]] = {}
    cur.execute(
        f"""SELECT chunk_id, doc_id
            FROM   {TABLE_CHUNKS}
            WHERE  text_tsv @@ plainto_tsquery('english', %s)
            ORDER  BY ts_rank(text_tsv, plainto_tsquery('english', %s)) DESC
            LIMIT  %s""",
        (query, query, TOP_K_PER_BRANCH),
    )
    for rank, (chunk_id, doc_id) in enumerate(cur.fetchall(), start=1):
        kw_scores[chunk_id] = (doc_id, (1 / (1 + rank)) * SCALE)

    # Fusion
    all_ids = set(vec_scores) | set(kw_scores)
    fused: list[tuple[str, float]] = []
    for cid in all_ids:
        doc_id = (vec_scores.get(cid) or kw_scores[cid])[0]
        vec   = vec_scores[cid][1] if cid in vec_scores else 0.0
        kw    = kw_scores[cid][1]  if cid in kw_scores  else 0.0
        final = VEC_W * vec + KW_W * kw
        if final >= THRESHOLD:
            fused.append((doc_id, final))

    fused.sort(key=lambda x: -x[1])

    # Deduplicate to doc_ids, preserving best-chunk-first order
    seen: list[str] = []
    for doc_id, _ in fused[: top_n * 3]:
        if doc_id not in seen:
            seen.append(doc_id)
        if len(seen) >= top_n:
            break
    return seen


# ── Metrics ────────────────────────────────────────────────────────────────────
def ndcg(predicted: list[str], expected: set[str], k: int) -> float:
    dcg  = sum(1 / math.log2(i + 2) for i, d in enumerate(predicted[:k]) if d in expected)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(expected))))
    return dcg / ideal if ideal > 0 else 0.0


def parse_expected(s) -> list[str]:
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x) for x in s]
    text = str(s).strip().replace("'", '"')
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate hybrid retriever on pgvector.")
    ap.add_argument("--questions", required=True, help="Path to questions_test.parquet")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate first N questions only (debug)")
    args = ap.parse_args()

    conn, cur = connect()
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS} WHERE embedding IS NOT NULL;")
    n_chunks = cur.fetchone()[0]
    print(f"[db   ] connected — {n_chunks} embedded chunks")
    print(f"[eval ] model={MODEL}  dim={DIM}")

    qt = pq.read_table(args.questions).to_pandas()
    if args.limit:
        qt = qt.head(args.limit)
    print(f"[eval ] {len(qt)} questions  top_k={args.top_k}")

    ks = [1, 3, 5, args.top_k]
    hits_at = {k: 0   for k in ks}
    mrr_at  = {k: 0.0 for k in ks}
    ndcg_at = {k: 0.0 for k in ks}
    n_valid = 0

    t0 = time.time()
    with httpx.Client(timeout=120) as client:
        for i, row in enumerate(qt.itertuples(index=False)):
            expected = set(parse_expected(row.expected_doc_ids))
            if not expected:
                continue
            predicted = search(cur, client, row.question, top_n=args.top_k)
            n_valid += 1
            for k in ks:
                top = predicted[:k]
                if expected & set(top):
                    hits_at[k] += 1
                for idx, d in enumerate(top, start=1):
                    if d in expected:
                        mrr_at[k] += 1.0 / idx
                        break
                ndcg_at[k] += ndcg(top, expected, k)
            if (i + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(qt)}  ({(i+1)/dt:.1f} q/s)")

    conn.close()
    print()
    print(f"{'k':>4} {'Recall':>8} {'MRR':>8} {'nDCG':>8}")
    for k in ks:
        print(f"{k:>4} {hits_at[k]/n_valid:>8.3f} {mrr_at[k]/n_valid:>8.3f} {ndcg_at[k]/n_valid:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
