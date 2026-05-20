"""Delete all chunks for a given source_type, re-chunk from rag_documents,
and re-embed with nomic-embed-text via Ollama. Use after editing chunkers.py.

Nuvolos counterpart to rechunk_source.py.

Usage:
    cd /files
    python indexing/rechunk_source_pg.py --source gmail
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chunkers import chunk                       # noqa: E402
from embed import DIM, MODEL, stream_embeddings  # noqa: E402

DB_HOST     = os.environ.get("PGHOST",     "nv-service-b01d63337fab32ac94f65eb2dc8a62ba")
DB_PORT     = int(os.environ.get("PGPORT", "5432"))
DB_USER     = os.environ.get("PGUSER",     "nuvolos")
DB_PASSWORD = os.environ.get("PGPASSWORD", "nuvolos")
DB_NAME     = os.environ.get("PGDATABASE", "nuvolos")

TABLE_DOCS   = "rag_documents"
TABLE_CHUNKS = "rag_chunks"


def connect():
    kwargs: dict = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, dbname=DB_NAME)
    if DB_PASSWORD:
        kwargs["password"] = DB_PASSWORD
    conn = psycopg2.connect(**kwargs)
    conn.autocommit = False
    register_vector(conn)
    return conn, conn.cursor()


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-chunk and re-embed one source_type in pgvector.")
    ap.add_argument("--source", required=True, help="source_type to rechunk (e.g. gmail, slack)")
    args = ap.parse_args()

    conn, cur = connect()

    cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS} WHERE source_type = %s", (args.source,))
    n_before = cur.fetchone()[0]
    print(f"[del  ] deleting {n_before} existing {args.source} chunks …")
    cur.execute(f"DELETE FROM {TABLE_CHUNKS} WHERE source_type = %s", (args.source,))
    conn.commit()

    cur.execute(
        f"SELECT doc_id, source_type, title, content FROM {TABLE_DOCS} WHERE source_type = %s",
        (args.source,),
    )
    docs = cur.fetchall()
    print(f"[chunk] re-chunking {len(docs)} {args.source} documents")
    t0 = time.time()
    n_chunks = 0
    write_cur = conn.cursor()
    for doc_id, st, title, content in docs:
        pieces = chunk(doc_id, st, title or "", content or "")
        for p in pieces:
            write_cur.execute(
                f"INSERT INTO {TABLE_CHUNKS}"
                f"(doc_id, source_type, ord, header, text, ts_from, ts_to, participants_json) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    doc_id, st, p["ord"], p["header"], p["text"],
                    p["ts_from"], p["ts_to"],
                    json.dumps(p["participants"]) if p["participants"] else None,
                ),
            )
            n_chunks += 1
    conn.commit()
    print(f"[chunk] inserted {n_chunks} new chunks in {time.time()-t0:.1f}s")

    cur.execute(
        f"SELECT chunk_id, text FROM {TABLE_CHUNKS} "
        f"WHERE source_type = %s AND embedding IS NULL ORDER BY chunk_id",
        (args.source,),
    )
    pending = cur.fetchall()
    print(f"[embed] {len(pending)} chunks  model={MODEL}  dim={DIM}")

    embed_cur = conn.cursor()
    counter = {"n": 0}

    def _on_result(item, vec):
        embed_cur.execute(
            f"UPDATE {TABLE_CHUNKS} SET embedding = %s::vector WHERE chunk_id = %s",
            (vec.tolist(), item[0]),
        )
        counter["n"] += 1
        if counter["n"] % 200 == 0:
            conn.commit()

    t0 = time.time()
    stream_embeddings(pending, text_index=1, on_result=_on_result, progress_every=200)
    conn.commit()
    print(f"[embed] done in {time.time()-t0:.1f}s")

    conn.close()
    print("[done ] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
