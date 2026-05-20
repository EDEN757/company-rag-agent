"""
indexing/build_index_pg.py — Index company documents into pgvector (Nuvolos)

Reads the same Parquet files as build_index.py but targets PostgreSQL/pgvector
instead of SQLite. Embedding is done via Ollama (nomic-embed-text) — the same
model and client code used by the local indexer.

The chunking + embedding phases are split so the script is RESUMABLE: running
it again will skip docs/chunks that already exist and only embed missing rows.
Use --rebuild to start fresh.

Run from the Backend VS Code app on Nuvolos:
    cd /files
    pip install -r backend/requirements.txt pyarrow pandas
    python indexing/build_index_pg.py --input data/raw/documents_subset.parquet

For a debug run on a small slice:
    python indexing/build_index_pg.py --input data/raw/documents_subset.parquet --limit 500

Environment variables (set in Nuvolos Backend app CONFIGURE):
    PGHOST        <hostname shown in Database app CONFIGURE>
    PGPORT        5432
    PGUSER        nuvolos
    PGPASSWORD    nuvolos
    PGDATABASE    nuvolos
    OLLAMA_HOST   http://localhost:11434   (default — Ollama on same Backend container)
    RAG_EMBED_MODEL  nomic-embed-text      (default — passed through to embed.py)
    HF_HOME       /space_mounts/pars/hf_cache  (not used; kept for compat)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg2
import pyarrow.parquet as pq
from pgvector.psycopg2 import register_vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chunkers import chunk          # noqa: E402
from embed import DIM, MODEL, stream_embeddings  # noqa: E402

# ── Database configuration ─────────────────────────────────────────────────────
DB_HOST     = os.environ.get("PGHOST",     "nv-service-b01d63337fab32ac94f65eb2dc8a62ba")
DB_PORT     = int(os.environ.get("PGPORT", "5432"))
DB_USER     = os.environ.get("PGUSER",     "nuvolos")
DB_PASSWORD = os.environ.get("PGPASSWORD", "nuvolos")
DB_NAME     = os.environ.get("PGDATABASE", "nuvolos")

TABLE_DOCS   = "rag_documents"
TABLE_CHUNKS = "rag_chunks"
TABLE_META   = "rag_meta"


# ── Database helpers ───────────────────────────────────────────────────────────
def connect() -> tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    kwargs: dict = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, dbname=DB_NAME)
    if DB_PASSWORD:
        kwargs["password"] = DB_PASSWORD
    conn = psycopg2.connect(**kwargs)
    conn.autocommit = False
    register_vector(conn)
    return conn, conn.cursor()


def apply_schema(conn, cur) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_DOCS} (
            doc_id        TEXT PRIMARY KEY,
            source_type   TEXT NOT NULL,
            title         TEXT,
            content       TEXT NOT NULL,
            metadata_json TEXT
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_rag_docs_source ON {TABLE_DOCS}(source_type);")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_CHUNKS} (
            chunk_id          SERIAL PRIMARY KEY,
            doc_id            TEXT NOT NULL REFERENCES {TABLE_DOCS}(doc_id),
            source_type       TEXT NOT NULL,
            ord               INTEGER NOT NULL,
            header            TEXT NOT NULL,
            text              TEXT NOT NULL,
            ts_from           TEXT,
            ts_to             TEXT,
            participants_json TEXT,
            embedding         vector({DIM}),
            text_tsv          tsvector GENERATED ALWAYS AS (
                                  to_tsvector('english', coalesce(text, ''))
                              ) STORED
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc    ON {TABLE_CHUNKS}(doc_id);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_rag_chunks_source ON {TABLE_CHUNKS}(source_type);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_rag_chunks_ts     ON {TABLE_CHUNKS}(ts_from, ts_to);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_rag_chunks_fts    ON {TABLE_CHUNKS} USING GIN (text_tsv);")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_META} (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()


def _existing_doc_ids(cur) -> set[str]:
    cur.execute(f"SELECT doc_id FROM {TABLE_DOCS};")
    return {row[0] for row in cur.fetchall()}


def _count_missing_embeddings(cur) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS} WHERE embedding IS NULL;")
    return cur.fetchone()[0]


# ── Phase 1: insert docs + chunks (no embeddings) ─────────────────────────────
def ingest_documents_and_chunks(conn, cur, df, rebuild: bool) -> None:
    if rebuild:
        print("[rebuild] truncating existing data …")
        cur.execute(f"TRUNCATE {TABLE_CHUNKS}, {TABLE_DOCS} CASCADE;")
        conn.commit()
        known: set[str] = set()
    else:
        known = _existing_doc_ids(cur)
        if known:
            print(f"[skip   ] {len(known)} docs already in DB — processing new ones only")

    docs_added = 0
    chunks_added = 0
    t0 = time.time()

    for row in df.itertuples(index=False):
        doc_id = row.doc_id
        if doc_id in known:
            continue
        st      = row.source_type
        title   = row.title or ""
        content = row.content or ""

        cur.execute(
            f"INSERT INTO {TABLE_DOCS}(doc_id, source_type, title, content, metadata_json) "
            f"VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (doc_id, st, title, content, None),
        )
        docs_added += 1

        pieces = chunk(doc_id, st, title, content)
        for p in pieces:
            cur.execute(
                f"INSERT INTO {TABLE_CHUNKS}"
                f"(doc_id, source_type, ord, header, text, ts_from, ts_to, participants_json) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    doc_id, st, p["ord"], p["header"], p["text"],
                    p["ts_from"], p["ts_to"],
                    json.dumps(p["participants"]) if p["participants"] else None,
                ),
            )
            chunks_added += 1

        if docs_added % 500 == 0:
            conn.commit()
            print(f"  {docs_added} docs processed ({chunks_added} chunks) …")

    conn.commit()
    print(f"[chunk  ] {docs_added} new docs → {chunks_added} chunks  ({time.time()-t0:.1f}s)")


# ── Phase 2: embed missing chunks ─────────────────────────────────────────────
def embed_missing(conn, cur) -> None:
    cur.execute(
        f"SELECT chunk_id, text FROM {TABLE_CHUNKS} WHERE embedding IS NULL ORDER BY chunk_id"
    )
    pending = cur.fetchall()
    total = len(pending)
    if total == 0:
        print("[embed  ] nothing to embed")
        return
    print(f"[embed  ] {total} chunks pending  model={MODEL}  dim={DIM}")

    write_cur = conn.cursor()
    counter = {"n": 0}

    def _on_result(item, vec):
        chunk_id = item[0]
        write_cur.execute(
            f"UPDATE {TABLE_CHUNKS} SET embedding = %s::vector WHERE chunk_id = %s",
            (vec.tolist(), chunk_id),
        )
        counter["n"] += 1
        if counter["n"] % 200 == 0:
            conn.commit()

    t0 = time.time()
    stream_embeddings(pending, text_index=1, on_result=_on_result, progress_every=200)
    conn.commit()
    dt = time.time() - t0
    print(f"[embed  ] done in {dt:.1f}s  ({total/max(dt, 0.001):.1f} chunks/s)")


# ── IVFFlat vector index ───────────────────────────────────────────────────────
def build_ivfflat_index(conn, cur) -> None:
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS} WHERE embedding IS NOT NULL;")
    n = cur.fetchone()[0]
    if n == 0:
        print("[index  ] no embeddings — skipping IVFFlat index")
        return
    lists = max(10, min(int(n ** 0.5), 300))
    print(f"[index  ] building IVFFlat cosine index  lists={lists}  n={n} …")
    cur.execute("DROP INDEX IF EXISTS idx_rag_chunks_vec;")
    cur.execute(
        f"CREATE INDEX idx_rag_chunks_vec ON {TABLE_CHUNKS} "
        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists});"
    )
    conn.commit()
    print("[index  ] IVFFlat index built.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Index company documents into pgvector.")
    ap.add_argument("--input",   required=True, help="Path to documents_subset.parquet")
    ap.add_argument("--limit",   type=int, default=0, help="Process first N docs only (debug)")
    ap.add_argument("--rebuild", action="store_true", help="Drop all existing data and re-index")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found.")
        return 1

    print("=" * 60)
    print("Company RAG — pgvector Indexer")
    print(f"  input  : {in_path}")
    print(f"  target : {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"  embed  : {MODEL}  dim={DIM}  (via Ollama)")
    print("=" * 60)

    conn, cur = connect()
    print(f"[db     ] connected to {DB_HOST}:{DB_PORT}/{DB_NAME}")
    apply_schema(conn, cur)
    print("[db     ] schema ready")

    cur2 = conn.cursor()
    cur2.execute(f"SELECT COUNT(*) FROM {TABLE_DOCS};")
    existing_docs = cur2.fetchone()[0]
    cur2.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS};")
    existing_chunks = cur2.fetchone()[0]
    missing = _count_missing_embeddings(cur2)
    print(f"[init   ] existing: {existing_docs} docs, {existing_chunks} chunks, {missing} missing embeddings")

    if existing_docs == 0 or args.rebuild:
        print(f"[load   ] reading {in_path} …")
        df = pq.read_table(in_path).to_pandas()
        before = len(df)
        df = df.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)
        if before != len(df):
            print(f"[load   ] dropped {before - len(df)} duplicate doc_ids")
        if args.limit:
            df = df.head(args.limit)
        print(f"[load   ] {len(df)} documents")
        ingest_documents_and_chunks(conn, cur, df, rebuild=args.rebuild)
    else:
        print("[skip   ] documents/chunks already populated — only embedding missing rows")

    embed_missing(conn, cur)
    build_ivfflat_index(conn, cur)

    for key, val in [
        ("embed_model", MODEL),
        ("embed_dim",   str(DIM)),
        ("built_at",    str(int(time.time()))),
    ]:
        cur.execute(
            f"INSERT INTO {TABLE_META}(key, value) VALUES (%s, %s) "
            f"ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
            (key, val),
        )
    conn.commit()
    conn.close()

    print("[done   ] indexing complete — Backend API is ready to serve queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
