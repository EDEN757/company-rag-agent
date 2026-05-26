"""Build the RAG SQLite index from documents_subset.parquet.

The chunking + embedding phases are split so the script is RESUMABLE:
running it again will skip chunks that already have an embedding and
just fill in the missing ones. Use --rebuild to start fresh.

Usage:
    python indexing/build_index.py \
        --input data/raw/documents_subset.parquet \
        --out   data/index/rag.db
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chunkers import chunk  # noqa: E402
from embed import DIM, MODEL, stream_embeddings  # noqa: E402


def _open_db(path: Path) -> sqlite3.Connection:
    schema_sql = (HERE / "schema.sql").read_text()
    con = sqlite3.connect(path)
    con.executescript(schema_sql)
    return con


def _has_documents(con: sqlite3.Connection) -> int:
    return con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def _has_chunks(con: sqlite3.Connection) -> int:
    return con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def _missing_embeddings(con: sqlite3.Connection) -> int:
    return con.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]


def ingest_documents_and_chunks(con: sqlite3.Connection, df) -> None:
    print("[chunk] generating chunks per source_type")
    by_source: dict[str, int] = {}
    t0 = time.time()
    cur = con.cursor()
    doc_count = 0
    chunk_count = 0
    for row in df.itertuples(index=False):
        doc_id = row.doc_id
        st = row.source_type
        title = row.title or ""
        content = row.content or ""
        cur.execute(
            "INSERT INTO documents(doc_id, source_type, title, content, metadata_json) VALUES (?,?,?,?,?)",
            (doc_id, st, title, content, None),
        )
        doc_count += 1
        pieces = chunk(doc_id, st, title, content)
        by_source[st] = by_source.get(st, 0) + len(pieces)
        for p in pieces:
            cur.execute(
                "INSERT INTO chunks(doc_id, source_type, ord, header, text, ts_from, ts_to, participants_json) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    st,
                    p["ord"],
                    p["header"],
                    p["text"],
                    p["ts_from"],
                    p["ts_to"],
                    json.dumps(p["participants"]) if p["participants"] else None,
                ),
            )
            chunk_count += 1
        if doc_count % 1000 == 0:
            con.commit()
    con.commit()
    print(f"[chunk] {chunk_count} chunks from {doc_count} docs in {time.time()-t0:.1f}s")
    for st, n in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"        {st:14s} {n:>6d}")


def embed_missing(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    pending = cur.execute(
        "SELECT chunk_id, text FROM chunks WHERE embedding IS NULL ORDER BY chunk_id"
    ).fetchall()
    total = len(pending)
    if total == 0:
        print("[embed] nothing to embed")
        return
    print(f"[embed] {total} chunks pending, model={MODEL} dim={DIM}")

    write_cur = con.cursor()
    counter = {"n": 0}

    def _on_result(item, vec):
        cid = item[0]
        write_cur.execute("UPDATE chunks SET embedding = ? WHERE chunk_id = ?", (vec.tobytes(), cid))
        counter["n"] += 1
        if counter["n"] % 200 == 0:
            con.commit()

    t0 = time.time()
    stream_embeddings(pending, text_index=1, on_result=_on_result, progress_every=200)
    con.commit()
    dt = time.time() - t0
    print(f"[embed] done in {dt:.1f}s ({total/max(dt,0.001):.1f} chunks/s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N docs (debug).")
    ap.add_argument("--rebuild", action="store_true", help="Delete existing DB first.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rebuild and out_path.exists():
        out_path.unlink()

    con = _open_db(out_path)

    existing_docs = _has_documents(con)
    existing_chunks = _has_chunks(con)
    print(f"[init ] existing: {existing_docs} docs, {existing_chunks} chunks, "
          f"{_missing_embeddings(con)} missing embeddings")

    if existing_docs == 0 and existing_chunks == 0:
        print(f"[load ] reading {in_path}")
        table = pq.read_table(in_path)
        df = table.to_pandas()
        before = len(df)
        df = df.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)
        if before != len(df):
            print(f"[load ] dropped {before - len(df)} duplicate doc_id rows")
        if args.limit:
            df = df.head(args.limit)
        print(f"[load ] {len(df)} documents")
        ingest_documents_and_chunks(con, df)
    else:
        print("[skip ] documents/chunks already populated — only embedding missing rows")

    embed_missing(con)

    # meta
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('embed_model', ?)", (MODEL,))
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('embed_dim', ?)", (str(DIM),))
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('built_at', ?)", (str(int(time.time())),))
    con.commit()

    print("[opt  ] ANALYZE + FTS optimize")
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    con.execute("ANALYZE")
    con.commit()
    con.close()

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[done ] {out_path}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
