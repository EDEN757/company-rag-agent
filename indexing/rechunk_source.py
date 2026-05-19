"""Delete all chunks for a given source_type, re-chunk from the documents
table, and re-embed only the new chunks. Use after editing chunkers.py.

Usage:
    python indexing/rechunk_source.py --db data/index/rag.db --source gmail
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chunkers import chunk  # noqa: E402
from embed import DIM, MODEL, stream_embeddings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--source", required=True)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    # Delete existing chunks for this source (FTS rows are removed by trigger).
    n_before = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE source_type = ?", (args.source,)
    ).fetchone()[0]
    print(f"[del  ] deleting {n_before} existing {args.source} chunks")
    con.execute("DELETE FROM chunks WHERE source_type = ?", (args.source,))
    con.commit()

    docs = con.execute(
        "SELECT doc_id, source_type, title, content FROM documents WHERE source_type = ?",
        (args.source,),
    ).fetchall()
    print(f"[chunk] re-chunking {len(docs)} {args.source} documents")
    t0 = time.time()
    cur = con.cursor()
    n_chunks = 0
    for doc_id, st, title, content in docs:
        pieces = chunk(doc_id, st, title or "", content or "")
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
            n_chunks += 1
    con.commit()
    print(f"[chunk] inserted {n_chunks} new chunks in {time.time()-t0:.1f}s")

    pending = cur.execute(
        "SELECT chunk_id, text FROM chunks WHERE source_type = ? AND embedding IS NULL ORDER BY chunk_id",
        (args.source,),
    ).fetchall()
    print(f"[embed] {len(pending)} chunks, model={MODEL} dim={DIM}")
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
    print(f"[embed] done in {time.time()-t0:.1f}s")

    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    con.execute("ANALYZE")
    con.commit()
    con.close()
    print("[done ] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
