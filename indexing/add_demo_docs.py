"""Ingest hand-authored demo docs from data/demo_docs/ into the RAG index.

Demo docs are markdown files with YAML frontmatter:

    ---
    doc_id: demo_bluefin_overview
    source_type: confluence
    title: Project Bluefin overview
    date: 2025-08-12
    participants: [jamie@acmeco.io, alex@acmeco.io]
    skill: trace
    ---
    <body>

Each one is INSERTed into the documents table with
metadata_json = {"synthetic": true, "skill": ..., "date": ..., "participants": [...]}
so a single WHERE json_extract(metadata_json,'$.synthetic')=1 enumerates
them. Existing doc_ids are skipped (idempotent — safe to re-run).

The script reuses indexing/chunkers.py:chunk() and
indexing/embed.py:stream_embeddings() — chunking and embedding logic is
NOT duplicated here.

Usage:
    python indexing/add_demo_docs.py            # uses $RAG_DB_PATH then data/index/rag.db
    python indexing/add_demo_docs.py --db /tmp/test-rag.db --docs-dir data/demo_docs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chunkers import chunk  # noqa: E402
from embed import DIM, MODEL, stream_embeddings  # noqa: E402


REPO_ROOT = HERE.parent
DEFAULT_DOCS_DIR = REPO_ROOT / "data" / "demo_docs"
DEFAULT_DB = REPO_ROOT / "data" / "index" / "rag.db"

REQUIRED_FIELDS = ("doc_id", "source_type", "title", "skill")


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _parse_list(s: str) -> list[str]:
    # "[a, b, c]" -> ["a", "b", "c"]
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return [_strip_quotes(s)] if s else []
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [_strip_quotes(p) for p in inner.split(",") if p.strip()]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse a tiny YAML-subset frontmatter block: scalar strings + bracket lists.

    Returns (fields, body). Raises if the file has no frontmatter or is missing
    a required field.
    """
    if not text.startswith("---"):
        raise ValueError("file has no leading '---' frontmatter delimiter")
    end = text.find("\n---", 3)
    if end == -1:
        raise ValueError("frontmatter has no closing '---'")
    header = text[3:end].strip("\n")
    body = text[end + len("\n---"):].lstrip("\n")

    fields: dict[str, object] = {}
    for line in header.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if not m:
            raise ValueError(f"frontmatter line not 'key: value': {line!r}")
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith("["):
            fields[key] = _parse_list(raw)
        else:
            fields[key] = _strip_quotes(raw)

    missing = [f for f in REQUIRED_FIELDS if f not in fields or fields[f] == ""]
    if missing:
        raise ValueError(f"frontmatter missing required fields: {missing}")
    return fields, body


def collect_files(docs_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(docs_dir.rglob("*.md")):
        if p.name == "QA.md":
            continue
        out.append(p)
    return out


def existing_doc_ids(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT doc_id FROM documents").fetchall()}


def open_db(path: Path) -> sqlite3.Connection:
    schema_sql = (HERE / "schema.sql").read_text()
    con = sqlite3.connect(path)
    con.executescript(schema_sql)
    return con


def ingest_documents(
    con: sqlite3.Connection,
    files: list[Path],
    already_have: set[str],
) -> tuple[int, int, int]:
    """INSERT documents + chunks for any file whose doc_id isn't already present.

    Returns (new_docs, new_chunks, skipped).
    """
    new_docs = 0
    new_chunks = 0
    skipped = 0
    cur = con.cursor()
    for path in files:
        text = path.read_text(encoding="utf-8")
        try:
            fields, body = parse_frontmatter(text)
        except ValueError as e:
            print(f"[err  ] {path}: {e}")
            raise
        doc_id = str(fields["doc_id"])
        if doc_id in already_have:
            skipped += 1
            continue

        source_type = str(fields["source_type"])
        title = str(fields["title"])
        skill = str(fields["skill"])
        date = fields.get("date")
        participants = fields.get("participants") or []
        metadata = {
            "synthetic": True,
            "skill": skill,
            "date": date,
            "participants": participants,
            "source_file": str(path.relative_to(REPO_ROOT)),
        }
        cur.execute(
            "INSERT INTO documents(doc_id, source_type, title, content, metadata_json) "
            "VALUES (?,?,?,?,?)",
            (doc_id, source_type, title, body, json.dumps(metadata)),
        )
        new_docs += 1

        pieces = chunk(doc_id, source_type, title, body)
        if not pieces:
            print(f"[warn ] {doc_id}: chunker produced no pieces; skipping chunks")
            continue
        for p in pieces:
            cur.execute(
                "INSERT INTO chunks(doc_id, source_type, ord, header, text, ts_from, ts_to, participants_json) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    source_type,
                    p["ord"],
                    p["header"],
                    p["text"],
                    p["ts_from"],
                    p["ts_to"],
                    json.dumps(p["participants"]) if p["participants"] else None,
                ),
            )
            new_chunks += 1
        already_have.add(doc_id)
    con.commit()
    return new_docs, new_chunks, skipped


def embed_pending_demo_chunks(con: sqlite3.Connection) -> int:
    """Embed any chunk that belongs to a synthetic doc and is still NULL."""
    pending = con.execute(
        "SELECT c.chunk_id, c.text "
        "FROM   chunks c "
        "JOIN   documents d ON d.doc_id = c.doc_id "
        "WHERE  c.embedding IS NULL "
        "  AND  json_extract(d.metadata_json, '$.synthetic') = 1 "
        "ORDER BY c.chunk_id"
    ).fetchall()
    total = len(pending)
    if total == 0:
        return 0
    print(f"[embed] {total} demo chunks pending, model={MODEL} dim={DIM}")
    write_cur = con.cursor()
    counter = {"n": 0}

    def _on_result(item, vec):
        cid = item[0]
        write_cur.execute(
            "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
            (vec.tobytes(), cid),
        )
        counter["n"] += 1
        if counter["n"] % 50 == 0:
            con.commit()

    t0 = time.time()
    stream_embeddings(pending, text_index=1, on_result=_on_result, progress_every=50)
    con.commit()
    dt = time.time() - t0
    print(f"[embed] done in {dt:.1f}s ({total/max(dt,0.001):.1f} chunks/s)")
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        default=os.environ.get("RAG_DB_PATH", str(DEFAULT_DB)),
        help="Path to rag.db (defaults to $RAG_DB_PATH then data/index/rag.db).",
    )
    ap.add_argument(
        "--docs-dir",
        default=str(DEFAULT_DOCS_DIR),
        help="Root of the demo docs tree (default: data/demo_docs).",
    )
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    docs_dir = Path(args.docs_dir).expanduser().resolve()

    if not docs_dir.exists():
        print(f"[err  ] docs dir not found: {docs_dir}")
        return 1

    db_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(docs_dir)
    if not files:
        print(f"[done ] no .md files under {docs_dir}")
        return 0
    print(f"[scan ] found {len(files)} demo doc(s) under {docs_dir}")
    print(f"[db   ] {db_path}")

    con = open_db(db_path)
    have = existing_doc_ids(con)
    new_docs, new_chunks, skipped = ingest_documents(con, files, have)
    print(f"[doc  ] inserted {new_docs} new doc(s); {new_chunks} new chunk(s); {skipped} already present")

    embed_pending_demo_chunks(con)

    if new_docs > 0:
        print("[opt  ] FTS optimize + ANALYZE")
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        con.execute("ANALYZE")
        con.commit()

    con.close()
    print("[done ] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
