import hashlib
import json
import re
import uuid

import bm25
import db
from embed import batch_embed
from config import TABLE_DOCS, TABLE_CHUNKS


def new_doc_id() -> str:
    return "dsid_" + hashlib.md5(uuid.uuid4().bytes).hexdigest()


def smart_chunk(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Split on paragraph boundaries; hard-split only when a paragraph exceeds chunk_size."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # Oversized paragraph — hard-split it, then continue building the next chunk
        if len(para) > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                chunks.append(para[start:end])
                if end == len(para):
                    break
                start += chunk_size - overlap
            continue

        sep = 2 if current else 0  # length of "\n\n" separator
        if current_len + sep + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Carry last paragraph as overlap into the next chunk
            last = current[-1]
            if len(last) <= overlap:
                current = [last, para]
                current_len = len(last) + 2 + len(para)
            else:
                current = [para]
                current_len = len(para)
        else:
            current.append(para)
            current_len += sep + len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text.strip()]


def _prepare_chunks(
    doc_id: str, source_type: str, title: str, content: str,
    participants: str | None, ts_from: str | None, ts_to: str | None,
) -> list[tuple]:
    """Pre-compute embeddings for all chunks before any DB writes.

    Returns a list of row tuples ready for INSERT. Raises on embedding failure
    so the caller can abort before touching the database.
    All chunks are embedded in a single batch call to Ollama.
    """
    participants_json = (
        json.dumps([p.strip() for p in participants.split(",")]) if participants else "[]"
    )
    full_texts: list[str] = []
    for chunk_text in smart_chunk(content):
        header_parts = [f"[source: {source_type}]", f"[title: {title}]"]
        if participants:
            header_parts.append(f"[participants: {participants}]")
        if ts_from:
            date_str = ts_from + (f" -> {ts_to}" if ts_to and ts_to != ts_from else "")
            header_parts.append(f"[dates: {date_str}]")
        full_texts.append(" ".join(header_parts) + "\n\n" + chunk_text)

    embeddings = batch_embed(full_texts)  # single round-trip; raises before any DB write
    return [
        (doc_id, source_type, ft, ts_from, ts_to, participants_json, emb, ft)
        for ft, emb in zip(full_texts, embeddings)
    ]


def _write_chunks(cur, rows: list[tuple]) -> None:
    for (doc_id, source_type, full_text, ts_from, ts_to,
         participants_json, embedding, tsv_text) in rows:
        cur.execute(
            f"""INSERT INTO {TABLE_CHUNKS}
                (doc_id, source_type, text, ts_from, ts_to, participants_json, embedding, text_tsv)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, to_tsvector('english', %s))""",
            (doc_id, source_type, full_text, ts_from, ts_to,
             participants_json, embedding, tsv_text),
        )


def add_document(
    source_type: str, title: str, content: str,
    participants: str | None = None, date: str | None = None,
) -> str:
    doc_id = new_doc_id()
    # Batch-embed all chunks first — raises before any DB state is touched
    chunk_rows = _prepare_chunks(doc_id, source_type, title, content, participants, date, date)
    with db.transaction() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {TABLE_DOCS} (doc_id, source_type, title, content) VALUES (%s, %s, %s, %s)",
                (doc_id, source_type, title, content),
            )
            _write_chunks(cur, chunk_rows)
    bm25.corpus_add_chunks([row[2] for row in chunk_rows])
    return doc_id


def edit_document(
    doc_id: str,
    new_content: str | None = None,
    old_string: str | None = None,
    new_string: str | None = None,
) -> str:
    doc = db.fetch_document(doc_id)
    if not doc:
        return f"Document not found: {doc_id}"

    if new_content is not None:
        updated = new_content
    elif old_string is not None and new_string is not None:
        count = doc["content"].count(old_string)
        if count == 0:
            return f"old_string not found in {doc_id}"
        if count > 1:
            return f"old_string appears {count} times — make it more specific"
        updated = doc["content"].replace(old_string, new_string, 1)
    else:
        return "Provide either new_content or both old_string and new_string."

    # Preserve participant and date metadata from the existing chunks
    meta = db.fetch_chunk_meta(doc_id)
    ts_from = meta["ts_from"] if meta else None
    ts_to = meta["ts_to"] if meta else None
    participants: str | None = None
    if meta and meta["participants_json"]:
        try:
            plist = json.loads(meta["participants_json"])
            participants = ", ".join(plist) if plist else None
        except (json.JSONDecodeError, TypeError):
            pass

    # Snapshot old chunk texts now so corpus stats can be decremented after commit
    with db.cursor() as cur:
        cur.execute(f"SELECT text FROM {TABLE_CHUNKS} WHERE doc_id = %s", (doc_id,))
        old_texts = [r[0] for r in cur.fetchall()]

    # Batch-embed new chunks before touching the database
    chunk_rows = _prepare_chunks(
        doc_id, doc["source_type"], doc["title"] or "", updated,
        participants, ts_from, ts_to,
    )

    with db.transaction() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE {TABLE_DOCS} SET content = %s WHERE doc_id = %s", (updated, doc_id))
            cur.execute(f"DELETE FROM {TABLE_CHUNKS} WHERE doc_id = %s", (doc_id,))
            _write_chunks(cur, chunk_rows)

    bm25.corpus_remove_chunks(old_texts)
    bm25.corpus_add_chunks([row[2] for row in chunk_rows])
    return f"Updated {doc_id} and re-indexed."
