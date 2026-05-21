import hashlib
import json
import uuid
import db
from embed import embed
from config import TABLE_DOCS, TABLE_CHUNKS


def new_doc_id() -> str:
    return "dsid_" + hashlib.md5(uuid.uuid4().bytes).hexdigest()


def simple_chunk(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def _insert_chunks(
    cur, doc_id: str, source_type: str, title: str, content: str,
    participants: str | None, ts_from: str | None, ts_to: str | None,
):
    participants_json = (
        json.dumps([p.strip() for p in participants.split(",")]) if participants else "[]"
    )
    for chunk_text in simple_chunk(content):
        header_parts = [f"[source: {source_type}]", f"[title: {title}]"]
        if participants:
            header_parts.append(f"[participants: {participants}]")
        if ts_from:
            date_str = ts_from + (f" -> {ts_to}" if ts_to and ts_to != ts_from else "")
            header_parts.append(f"[dates: {date_str}]")
        full_text = " ".join(header_parts) + "\n\n" + chunk_text
        embedding = embed(full_text)
        cur.execute(
            f"""INSERT INTO {TABLE_CHUNKS}
                (doc_id, source_type, text, ts_from, ts_to, participants_json, embedding, text_tsv)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, to_tsvector('english', %s))""",
            (doc_id, source_type, full_text, ts_from, ts_to,
             participants_json, embedding, full_text),
        )


def add_document(
    source_type: str, title: str, content: str,
    participants: str | None = None, date: str | None = None,
) -> str:
    doc_id = new_doc_id()
    cur = db.conn.cursor()
    try:
        cur.execute(
            f"INSERT INTO {TABLE_DOCS} (doc_id, source_type, title, content) VALUES (%s, %s, %s, %s)",
            (doc_id, source_type, title, content),
        )
        _insert_chunks(cur, doc_id, source_type, title, content, participants, date, date)
    finally:
        cur.close()
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
    cur = db.conn.cursor()
    try:
        cur.execute(f"UPDATE {TABLE_DOCS} SET content = %s WHERE doc_id = %s", (updated, doc_id))
        cur.execute(f"DELETE FROM {TABLE_CHUNKS} WHERE doc_id = %s", (doc_id,))
        _insert_chunks(cur, doc_id, doc["source_type"], doc["title"] or "", updated, None, None, None)
    finally:
        cur.close()
    return f"Updated {doc_id} and re-indexed."
