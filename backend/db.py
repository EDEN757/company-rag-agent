import logging
import psycopg2
from pgvector.psycopg2 import register_vector
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, TABLE_DOCS, TABLE_CHUNKS

log = logging.getLogger(__name__)

conn = None
cur  = None


def connect():
    global conn, cur
    log.info(f"Connecting to pgvector @ {DB_HOST}:{DB_PORT}/{DB_NAME}")
    kwargs: dict = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, dbname=DB_NAME)
    if DB_PASSWORD:
        kwargs["password"] = DB_PASSWORD
    conn = psycopg2.connect(**kwargs)
    conn.autocommit = True
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SET ivfflat.probes = 10;")
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS};")
    n = cur.fetchone()[0]
    log.info(f"pgvector connected — {n} chunks indexed.")


def disconnect():
    global conn
    if conn:
        conn.close()


def fetch_document(doc_id: str) -> dict | None:
    c = conn.cursor()
    try:
        c.execute(
            f"SELECT doc_id, source_type, title, content FROM {TABLE_DOCS} WHERE doc_id = %s",
            (doc_id,),
        )
        row = c.fetchone()
    finally:
        c.close()
    if not row:
        return None
    return {"doc_id": row[0], "source_type": row[1], "title": row[2], "content": row[3]}
