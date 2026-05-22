import logging
import threading
from contextlib import contextmanager

import psycopg2
from pgvector.psycopg2 import register_vector

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, TABLE_DOCS, TABLE_CHUNKS

log = logging.getLogger(__name__)

# Each worker thread gets its own connection so cursors never race.
_local = threading.local()
_connect_kwargs: dict = {}


def _get_conn():
    """Return the thread-local psycopg2 connection, creating it on first use."""
    if getattr(_local, "conn", None) is None or _local.conn.closed:
        c = psycopg2.connect(**_connect_kwargs)
        c.autocommit = True
        register_vector(c)
        with c.cursor() as cur:
            cur.execute("SET ivfflat.probes = 10;")
        _local.conn = c
    return _local.conn


def cursor():
    """Return a new cursor on the calling thread's connection."""
    return _get_conn().cursor()


@contextmanager
def transaction():
    """Wrap multiple statements in a single atomic transaction.

    Temporarily disables autocommit, commits on success, rolls back on any
    exception, and restores autocommit regardless of outcome.
    """
    conn = _get_conn()
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True


def connect():
    global _connect_kwargs
    log.info(f"Connecting to pgvector @ {DB_HOST}:{DB_PORT}/{DB_NAME}")
    kwargs: dict = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, dbname=DB_NAME)
    if DB_PASSWORD:
        kwargs["password"] = DB_PASSWORD
    _connect_kwargs = kwargs
    with cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS};")
        n = cur.fetchone()[0]
    log.info(f"pgvector connected — {n} chunks indexed.")


def disconnect():
    if getattr(_local, "conn", None) and not _local.conn.closed:
        _local.conn.close()
        _local.conn = None


def fetch_document(doc_id: str) -> dict | None:
    with cursor() as cur:
        cur.execute(
            f"SELECT doc_id, source_type, title, content FROM {TABLE_DOCS} WHERE doc_id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"doc_id": row[0], "source_type": row[1], "title": row[2], "content": row[3]}


def fetch_chunk_meta(doc_id: str) -> dict | None:
    """Return participants_json, ts_from, ts_to from the first chunk of a document."""
    with cursor() as cur:
        cur.execute(
            f"SELECT participants_json, ts_from, ts_to FROM {TABLE_CHUNKS} "
            f"WHERE doc_id = %s LIMIT 1",
            (doc_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"participants_json": row[0], "ts_from": row[1], "ts_to": row[2]}
