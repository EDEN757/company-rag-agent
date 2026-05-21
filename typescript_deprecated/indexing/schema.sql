PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    source_type   TEXT NOT NULL,
    title         TEXT,
    content       TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_type);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id            TEXT NOT NULL REFERENCES documents(doc_id),
    source_type       TEXT NOT NULL,
    ord               INTEGER NOT NULL,
    header            TEXT NOT NULL,
    text              TEXT NOT NULL,
    ts_from           TEXT,
    ts_to             TEXT,
    participants_json TEXT,
    embedding         BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc    ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_ts     ON chunks(ts_from, ts_to);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.chunk_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.chunk_id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
