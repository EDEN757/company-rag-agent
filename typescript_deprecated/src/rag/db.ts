import Database from "better-sqlite3";
import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

export interface ChunkRow {
  chunk_id: number;
  doc_id: string;
  source_type: string;
  title: string | null;
  header: string;
  text: string;
  ts_from: string | null;
  ts_to: string | null;
  participants_json: string | null;
  embedding: Buffer;
}

export interface DocumentRow {
  doc_id: string;
  source_type: string;
  title: string | null;
  content: string;
  metadata_json: string | null;
}

const here = dirname(fileURLToPath(import.meta.url));
const DEFAULT_DB = resolve(here, "../../data/index/rag.db");

let _db: Database.Database | null = null;
let _all: Float32Array | null = null;
let _meta: { chunk_ids: number[]; dim: number } | null = null;

export function getDb(): Database.Database {
  if (_db) return _db;
  const path = process.env.RAG_DB_PATH ?? DEFAULT_DB;
  if (!existsSync(path)) {
    throw new Error(
      `RAG index not found at ${path}. Build it with: python indexing/build_index.py --input data/raw/documents_subset.parquet --out data/index/rag.db`,
    );
  }
  _db = new Database(path, { readonly: true, fileMustExist: true });
  _db.pragma("query_only = ON");
  return _db;
}

/** Lazily load every chunk's embedding into one Float32Array for fast matmul. */
export function loadAllEmbeddings(): { matrix: Float32Array; chunkIds: number[]; dim: number } {
  if (_all && _meta) return { matrix: _all, chunkIds: _meta.chunk_ids, dim: _meta.dim };
  const db = getDb();
  const dimRow = db.prepare("SELECT value FROM meta WHERE key = 'embed_dim'").get() as
    | { value: string }
    | undefined;
  const dim = dimRow ? parseInt(dimRow.value, 10) : 768;

  const rows = db
    .prepare("SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY chunk_id ASC")
    .all() as { chunk_id: number; embedding: Buffer }[];

  const matrix = new Float32Array(rows.length * dim);
  const ids: number[] = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const buf = rows[i].embedding;
    if (buf.length !== dim * 4) {
      throw new Error(`embedding[${rows[i].chunk_id}] has ${buf.length} bytes, expected ${dim * 4}`);
    }
    const slice = new Float32Array(buf.buffer, buf.byteOffset, dim);
    matrix.set(slice, i * dim);
    ids[i] = rows[i].chunk_id;
  }
  _all = matrix;
  _meta = { chunk_ids: ids, dim };
  return { matrix, chunkIds: ids, dim };
}

export function fetchChunksByIds(ids: number[]): Map<number, ChunkRow> {
  if (ids.length === 0) return new Map();
  const db = getDb();
  const placeholders = ids.map(() => "?").join(",");
  const rows = db
    .prepare(
      `SELECT c.chunk_id, c.doc_id, c.source_type, d.title, c.header, c.text,
              c.ts_from, c.ts_to, c.participants_json
       FROM chunks c LEFT JOIN documents d ON c.doc_id = d.doc_id
       WHERE c.chunk_id IN (${placeholders})`,
    )
    .all(...ids) as ChunkRow[];
  const out = new Map<number, ChunkRow>();
  for (const r of rows) out.set(r.chunk_id, r);
  return out;
}

export function fetchDocument(doc_id: string): DocumentRow | undefined {
  const db = getDb();
  return db
    .prepare("SELECT doc_id, source_type, title, content, metadata_json FROM documents WHERE doc_id = ?")
    .get(doc_id) as DocumentRow | undefined;
}
