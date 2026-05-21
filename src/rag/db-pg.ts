import pg from "pg";
const { Pool } = pg;

export const pool = new Pool({
  host:     process.env.PGHOST     ?? "localhost",
  port:     parseInt(process.env.PGPORT    ?? "5432"),
  database: process.env.PGDATABASE ?? "defaultdb",
  user:     process.env.PGUSER     ?? "nuvolos",
  password: process.env.PGPASSWORD ?? "",
  ssl:      process.env.PGSSLMODE === "require" ? { rejectUnauthorized: false } : false,
  connectionTimeoutMillis: 10_000,
  idleTimeoutMillis:       30_000,
});

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
}

export interface DocumentRow {
  doc_id: string;
  source_type: string;
  title: string | null;
  content: string;
  metadata_json: string | null;
}

export async function fetchDocument(docId: string): Promise<DocumentRow | undefined> {
  const { rows } = await pool.query<DocumentRow>(
    `SELECT doc_id, source_type, title, content, metadata_json
     FROM rag_documents WHERE doc_id = $1`,
    [docId],
  );
  return rows[0];
}
