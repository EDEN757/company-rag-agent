import { fetchChunksByIds, getDb, loadAllEmbeddings, type ChunkRow } from "./db.js";
import { embedQuery } from "./embed.js";
import { rerank } from "./rerank.js";

export const TOP_K_PER_BRANCH = 16;
export const KW_WEIGHT = 0.3;
export const VEC_WEIGHT = 0.7;
export const SCORE_THRESHOLD = 0.30;
export const SCORE_SCALE = 4;

export interface SearchFilters {
  source_types?: string[];
  date_from?: string;
  date_to?: string;
  participant?: string;
}

export interface SearchHit {
  chunk_id: number;
  doc_id: string;
  source_type: string;
  title: string | null;
  score: number;
  vec_score: number;
  kw_score: number;
  rerank_score?: number;
  preview: string;
  ts_from: string | null;
  ts_to: string | null;
}

function buildFilterClause(filters: SearchFilters): { sql: string; params: unknown[] } {
  const clauses: string[] = [];
  const params: unknown[] = [];
  if (filters.source_types && filters.source_types.length > 0) {
    const placeholders = filters.source_types.map(() => "?").join(",");
    clauses.push(`source_type IN (${placeholders})`);
    params.push(...filters.source_types);
  }
  if (filters.date_from) {
    clauses.push("(ts_to IS NULL OR ts_to >= ?)");
    params.push(filters.date_from);
  }
  if (filters.date_to) {
    clauses.push("(ts_from IS NULL OR ts_from <= ?)");
    params.push(filters.date_to);
  }
  if (filters.participant) {
    clauses.push("participants_json LIKE ?");
    params.push(`%${filters.participant}%`);
  }
  return { sql: clauses.length ? `WHERE ${clauses.join(" AND ")}` : "", params };
}

function ftsQuery(q: string): string {
  // Strip operators FTS5 would interpret, then OR-join words so any-of matches.
  const words = q
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((w) => w.length > 1);
  if (words.length === 0) return '""';
  return words.map((w) => `"${w}"`).join(" OR ");
}

function preview(text: string, maxChars = 320): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length <= maxChars ? clean : `${clean.slice(0, maxChars).trim()}…`;
}

export async function search(query: string, filters: SearchFilters = {}, topN = 6): Promise<SearchHit[]> {
  const db = getDb();
  const { sql: whereSql, params } = buildFilterClause(filters);

  // --- Keyword branch ---
  const kwQuery = ftsQuery(query);
  const kwSql = `
    SELECT c.chunk_id, c.doc_id, c.source_type
    FROM chunks_fts f
    JOIN chunks c ON c.chunk_id = f.rowid
    ${whereSql ? whereSql + " AND" : "WHERE"} chunks_fts MATCH ?
    ORDER BY bm25(chunks_fts) ASC
    LIMIT ?
  `;
  const kwRows = db.prepare(kwSql).all(...params, kwQuery, TOP_K_PER_BRANCH) as {
    chunk_id: number;
    doc_id: string;
    source_type: string;
  }[];

  const kwScores = new Map<number, number>();
  kwRows.forEach((r, i) => {
    const rank = i + 1;
    kwScores.set(r.chunk_id, (1 / (1 + rank)) * SCORE_SCALE);
  });

  // --- Vector branch ---
  const qvec = await embedQuery(query);
  const { matrix, chunkIds, dim } = loadAllEmbeddings();

  // Build allowed-id set for filtering, if any filter applies.
  let allowed: Set<number> | null = null;
  if (whereSql) {
    const rows = db
      .prepare(`SELECT chunk_id FROM chunks ${whereSql}`)
      .all(...params) as { chunk_id: number }[];
    allowed = new Set(rows.map((r) => r.chunk_id));
  }

  // Cosine similarity = dot product (vectors are L2-normalized).
  const sims = new Float32Array(chunkIds.length);
  for (let i = 0; i < chunkIds.length; i++) {
    if (allowed && !allowed.has(chunkIds[i])) {
      sims[i] = -Infinity;
      continue;
    }
    let s = 0;
    const off = i * dim;
    for (let d = 0; d < dim; d++) s += matrix[off + d] * qvec[d];
    sims[i] = s;
  }
  // Top-K from sims.
  const order: number[] = Array.from({ length: chunkIds.length }, (_, i) => i);
  order.sort((a, b) => sims[b] - sims[a]);
  const vecScores = new Map<number, number>();
  for (let i = 0; i < Math.min(TOP_K_PER_BRANCH, order.length); i++) {
    const idx = order[i];
    if (!isFinite(sims[idx])) break;
    // sims is cosine similarity in [-1, 1] (L2-normalized vectors → dot product).
    // Scale to put it on the same magnitude as the kw branch ((1/(1+rank))*SCALE).
    vecScores.set(chunkIds[idx], sims[idx] * SCORE_SCALE);
  }

  // --- Fusion ---
  const allIds = new Set<number>([...kwScores.keys(), ...vecScores.keys()]);
  const fused: { chunk_id: number; final: number; kw: number; vec: number }[] = [];
  for (const id of allIds) {
    const kw = kwScores.get(id) ?? 0;
    const vec = vecScores.get(id) ?? 0;
    const final = VEC_WEIGHT * vec + KW_WEIGHT * kw;
    if (final >= SCORE_THRESHOLD) fused.push({ chunk_id: id, final, kw, vec });
  }
  fused.sort((a, b) => b.final - a.final);
  // Take the full candidate pool (at most 16: 2 branches × 8 each) for the
  // cross-encoder, then slice to topN after reranking.
  const pool = fused.slice(0, Math.min(TOP_K_PER_BRANCH * 2, fused.length));

  if (pool.length === 0) return [];

  const rows = fetchChunksByIds(pool.map((t) => t.chunk_id));

  // Cross-encoder reranking: the model scores each (query, passage) pair
  // jointly, catching relevance signals the bi-encoder dot product misses.
  // Falls back to fusion order if the reranker service is unavailable.
  const passages = pool.map((t) => {
    const r = rows.get(t.chunk_id);
    return r ? r.text : "";  // r.text already contains the header prepended
  });
  const rerankScores = await rerank(query, passages);

  // Sort full pool by reranker score, fall back to fusion order.
  const sortedPool = rerankScores !== null
    ? pool.map((c, i) => ({ ...c, rerankScore: rerankScores[i] }))
          .sort((a, b) => b.rerankScore - a.rerankScore)
    : pool;

  // Deduplicate by doc_id: keep the best-scoring chunk per document.
  // Multiple chunks from the same document waste result slots and can cause
  // the agent to open the same document twice. Mirrors the eval script logic.
  const seenDocIds = new Set<string>();
  const hits: SearchHit[] = [];
  for (const t of sortedPool) {
    if (hits.length >= topN) break;
    const r = rows.get(t.chunk_id);
    if (!r) continue;
    if (seenDocIds.has(r.doc_id)) continue;
    seenDocIds.add(r.doc_id);
    hits.push({
      chunk_id: r.chunk_id,
      doc_id: r.doc_id,
      source_type: r.source_type,
      title: r.title,
      score: Number(t.final.toFixed(3)),
      vec_score: Number(t.vec.toFixed(3)),
      kw_score: Number(t.kw.toFixed(3)),
      rerank_score: "rerankScore" in t ? Number((t.rerankScore as number).toFixed(3)) : undefined,
      preview: preview(r.text),
      ts_from: r.ts_from,
      ts_to: r.ts_to,
    });
  }
  return hits;
}
