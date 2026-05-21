import { pool } from "./db-pg.js";
import { embedQuery } from "./embed.js";
import { tokenize, bm25Scores } from "./bm25.js";

export const TOP_K_PER_BRANCH = 8;
export const KW_WEIGHT        = 0.3;
export const VEC_WEIGHT       = 0.7;
export const SCORE_THRESHOLD  = 0.35;
export const SCORE_SCALE      = 4;

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
  preview: string;
  ts_from: string | null;
  ts_to: string | null;
}

type ChunkData = {
  doc_id: string; source_type: string; text: string;
  ts_from: string | null; ts_to: string | null; score: number;
};

function ftsOrQuery(q: string): string | null {
  const words = (q.toLowerCase().match(/[a-zA-Z]\w*/g) ?? []).filter((w) => w.length > 1);
  return words.length ? words.join(" | ") : null;
}

function preview(text: string, maxChars = 320): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length <= maxChars ? clean : `${clean.slice(0, maxChars).trim()}…`;
}

function buildFilters(
  filters: SearchFilters,
  startIdx: number,
): { clauses: string[]; params: unknown[]; nextIdx: number } {
  const clauses: string[] = [];
  const params: unknown[] = [];
  let idx = startIdx;
  if (filters.source_types?.length) {
    clauses.push(`c.source_type = ANY($${idx++})`);
    params.push(filters.source_types);
  }
  if (filters.date_from) {
    clauses.push(`(c.ts_to IS NULL OR c.ts_to >= $${idx++})`);
    params.push(filters.date_from);
  }
  if (filters.date_to) {
    clauses.push(`(c.ts_from IS NULL OR c.ts_from <= $${idx++})`);
    params.push(filters.date_to);
  }
  if (filters.participant) {
    clauses.push(`c.participants_json LIKE $${idx++}`);
    params.push(`%${filters.participant}%`);
  }
  return { clauses, params, nextIdx: idx };
}

export async function search(
  query: string,
  filters: SearchFilters = {},
  topN = 6,
): Promise<SearchHit[]> {
  topN = Math.max(1, Math.min(20, topN));
  const qvec = await embedQuery(query);
  const vecStr = "[" + Array.from(qvec).join(",") + "]";

  // --- Vector branch ---
  const { clauses: vClauses, params: vParams, nextIdx: vNext } = buildFilters(filters, 2);
  const vWhere = vClauses.length ? "AND " + vClauses.join(" AND ") : "";
  const { rows: vecRows } = await pool.query<{
    chunk_id: number; doc_id: string; source_type: string;
    text: string; ts_from: string | null; ts_to: string | null; vec_score: string;
  }>(
    `SELECT c.chunk_id, c.doc_id, c.source_type, c.text, c.ts_from, c.ts_to,
            1 - (c.embedding <=> $1::vector) AS vec_score
     FROM rag_chunks c
     WHERE c.embedding IS NOT NULL ${vWhere}
     ORDER BY c.embedding <=> $1::vector
     LIMIT $${vNext}`,
    [vecStr, ...vParams, TOP_K_PER_BRANCH],
  );

  const vecScores = new Map<number, ChunkData>();
  for (const r of vecRows) {
    vecScores.set(r.chunk_id, {
      doc_id: r.doc_id, source_type: r.source_type, text: r.text,
      ts_from: r.ts_from, ts_to: r.ts_to,
      score: Number(r.vec_score) * SCORE_SCALE,
    });
  }

  // --- Keyword branch ---
  const kwScores = new Map<number, ChunkData>();
  const ftsQ = ftsOrQuery(query);
  if (ftsQ) {
    try {
      const { clauses: kClauses, params: kParams, nextIdx: kNext } = buildFilters(filters, 2);
      const kConditions = [`c.text_tsv @@ to_tsquery('english', $1)`, ...kClauses].join(" AND ");
      const { rows: ftsRows } = await pool.query<{
        chunk_id: number; doc_id: string; source_type: string;
        text: string; ts_from: string | null; ts_to: string | null;
      }>(
        `SELECT c.chunk_id, c.doc_id, c.source_type, c.text, c.ts_from, c.ts_to
         FROM rag_chunks c
         WHERE ${kConditions}
         LIMIT $${kNext}`,
        [ftsQ, ...kParams, TOP_K_PER_BRANCH * 3],
      );

      const queryTerms = tokenize(query);
      const bm25 = bm25Scores(queryTerms, ftsRows.map((r) => ({ id: r.chunk_id, text: r.text })));
      const ranked = [...ftsRows]
        .sort((a, b) => (bm25.get(b.chunk_id) ?? 0) - (bm25.get(a.chunk_id) ?? 0))
        .slice(0, TOP_K_PER_BRANCH);

      ranked.forEach((r, i) => {
        kwScores.set(r.chunk_id, {
          doc_id: r.doc_id, source_type: r.source_type, text: r.text,
          ts_from: r.ts_from, ts_to: r.ts_to,
          score: (1 / (1 + i + 1)) * SCORE_SCALE,
        });
      });
    } catch (e) {
      console.warn("Keyword branch failed (vector-only fallback):", e);
    }
  }

  const allChunkIds = new Set([...vecScores.keys(), ...kwScores.keys()]);
  if (!allChunkIds.size) return [];

  // Fetch titles from documents table
  const allDocIds = [...new Set([
    ...[...vecScores.values()].map((v) => v.doc_id),
    ...[...kwScores.values()].map((v) => v.doc_id),
  ])];
  const { rows: titleRows } = await pool.query<{ doc_id: string; title: string | null }>(
    `SELECT doc_id, title FROM rag_documents WHERE doc_id = ANY($1)`,
    [allDocIds],
  );
  const titles = new Map(titleRows.map((r) => [r.doc_id, r.title]));

  // --- Fusion ---
  const fused: SearchHit[] = [];
  for (const cid of allChunkIds) {
    const vd = vecScores.get(cid);
    const kd = kwScores.get(cid);
    const data = (vd ?? kd)!;
    const vec   = vd?.score ?? 0;
    const kw    = kd?.score ?? 0;
    const final = VEC_WEIGHT * vec + KW_WEIGHT * kw;
    if (final < SCORE_THRESHOLD) continue;
    fused.push({
      chunk_id:    cid,
      doc_id:      data.doc_id,
      source_type: data.source_type,
      title:       titles.get(data.doc_id) ?? null,
      score:       Number(final.toFixed(3)),
      vec_score:   Number(vec.toFixed(3)),
      kw_score:    Number(kw.toFixed(3)),
      preview:     preview(data.text),
      ts_from:     data.ts_from,
      ts_to:       data.ts_to,
    });
  }
  fused.sort((a, b) => b.score - a.score);
  return fused.slice(0, topN);
}
