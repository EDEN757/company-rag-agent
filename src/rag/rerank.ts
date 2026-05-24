const RERANKER = process.env.RERANKER_URL ?? "http://127.0.0.1:8001";

/**
 * Call the Python cross-encoder service to score (query, passage) pairs.
 * Returns null if the service is unreachable or times out — callers fall back
 * to the existing fusion ranking in that case.
 */
export async function rerank(query: string, passages: string[]): Promise<number[] | null> {
  if (passages.length === 0) return null;
  try {
    const r = await fetch(`${RERANKER}/rerank`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query, passages }),
      signal: AbortSignal.timeout(4000),
    });
    if (!r.ok) return null;
    const j = (await r.json()) as { scores: unknown };
    if (!Array.isArray(j.scores) || j.scores.length !== passages.length) return null;
    return j.scores as number[];
  } catch {
    return null;
  }
}
