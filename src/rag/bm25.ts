export const BM25_K1 = 1.5;
export const BM25_B  = 0.75;

export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-zA-Z]\w*/g) ?? []).filter((w) => w.length > 1);
}

export function bm25Scores(
  queryTerms: string[],
  candidates: { id: number; text: string }[],
): Map<number, number> {
  if (!candidates.length || !queryTerms.length) return new Map();
  const tokenized = candidates.map((c) => ({ id: c.id, tokens: tokenize(c.text) }));
  const avgdl = tokenized.reduce((s, t) => s + t.tokens.length, 0) / tokenized.length || 1;
  const N = tokenized.length;

  const df = new Map<string, number>();
  for (const { tokens } of tokenized) {
    for (const term of new Set(tokens)) df.set(term, (df.get(term) ?? 0) + 1);
  }

  const scores = new Map<number, number>();
  for (const { id, tokens } of tokenized) {
    const dl = tokens.length;
    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    let score = 0;
    for (const term of queryTerms) {
      const dfVal = df.get(term) ?? 0;
      if (!dfVal) continue;
      const f   = tf.get(term) ?? 0;
      const idf = Math.log((N - dfVal + 0.5) / (dfVal + 0.5) + 1);
      score += idf * (f * (BM25_K1 + 1)) / (f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl));
    }
    scores.set(id, score);
  }
  return scores;
}
