const OLLAMA = process.env.OLLAMA_HOST ?? "http://127.0.0.1:11434";
const MODEL = process.env.RAG_EMBED_MODEL ?? "nomic-embed-text";

function l2normalize(v: number[]): Float32Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

export async function embedQuery(text: string): Promise<Float32Array> {
  const r = await fetch(`${OLLAMA}/api/embeddings`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, prompt: text }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!r.ok) {
    throw new Error(`Ollama embedding request failed: ${r.status} ${await r.text()}`);
  }
  const j = (await r.json()) as { embedding: number[] };
  return l2normalize(j.embedding);
}
