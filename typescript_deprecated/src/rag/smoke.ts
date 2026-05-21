// Quick CLI smoke test for the hybrid search — usable without the TUI.
//   npx tsx src/rag/smoke.ts "who complained about the November invoice spike?"

import { search } from "./fusion.js";

async function main() {
  const query = process.argv.slice(2).join(" ").trim();
  if (!query) {
    console.error('Usage: npx tsx src/rag/smoke.ts "<query>"');
    process.exit(1);
  }
  const t0 = Date.now();
  const hits = await search(query, {}, 8);
  const dt = Date.now() - t0;
  console.log(`# ${hits.length} hits in ${dt}ms\n`);
  for (const h of hits) {
    const ts = h.ts_from ? ` [${h.ts_from}]` : "";
    console.log(
      `#${h.chunk_id}  score=${h.score}  vec=${h.vec_score}  kw=${h.kw_score}  ${h.source_type}${ts}  doc=${h.doc_id}`,
    );
    console.log(`  title: ${h.title ?? ""}`);
    console.log(`  ${h.preview}\n`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
