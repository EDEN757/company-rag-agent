import { search } from "./fusion.js";

async function main() {
  console.log("=== participant filter: alex@hybridai.io ===");
  const a = await search("invoice spike", { participant: "alex@hybridai.io" }, 3);
  for (const h of a)
    console.log(`#${h.chunk_id}  score=${h.score}  ${h.source_type}  doc=${h.doc_id}  title=${h.title}`);

  console.log("\n=== source filter: slack only ===");
  const b = await search("invoice spike", { source_types: ["slack"] }, 3);
  for (const h of b)
    console.log(`#${h.chunk_id}  score=${h.score}  ${h.source_type}  doc=${h.doc_id}  title=${h.title}`);

  console.log("\n=== date filter: 2026-11 ===");
  const c = await search("invoice spike", { date_from: "2026-11-01", date_to: "2026-11-30" }, 3);
  for (const h of c)
    console.log(`#${h.chunk_id}  score=${h.score}  ${h.source_type}  doc=${h.doc_id}  ts=${h.ts_from}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
