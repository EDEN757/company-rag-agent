import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Type } from "@mariozechner/pi-ai";

import { search } from "../rag/fusion.js";

export const searchTool: AgentTool = {
  name: "search",
  label: "Hybrid search",
  description:
    "Search the company knowledge base (documents, emails, chats) using a hybrid of BM25 keyword search and dense vector embeddings. Returns ranked chunks with a short preview so you can decide which document to open in full. Use optional filters to narrow by source, date range, or participant.",
  parameters: Type.Object({
    query: Type.String({ description: "Natural-language search query." }),
    source_types: Type.Optional(
      Type.Array(Type.String(), {
        description:
          "Optional list of source types to restrict the search to. One or more of: slack, gmail, linear, jira, confluence, google_drive, hubspot, github, fireflies.",
      }),
    ),
    date_from: Type.Optional(
      Type.String({
        description:
          "Optional ISO-8601 date/datetime lower bound. NOTE: only gmail chunks are time-stamped; other sources (slack, confluence, jira, linear, hubspot, github, fireflies, google_drive) pass through this filter unchanged. Use this when the user explicitly asks about emails in a date range. See docs/date-filter.md.",
      }),
    ),
    date_to: Type.Optional(
      Type.String({
        description:
          "Optional ISO-8601 date/datetime upper bound. Same caveat as date_from — effectively gmail-only. See docs/date-filter.md.",
      }),
    ),
    participant: Type.Optional(
      Type.String({
        description:
          "Optional substring to match against participants (email or Slack handle). Use when the user asks who said/sent something.",
      }),
    ),
    top_n: Type.Optional(
      Type.Number({
        description: "How many fused hits to return (default 6, max 20).",
      }),
    ),
  }),
  execute: async (_id, params) => {
    const p = params as {
      query: string;
      source_types?: string[];
      date_from?: string;
      date_to?: string;
      participant?: string;
      top_n?: number;
    };
    const topN = Math.max(1, Math.min(20, p.top_n ?? 6));
    const hits = await search(
      p.query,
      {
        source_types: p.source_types,
        date_from: p.date_from,
        date_to: p.date_to,
        participant: p.participant,
      },
      topN,
    );
    if (hits.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: "No results above the fusion threshold. Try broadening the query or removing filters.",
          },
        ],
        details: { hits: 0 },
      };
    }
    const lines: string[] = [];
    for (const h of hits) {
      const time = h.ts_from ? ` [${h.ts_from}${h.ts_to && h.ts_to !== h.ts_from ? ` → ${h.ts_to}` : ""}]` : "";
      lines.push(
        `#${h.chunk_id} (doc=${h.doc_id}, source=${h.source_type}${time}, score=${h.score} vec=${h.vec_score} kw=${h.kw_score})\n` +
          `title: ${h.title ?? ""}\n` +
          `preview: ${h.preview}`,
      );
    }
    return {
      content: [{ type: "text", text: lines.join("\n\n") }],
      details: { hits: hits.length, results: hits },
    };
  },
};
