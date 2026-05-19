import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Type } from "@mariozechner/pi-ai";

import { fetchDocument } from "../rag/db.js";

export const openDocumentTool: AgentTool = {
  name: "open_document",
  label: "Open document",
  description:
    "Fetch the full text of a document from the knowledge base by doc_id (as returned by `search`). Use this after `search` when a preview looks relevant and you need the complete content to answer.",
  parameters: Type.Object({
    doc_id: Type.String({ description: "The doc_id from a search result, e.g. dsid_abc123…" }),
  }),
  execute: async (_id, params) => {
    const { doc_id } = params as { doc_id: string };
    const row = fetchDocument(doc_id);
    if (!row) {
      return {
        content: [{ type: "text", text: `No document with doc_id=${doc_id}` }],
        details: { found: false },
      };
    }
    const header = `doc_id: ${row.doc_id}\nsource: ${row.source_type}\ntitle: ${row.title ?? ""}\n\n`;
    return {
      content: [{ type: "text", text: header + row.content }],
      details: {
        found: true,
        doc_id: row.doc_id,
        source_type: row.source_type,
        title: row.title,
        chars: row.content.length,
      },
    };
  },
};
