import { Agent } from "@mariozechner/pi-agent-core";
import { ollamaModel } from "../model.js";
import { basePrompt } from "../prompt.js";
import { searchTool, openDocumentTool } from "../tools/index.js";

/** Build a fresh Agent wired only for retrieval-style RAG tools (no filesystem/bash). */
export function newAgent(): Agent {
  return new Agent({
    initialState: {
      systemPrompt: basePrompt,
      model: ollamaModel,
      tools: [searchTool, openDocumentTool],
      messages: [],
      thinkingLevel: "off",
    },
    getApiKey: () => "ollama",
    toolExecution: "sequential",
    // search and open_document are read-only — no confirmation needed.
    beforeToolCall: async () => undefined,
  });
}
