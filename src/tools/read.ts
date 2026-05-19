import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Type } from "@mariozechner/pi-ai";
import { readFile } from "node:fs/promises";

export const readTool: AgentTool = {
  name: "read",
  label: "Read file",
  description: "Read a UTF-8 text file from disk and return its contents.",
  parameters: Type.Object({
    path: Type.String({ description: "Absolute or ~-prefixed path." }),
  }),
  execute: async (_id, params) => {
    const { path } = params as { path: string };
    const resolved = path.startsWith("~/")
      ? path.replace(/^~/, process.env.HOME ?? "")
      : path;
    const text = await readFile(resolved, "utf8");
    return {
      content: [{ type: "text", text }],
      details: { path: resolved, bytes: text.length },
    };
  },
};
