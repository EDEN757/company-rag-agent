import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Type } from "@mariozechner/pi-ai";
import { writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";

export const writeTool: AgentTool = {
  name: "write",
  label: "Write file",
  description:
    "Write UTF-8 text to a file, overwriting any existing content. Creates parent directories.",
  parameters: Type.Object({
    path: Type.String({ description: "Absolute or ~-prefixed path." }),
    content: Type.String(),
  }),
  execute: async (_id, params) => {
    const { path, content } = params as { path: string; content: string };
    const resolved = path.startsWith("~/")
      ? path.replace(/^~/, process.env.HOME ?? "")
      : path;
    await mkdir(dirname(resolved), { recursive: true });
    await writeFile(resolved, content, "utf8");
    return {
      content: [{ type: "text", text: `Wrote ${content.length} bytes to ${resolved}` }],
      details: { path: resolved, bytes: content.length },
    };
  },
};
