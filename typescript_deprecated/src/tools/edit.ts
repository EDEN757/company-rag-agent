import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Type } from "@mariozechner/pi-ai";
import { readFile, writeFile } from "node:fs/promises";

export const editTool: AgentTool = {
  name: "edit",
  label: "Edit file",
  description:
    "Replace the first occurrence of `old_string` with `new_string` in a file. Fails if `old_string` does not appear exactly once.",
  parameters: Type.Object({
    path: Type.String(),
    old_string: Type.String(),
    new_string: Type.String(),
  }),
  execute: async (_id, params) => {
    const { path, old_string, new_string } = params as {
      path: string;
      old_string: string;
      new_string: string;
    };
    const resolved = path.startsWith("~/")
      ? path.replace(/^~/, process.env.HOME ?? "")
      : path;
    const text = await readFile(resolved, "utf8");
    const occurrences = text.split(old_string).length - 1;
    if (occurrences === 0) throw new Error(`old_string not found in ${resolved}`);
    if (occurrences > 1)
      throw new Error(
        `old_string appears ${occurrences} times in ${resolved} — make it more specific`,
      );
    const updated = text.replace(old_string, new_string);
    await writeFile(resolved, updated, "utf8");
    return {
      content: [{ type: "text", text: `Edited ${resolved}` }],
      details: { path: resolved, before: text.length, after: updated.length },
    };
  },
};
