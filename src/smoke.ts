import { Agent } from "@mariozechner/pi-agent-core";
import { ollamaModel } from "./model.js";
import { systemPrompt } from "./prompt.js";
import { bashTool, editTool, readTool, writeTool } from "./tools/index.js";

const agent = new Agent({
  initialState: {
    systemPrompt,
    model: ollamaModel,
    tools: [readTool, writeTool, editTool, bashTool],
    messages: [],
    thinkingLevel: "off",
  },
  getApiKey: () => "ollama",
});

agent.subscribe((event) => {
  if (event.type === "message_update") {
    if (event.assistantMessageEvent?.type === "text_delta") {
      process.stdout.write((event.assistantMessageEvent as { delta: string }).delta);
    }
  }
  if (event.type === "tool_execution_start") {
    console.log(`\n[tool_start] ${event.toolName} ${JSON.stringify(event.args)}`);
  }
  if (event.type === "tool_execution_end") {
    const r = event.result as { content?: { type: string; text?: string }[] };
    const txt = r?.content?.[0]?.text ?? "";
    console.log(`[tool_end ${event.isError ? "ERR" : "ok"}] ${txt.slice(0, 200)}${txt.length > 200 ? "…" : ""}`);
  }
  if (event.type === "agent_end") {
    console.log("\n[agent_end]");
  }
});

const userPrompt = process.argv.slice(2).join(" ") || "What's in my Downloads folder? Use the bash tool with `ls ~/Downloads` and tell me how many items.";
console.log(`> ${userPrompt}\n`);
await agent.prompt(userPrompt);
await agent.waitForIdle();
process.exit(0);
