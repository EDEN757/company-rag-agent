import { Agent } from "@mariozechner/pi-agent-core";
import {
  CombinedAutocompleteProvider,
  Editor,
  type EditorTheme,
  ProcessTerminal,
  type SlashCommand,
  Text,
  TUI,
} from "@mariozechner/pi-tui";

import { ollamaModel } from "./model.js";
import { confirmToolCall } from "./permissions.js";
import { systemPrompt } from "./prompt.js";
import { bashTool, editTool, openDocumentTool, readTool, searchTool, writeTool } from "./tools/index.js";

const identity = (s: string) => s;

const editorTheme: EditorTheme = {
  borderColor: identity,
  selectList: {
    selectedPrefix: identity,
    selectedText: identity,
    description: identity,
    scrollInfo: identity,
    noMatch: identity,
  },
};

const AUTO_ALLOWED = new Set<string>(["read", "search", "open_document"]);
const sessionAllowed = new Set<string>();

const agent = new Agent({
  initialState: {
    systemPrompt,
    model: ollamaModel,
    tools: [searchTool, openDocumentTool, readTool, writeTool, editTool, bashTool],
    messages: [],
    thinkingLevel: "off",
  },
  getApiKey: () => "ollama",
  // Tool execution must be sequential so overlays don't stack.
  toolExecution: "sequential",
  beforeToolCall: async ({ toolCall, args }) => {
    if (AUTO_ALLOWED.has(toolCall.name)) return undefined;
    if (sessionAllowed.has(toolCall.name)) return undefined;
    const decision = await confirmToolCall(tui, toolCall.name, args);
    if (decision === "always") {
      sessionAllowed.add(toolCall.name);
      return undefined;
    }
    if (decision === "once") return undefined;
    return { block: true, reason: `User denied "${toolCall.name}".` };
  },
});

const terminal = new ProcessTerminal();
const tui = new TUI(terminal);

let buffer = "Personal agent — type and press Enter. Ctrl+C or /quit to exit.\n";
const transcript = new Text(buffer);
tui.addChild(transcript);

const editor = new Editor(tui, editorTheme);
tui.addChild(editor);
tui.setFocus(editor);

const slashCommands: SlashCommand[] = [
  { name: "quit", description: "Exit the agent" },
  { name: "exit", description: "Exit the agent" },
  { name: "reset", description: "Clear the conversation transcript" },
];
editor.setAutocompleteProvider(
  new CombinedAutocompleteProvider(slashCommands, process.cwd()),
);

const append = (chunk: string) => {
  buffer += chunk;
  transcript.setText(buffer);
  tui.requestRender();
};

agent.subscribe((event) => {
  switch (event.type) {
    case "message_start":
      if (event.message.role === "assistant") {
        append("\nassistant: ");
      }
      break;
    case "message_update":
      if (
        event.assistantMessageEvent &&
        event.assistantMessageEvent.type === "text_delta"
      ) {
        const delta = (event.assistantMessageEvent as { delta: string }).delta;
        append(delta);
      }
      break;
    case "message_end":
      if (event.message.role === "assistant") {
        // Newline after each assistant message
        append("\n");
      }
      break;
    case "tool_execution_start":
      append(`\n[tool] ${event.toolName} ${JSON.stringify(event.args)} ...`);
      break;
    case "tool_execution_end":
      append(event.isError ? " ERROR\n" : " done\n");
      break;
    case "agent_end":
      append("\n");
      break;
  }
});

const quit = () => {
  agent.abort();
  tui.stop();
  process.exit(0);
};

editor.onSubmit = async (text: string) => {
  const trimmed = text.trim();
  if (!trimmed) return;
  editor.setText("");
  if (trimmed.startsWith("/")) {
    const cmd = trimmed.slice(1).split(/\s+/)[0];
    if (cmd === "quit" || cmd === "exit") {
      quit();
      return;
    }
    if (cmd === "reset") {
      agent.reset();
      buffer = "Personal agent — type and press Enter. Ctrl+C or /quit to exit.\n";
      transcript.setText(buffer);
      tui.requestRender();
      return;
    }
    append(`\n[unknown command] /${cmd}\n`);
    return;
  }
  append(`\nyou: ${trimmed}\n`);
  try {
    await agent.prompt(trimmed);
  } catch (err) {
    append(`\n[error] ${(err as Error).message}\n`);
  }
};

// In raw mode, Ctrl+C arrives as 0x03 and does NOT raise SIGINT automatically.
tui.addInputListener((data) => {
  if (data === "\x03") {
    quit();
    return { consume: true };
  }
  return undefined;
});

tui.start();
