import type { Model } from "@mariozechner/pi-ai";

const OLLAMA_HOST = process.env.OLLAMA_HOST ?? "http://127.0.0.1:11434";
const LLM_MODEL = process.env.LLM_MODEL ?? "qwen3.5-9b-32k";

export const ollamaModel: Model<"openai-completions"> = {
  id: LLM_MODEL,
  name: `Ollama: ${LLM_MODEL}`,
  api: "openai-completions",
  provider: "ollama",
  baseUrl: `${OLLAMA_HOST.replace(/\/$/, "")}/v1`,
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 32_000,
  maxTokens: 4096,
  compat: {
    supportsStore: false,
    supportsDeveloperRole: false,
    supportsReasoningEffort: false,
  },
};
