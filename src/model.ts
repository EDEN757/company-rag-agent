import type { Model } from "@mariozechner/pi-ai";

export const ollamaModel: Model<"openai-completions"> = {
  id: "qwen3-8b-32k",
  name: "Qwen 3 8B 32k (Ollama, custom)",
  api: "openai-completions",
  provider: "ollama",
  baseUrl: "http://localhost:11434/v1",
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
