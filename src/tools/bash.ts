import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Type } from "@mariozechner/pi-ai";
import { spawn } from "node:child_process";

const DENY_PATTERNS: RegExp[] = [
  /\brm\s+-rf\b/,
  /\bsudo\b/,
  /\bdd\s+if=/,
  /\bmkfs\b/,
  /\b>\s*\/dev\//,
  /\bshutdown\b/,
  /\breboot\b/,
];

function runBash(cmd: string, signal?: AbortSignal): Promise<{ stdout: string; stderr: string; code: number }> {
  return new Promise((resolve, reject) => {
    const child = spawn("/bin/bash", ["-lc", cmd], { signal });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d.toString()));
    child.stderr.on("data", (d) => (stderr += d.toString()));
    child.on("close", (code) => resolve({ stdout, stderr, code: code ?? 0 }));
    child.on("error", reject);
  });
}

export const bashTool: AgentTool = {
  name: "bash",
  label: "Run bash",
  description:
    "Run a shell command via /bin/bash -lc. Returns stdout, stderr, and exit code. Refuses dangerous patterns.",
  parameters: Type.Object({
    command: Type.String({ description: "The shell command to run." }),
  }),
  execute: async (_id, params, signal) => {
    const { command } = params as { command: string };
    const matched = DENY_PATTERNS.find((p) => p.test(command));
    if (matched) {
      throw new Error(
        `Refused: command matches deny pattern ${matched}. Ask the user to run it manually.`,
      );
    }
    const { stdout, stderr, code } = await runBash(command, signal);
    const text = `exit=${code}\n--- stdout ---\n${stdout}${stderr ? `\n--- stderr ---\n${stderr}` : ""}`;
    return {
      content: [{ type: "text", text }],
      details: { command, exitCode: code },
    };
  },
};
