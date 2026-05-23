import type { Agent } from "@mariozechner/pi-agent-core";
import { newAgent } from "./agent_factory.js";
import { basePrompt } from "../prompt.js";
import type { Skill } from "./skills.js";

interface SessionEntry {
  agent: Agent;
  lastActive: number;
}

const TTL_MS = 30 * 60 * 1000; // 30 minutes
const SWEEP_INTERVAL_MS = 5 * 60 * 1000; // every 5 minutes

export class Sessions {
  private readonly entries = new Map<string, SessionEntry>();
  private readonly sweep: NodeJS.Timeout;

  constructor() {
    this.sweep = setInterval(() => this.evictStale(), SWEEP_INTERVAL_MS);
    // Don't keep the process alive just for the sweep timer.
    this.sweep.unref?.();
  }

  /** Return the existing Agent for `id`, or lazily create a fresh one. */
  get(id: string): Agent {
    const now = Date.now();
    const existing = this.entries.get(id);
    if (existing) {
      existing.lastActive = now;
      return existing.agent;
    }
    const agent = newAgent();
    this.entries.set(id, { agent, lastActive: now });
    return agent;
  }

  /** Activate or clear the skill for a given session by mutating the agent's systemPrompt. */
  setSkill(id: string, skill: Skill | null): void {
    const agent = this.get(id);
    agent.state.systemPrompt = skill ? `${basePrompt}\n\n${skill.body}` : basePrompt;
  }

  /** Reset the conversation for a session (clears transcript and restores base prompt). */
  reset(id: string): void {
    const entry = this.entries.get(id);
    if (!entry) return;
    entry.agent.reset();
    entry.agent.state.systemPrompt = basePrompt;
    entry.lastActive = Date.now();
  }

  size(): number {
    return this.entries.size;
  }

  private evictStale(): void {
    const cutoff = Date.now() - TTL_MS;
    for (const [id, entry] of this.entries) {
      if (entry.lastActive < cutoff) {
        entry.agent.abort();
        this.entries.delete(id);
      }
    }
  }
}
