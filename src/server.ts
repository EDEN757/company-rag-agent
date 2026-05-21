import express from "express";
import { createHash, randomBytes } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { resolve, dirname } from "node:path";
import { exec } from "node:child_process";
import { fileURLToPath } from "node:url";

import { pool, fetchDocument } from "./rag/db-pg.js";
import { search, type SearchFilters, type SearchHit } from "./rag/fusion-pg.js";
import { embedQuery } from "./rag/embed.js";
import { systemPrompt } from "./prompt.js";

const OLLAMA_HOST    = process.env.OLLAMA_HOST  ?? "http://localhost:11434";
const LLM_MODEL      = process.env.LLM_MODEL    ?? "qwen3:8b";
const MAX_NEW_TOKENS = 512;
const TEMPERATURE    = 0.1;
const MAX_TURNS      = 6;
const THINK_RE       = /<think>([\s\S]*?)<\/think>/;

const here = dirname(fileURLToPath(import.meta.url));

// ── Tool definitions (OpenAI format) ─────────────────────────────────────────
const TOOL_DEFS = [
  {
    type: "function",
    function: {
      name: "search",
      description:
        "Search the company knowledge base using hybrid BM25 + vector retrieval. Returns ranked chunks with previews.",
      parameters: {
        type: "object",
        properties: {
          query:        { type: "string", description: "Natural-language search query." },
          source_types: { type: "array", items: { type: "string" }, description: "Filter by source: slack, gmail, linear, jira, confluence, google_drive, hubspot, github, fireflies." },
          date_from:    { type: "string", description: "ISO-8601 date lower bound (ts_to >= date_from)." },
          date_to:      { type: "string", description: "ISO-8601 date upper bound (ts_from <= date_to)." },
          participant:  { type: "string", description: "Substring match against participants." },
          top_n:        { type: "number", description: "Max hits to return (default 6, max 20)." },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "open_document",
      description: "Fetch the full text of a document by doc_id (as returned by search).",
      parameters: {
        type: "object",
        properties: { doc_id: { type: "string" } },
        required: ["doc_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "add_document",
      description:
        "Create and index a new document in the knowledge base. Use this when the user asks to write or save something — e.g. 'write a gmail about the meeting'. Do NOT use local write tool for knowledge-base content.",
      parameters: {
        type: "object",
        properties: {
          source_type:  { type: "string", description: "One of: gmail, slack, confluence, jira, linear, google_drive, hubspot, github, fireflies." },
          title:        { type: "string" },
          content:      { type: "string", description: "Document content. Keep under 150 words unless asked for more." },
          participants: { type: "string", description: "Comma-separated names/emails." },
          date:         { type: "string", description: "ISO-8601 date (YYYY-MM-DD)." },
        },
        required: ["source_type", "title", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "edit_document",
      description: "Update an existing knowledge-base document and re-index it.",
      parameters: {
        type: "object",
        properties: {
          doc_id:      { type: "string" },
          new_content: { type: "string", description: "Replace the entire content." },
          old_string:  { type: "string", description: "Exact text to replace (alternative to new_content)." },
          new_string:  { type: "string", description: "Replacement text." },
        },
        required: ["doc_id"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read",
      description: "Read a local file on the server filesystem.",
      parameters: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write",
      description: "Write text to a local file on the server filesystem.",
      parameters: {
        type: "object",
        properties: {
          path:    { type: "string" },
          content: { type: "string" },
        },
        required: ["path", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "edit",
      description: "Replace one exact occurrence of old_string with new_string in a local file.",
      parameters: {
        type: "object",
        properties: {
          path:       { type: "string" },
          old_string: { type: "string" },
          new_string: { type: "string" },
        },
        required: ["path", "old_string", "new_string"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "bash",
      description: "Run a shell command on the server via /bin/bash.",
      parameters: {
        type: "object",
        properties: { command: { type: "string" } },
        required: ["command"],
      },
    },
  },
];

// ── Knowledge-base write helpers ──────────────────────────────────────────────
function newDocId(): string {
  return "dsid_" + createHash("md5").update(randomBytes(16)).digest("hex");
}

function simpleChunk(text: string, chunkSize = 2000, overlap = 200): string[] {
  if (text.length <= chunkSize) return [text];
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    chunks.push(text.slice(start, end));
    if (end === text.length) break;
    start += chunkSize - overlap;
  }
  return chunks;
}

async function insertChunks(
  docId: string, sourceType: string, title: string, content: string,
  participants: string | null, tsFrom: string | null, tsTo: string | null,
): Promise<void> {
  const participantsJson = participants
    ? JSON.stringify(participants.split(",").map((p) => p.trim()))
    : "[]";
  const chunks = simpleChunk(content);
  for (let ord = 0; ord < chunks.length; ord++) {
    const headerParts = [`[source: ${sourceType}]`, `[title: ${title}]`];
    if (participants) headerParts.push(`[participants: ${participants}]`);
    if (tsFrom) {
      const dateStr = tsFrom + (tsTo && tsTo !== tsFrom ? ` -> ${tsTo}` : "");
      headerParts.push(`[dates: ${dateStr}]`);
    }
    const fullText  = headerParts.join(" ") + "\n\n" + chunks[ord];
    const embedding = await embedQuery(fullText);
    const vecStr    = "[" + Array.from(embedding).join(",") + "]";
    await pool.query(
      `INSERT INTO rag_chunks
         (doc_id, source_type, ord, header, text, ts_from, ts_to, participants_json, embedding)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector)`,
      [docId, sourceType, ord, "", fullText, tsFrom, tsTo, participantsJson, vecStr],
    );
  }
}

async function dbAddDocument(
  sourceType: string, title: string, content: string,
  participants?: string, date?: string,
): Promise<string> {
  const docId = newDocId();
  const ts    = date ?? null;
  await pool.query(
    `INSERT INTO rag_documents (doc_id, source_type, title, content) VALUES ($1, $2, $3, $4)`,
    [docId, sourceType, title, content],
  );
  await insertChunks(docId, sourceType, title, content, participants ?? null, ts, ts);
  return docId;
}

async function dbEditDocument(
  docId: string, newContent?: string, oldString?: string, newString?: string,
): Promise<string> {
  const doc = await fetchDocument(docId);
  if (!doc) return `Document not found: ${docId}`;
  let updated: string;
  if (newContent !== undefined) {
    updated = newContent;
  } else if (oldString !== undefined && newString !== undefined) {
    const count = (doc.content.match(new RegExp(oldString.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g")) ?? []).length;
    if (count === 0) return `old_string not found in ${docId}`;
    if (count > 1)   return `old_string appears ${count} times — make it more specific`;
    updated = doc.content.replace(oldString, newString);
  } else {
    return "Provide either new_content or both old_string and new_string.";
  }
  await pool.query(`UPDATE rag_documents SET content = $1 WHERE doc_id = $2`, [updated, docId]);
  await pool.query(`DELETE FROM rag_chunks WHERE doc_id = $1`, [docId]);
  await insertChunks(docId, doc.source_type, doc.title ?? "", updated, null, null, null);
  return `Updated ${docId} and re-indexed.`;
}

// ── File tool helpers ─────────────────────────────────────────────────────────
function expandPath(p: string): string {
  return p.startsWith("~/") ? resolve(homedir(), p.slice(2)) : resolve(p);
}

const BASH_DENY = [/\brm\s+-rf\b/, /\bsudo\b/, /\bdd\s+if=/, /\bmkfs\b/, /\b>\s*\/dev\//, /\bshutdown\b/, /\breboot\b/];

async function runBash(command: string, signal: AbortSignal): Promise<string> {
  for (const pat of BASH_DENY) {
    if (pat.test(command)) return `Refused: command matches denied pattern ${pat}`;
  }
  return new Promise((res) => {
    const child = exec(`/bin/bash -lc ${JSON.stringify(command)}`, { timeout: 30_000 }, (err, stdout, stderr) => {
      res(stdout + (stderr ? `\nstderr: ${stderr}` : "") + (err ? `\nexit: ${err.code}` : ""));
    });
    signal.addEventListener("abort", () => child.kill());
  });
}

// ── Source type ───────────────────────────────────────────────────────────────
interface Source {
  doc_id: string;
  source_type: string;
  title: string | null;
  score: number;
  preview: string;
  opened: boolean;
}

// ── Tool executor ─────────────────────────────────────────────────────────────
async function executeTool(
  name: string,
  args: Record<string, unknown>,
  signal: AbortSignal,
): Promise<{ text: string; sources: Source[] }> {
  switch (name) {
    case "search": {
      const hits = await search(
        args.query as string,
        {
          source_types: args.source_types as string[] | undefined,
          date_from:    args.date_from    as string | undefined,
          date_to:      args.date_to      as string | undefined,
          participant:  args.participant  as string | undefined,
        } satisfies SearchFilters,
        Math.max(1, Math.min(20, (args.top_n as number | undefined) ?? 6)),
      );
      if (!hits.length) {
        return { text: "No results above the fusion threshold. Try broadening the query or removing filters.", sources: [] };
      }
      const lines = hits.map((h) => {
        const time = h.ts_from ? ` [${h.ts_from}${h.ts_to && h.ts_to !== h.ts_from ? ` → ${h.ts_to}` : ""}]` : "";
        return `#${h.chunk_id} (doc=${h.doc_id}, source=${h.source_type}${time}, score=${h.score} vec=${h.vec_score} kw=${h.kw_score})\ntitle: ${h.title ?? ""}\npreview: ${h.preview}`;
      });
      const sources: Source[] = hits.map((h) => ({
        doc_id: h.doc_id, source_type: h.source_type, title: h.title,
        score: h.score, preview: h.preview, opened: false,
      }));
      return { text: lines.join("\n\n"), sources };
    }

    case "open_document": {
      const doc = await fetchDocument(args.doc_id as string);
      if (!doc) return { text: `No document found with doc_id=${args.doc_id}`, sources: [] };
      return {
        text: `doc_id: ${doc.doc_id}\nsource: ${doc.source_type}\ntitle: ${doc.title ?? ""}\n\n${doc.content}`,
        sources: [{
          doc_id: doc.doc_id, source_type: doc.source_type, title: doc.title,
          score: 0, preview: doc.content.slice(0, 320).replace(/\n/g, " ").trim(), opened: true,
        }],
      };
    }

    case "add_document": {
      const docId = await dbAddDocument(
        args.source_type as string,
        args.title       as string,
        args.content     as string,
        args.participants as string | undefined,
        args.date         as string | undefined,
      );
      return {
        text: `Created and indexed ${docId} (source=${args.source_type}, title=${JSON.stringify(args.title)}). Include ${docId} in your answer.`,
        sources: [{
          doc_id: docId, source_type: args.source_type as string, title: args.title as string,
          score: 0, preview: (args.content as string).slice(0, 320).replace(/\n/g, " ").trim(), opened: true,
        }],
      };
    }

    case "edit_document": {
      const result = await dbEditDocument(
        args.doc_id      as string,
        args.new_content as string | undefined,
        args.old_string  as string | undefined,
        args.new_string  as string | undefined,
      );
      const doc = await fetchDocument(args.doc_id as string);
      return {
        text: result,
        sources: doc ? [{
          doc_id: doc.doc_id, source_type: doc.source_type, title: doc.title,
          score: 0, preview: doc.content.slice(0, 320).replace(/\n/g, " ").trim(), opened: true,
        }] : [],
      };
    }

    case "read": {
      const p = expandPath(args.path as string);
      try {
        const content = await readFile(p, "utf-8");
        return { text: content, sources: [] };
      } catch (e) {
        return { text: `Error reading ${p}: ${(e as Error).message}`, sources: [] };
      }
    }

    case "write": {
      const p = expandPath(args.path as string);
      await mkdir(dirname(p), { recursive: true });
      await writeFile(p, args.content as string, "utf-8");
      return { text: `Wrote ${p}`, sources: [] };
    }

    case "edit": {
      const p = expandPath(args.path as string);
      let text: string;
      try { text = await readFile(p, "utf-8"); } catch (e) {
        return { text: `Error reading ${p}: ${(e as Error).message}`, sources: [] };
      }
      const old = args.old_string as string;
      const count = (text.match(new RegExp(old.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g")) ?? []).length;
      if (count === 0) return { text: `old_string not found in ${p}`, sources: [] };
      if (count > 1)   return { text: `old_string appears ${count} times — make it more specific`, sources: [] };
      await writeFile(p, text.replace(old, args.new_string as string), "utf-8");
      return { text: `Edited ${p}`, sources: [] };
    }

    case "bash": {
      const out = await runBash(args.command as string, signal);
      return { text: out, sources: [] };
    }

    default:
      return { text: `Unknown tool: ${name}`, sources: [] };
  }
}

// ── Agent loop ────────────────────────────────────────────────────────────────
type Message = Record<string, unknown>;

async function runAgent(
  question: string,
  history: { role: string; content: string }[],
  thinkingMode: boolean,
  signal: AbortSignal,
  emit: (event: Record<string, unknown>) => void,
): Promise<{ answer: string; sources: Source[]; traces: string[] }> {
  const messages: Message[] = [
    { role: "system", content: systemPrompt },
    ...history,
    { role: "user", content: question },
  ];

  const sources: Source[] = [];
  const traces:  string[] = [];
  let   answer  = "";

  for (let turn = 0; turn < MAX_TURNS; turn++) {
    if (signal.aborted) break;

    let resp: Response;
    try {
      resp = await fetch(`${OLLAMA_HOST}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": "Bearer ollama" },
        body: JSON.stringify({
          model: LLM_MODEL,
          messages,
          tools: TOOL_DEFS,
          max_tokens: MAX_NEW_TOKENS,
          temperature: TEMPERATURE,
          options: { num_ctx: 32768, ...(thinkingMode ? { think: true } : {}) },
        }),
        signal,
      });
    } catch (e) {
      if ((e as Error).name === "AbortError") break;
      throw e;
    }

    if (!resp.ok) throw new Error(`LLM error ${resp.status}: ${await resp.text()}`);
    const data = await resp.json() as {
      choices: { message: { role: string; content: string | null; tool_calls?: { id: string; function: { name: string; arguments: string } }[] } }[]
    };

    const msg = data.choices[0].message;
    const rawContent = msg.content ?? "";

    // Extract thinking block
    const thinkMatch = THINK_RE.exec(rawContent);
    if (thinkMatch) {
      emit({ type: "thinking", content: thinkMatch[1].trim() });
    }
    const visibleContent = rawContent.replace(THINK_RE, "").trim();

    if (msg.tool_calls && msg.tool_calls.length > 0) {
      messages.push({ role: "assistant", content: rawContent, tool_calls: msg.tool_calls });
      for (const tc of msg.tool_calls) {
        if (signal.aborted) break;
        let args: Record<string, unknown> = {};
        try { args = JSON.parse(tc.function.arguments || "{}"); } catch {}
        emit({ type: "tool_start", name: tc.function.name, args });
        const { text, sources: newSources } = await executeTool(tc.function.name, args, signal);
        sources.push(...newSources);
        traces.push(`${tc.function.name}(${tc.function.arguments}) → ${text.slice(0, 120)}`);
        emit({ type: "tool_end", name: tc.function.name });
        messages.push({ role: "tool", tool_call_id: tc.id, content: text });
      }
    } else {
      answer = visibleContent;
      break;
    }
  }

  if (!answer && signal.aborted) answer = "Search interrupted by user.";
  return { answer, sources, traces };
}

// ── Express app ───────────────────────────────────────────────────────────────
const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(express.static(resolve(here, "../frontend")));

// Health check
app.get("/health", (_req, res) => res.json({ status: "ok", model: LLM_MODEL }));

// Document fetch
app.get("/doc/:id", async (req, res) => {
  try {
    const doc = await fetchDocument(req.params.id);
    if (!doc) { res.status(404).json({ error: "Not found" }); return; }
    res.json(doc);
  } catch (e) {
    res.status(500).json({ error: (e as Error).message });
  }
});

// Main query endpoint — SSE
app.post("/query", async (req, res) => {
  const { question, history = [], thinking_mode = false } = req.body as {
    question: string;
    history: { role: string; content: string }[];
    thinking_mode: boolean;
  };

  if (!question?.trim()) { res.status(400).json({ error: "question required" }); return; }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const emit = (data: Record<string, unknown>) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  const controller = new AbortController();
  req.on("close", () => controller.abort());

  const t0 = Date.now();
  try {
    const { answer, sources, traces } = await runAgent(
      question, history, thinking_mode, controller.signal, emit,
    );
    emit({ type: "done", answer, sources, traces, latency_ms: Date.now() - t0 });
  } catch (e) {
    emit({ type: "error", message: (e as Error).message });
  }
  res.end();
});

export { app };
