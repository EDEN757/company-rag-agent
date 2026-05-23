import { createReadStream, statSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { fetchDocument } from "../rag/db.js";
import { loadSkills, type Skill } from "./skills.js";
import { Sessions } from "./sessions.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../..");
const SKILLS_DIR = resolve(REPO_ROOT, "skills");
const FRONTEND_DIR = resolve(REPO_ROOT, "frontend");
const PORT = Number(process.env.PORT ?? 3000);

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".png": "image/png",
};

let skills = loadSkills(SKILLS_DIR);
const sessions = new Sessions();

function sendJson(res: ServerResponse, status: number, body: unknown) {
  const json = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(json),
  });
  res.end(json);
}

function sendText(res: ServerResponse, status: number, body: string) {
  res.writeHead(status, { "content-type": "text/plain; charset=utf-8" });
  res.end(body);
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolveBody, rejectBody) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolveBody(Buffer.concat(chunks).toString("utf8")));
    req.on("error", rejectBody);
  });
}

function serveStatic(res: ServerResponse, urlPath: string) {
  const wanted = urlPath === "/" ? "/index.html" : urlPath;
  const safePath = normalize(join(FRONTEND_DIR, wanted));
  if (!safePath.startsWith(FRONTEND_DIR)) {
    sendText(res, 403, "forbidden");
    return;
  }
  let stat;
  try {
    stat = statSync(safePath);
  } catch {
    sendText(res, 404, "not found");
    return;
  }
  if (stat.isDirectory()) {
    sendText(res, 404, "not found");
    return;
  }
  const mime = MIME[extname(safePath).toLowerCase()] ?? "application/octet-stream";
  res.writeHead(200, {
    "content-type": mime,
    "content-length": stat.size,
    "cache-control": "no-cache",
  });
  createReadStream(safePath).pipe(res);
}

function extractTextContent(content: unknown): string {
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const c of content as { type?: string; text?: string }[]) {
    if (c && c.type === "text" && typeof c.text === "string") parts.push(c.text);
  }
  return parts.join("\n");
}

async function handleChat(req: IncomingMessage, res: ServerResponse) {
  const raw = await readBody(req);
  let payload: { session_id?: string; skill_name?: string | null; message?: string };
  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch {
    sendJson(res, 400, { error: "invalid JSON body" });
    return;
  }
  const sessionId = payload.session_id?.trim();
  const message = payload.message?.trim();
  if (!sessionId || !message) {
    sendJson(res, 400, { error: "session_id and message are required" });
    return;
  }
  const skillName = payload.skill_name?.trim() || null;
  const skill: Skill | null = skillName ? skills.get(skillName) ?? null : null;
  if (skillName && !skill) {
    sendJson(res, 400, { error: `unknown skill '${skillName}'` });
    return;
  }
  const agent = sessions.get(sessionId);
  // If a previous request on this session was interrupted, abort whatever
  // run might still be in flight before starting a new one.
  if (agent.state.isStreaming) {
    agent.abort();
    await agent.waitForIdle();
  }
  sessions.setSkill(sessionId, skill);

  // Open SSE stream.
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache, no-transform",
    "connection": "keep-alive",
    "x-accel-buffering": "no",
  });
  res.flushHeaders?.();

  const write = (event: Record<string, unknown>) => {
    if (res.writableEnded) return;
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  };

  // Heartbeat — keeps proxies from killing the connection on long LLM calls.
  const heartbeat = setInterval(() => {
    if (!res.writableEnded) res.write(": keep-alive\n\n");
  }, 15_000);
  heartbeat.unref?.();

  let closed = false;
  req.on("close", () => {
    closed = true;
    agent.abort();
  });

  const unsubscribe = agent.subscribe(async (event) => {
    if (closed) return;
    switch (event.type) {
      case "message_update": {
        const ev = event.assistantMessageEvent;
        if (ev && ev.type === "text_delta") {
          const delta = (ev as { delta?: string }).delta ?? "";
          if (delta) write({ type: "text", delta });
        }
        break;
      }
      case "tool_execution_start": {
        write({
          type: "tool_start",
          id: event.toolCallId,
          name: event.toolName,
          args: event.args,
        });
        break;
      }
      case "tool_execution_end": {
        const result = event.result as { content?: unknown; details?: unknown } | undefined;
        const text = extractTextContent(result?.content);
        write({
          type: "tool_end",
          id: event.toolCallId,
          name: event.toolName,
          isError: event.isError,
          summary: text.length > 600 ? `${text.slice(0, 600)}…` : text,
          details: result?.details ?? null,
        });
        break;
      }
      default:
        break;
    }
  });

  try {
    await agent.prompt(message);
    await agent.waitForIdle();
  } catch (err) {
    write({ type: "error", message: (err as Error).message });
  } finally {
    unsubscribe();
    clearInterval(heartbeat);
    write({ type: "done" });
    res.end();
  }
}

const server = createServer(async (req, res) => {
  const method = req.method ?? "GET";
  const url = req.url ?? "/";
  // Strip query string for routing.
  const path = url.split("?")[0];

  try {
    if (method === "GET" && path === "/api/skills") {
      const out = Array.from(skills.values()).map((s) => ({
        name: s.name,
        description: s.description,
        suggested_question: s.suggested_question,
      }));
      sendJson(res, 200, out);
      return;
    }
    if (method === "POST" && path === "/api/skills/reload") {
      skills = loadSkills(SKILLS_DIR);
      sendJson(res, 200, { reloaded: skills.size });
      return;
    }
    if (method === "GET" && path.startsWith("/api/doc/")) {
      const docId = decodeURIComponent(path.slice("/api/doc/".length));
      if (!docId) {
        sendJson(res, 400, { error: "doc_id required" });
        return;
      }
      const row = fetchDocument(docId);
      if (!row) {
        sendJson(res, 404, { error: `doc ${docId} not found` });
        return;
      }
      let metadata: unknown = null;
      if (row.metadata_json) {
        try {
          metadata = JSON.parse(row.metadata_json);
        } catch {
          metadata = row.metadata_json;
        }
      }
      sendJson(res, 200, {
        doc_id: row.doc_id,
        source_type: row.source_type,
        title: row.title,
        content: row.content,
        metadata,
      });
      return;
    }
    if (method === "POST" && path === "/api/chat") {
      await handleChat(req, res);
      return;
    }
    if (method === "POST" && path === "/api/session/reset") {
      const raw = await readBody(req);
      let payload: { session_id?: string };
      try {
        payload = raw ? JSON.parse(raw) : {};
      } catch {
        sendJson(res, 400, { error: "invalid JSON body" });
        return;
      }
      if (!payload.session_id) {
        sendJson(res, 400, { error: "session_id required" });
        return;
      }
      sessions.reset(payload.session_id);
      sendJson(res, 200, { ok: true });
      return;
    }
    if (method === "GET") {
      serveStatic(res, path);
      return;
    }
    sendText(res, 405, "method not allowed");
  } catch (err) {
    console.error("[server] unhandled error:", err);
    if (!res.headersSent) sendJson(res, 500, { error: (err as Error).message });
    else res.end();
  }
});

server.listen(PORT, () => {
  console.log(`[server] listening on http://localhost:${PORT}`);
  console.log(`[server] loaded ${skills.size} skill(s): ${[...skills.keys()].join(", ")}`);
  console.log(`[server] frontend root: ${FRONTEND_DIR}`);
});
