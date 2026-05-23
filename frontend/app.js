// Single-file frontend client. No build step, no framework.

const DOC_ID_REGEX = /\b(dsid_[a-f0-9]+|demo_[a-z0-9_]+)\b/gi;

const state = {
  sessionId: ensureSessionId(),
  skills: new Map(), // name -> {name, description, suggested_question}
  activeSkill: null, // skill name or null
  sending: false,
};

function ensureSessionId() {
  let id = localStorage.getItem("ragSessionId");
  if (!id) {
    id = (crypto.randomUUID && crypto.randomUUID()) || `s-${Math.random().toString(36).slice(2)}`;
    localStorage.setItem("ragSessionId", id);
  }
  return id;
}

function $(sel) { return document.querySelector(sel); }
function clearChildren(node) { while (node.firstChild) node.removeChild(node.firstChild); }
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

async function loadSkills() {
  const r = await fetch("api/skills");
  if (!r.ok) throw new Error(`api/skills failed: ${r.status}`);
  const arr = await r.json();
  state.skills = new Map(arr.map((s) => [s.name, s]));
  renderSkills();
}

function renderSkills() {
  const list = $("#skills-list");
  clearChildren(list);
  for (const s of state.skills.values()) {
    const card = el("button", { type: "button", class: "skill-card", "data-name": s.name }, [
      el("div", { class: "skill-name" }, s.name),
      el("div", { class: "skill-desc" }, s.description),
      el("div", { class: "skill-suggest" }, `Try: "${s.suggested_question}"`),
    ]);
    card.addEventListener("click", () => activateSkill(s.name));
    list.appendChild(card);
  }
}

function activateSkill(name) {
  const s = state.skills.get(name);
  if (!s) return;
  state.activeSkill = name;
  for (const card of document.querySelectorAll(".skill-card")) {
    card.classList.toggle("active", card.dataset.name === name);
  }
  $("#active-skill-badge").classList.remove("hidden");
  $("#active-skill-name").textContent = name;
  const input = $("#composer-input");
  input.value = s.suggested_question;
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
  autosize(input);
}

function clearSkill() {
  state.activeSkill = null;
  for (const card of document.querySelectorAll(".skill-card")) card.classList.remove("active");
  $("#active-skill-badge").classList.add("hidden");
}

function autosize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(200, textarea.scrollHeight) + "px";
}

function ensureEmptyPlaceholder() {
  const t = $("#transcript");
  if (t.children.length === 0) {
    t.appendChild(el("div", { class: "empty" }, "No messages yet. Pick a skill above, or just ask."));
  } else {
    const empty = t.querySelector(".empty");
    if (empty) empty.remove();
  }
}

function appendUserBubble(text) {
  const t = $("#transcript");
  t.querySelector(".empty")?.remove();
  t.appendChild(el("div", { class: "bubble user" }, text));
  t.scrollTop = t.scrollHeight;
}

function newAssistantTurn() {
  const t = $("#transcript");
  t.querySelector(".empty")?.remove();
  const bubble = el("div", { class: "bubble asst" }, "");
  const sourcesBar = el("div", { class: "sources hidden" }, [
    el("span", { class: "sources-label" }, "Sources"),
  ]);
  const turn = el("div", { class: "turn" }, []);
  t.appendChild(turn);

  return {
    turn,
    bubble,
    sourcesBar,
    docIds: new Set(),
    text: "",
    bubbleAttached: false,
    indicator: null,
    toolCards: new Map(),
  };
}

function ensureBubble(ts) {
  if (!ts.bubbleAttached) {
    ts.turn.appendChild(ts.bubble);
    ts.indicator = el("span", { class: "streaming-indicator" });
    ts.bubble.appendChild(ts.indicator);
    ts.bubbleAttached = true;
  }
}

function appendDelta(ts, delta) {
  ensureBubble(ts);
  ts.text += delta;
  const textNode = document.createTextNode(delta);
  ts.bubble.insertBefore(textNode, ts.indicator);
  $("#transcript").scrollTop = $("#transcript").scrollHeight;
}

function addToolStart(ts, ev) {
  const argsLine = ev.args ? truncate(JSON.stringify(ev.args), 120) : "";
  const head = el("summary", {}, [
    el("span", { class: "tool-name" }, `tool: ${ev.name}`),
    el("span", { class: "tool-args-summary" }, argsLine),
  ]);
  const body = el("div", { class: "tool-body" }, [
    el("h4", {}, "args"),
    el("pre", {}, ev.args ? JSON.stringify(ev.args, null, 2) : "(none)"),
    el("h4", {}, "result"),
    el("pre", { class: "tool-result" }, "running…"),
  ]);
  const details = el("details", { class: "tool-card" }, [head, body]);
  ts.turn.appendChild(details);
  ts.toolCards.set(ev.id, { details, resultEl: body.querySelector(".tool-result") });
  $("#transcript").scrollTop = $("#transcript").scrollHeight;
}

function addToolEnd(ts, ev) {
  const card = ts.toolCards.get(ev.id);
  if (card) {
    card.resultEl.textContent = ev.summary || "(empty)";
    if (ev.isError) card.details.classList.add("err");
  }
  collectDocIdsFromToolDetails(ts, ev);
}

function collectDocIdsFromToolDetails(ts, ev) {
  const d = ev.details;
  if (!d) return;
  if (ev.name === "search" && Array.isArray(d.results)) {
    for (const r of d.results) if (r && r.doc_id) ts.docIds.add(r.doc_id);
  } else if (ev.name === "open_document" && d.doc_id) {
    ts.docIds.add(d.doc_id);
  }
}

function finalizeTurn(ts) {
  if (ts.indicator) ts.indicator.remove();
  if (ts.text) {
    for (const m of ts.text.matchAll(DOC_ID_REGEX)) ts.docIds.add(m[1]);
  }
  if (ts.docIds.size > 0) {
    for (const id of ts.docIds) {
      const chip = el("button", {
        type: "button",
        class: id.startsWith("demo_") ? "doc-chip synthetic" : "doc-chip",
      }, id);
      chip.addEventListener("click", () => openDoc(id));
      ts.sourcesBar.appendChild(chip);
    }
    ts.sourcesBar.classList.remove("hidden");
    ts.turn.appendChild(ts.sourcesBar);
  }
}

function appendErrorBubble(ts, message) {
  ensureBubble(ts);
  if (ts.indicator) ts.indicator.remove();
  ts.bubble.appendChild(el("div", { class: "tool-card err", style: "margin-top:6px;padding:6px 10px;" }, `error: ${message}`));
}

function truncate(s, n) { return s.length <= n ? s : s.slice(0, n) + "…"; }

async function openDoc(docId) {
  $("#doc-modal-id").textContent = docId;
  $("#doc-modal-body").textContent = "Loading…";
  clearChildren($("#doc-modal-meta"));
  $("#doc-modal").classList.remove("hidden");
  try {
    const r = await fetch(`api/doc/${encodeURIComponent(docId)}`);
    if (r.status === 404) {
      $("#doc-modal-body").textContent = `Document ${docId} not found in the index.`;
      return;
    }
    if (!r.ok) {
      $("#doc-modal-body").textContent = `Error: ${r.status} ${r.statusText}`;
      return;
    }
    const data = await r.json();
    const meta = $("#doc-modal-meta");
    clearChildren(meta);
    if (data.title) meta.appendChild(metaSpan("Title", data.title));
    if (data.source_type) meta.appendChild(metaSpan("Source", data.source_type));
    if (data.metadata && data.metadata.date) meta.appendChild(metaSpan("Date", data.metadata.date));
    if (data.metadata && data.metadata.skill) meta.appendChild(metaSpan("Skill", data.metadata.skill));
    if (data.metadata && data.metadata.synthetic) meta.appendChild(metaSpan("Synthetic", "yes"));
    $("#doc-modal-body").textContent = data.content || "(empty)";
  } catch (e) {
    $("#doc-modal-body").textContent = `Error: ${e.message}`;
  }
}
function metaSpan(label, value) {
  return el("span", {}, [el("b", {}, `${label}:`), document.createTextNode(value)]);
}

function closeModal() { $("#doc-modal").classList.add("hidden"); }

async function send() {
  if (state.sending) return;
  const input = $("#composer-input");
  const message = input.value.trim();
  if (!message) return;
  state.sending = true;
  $("#send-btn").disabled = true;

  appendUserBubble(message);
  input.value = "";
  autosize(input);

  const ts = newAssistantTurn();

  try {
    const resp = await fetch("api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        skill_name: state.activeSkill,
        message,
      }),
    });
    if (!resp.ok || !resp.body) {
      const text = await resp.text();
      appendErrorBubble(ts, `${resp.status} ${text}`);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let done = false;
    while (!done) {
      const r = await reader.read();
      done = r.done;
      if (r.value) buf += decoder.decode(r.value, { stream: !done });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          let ev;
          try { ev = JSON.parse(payload); } catch { continue; }
          handleEvent(ts, ev);
        }
      }
    }
  } catch (err) {
    appendErrorBubble(ts, err.message);
  } finally {
    finalizeTurn(ts);
    state.sending = false;
    $("#send-btn").disabled = false;
  }
}

function handleEvent(ts, ev) {
  switch (ev.type) {
    case "text":
      appendDelta(ts, ev.delta || "");
      break;
    case "tool_start":
      addToolStart(ts, ev);
      break;
    case "tool_end":
      addToolEnd(ts, ev);
      break;
    case "error":
      appendErrorBubble(ts, ev.message || "unknown");
      break;
    case "done":
      break;
  }
}

async function resetConversation() {
  if (!confirm("Clear the conversation?")) return;
  await fetch("api/session/reset", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId }),
  });
  clearChildren($("#transcript"));
  clearSkill();
  ensureEmptyPlaceholder();
}

function wireEvents() {
  $("#composer-form").addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });
  $("#composer-input").addEventListener("input", (e) => autosize(e.target));
  $("#composer-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  $("#clear-skill").addEventListener("click", clearSkill);
  $("#reset-btn").addEventListener("click", resetConversation);
  for (const close of document.querySelectorAll('[data-close="1"]')) {
    close.addEventListener("click", closeModal);
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

(async function main() {
  wireEvents();
  ensureEmptyPlaceholder();
  try {
    await loadSkills();
  } catch (e) {
    $("#skills-list").appendChild(el("div", { style: "color:var(--err)" }, `Failed to load skills: ${e.message}`));
  }
})();
