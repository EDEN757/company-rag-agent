"""
frontend/app.py — Company RAG Agent UI (Nuvolos)
=================================================
Gradio chat interface running on the Frontend VS Code app.

Run:
    cd /files/frontend
    pip install -r requirements.txt
    python app.py

Access via Nuvolos proxy:
    https://<hash>.proxy-eu1.nuvolos.cloud/proxy/7860/

Environment variables (set in ~/.bashrc on the Frontend app):
    BACKEND_URL   http://<backend-hostname>:8500
                  (hostname shown in Backend app network info)
"""

import html
import json as _json
import os
import re
import threading
import urllib.parse

import requests
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

BACKEND_URL = os.environ.get("BACKEND_URL", "http://nv-service-e4bb2876d3e69f18fd98d56e852aa814:8500").rstrip("/")

MAX_QUERY_HISTORY = 5
DOC_ID_RE    = re.compile(r'\b(dsid_[a-f0-9]+)\b')
_HTML_TAG_RE = re.compile(r'<[^>]+>')

_cancel_event = threading.Event()

# Shared state so JavaScript can poll live trace progress
_trace_lock  = threading.Lock()
_trace_state: dict = {"traces": [], "current": "", "running": False}

CSS = """
.thinking-panel,
.thinking-panel > div,
.thinking-panel .prose { max-height: 300px !important; overflow-y: auto !important; }

.results-panel,
.results-panel > div,
.results-panel .prose  { max-height: 380px !important; overflow-y: auto !important; }

.history-panel,
.history-panel > div,
.history-panel .prose  { max-height: 380px !important; overflow-y: auto !important; }
"""


# ── Backend calls ──────────────────────────────────────────────────────────────
def _get_health() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        r.raise_for_status()
        d = r.json()
        return f"**Backend:** OK &nbsp;|&nbsp; model: `{d.get('model')}` &nbsp;|&nbsp; embed: `{d.get('embed')}`"
    except Exception:
        return "**Backend:** unreachable — make sure the Backend app is running."


def _get_stats() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        r.raise_for_status()
        d = r.json()
        rows = "\n".join(f"- **{k}**: {v:,}" for k, v in d.get("by_source", {}).items())
        return (
            f"**{d.get('documents', 0):,} documents** &nbsp;|&nbsp; "
            f"**{d.get('chunks', 0):,} chunks**\n\n{rows}"
        )
    except Exception:
        return "Stats unavailable — backend not reachable."


def _api_history(history: list[dict]) -> list[dict]:
    """Strip HTML and normalise Gradio message dicts for the backend API."""
    out = []
    for m in history:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        content = _HTML_TAG_RE.sub("", str(content) if content is not None else "")
        if m.get("role") in ("user", "assistant"):
            out.append({"role": m["role"], "content": content})
    return out


def _linkify(text: str, question: str) -> str:
    """Make doc IDs in text clickable."""
    q_enc = urllib.parse.quote(question, safe="")
    def _link(m):
        eid = html.escape(m.group(1))
        return f'<a href="doc/{eid}?q={q_enc}" target="_blank" rel="noopener"><code>{eid}</code></a>'
    return DOC_ID_RE.sub(_link, text)


def _build_sources_md(question: str, sources: list[dict]) -> str:
    if not sources:
        return "*No sources retrieved for this query.*"
    q_enc = urllib.parse.quote(question, safe="")
    def _doc_link(doc_id: str) -> str:
        eid = html.escape(doc_id)
        return f'<a href="doc/{eid}?q={q_enc}" target="_blank" rel="noopener"><code>{eid}</code></a>'
    opened = [s for s in sources if s.get("opened")]
    found  = [s for s in sources if not s.get("opened")]
    lines: list[str] = []
    if opened:
        lines.append("**Read in full:**\n")
        for s in opened:
            title = s.get("title") or s["doc_id"]
            lines.append(
                f"**{title}**\n"
                f"`{s['source_type']}` &nbsp; {_doc_link(s['doc_id'])}\n"
                f"> {s.get('preview', '')[:200]}…"
            )
    if found:
        lines.append(f"**{len(found)} retrieved chunk(s):**\n")
        for i, s in enumerate(found, 1):
            title = s.get("title") or s["doc_id"]
            lines.append(
                f"**{i}. {title}**\n"
                f"`{s['source_type']}` &nbsp; score: `{s['score']}` &nbsp; {_doc_link(s['doc_id'])}\n"
                f"> {s.get('preview', '')[:200]}…"
            )
    return "\n\n".join(lines)


# ── Query history helpers ──────────────────────────────────────────────────────
def _render_history(entries: list[dict]) -> str:
    if not entries:
        return "*No previous queries in this session.*"
    lines = []
    for i, e in enumerate(entries, 1):
        q = e["question"]
        q_label = q[:120] + ("…" if len(q) > 120 else "")
        lines.append(f"{i}. {q_label}")
    return "\n".join(lines)


# ── Loading state helper (sync — renders immediately before generator starts) ──
def _add_loading(question: str, history: list[dict]) -> list[dict]:
    if not question.strip():
        return history
    return history + [{"role": "user", "content": question}]


# ── Chat handler (streaming SSE from /query/stream) ────────────────────────────
def chat(
    question: str,
    history_with_loading: list[dict],
    show_sources: bool,
    enable_thinking: bool,
    query_history: list[dict],
):
    if not question.strip():
        yield history_with_loading, "", "", query_history, _render_history(query_history)
        return

    actual_history = history_with_loading[:-1] if history_with_loading else []
    _cancel_event.clear()
    with _trace_lock:
        _trace_state.update({"traces": [], "current": "", "running": True})

    thinking_md  = ("*Reasoning mode is on — waiting for model…*" if enable_thinking
                    else "*Reasoning mode is off — enable the toggle to activate model thinking.*")
    traces: list[str] = []
    current_trace = ""   # label of the tool currently running
    traces_md    = ""
    current_ans  = ""
    all_sources: list[dict] = []
    latency_ms   = 0.0

    def _h(answer_so_far: str = "") -> list[dict]:
        h = actual_history + [{"role": "user", "content": question}]
        if answer_so_far:
            h = h + [{"role": "assistant", "content": answer_so_far}]
        return h

    def _cur_traces_md(running_label: str = "") -> str:
        items = [f"- `{t}`" for t in traces]
        if running_label:
            items.append(f"- `{running_label}` ⏳")
        return ("**Agent steps:**\n\n" + "\n".join(items)) if items else ""

    try:
        with requests.post(
            f"{BACKEND_URL}/query/stream",
            json={"question": question, "history": _api_history(actual_history),
                  "thinking_mode": enable_thinking},
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()

            for raw in resp.iter_lines():
                if _cancel_event.is_set():
                    with _trace_lock:
                        _trace_state["running"] = False
                    yield _h("⚠️ Search interrupted by user."), thinking_md, traces_md, query_history, _render_history(query_history)
                    return

                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    ev = _json.loads(data)
                except Exception:
                    continue

                etype = ev.get("type")

                if etype == "trace_start":
                    current_trace = ev["label"]
                    with _trace_lock:
                        _trace_state["current"] = current_trace
                    traces_md = _cur_traces_md(current_trace)
                    yield _h(current_ans), thinking_md, traces_md, query_history, _render_history(query_history)

                elif etype == "trace_done":
                    traces.append(ev["content"])
                    current_trace = ""
                    with _trace_lock:
                        _trace_state["traces"] = traces.copy()
                        _trace_state["current"] = ""
                    traces_md = _cur_traces_md()
                    yield _h(current_ans), thinking_md, traces_md, query_history, _render_history(query_history)

                elif etype == "token":
                    current_ans += ev["content"]
                    yield _h(current_ans), thinking_md, traces_md, query_history, _render_history(query_history)

                elif etype == "retract":
                    current_ans = ""
                    yield _h(), thinking_md, traces_md, query_history, _render_history(query_history)

                elif etype == "answer":
                    current_ans = _linkify(ev["content"], question)
                    yield _h(current_ans), thinking_md, traces_md, query_history, _render_history(query_history)

                elif etype == "thinking":
                    thinking_md = f"**Model reasoning:**\n\n{ev['content']}"

                elif etype == "done":
                    latency_ms  = ev.get("latency_ms", 0)
                    all_sources = ev.get("sources", [])
                    with _trace_lock:
                        _trace_state["running"] = False
                    if traces_md:
                        traces_md = traces_md.replace(
                            "**Agent steps:**",
                            f"**Agent steps** &nbsp;*(completed in {latency_ms:.0f} ms)*",
                        )

    except requests.exceptions.ConnectionError:
        with _trace_lock:
            _trace_state["running"] = False
        yield _h("Cannot reach the backend. Make sure the Backend app is running."), thinking_md, "", query_history, _render_history(query_history)
        return
    except requests.exceptions.Timeout:
        with _trace_lock:
            _trace_state["running"] = False
        yield _h("Backend timed out (>300 s)."), thinking_md, traces_md, query_history, _render_history(query_history)
        return
    except Exception as e:
        with _trace_lock:
            _trace_state["running"] = False
        yield _h(f"Error: {e}"), thinking_md, traces_md, query_history, _render_history(query_history)
        return

    # Apply doc-ID links to streamed answer (tokens arrive without HTML)
    if current_ans and '<a href="doc/' not in current_ans:
        current_ans = _linkify(current_ans, question)

    if enable_thinking and "**Model reasoning:**" not in thinking_md:
        thinking_md = "*Reasoning mode is on but the model did not emit any reasoning.*"

    sources_md = _build_sources_md(question, all_sources)
    new_qhist  = ([{"question": question, "traces_md": traces_md, "sources_md": sources_md}]
                  + query_history)[:MAX_QUERY_HISTORY]

    parts: list[str] = []
    if traces_md:
        parts.append(traces_md)
    if show_sources:
        parts.append(sources_md)

    yield (
        _h(current_ans),
        thinking_md,
        "\n\n---\n\n".join(parts),
        new_qhist,
        _render_history(new_qhist),
    )


def _do_cancel():
    _cancel_event.set()
    with _trace_lock:
        _trace_state["running"] = False


# ── Gradio UI ──────────────────────────────────────────────────────────────────
_LINK_FIX_JS = """
<script>
(function() {
  // ── Open all chatbot links in a new tab ──────────────────────────────────
  const fixLinks = node => {
    if (node.nodeType !== 1) return;
    node.querySelectorAll && node.querySelectorAll('a[href]').forEach(a => {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    });
  };
  new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(fixLinks)))
    .observe(document.body, { childList: true, subtree: true });

  // ── Live agent-step polling (bypasses proxy WebSocket buffering) ──────────
  let _poll = null;

  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function getPanel() {
    return document.querySelector('.results-panel .prose');
  }

  function renderTraces(d) {
    const el = getPanel();
    if (!el) return;
    if (!d.traces.length && !d.current) {
      if (d.running) el.innerHTML = '<em style="color:#888">Searching…</em>';
      return;
    }
    let html = '<strong>Agent steps:</strong><ul style="margin:6px 0 0 18px;padding:0;list-style:disc">';
    d.traces.forEach(t => { html += '<li><code>' + esc(t) + '</code></li>'; });
    if (d.current) html += '<li><code>' + esc(d.current) + '</code> ⏳</li>';
    html += '</ul>';
    el.innerHTML = html;
  }

  function startPoll() {
    stopPoll();
    const el = getPanel();
    if (el) el.innerHTML = '<em style="color:#888">Searching…</em>';
    _poll = setInterval(async () => {
      try {
        const r = await fetch('api/traces');
        if (!r.ok) return;
        const d = await r.json();
        renderTraces(d);
        if (!d.running) stopPoll();
      } catch(e) {}
    }, 600);
  }

  function stopPoll() {
    if (_poll) { clearInterval(_poll); _poll = null; }
  }

  // Start polling on Send button click or Enter in the input
  // (lines=1 renders <input>, not <textarea>, so check both)
  document.addEventListener('click', e => {
    const btn = e.target.closest('button');
    if (btn && /^send$/i.test(btn.textContent.trim())) startPoll();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') startPoll();
    }
  });
})();
</script>
"""

with gr.Blocks(title="Company Knowledge Assistant", css=CSS, head=_LINK_FIX_JS) as demo:
    query_history_state = gr.State([])
    gr.Markdown(
        "# Company Knowledge Assistant\n"
        "Ask questions about company documents, emails, Slack conversations, "
        "Jira tickets, Confluence pages, and more."
    )

    with gr.Row():
        # ── Left column: chat ──────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
            )
            msg_box = gr.Textbox(
                placeholder="Ask a question about company knowledge…",
                label="Your question",
                lines=1,
                autofocus=True,
            )
            gr.Examples(
                examples=[
                    "Who complained about the November invoice spike?",
                    "What were the action items from the last engineering meeting?",
                    "Summarize recent Slack discussions about the API migration.",
                    "Find emails about the Q4 budget review.",
                    "What decisions were made in the latest Jira sprint planning?",
                ],
                inputs=msg_box,
                label="Example questions",
            )
            with gr.Row():
                submit_btn  = gr.Button("Send", variant="primary")
                stop_btn    = gr.Button("Stop")
                clear_btn   = gr.Button("Clear conversation")
            show_sources    = gr.Checkbox(label="Show retrieved sources", value=True)
            enable_thinking = gr.Checkbox(label="Reasoning mode (slower — model thinks before answering)", value=False)

        # ── Right column: reasoning + results + history + info ────────────────
        with gr.Column(scale=1):
            with gr.Accordion("Model reasoning", open=False):
                thinking_box = gr.Markdown(
                    value="*Reasoning mode is off — enable the toggle to activate model thinking.*",
                    elem_classes=["thinking-panel"],
                )
            with gr.Accordion("Agent steps & sources", open=True):
                results_box = gr.Markdown(
                    value="",
                    elem_classes=["results-panel"],
                )
            with gr.Accordion(f"Query history (last {MAX_QUERY_HISTORY})", open=False):
                history_view = gr.Markdown(
                    "*No previous queries in this session.*",
                    elem_classes=["history-panel"],
                )
            with gr.Accordion("System info", open=False):
                health_md   = gr.Markdown(_get_health())
                stats_md    = gr.Markdown(_get_stats())
                refresh_btn = gr.Button("Refresh")

    # ── Event wiring ───────────────────────────────────────────────────────────
    # Two-step: sync _add_loading renders immediately (bypasses proxy buffering),
    # then the generator chat() runs the backend call.
    _load_btn = submit_btn.click(
        fn=_add_loading, inputs=[msg_box, chatbot], outputs=[chatbot],
    )
    _submit_event = _load_btn.then(
        fn=chat,
        inputs=[msg_box, chatbot, show_sources, enable_thinking, query_history_state],
        outputs=[chatbot, thinking_box, results_box, query_history_state, history_view],
    )
    _submit_event.then(fn=lambda: "", outputs=msg_box)

    _load_msg = msg_box.submit(
        fn=_add_loading, inputs=[msg_box, chatbot], outputs=[chatbot],
    )
    _msg_event = _load_msg.then(
        fn=chat,
        inputs=[msg_box, chatbot, show_sources, enable_thinking, query_history_state],
        outputs=[chatbot, thinking_box, results_box, query_history_state, history_view],
    )
    _msg_event.then(fn=lambda: "", outputs=msg_box)

    stop_btn.click(fn=_do_cancel, cancels=[_submit_event, _msg_event])

    clear_btn.click(
        fn=lambda: ([], "*Reasoning mode is off — enable the toggle to activate model thinking.*",
                    "", [], "*No previous queries in this session.*"),
        outputs=[chatbot, thinking_box, results_box, query_history_state, history_view],
    )

    refresh_btn.click(
        fn=lambda: (_get_health(), _get_stats()),
        outputs=[health_md, stats_md],
    )


# ── Document viewer ────────────────────────────────────────────────────────────
fastapi_app = FastAPI()


@fastapi_app.get("/api/traces")
def api_traces():
    """Polled by JavaScript to show live agent steps while Gradio generator is blocked by proxy."""
    with _trace_lock:
        return dict(_trace_state)

# Highlight query terms that actually appear in the document, ranked by TF.
# This shows WHICH words drove the BM25 score (terms the model actually matched),
# not just all words from the query regardless of whether they appear.
_HIGHLIGHT_JS = """
<script>
(function() {
  const q = new URLSearchParams(window.location.search).get('q') || '';
  if (!q) return;
  const pre = document.querySelector('pre');
  if (!pre) return;

  // Tokenize query into words > 3 chars
  const queryTerms = (q.toLowerCase().match(/[a-zA-Z]\\w*/g) || []).filter(w => w.length > 3);
  if (!queryTerms.length) return;

  // Score each query term by term-frequency in the document (TF component of BM25).
  // A term that appears often in this document contributed more to its retrieval score.
  const docText  = pre.textContent.toLowerCase();
  const docWords = (docText.match(/[a-zA-Z]\\w*/g) || []);
  const docLen   = docWords.length || 1;
  const tf = {};
  for (const w of docWords) tf[w] = (tf[w] || 0) + 1;

  const scored = queryTerms
    .map(t => ({ t, s: (tf[t] || 0) / docLen }))
    .filter(x => x.s > 0)
    .sort((a, b) => b.s - a.s)
    .slice(0, 12)
    .map(x => x.t);

  if (!scored.length) return;

  const escaped = scored.map(w => w.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'));
  const pat = new RegExp('(' + escaped.join('|') + ')', 'gi');
  pre.innerHTML = pre.innerHTML.replace(pat,
    '<mark style="background:#ff9800;color:#000;border-radius:2px">$1</mark>');
})();
</script>
"""


@fastapi_app.get("/doc/{doc_id:path}", response_class=HTMLResponse)
def view_document(doc_id: str, q: str = ""):
    try:
        r = requests.get(f"{BACKEND_URL}/document/{doc_id}", timeout=10)
        if r.status_code == 404:
            return HTMLResponse(
                f"<h2>Document not found</h2><p><code>{html.escape(doc_id)}</code></p>",
                status_code=404,
            )
        r.raise_for_status()
        doc = r.json()
    except Exception as e:
        return HTMLResponse(
            f"<h2>Error loading document</h2><p>{html.escape(str(e))}</p>",
            status_code=502,
        )

    title   = html.escape(doc.get("title") or doc_id)
    source  = html.escape(doc.get("source_type", ""))
    content = html.escape(doc.get("content", ""))
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
    h1 {{ font-size: 1.4rem; margin-bottom: .25rem; }}
    .meta {{ color: #666; font-size: .85rem; margin-bottom: 1.5rem; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f6f8fa;
           padding: 1rem; border-radius: 6px; font-size: .88rem; line-height: 1.5; }}
    a {{ color: #0969da; }}
  </style>
</head>
<body>
  <a href="javascript:history.back()">&larr; Back</a>
  <h1>{title}</h1>
  <div class="meta">Source type: <strong>{source}</strong> &nbsp;|&nbsp; ID: <code>{html.escape(doc_id)}</code></div>
  <pre>{content}</pre>
  {_HIGHLIGHT_JS}
</body>
</html>"""
    return HTMLResponse(body)


demo.queue()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        root_path="/proxy/7860",
    )
