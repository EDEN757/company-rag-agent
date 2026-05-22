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
import os
import re
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


# ── Chat handler ───────────────────────────────────────────────────────────────
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

    thinking_md = ("*Reasoning mode is on — thinking…*" if enable_thinking
                   else "*Reasoning mode is off — enable the toggle to activate model thinking.*")

    # Show "..." immediately while the backend processes
    yield (
        history_with_loading + [{"role": "assistant", "content": "…"}],
        thinking_md, "", query_history, _render_history(query_history),
    )

    try:
        r = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question, "history": _api_history(actual_history),
                  "thinking_mode": enable_thinking},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.ConnectionError:
        h = actual_history + [{"role": "user", "content": question},
                               {"role": "assistant", "content": "Cannot reach the backend. Make sure the Backend app is running."}]
        yield h, thinking_md, "", query_history, _render_history(query_history)
        return
    except requests.exceptions.Timeout:
        h = actual_history + [{"role": "user", "content": question},
                               {"role": "assistant", "content": "Backend timed out (>300 s)."}]
        yield h, thinking_md, "", query_history, _render_history(query_history)
        return
    except Exception as e:
        h = actual_history + [{"role": "user", "content": question},
                               {"role": "assistant", "content": f"Error: {e}"}]
        yield h, thinking_md, "", query_history, _render_history(query_history)
        return

    answer       = _linkify(data.get("answer", ""), question)
    traces       = data.get("tool_traces", [])
    all_sources  = data.get("sources", [])
    latency_ms   = data.get("latency_ms", 0)
    thinking_steps = data.get("thinking_steps", [])

    traces_md = ""
    if traces:
        traces_md = (
            f"**Agent steps** &nbsp;*(completed in {latency_ms:.0f} ms)*\n\n"
            + "\n".join(f"- `{t}`" for t in traces)
        )

    if enable_thinking and thinking_steps:
        thinking_md = f"**Model reasoning:**\n\n{thinking_steps[-1]}"
    elif enable_thinking:
        thinking_md = "*Reasoning mode is on but the model did not emit any reasoning.*"
    else:
        thinking_md = "*Reasoning mode is off — enable the toggle to activate model thinking.*"

    sources_md = _build_sources_md(question, all_sources)
    new_qhist  = ([{"question": question, "traces_md": traces_md, "sources_md": sources_md}]
                  + query_history)[:MAX_QUERY_HISTORY]

    parts: list[str] = []
    if traces_md:
        parts.append(traces_md)
    if show_sources:
        parts.append(sources_md)

    h = actual_history + [{"role": "user", "content": question},
                           {"role": "assistant", "content": answer}]
    yield h, thinking_md, "\n\n---\n\n".join(parts), new_qhist, _render_history(new_qhist)


# ── Gradio UI ──────────────────────────────────────────────────────────────────
_LINK_FIX_JS = """
<script defer>
(function() {
  const fixLinks = node => {
    if (node.nodeType !== 1) return;
    node.querySelectorAll && node.querySelectorAll('a[href]').forEach(a => {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    });
  };
  new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(fixLinks)))
    .observe(document.body, { childList: true, subtree: true });
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
                    "What is Redwood Inference's mission statement?",
                    "What are the four main revenue streams in Redwood Inference's business model?",
                    "In the SOC2 readiness notes, what log retention duration is mentioned as a risk for some environments?",
                    "When is the 48-hour throughput and latency benchmark on the dedicated pool due to be completed?",
                    "What caused the EU-West activation funnel and onboarding email issues in late January, and which code/config changes fixed it?",
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
                    value="*Send a query to see agent steps here.*",
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

    stop_btn.click(fn=None, cancels=[_submit_event, _msg_event])

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

_HIGHLIGHT_JS = """
<script>
(function() {
  const q = new URLSearchParams(window.location.search).get('q') || '';
  if (!q) return;
  const pre = document.querySelector('pre');
  if (!pre) return;

  const queryTerms = (q.toLowerCase().match(/[a-zA-Z]\\w*/g) || []).filter(w => w.length > 3);
  if (!queryTerms.length) return;

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
