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

import requests
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

BACKEND_URL = os.environ.get("BACKEND_URL", "http://nv-service-e4bb2876d3e69f18fd98d56e852aa814:8500").rstrip("/")

MAX_QUERY_HISTORY = 5   # how many past queries to keep in the history panel
DOC_ID_RE = re.compile(r'\b(dsid_[a-f0-9]+)\b')

CSS = """
.thinking-panel { max-height: 300px; overflow-y: auto; padding-right: 4px; }
.traces-panel   { max-height: 180px; overflow-y: auto; padding-right: 4px; }
.history-panel  { max-height: 400px; overflow-y: auto; padding-right: 4px; }
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


def _query_backend(
    question: str,
    history: list[dict],
    enable_thinking: bool = False,
) -> tuple[str, str, str, str]:
    """Call /query and return (answer, thinking_md, traces_md, sources_md)."""
    api_history = [{"role": m["role"], "content": m["content"]} for m in history]
    question_to_send = question + " /think" if enable_thinking else question

    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question_to_send, "history": api_history},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return "Cannot reach the backend. Make sure the Backend app is running.", "", "", "*No sources retrieved for this query.*"
    except requests.exceptions.Timeout:
        return "Backend timed out (>180 s). The query may be too complex.", "", "", "*No sources retrieved for this query.*"
    except Exception as e:
        return f"Error: {e}", "", "", "*No sources retrieved for this query.*"

    answer         = DOC_ID_RE.sub(lambda m: f'[{m.group(1)}](doc/{m.group(1)})', data.get("answer", ""))
    sources        = data.get("sources", [])
    latency        = data.get("latency_ms", 0)
    tool_traces    = data.get("tool_traces", [])
    thinking_steps = data.get("thinking_steps", [])

    # ── Model reasoning ────────────────────────────────────────────────────────
    if thinking_steps:
        blocks = "\n\n---\n\n".join(thinking_steps)
        thinking_md = f"**Model reasoning:**\n\n{blocks}"
    elif enable_thinking:
        thinking_md = "*Reasoning mode is on but the model did not emit any reasoning for this query.*"
    else:
        thinking_md = "*Reasoning mode is off — enable the toggle to activate model thinking.*"

    # ── Agent step traces ──────────────────────────────────────────────────────
    traces_md = ""
    if tool_traces:
        items = "\n".join(f"- `{t}`" for t in tool_traces)
        traces_md = f"**Agent steps** &nbsp;*(completed in {latency:.0f} ms)*\n\n{items}"

    # ── Sources ────────────────────────────────────────────────────────────────
    sources_md = "*No sources retrieved for this query.*"
    if sources:
        opened  = [s for s in sources if s.get("opened")]
        found   = [s for s in sources if not s.get("opened")]
        lines: list[str] = []

        def _doc_link(doc_id: str) -> str:
            eid = html.escape(doc_id)
            return f'<a href="doc/{eid}" target="_blank" rel="noopener"><code>{eid}</code></a>'

        if opened:
            lines.append("**Read in full:**\n")
            for s in opened:
                title    = s.get("title") or s["doc_id"]
                preview  = s.get("preview", "")[:200]
                lines.append(
                    f"**{title}**\n"
                    f"`{s['source_type']}` &nbsp; {_doc_link(s['doc_id'])}\n"
                    f"> {preview}…"
                )

        if found:
            lines.append(f"**{len(found)} retrieved chunk(s):**\n")
            for i, s in enumerate(found, 1):
                title    = s.get("title") or s["doc_id"]
                preview  = s.get("preview", "")[:200]
                lines.append(
                    f"**{i}. {title}**\n"
                    f"`{s['source_type']}` &nbsp; score: `{s['score']}` &nbsp; {_doc_link(s['doc_id'])}\n"
                    f"> {preview}…"
                )

        sources_md = "\n\n".join(lines)

    return answer, thinking_md, traces_md, sources_md


# ── Query history helpers ──────────────────────────────────────────────────────
def _render_history(entries: list[dict]) -> str:
    """Format the session query history as markdown (latest first)."""
    if not entries:
        return "*No previous queries in this session.*"
    parts = []
    for i, e in enumerate(entries, 1):
        q = e["question"]
        q_label = q[:90] + ("…" if len(q) > 90 else "")
        section = f"**{i}. {q_label}**"
        if e["traces_md"]:
            section += f"\n\n{e['traces_md']}"
        if e["sources_md"]:
            section += f"\n\n{e['sources_md']}"
        parts.append(section)
    return "\n\n---\n\n".join(parts)


# ── Chat handler ───────────────────────────────────────────────────────────────
def chat(
    question: str,
    history: list[dict],
    show_sources: bool,
    enable_thinking: bool,
    query_history: list[dict],
) -> tuple[list[dict], str, str, str, list[dict], str]:
    if not question.strip():
        return history, "", "", "", query_history, _render_history(query_history)
    answer, thinking_md, traces_md, sources_md = _query_backend(question, history, enable_thinking)
    history = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    # Prepend to session history, keep last MAX_QUERY_HISTORY entries
    new_qhist = ([{
        "question":   question,
        "traces_md":  traces_md,
        "sources_md": sources_md,
    }] + query_history)[:MAX_QUERY_HISTORY]

    return (
        history,
        thinking_md,
        traces_md,
        sources_md if show_sources else "",
        new_qhist,
        _render_history(new_qhist),
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Company Knowledge Assistant", css=CSS) as demo:
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
                height=380,
            )
            msg_box = gr.Textbox(
                placeholder="Ask a question about company knowledge…",
                label="Your question",
                lines=2,
                autofocus=True,
            )
            with gr.Row():
                submit_btn  = gr.Button("Send", variant="primary")
                clear_btn   = gr.Button("Clear conversation")
            show_sources    = gr.Checkbox(label="Show retrieved sources", value=True)
            enable_thinking = gr.Checkbox(label="Reasoning mode (slower — model thinks before answering)", value=False)

        # ── Right column: reasoning + traces + sources + info ─────────────────
        with gr.Column(scale=1):
            with gr.Accordion("Model reasoning", open=False):
                thinking_box = gr.Markdown(
                    value="*Reasoning mode is off — enable the toggle to activate model thinking.*",
                    elem_classes=["thinking-panel"],
                )
            with gr.Accordion("Agent steps", open=True):
                traces_box = gr.Markdown(
                    value="",
                    elem_classes=["traces-panel"],
                )
            sources_box = gr.Markdown(
                value="*Sources will appear here after each query.*",
                label="Sources",
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

    gr.Examples(
        examples=[
            "Who complained about the November invoice spike?",
            "What were the action items from the last engineering meeting?",
            "Summarize recent Slack discussions about the API migration.",
            "Find emails about the Q4 budget review.",
            "What decisions were made in the latest Jira sprint planning?",
        ],
        inputs=msg_box,
    )

    # ── Event wiring ───────────────────────────────────────────────────────────
    submit_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, show_sources, enable_thinking, query_history_state],
        outputs=[chatbot, thinking_box, traces_box, sources_box,
                 query_history_state, history_view],
    ).then(fn=lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, show_sources, enable_thinking, query_history_state],
        outputs=[chatbot, thinking_box, traces_box, sources_box,
                 query_history_state, history_view],
    ).then(fn=lambda: "", outputs=msg_box)

    clear_btn.click(
        fn=lambda: ([], "*Reasoning mode is off — enable the toggle to activate model thinking.*", "",
                    "*Sources will appear here after each query.*",
                    [], "*No previous queries in this session.*"),
        outputs=[chatbot, thinking_box, traces_box, sources_box,
                 query_history_state, history_view],
    )

    refresh_btn.click(
        fn=lambda: (_get_health(), _get_stats()),
        outputs=[health_md, stats_md],
    )


# ── Document viewer ────────────────────────────────────────────────────────────
fastapi_app = FastAPI()


@fastapi_app.get("/doc/{doc_id:path}", response_class=HTMLResponse)
def view_document(doc_id: str):
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
</body>
</html>"""
    return HTMLResponse(body)


app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        root_path="/proxy/7860",
    )
