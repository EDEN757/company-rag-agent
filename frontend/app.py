"""
frontend/app.py — Company RAG Agent UI (Nuvolos)
=================================================
Gradio chat interface running on the Frontend VS Code app.

Run:
    cd /files/frontend
    pip install -r requirements.txt
    python app.py

Access via Nuvolos proxy:
    https://<hash>.app.az.nuvolos.cloud/proxy/7860/

Environment variables (set in Nuvolos Frontend app CONFIGURE):
    BACKEND_URL   http://<backend-hostname>:8500
                  (hostname shown in Backend app CONFIGURE page)
"""

import os
import requests
import gradio as gr

BACKEND_URL = os.environ.get("BACKEND_URL", "http://nv-service-e4bb2876d3e69f18fd98d56e852aa814:8500").rstrip("/")

MAX_QUERY_HISTORY = 5   # how many past queries to keep in the history panel


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
    history: list[tuple[str, str]],
) -> tuple[str, str, str, str]:
    """Call /query and return (answer, thinking_md, traces_md, sources_md)."""
    api_history = []
    for user_msg, assistant_msg in history:
        api_history.append({"role": "user",      "content": user_msg})
        api_history.append({"role": "assistant", "content": assistant_msg})

    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question, "history": api_history},
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return "Cannot reach the backend. Make sure the Backend app is running.", "", "", "*No sources retrieved for this query.*"
    except requests.exceptions.Timeout:
        return "Backend timed out (>180 s). The query may be too complex.", "", "", "*No sources retrieved for this query.*"
    except Exception as e:
        return f"Error: {e}", "", "", "*No sources retrieved for this query.*"

    answer         = data.get("answer", "")
    sources        = data.get("sources", [])
    latency        = data.get("latency_ms", 0)
    tool_traces    = data.get("tool_traces", [])
    thinking_steps = data.get("thinking_steps", [])

    # ── Model reasoning ────────────────────────────────────────────────────────
    thinking_md = ""
    if thinking_steps:
        blocks = "\n\n---\n\n".join(thinking_steps)
        thinking_md = f"**Model reasoning:**\n\n{blocks}"

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

        if opened:
            lines.append("**Read in full:**\n")
            for s in opened:
                title = s.get("title") or s["doc_id"]
                preview = s.get("preview", "")[:200]
                lines.append(
                    f"**{title}**\n"
                    f"`{s['source_type']}` &nbsp; `{s['doc_id']}`\n"
                    f"> {preview}…"
                )

        if found:
            lines.append(f"**{len(found)} retrieved chunk(s):**\n")
            for i, s in enumerate(found, 1):
                title   = s.get("title") or s["doc_id"]
                preview = s.get("preview", "")[:200]
                lines.append(
                    f"**{i}. {title}**\n"
                    f"`{s['source_type']}` &nbsp; score: `{s['score']}`\n"
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
    history: list[tuple[str, str]],
    show_sources: bool,
    query_history: list[dict],
) -> tuple[list[tuple[str, str]], str, str, str, list[dict], str]:
    if not question.strip():
        return history, "", "", "", query_history, _render_history(query_history)
    answer, thinking_md, traces_md, sources_md = _query_backend(question, history)
    history = history + [(question, answer)]

    # Prepend to session history, keep last MAX_QUERY_HISTORY entries
    new_qhist = ([{
        "question":   question,
        "traces_md":  traces_md,
        "sources_md": sources_md,  # always store in history regardless of checkbox
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
with gr.Blocks(title="Company Knowledge Assistant") as demo:
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
                type="tuples",
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
            show_sources = gr.Checkbox(label="Show retrieved sources", value=True)

        # ── Right column: reasoning + traces + sources + info ─────────────────
        with gr.Column(scale=1):
            with gr.Accordion("Model reasoning", open=False):
                thinking_box = gr.Markdown(value="*Model reasoning will appear here.*")
            traces_box = gr.Markdown(
                value="",
                label="Agent steps",
            )
            sources_box = gr.Markdown(
                value="*Sources will appear here after each query.*",
                label="Sources",
            )
            with gr.Accordion(f"Query history (last {MAX_QUERY_HISTORY})", open=False):
                history_view = gr.Markdown("*No previous queries in this session.*")
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
        inputs=[msg_box, chatbot, show_sources, query_history_state],
        outputs=[chatbot, thinking_box, traces_box, sources_box,
                 query_history_state, history_view],
    ).then(fn=lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, show_sources, query_history_state],
        outputs=[chatbot, thinking_box, traces_box, sources_box,
                 query_history_state, history_view],
    ).then(fn=lambda: "", outputs=msg_box)

    clear_btn.click(
        fn=lambda: ([], "*Model reasoning will appear here.*", "",
                    "*Sources will appear here after each query.*",
                    [], "*No previous queries in this session.*"),
        outputs=[chatbot, thinking_box, traces_box, sources_box,
                 query_history_state, history_view],
    )

    refresh_btn.click(
        fn=lambda: (_get_health(), _get_stats()),
        outputs=[health_md, stats_md],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # bind to all interfaces in the container
        server_port=7860,
        root_path="/proxy/7860",  # Nuvolos HTTPS proxy path
        theme=gr.themes.Soft(),
    )
