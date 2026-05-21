import json
import logging
import re
import time

from openai import OpenAI

import db
from config import (
    OLLAMA_HOST, LLM_MODEL,
    MAX_AGENT_TURNS, MAX_HISTORY_TURNS, MAX_NEW_TOKENS, TEMPERATURE,
)
from prompt import SYSTEM_PROMPT, TOOLS
from tools import execute_tool

log = logging.getLogger(__name__)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

_client: OpenAI | None = None


def init_client():
    global _client
    _client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")


def _extract_thinking(content: str) -> tuple[str, str]:
    thoughts: list[str] = []
    clean = THINK_RE.sub(lambda m: thoughts.append(m.group(1).strip()) or "", content)
    return "\n\n".join(thoughts), clean.strip()


def _trace_args(name: str, args: dict) -> str:
    if name == "search":
        q = args.get("query", "")[:60]
        extras: list[str] = []
        if args.get("source_types"):
            extras.append(f"source={args['source_types']}")
        if args.get("participant"):
            extras.append(f"participant={args['participant']}")
        if args.get("date_from") or args.get("date_to"):
            extras.append(f"date={args.get('date_from', '')}..{args.get('date_to', '')}")
        return f'"{q}"' + (f" ({', '.join(extras)})" if extras else "")
    if name == "open_document":
        return args.get("doc_id", "")
    if name == "add_document":
        return f"{args.get('source_type', '')} / {args.get('title', '')[:50]}"
    if name == "edit_document":
        return args.get("doc_id", "")
    if name in ("read", "write", "edit"):
        return args.get("path", "")
    if name == "bash":
        return args.get("command", "")[:80]
    return ""


def _trace_result(name: str, result_text: str, hits: list) -> str:
    if name == "search":
        return f"{len(hits)} result(s)" if hits else "no results"
    if name == "open_document":
        return "not found" if result_text.startswith("No document") else f"{len(result_text):,} chars"
    if name == "add_document":
        m = re.search(r"dsid_[a-f0-9]+", result_text)
        return m.group(0) if m else "created"
    if name == "edit_document":
        return "updated" if "Updated" in result_text else result_text[:40]
    if name == "bash":
        m = re.match(r"exit=(\d+)", result_text)
        return f"exit={m.group(1)}" if m else "done"
    if result_text.startswith("Error"):
        return result_text[:60]
    return "done"


def run_agent(
    question: str,
    history: list[dict],
    thinking_mode: bool = False,
) -> tuple[str, list[dict], list[str], list[str]]:
    system_content = "/think\n\n" + SYSTEM_PROMPT if thinking_mode else SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system_content}]
    for h in history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    all_sources: list[dict] = []
    traces: list[str] = []
    thinking_steps: list[str] = []

    for _ in range(MAX_AGENT_TURNS):
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            extra_body={"options": {"num_ctx": 12288, **({"think": True} if thinking_mode else {})}},
        )
        msg = resp.choices[0].message
        raw_content = msg.content or ""

        think_text, visible_content = _extract_thinking(raw_content)
        if not think_text and thinking_mode:
            model_extra = getattr(msg, "model_extra", None) or {}
            extra_think = model_extra.get("thinking") or model_extra.get("think_content")
            if extra_think:
                think_text = str(extra_think)
                visible_content = raw_content
        if think_text:
            thinking_steps.append(think_text)

        if not msg.tool_calls:
            answer = visible_content.strip()
            if not answer and thinking_steps:
                answer = thinking_steps[-1]
            elif not answer:
                answer = "The model did not produce an answer. Please try again."
            return answer, all_sources, traces, thinking_steps

        if visible_content.strip():
            thinking_steps.append(visible_content.strip())

        messages.append({
            "role":       "assistant",
            "content":    raw_content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args_parsed = json.loads(tc.function.arguments)
            try:
                result_text, hits = execute_tool(tc.function.name, args_parsed)
                all_sources.extend(hits)
                summary = _trace_result(tc.function.name, result_text, hits)
            except Exception as e:
                result_text = f"Tool error: {e}"
                summary = f"error: {str(e)[:50]}"
            traces.append(
                f"[{tc.function.name}] {_trace_args(tc.function.name, args_parsed)} → {summary}"
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

    return (
        "Agent reached the maximum number of turns without a final answer.",
        all_sources, traces, thinking_steps,
    )


def run_agent_streaming(
    question: str,
    history: list[dict],
    thinking_mode: bool = False,
):
    """Generator that yields SSE-ready dicts as the agent runs.

    Event types:
      trace_start  {"type": "trace_start", "label": str}
      trace_done   {"type": "trace_done",  "content": str}
      token        {"type": "token",       "content": str}   # final-answer tokens
      retract      {"type": "retract"}                       # discard speculative tokens
      answer       {"type": "answer",      "content": str}   # used when no tokens were streamed
      thinking     {"type": "thinking",    "content": str}
      done         {"type": "done", "sources": list, "latency_ms": float}
    """
    t0 = time.time()
    system_content = "/think\n\n" + SYSTEM_PROMPT if thinking_mode else SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system_content}]
    for h in history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    all_sources: list[dict] = []
    traces: list[str] = []

    for _turn in range(MAX_AGENT_TURNS):
        stream = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            extra_body={"options": {"num_ctx": 12288, **({"think": True} if thinking_mode else {})}},
        )

        content_acc: list[str] = []
        tool_acc: dict[int, dict] = {}
        speculative_sent = False
        has_tool_calls = False

        for chunk in stream:
            delta = chunk.choices[0].delta

            if getattr(delta, "tool_calls", None):
                has_tool_calls = True
                for tc in delta.tool_calls:
                    i = tc.index
                    if i not in tool_acc:
                        tool_acc[i] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_acc[i]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_acc[i]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_acc[i]["arguments"] += tc.function.arguments

            if delta.content:
                content_acc.append(delta.content)
                # Stream tokens speculatively only on non-thinking, non-tool turns
                if not has_tool_calls and not thinking_mode:
                    speculative_sent = True
                    yield {"type": "token", "content": delta.content}

        full_content = "".join(content_acc)
        think_text, visible = _extract_thinking(full_content)
        if think_text:
            yield {"type": "thinking", "content": think_text}

        if has_tool_calls:
            if speculative_sent:
                yield {"type": "retract"}

            messages.append({
                "role": "assistant",
                "content": full_content,
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_acc.values()
                ],
            })

            for tc in sorted(tool_acc.items()):
                tc = tc[1]
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    args = {}

                label = _trace_args(tc["name"], args)
                yield {"type": "trace_start", "label": f"[{tc['name']}] {label}"}

                try:
                    result_text, hits = execute_tool(tc["name"], args)
                    all_sources.extend(hits)
                    summary = _trace_result(tc["name"], result_text, hits)
                except Exception as e:
                    result_text = f"Tool error: {e}"
                    summary = f"error: {str(e)[:50]}"
                    hits = []

                full_trace = f"[{tc['name']}] {label} → {summary}"
                traces.append(full_trace)
                yield {"type": "trace_done", "content": full_trace}
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_text})

        else:
            answer = visible.strip()
            if not answer and think_text:
                answer = think_text
            if not answer:
                answer = "The model did not produce an answer. Please try again."

            if not speculative_sent:
                yield {"type": "answer", "content": answer}

            yield {
                "type": "done",
                "sources": all_sources,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
            return

    yield {"type": "answer", "content": "Agent reached the maximum number of turns without a final answer."}
    yield {"type": "done", "sources": all_sources, "latency_ms": round((time.time() - t0) * 1000, 1)}
