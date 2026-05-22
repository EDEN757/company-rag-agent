import os
import re
import subprocess
import db
import fusion
import kb
from config import BASH_DENY


def execute_tool(name: str, args: dict) -> tuple[str, list[dict]]:
    """Dispatch a tool call. Returns (text_for_llm, source_hits)."""

    if name == "search":
        hits = fusion.search(
            query=args["query"],
            source_types=args.get("source_types"),
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            participant=args.get("participant"),
            top_n=args.get("top_n", 6),
        )
        if not hits:
            return "No results found. Try broadening the query or removing filters.", []
        lines = []
        for h in hits:
            ts_from, ts_to = h.get("ts_from") or "", h.get("ts_to") or ""
            time_str = (
                f" [{ts_from} → {ts_to}]" if ts_from and ts_to and ts_to != ts_from
                else (f" [{ts_from}]" if ts_from else "")
            )
            score_str = f"score={h['score']} vec={h['vec_score']} kw={h['kw_score']}"
            if "rerank_score" in h:
                score_str += f" rerank={h['rerank_score']:.2f}"
            lines.append(
                f"#{h['chunk_id']} (doc={h['doc_id']}, source={h['source_type']}{time_str}, "
                f"{score_str})\ntitle: {h['title'] or ''}\npreview: {h['preview']}"
            )
        return "\n\n".join(lines), hits

    if name == "open_document":
        doc = db.fetch_document(args["doc_id"])
        if not doc:
            return f"No document found with doc_id={args['doc_id']}", []
        hit = {
            "chunk_id": -1, "doc_id": doc["doc_id"], "source_type": doc["source_type"],
            "title": doc["title"], "score": 0.0, "vec_score": 0.0, "kw_score": 0.0,
            "preview": doc["content"][:600].replace("\n", " ").strip(),
            "ts_from": None, "ts_to": None, "opened": True,
        }
        return (
            f"doc_id: {doc['doc_id']}\nsource: {doc['source_type']}\n"
            f"title: {doc['title'] or ''}\n\n{doc['content']}"
        ), [hit]

    if name == "add_document":
        doc_id = kb.add_document(
            source_type=args["source_type"],
            title=args["title"],
            content=args["content"],
            participants=args.get("participants"),
            date=args.get("date"),
        )
        hit = {
            "chunk_id": -1, "doc_id": doc_id,
            "source_type": args["source_type"], "title": args["title"],
            "score": 0.0, "vec_score": 0.0, "kw_score": 0.0,
            "preview": args["content"][:600].replace("\n", " ").strip(),
            "ts_from": args.get("date"), "ts_to": args.get("date"), "opened": True,
        }
        return (
            f"Created and indexed {doc_id} "
            f"(source={args['source_type']}, title={args['title']!r}). "
            f"Include {doc_id} in your answer."
        ), [hit]

    if name == "edit_document":
        result = kb.edit_document(
            doc_id=args["doc_id"],
            new_content=args.get("new_content"),
            old_string=args.get("old_string"),
            new_string=args.get("new_string"),
        )
        doc = db.fetch_document(args["doc_id"])
        hit = {
            "chunk_id": -1, "doc_id": args["doc_id"],
            "source_type": doc["source_type"] if doc else "",
            "title": doc["title"] if doc else args["doc_id"],
            "score": 0.0, "vec_score": 0.0, "kw_score": 0.0,
            "preview": (doc["content"][:600].replace("\n", " ").strip()) if doc else "",
            "ts_from": None, "ts_to": None, "opened": True,
        }
        return result + f" Include {args['doc_id']} in your answer.", [hit]

    if name == "read":
        path = os.path.expanduser(args["path"])
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), []

    if name == "write":
        path = os.path.expanduser(args["path"])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return f"Wrote {len(args['content'])} bytes to {path}", []

    if name == "edit":
        path = os.path.expanduser(args["path"])
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        old_s, new_s = args["old_string"], args["new_string"]
        count = text.count(old_s)
        if count == 0:
            return f"Error: old_string not found in {path}", []
        if count > 1:
            return f"Error: old_string appears {count} times — make it more specific", []
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old_s, new_s, 1))
        return f"Edited {path}", []

    if name == "bash":
        command = args["command"]
        for pattern in BASH_DENY:
            if re.search(pattern, command):
                return f"Refused: command matches deny pattern '{pattern}'.", []
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, executable="/bin/bash",
        )
        output = f"exit={result.returncode}\n--- stdout ---\n{result.stdout}"
        if result.stderr:
            output += f"\n--- stderr ---\n{result.stderr}"
        return output, []

    return f"Unknown tool: {name}", []
