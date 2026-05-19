"""Per-source-type chunkers. All return a list of dicts with keys:
    text, header, ord, ts_from, ts_to, participants (list[str] or None)

The `text` field is what gets embedded and FTS-indexed; it ALREADY
contains the header prepended, so both BM25 and the embedding see the
metadata.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

# Approximate token count: 1 token ~= 4 chars for English text.
TARGET_TOKENS = 500
OVERLAP_TOKENS = 50
TARGET_CHARS = TARGET_TOKENS * 4
OVERLAP_CHARS = OVERLAP_TOKENS * 4

DOCUMENT_LIKE = {
    "confluence",
    "google_drive",
    "jira",
    "linear",
    "hubspot",
    "github",
    "fireflies",
}

EMAIL_HEADER_RE = re.compile(
    r"^From:\s*(?P<from>.+?)\s*\n"
    r"To:\s*(?P<to>.+?)\s*\n"
    r"(?:Cc:\s*(?P<cc>.+?)\s*\n)?"
    r"Date:\s*(?P<date>.+?)\s*\n"
    r"Subject:\s*(?P<subject>.+?)\s*\n",
    re.MULTILINE,
)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
SLACK_SPEAKER_RE = re.compile(r"^([\w.\-]+):\s", re.MULTILINE)


def _unescape_email_blob(s: str) -> str:
    # The dataset stores email messages with DOUBLE-escaped newlines —
    # the actual characters are: backslash, backslash, 'n' (codepoints
    # 92, 92, 110). Collapse the double-escape variants first, then the
    # single-escape ones, then the common quote escapes.
    return (
        s.replace("\\\\r\\\\n", "\n")
        .replace("\\\\n", "\n")
        .replace("\\\\t", "\t")
        .replace('\\\\"', '"')
        .replace("\\\\'", "'")
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )


# Used to scan for the start of each email message inside a Python-repr'd
# list. Each message begins with "From: " (occasionally preceded by quote +
# whitespace). A regex-based quoted-string extractor on these long blobs
# triggers catastrophic backtracking, so we use a forward scan instead.
_EMAIL_START_RE = re.compile(r"From:\s")


def _split_email_blob(raw: str) -> list[str]:
    """Linear scan that splits a Python-repr'd list-of-emails blob into
    individual messages. Robust to mixed quote styles and pathological
    backslash sequences — runs in O(n)."""
    # Find every "From: " occurrence. The substring between two consecutive
    # occurrences is one message (minus any trailing quote/comma/space).
    starts = [m.start() for m in _EMAIL_START_RE.finditer(raw)]
    if not starts:
        return [raw]
    starts.append(len(raw))
    out: list[str] = []
    for i in range(len(starts) - 1):
        seg = raw[starts[i] : starts[i + 1]]
        # Strip trailing closing-quote / comma / whitespace before the next msg.
        seg = re.sub(r"['\"\\,\s]+$", "", seg)
        if seg:
            out.append(seg)
    return out


def _slide(text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    if len(text) <= target:
        return [text]
    out: list[str] = []
    step = target - overlap
    i = 0
    while i < len(text):
        out.append(text[i : i + target])
        if i + target >= len(text):
            break
        i += step
    return out


def chunk_document_like(doc_id: str, source_type: str, title: str, content: str) -> list[dict]:
    header = f"[source: {source_type}] [title: {title or ''}]"
    pieces = _slide(content)
    return [
        {
            "ord": i,
            "header": header,
            "text": f"{header}\n\n{p}",
            "ts_from": None,
            "ts_to": None,
            "participants": None,
        }
        for i, p in enumerate(pieces)
    ]


def _parse_gmail_messages(raw: str) -> list[dict]:
    """The `content` column for gmail is a Python-repr'd list of strings; some
    rows are valid JSON, some use single quotes, some mix both. We try
    JSON first and fall back to a deterministic split on the "From: " header
    that begins every email — robust to mixed quote styles and avoids the
    catastrophic backtracking that a generic quoted-string regex would hit
    on long, escape-heavy bodies."""
    msgs: list[str] = []

    try:
        v = json.loads(raw)
        if isinstance(v, list):
            msgs = [m for m in v if isinstance(m, str)]
    except Exception:
        pass

    if not msgs:
        msgs = _split_email_blob(raw)

    parsed: list[dict] = []
    for m in msgs:
        body = _unescape_email_blob(m)
        match = EMAIL_HEADER_RE.search(body)
        if not match:
            parsed.append({"from": "", "to": "", "cc": "", "date": "", "subject": "", "body": body})
            continue
        parsed.append(
            {
                "from": match.group("from"),
                "to": match.group("to"),
                "cc": match.group("cc") or "",
                "date": match.group("date"),
                "subject": match.group("subject"),
                "body": body[match.end() :].strip(),
            }
        )
    return parsed


def _norm_subject(s: str) -> str:
    return re.sub(r"^(re|fwd|fw):\s*", "", s.strip(), flags=re.IGNORECASE).strip().lower()


def chunk_gmail(doc_id: str, source_type: str, title: str, content: str) -> list[dict]:
    msgs = _parse_gmail_messages(content)
    if not msgs:
        return []

    # Group by normalized subject; collect participants and date range.
    threads: list[list[dict]] = []
    current: list[dict] = []
    current_subj: str | None = None
    for m in msgs:
        ns = _norm_subject(m["subject"]) or _norm_subject(title)
        if current_subj is None or ns == current_subj:
            current.append(m)
            current_subj = ns
        else:
            threads.append(current)
            current = [m]
            current_subj = ns
    if current:
        threads.append(current)

    chunks: list[dict] = []
    ord_i = 0
    for thread in threads:
        subject = thread[0]["subject"] or title
        emails = set()
        for m in thread:
            for field in ("from", "to", "cc"):
                emails.update(EMAIL_RE.findall(m[field]))
        dates = [m["date"] for m in thread if m["date"]]
        ts_from = min(dates) if dates else None
        ts_to = max(dates) if dates else None

        # Window of 3-5 messages with 1-message overlap.
        win = 4
        step = win - 1
        i = 0
        first = True
        while i < len(thread):
            window = thread[i : i + win]
            if not window:
                break
            participants_line = ", ".join(sorted(emails))
            dates_line = (
                f"{ts_from} -> {ts_to}" if ts_from and ts_to else (ts_from or ts_to or "")
            )
            header = (
                f"[source: gmail] [thread: {subject}]\n"
                f"[participants: {participants_line}]\n"
                f"[dates: {dates_line}]"
            )
            body_parts = []
            for m in window:
                body_parts.append(
                    f"From: {m['from']}\nTo: {m['to']}\nDate: {m['date']}\nSubject: {m['subject']}\n\n{m['body']}"
                )
            body = "\n\n---\n\n".join(body_parts)
            text = f"{header}\n\n{body}"
            # If a single window is huge, slide inside the body.
            for j, piece in enumerate(_slide(text)):
                chunks.append(
                    {
                        "ord": ord_i,
                        "header": header,
                        "text": piece if j == 0 else f"{header}\n\n{piece}",
                        "ts_from": ts_from,
                        "ts_to": ts_to,
                        "participants": sorted(emails) or None,
                    }
                )
                ord_i += 1
            if i + win >= len(thread):
                break
            i += step
            first = False
    return chunks


def chunk_slack(doc_id: str, source_type: str, title: str, content: str) -> list[dict]:
    channel = title or ""
    # Split into conversational blocks on blank lines; if there are no blank
    # lines, treat the whole thing as one block.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    if not blocks:
        blocks = [content]

    speakers_all = set(SLACK_SPEAKER_RE.findall(content))

    chunks: list[dict] = []
    ord_i = 0
    # Window blocks so each chunk is around TARGET_CHARS.
    buf: list[str] = []
    buf_len = 0
    for block in blocks:
        if buf and buf_len + len(block) > TARGET_CHARS:
            joined = "\n\n".join(buf)
            local_speakers = sorted(set(SLACK_SPEAKER_RE.findall(joined)) or speakers_all)
            header = (
                f"[source: slack] [channel: {channel}]\n"
                f"[participants: {', '.join(local_speakers)}]"
            )
            text = f"{header}\n\n{joined}"
            for j, piece in enumerate(_slide(text)):
                chunks.append(
                    {
                        "ord": ord_i,
                        "header": header,
                        "text": piece if j == 0 else f"{header}\n\n{piece}",
                        "ts_from": None,
                        "ts_to": None,
                        "participants": local_speakers or None,
                    }
                )
                ord_i += 1
            # Keep last block as overlap.
            buf = [block]
            buf_len = len(block)
        else:
            buf.append(block)
            buf_len += len(block) + 2

    if buf:
        joined = "\n\n".join(buf)
        local_speakers = sorted(set(SLACK_SPEAKER_RE.findall(joined)) or speakers_all)
        header = (
            f"[source: slack] [channel: {channel}]\n"
            f"[participants: {', '.join(local_speakers)}]"
        )
        text = f"{header}\n\n{joined}"
        for j, piece in enumerate(_slide(text)):
            chunks.append(
                {
                    "ord": ord_i,
                    "header": header,
                    "text": piece if j == 0 else f"{header}\n\n{piece}",
                    "ts_from": None,
                    "ts_to": None,
                    "participants": local_speakers or None,
                }
            )
            ord_i += 1
    return chunks


def chunk(doc_id: str, source_type: str, title: str, content: str) -> list[dict]:
    if source_type == "gmail":
        return chunk_gmail(doc_id, source_type, title or "", content or "")
    if source_type == "slack":
        return chunk_slack(doc_id, source_type, title or "", content or "")
    if source_type in DOCUMENT_LIKE:
        return chunk_document_like(doc_id, source_type, title or "", content or "")
    # Unknown source: fall back to document-like behavior.
    return chunk_document_like(doc_id, source_type, title or "", content or "")
