"""
indexing/eval_agent.py — End-to-end agent evaluation against questions_test.parquet.

Calls the backend /query endpoint for each question and measures:
  - Source hit rate : did the agent include an expected_doc_id in its sources?
  - Fact coverage   : what fraction of answer_facts keywords appear in the answer?

The fact check is keyword-overlap based (>= 50% of non-stopword tokens from the
fact sentence must appear in the answer). It is not an LLM judge, but it gives a
fast, reproducible signal without any extra API calls.

Results are broken down by question_type (basic, multi-hop, etc.) at the end.

Run from the Backend VS Code app on Nuvolos (uvicorn must already be running):
    cd /files/company-rag-agent
    python indexing/eval_agent.py \
        --questions data/raw/questions_test.parquet \
        --backend   http://localhost:8500

Quick sanity check on the first 20 questions:
    python indexing/eval_agent.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict

import httpx
import pyarrow.parquet as pq

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "it", "its", "this", "that", "with",
    "from", "by", "as", "be", "been", "have", "has", "had", "which", "that",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def _fact_covered(fact: str, answer: str, threshold: float = 0.5) -> bool:
    """True if >= threshold of the fact's key tokens appear anywhere in the answer."""
    fact_tokens = _tokenize(fact)
    if not fact_tokens:
        return True
    answer_tokens = _tokenize(answer)
    return len(fact_tokens & answer_tokens) / len(fact_tokens) >= threshold


def _parse_list(val) -> list[str]:
    if val is None:
        return []
    if hasattr(val, "tolist"):
        return [str(x) for x in val.tolist()]
    if isinstance(val, list):
        return [str(x) for x in val]
    text = str(val).strip().replace("'", '"')
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return [text] if text else []


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end agent eval via /query endpoint.")
    ap.add_argument("--questions", default="data/raw/questions_test.parquet")
    ap.add_argument("--backend",   default="http://localhost:8500")
    ap.add_argument("--limit",     type=int,   default=0,     help="Evaluate first N questions only (debug)")
    ap.add_argument("--timeout",   type=float, default=180.0, help="Per-request timeout in seconds")
    ap.add_argument("--fact-threshold", type=float, default=0.5,
                    help="Min token-overlap fraction to count a fact as covered (default 0.5)")
    args = ap.parse_args()

    qt = pq.read_table(args.questions).to_pandas()
    if args.limit:
        qt = qt.head(args.limit)
    print(f"[eval] {len(qt)} questions  backend={args.backend}  timeout={args.timeout}s")

    source_hits = 0
    total_facts = 0
    covered_facts = 0
    errors = 0
    by_type: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "src_hits": 0, "facts": 0, "facts_covered": 0}
    )

    t0 = time.time()
    with httpx.Client(timeout=args.timeout, base_url=args.backend) as client:
        for i, row in enumerate(qt.itertuples(index=False)):
            q_type    = str(row.question_type)
            expected  = set(_parse_list(row.expected_doc_ids))
            facts     = _parse_list(row.answer_facts)

            try:
                resp = client.post("/query", json={"question": row.question})
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                print(f"  ERROR {getattr(row, 'question_id', i)}: {e}")
                errors += 1
                continue

            # Source hit: agent cited at least one expected doc in its sources
            returned_ids = {s["doc_id"] for s in result.get("sources", [])}
            src_hit = bool(expected & returned_ids)
            if src_hit:
                source_hits += 1

            # Fact coverage: keyword-overlap check over each answer_fact
            answer = result.get("answer", "")
            n_covered = sum(1 for f in facts if _fact_covered(f, answer, args.fact_threshold))
            total_facts   += len(facts)
            covered_facts += n_covered

            by_type[q_type]["n"]              += 1
            by_type[q_type]["src_hits"]       += int(src_hit)
            by_type[q_type]["facts"]          += len(facts)
            by_type[q_type]["facts_covered"]  += n_covered

            if (i + 1) % 10 == 0:
                done = i + 1 - errors
                dt   = time.time() - t0
                print(
                    f"  {i+1}/{len(qt)}  ({(i+1)/dt:.1f} q/s)  "
                    f"src_hit={source_hits/max(done,1):.2f}  "
                    f"fact_cov={covered_facts/max(total_facts,1):.2f}"
                )

    n = len(qt) - errors
    print()
    print(f"Results over {n} questions  ({errors} errors / timeouts)")
    print(f"  Source hit rate : {source_hits/max(n,1):.3f}  "
          f"(agent cited an expected doc in its sources)")
    print(f"  Fact coverage   : {covered_facts/max(total_facts,1):.3f}  "
          f"(answer_facts token-overlap >= {args.fact_threshold})")
    print()
    print(f"{'Type':<20} {'N':>5} {'Src hit':>9} {'Fact cov':>10}")
    print("-" * 47)
    for qtype, d in sorted(by_type.items()):
        src  = d["src_hits"] / d["n"]
        fact = d["facts_covered"] / max(d["facts"], 1)
        print(f"{qtype:<20} {d['n']:>5} {src:>9.3f} {fact:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
