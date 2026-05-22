"""
backend/eval.py — Retrieval quality evaluation harness

Measures Recall@k, MRR@k, nDCG@k for k ∈ {1, 3, 5, 10} against a labeled test set.

Test file format — one JSON object per line:
  {"question": "...", "expected_doc_ids": ["dsid_abc", "dsid_xyz"]}

Usage (from the backend container):
    cd /files/company-rag-agent/backend
    python eval.py --questions eval_questions.jsonl
    python eval.py --questions eval_questions.jsonl --top-k 10 --limit 50 --no-hyde
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


# ── Metrics ───────────────────────────────────────────────────────────────────

def _ndcg(predicted: list[str], expected: set[str], k: int) -> float:
    dcg = sum(
        1 / math.log2(i + 2)
        for i, d in enumerate(predicted[:k])
        if d in expected
    )
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(expected))))
    return dcg / ideal if ideal > 0 else 0.0


def _mrr(predicted: list[str], expected: set[str], k: int) -> float:
    for i, d in enumerate(predicted[:k], 1):
        if d in expected:
            return 1.0 / i
    return 0.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj or "expected_doc_ids" not in obj:
                continue
            expected = obj["expected_doc_ids"]
            if isinstance(expected, str):
                expected = json.loads(expected.replace("'", '"'))
            questions.append({"question": obj["question"], "expected_doc_ids": list(expected)})
    return questions


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate hybrid RAG retrieval quality.")
    ap.add_argument("--questions", required=True, help="Path to JSONL test file")
    ap.add_argument("--top-k", type=int, default=10, help="Maximum results to retrieve per query")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only first N questions (0 = all)")
    ap.add_argument("--no-hyde", action="store_true", help="Disable HyDE for faster evaluation")
    args = ap.parse_args()

    if args.no_hyde:
        os.environ["HYDE_ENABLED"] = "false"

    import db
    import bm25
    import fusion
    import hyde
    import reranker
    from config import OLLAMA_HOST, LLM_MODEL, EMBED_MODEL, HYDE_ENABLED, RERANKER_MODEL

    db.connect()
    bm25.init_corpus_stats()
    hyde.init_hyde()
    reranker.init_reranker()

    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        print("No valid questions found in test file.")
        return 1

    hyde_status = "enabled" if HYDE_ENABLED else "disabled"
    reranker_status = RERANKER_MODEL if RERANKER_MODEL else "disabled"
    print(f"Evaluating {len(questions)} questions  top-k={args.top_k}")
    print(f"  LLM: {LLM_MODEL}  embed: {EMBED_MODEL}  ollama: {OLLAMA_HOST}")
    print(f"  HyDE: {hyde_status}  reranker: {reranker_status}\n")

    ks = sorted({1, 3, 5, args.top_k})
    recall  = {k: 0   for k in ks}
    mrr_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    valid = 0
    errors = 0
    t0 = time.time()

    for i, item in enumerate(questions):
        expected = set(item["expected_doc_ids"])
        if not expected:
            continue
        try:
            hits = fusion.search(item["question"], top_n=args.top_k)
            predicted = [h["doc_id"] for h in hits]
        except Exception as e:
            print(f"  ERROR on q{i+1}: {e}")
            errors += 1
            continue

        for k in ks:
            top = predicted[:k]
            if expected & set(top):
                recall[k] += 1
            mrr_sum[k]  += _mrr(predicted, expected, k)
            ndcg_sum[k] += _ndcg(predicted, expected, k)
        valid += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(questions)}  ({(i+1)/elapsed:.1f} q/s)")

    if not valid:
        print("No valid questions evaluated.")
        return 1

    elapsed = time.time() - t0
    print(f"\n{'k':>4}  {'Recall':>8}  {'MRR':>8}  {'nDCG':>8}")
    print("-" * 38)
    for k in ks:
        print(f"{k:>4}  {recall[k]/valid:>8.3f}  {mrr_sum[k]/valid:>8.3f}  {ndcg_sum[k]/valid:>8.3f}")

    print(f"\n{valid} questions in {elapsed:.1f}s  ({valid/elapsed:.1f} q/s)"
          + (f"  {errors} errors" if errors else ""))
    db.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
