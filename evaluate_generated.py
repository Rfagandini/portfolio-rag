"""
Retrieval evaluation over the auto-generated, ground-truth question set.

HOW THIS DIFFERS FROM evaluate_retrieval.py
-------------------------------------------
evaluate_retrieval.py matches on the article KEY, stripping the category
prefix: docs/players_big3__Roger_Federer.txt -> "Roger_Federer". That was
right for the original corpus, where each subject had exactly one document.

It is no longer sufficient. A player now has up to three documents:

    players_top50__Pete_Sampras   <- Wikipedia biography
    career__Pete_Sampras          <- aggregated career statistics

Those share the key "Pete_Sampras". A question like "how many titles did
Sampras win" is only genuinely answered by the career document, but the old
matcher would score the biography as a hit. So this evaluator compares FULL
stems, including the category prefix.

The metric functions themselves are imported from evaluate_retrieval.py --
Hit@k, MRR@k and Recall@k are independent of how documents are identified,
and duplicating them would let the two evaluators silently drift apart.

Usage:
    python evaluate_generated.py
    python evaluate_generated.py --mode dense     # ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from evaluate_retrieval import hit_at_k, mrr_at_k, recall_at_k, _mean
from rag_chain import (
    COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL,
    build_reranking_retriever, get_vector_store,
)

QUESTIONS_PATH = Path("eval_questions_generated.json")


def stem_from_source(source: str) -> str:
    """docs/results__1995_Wimbledon.txt -> 'results__1995_Wimbledon'."""
    return Path(source.replace("\\", "/")).stem


def retrieved_ranking(docs) -> list[str]:
    """Unique document stems in first-appearance order."""
    seen: list[str] = []
    for d in docs:
        stem = stem_from_source(d.metadata.get("source", ""))
        if stem not in seen:
            seen.append(stem)
    return seen


def build_retriever(mode: str, k: int):
    """
    Build one of three retrievers, so the ablation can separate the effect of
    corpus size from the effect of hybrid fusion and reranking.
    """
    if mode == "dense":
        # Deliberately DENSE-only against the hybrid collection. This does not
        # error -- it silently ignores the sparse vectors -- which is exactly
        # the failure mode we want to quantify.
        store = QdrantVectorStore(
            client=get_vector_store().client,
            collection_name=COLLECTION_NAME,
            embedding=HuggingFaceEmbeddings(model_name=DENSE_MODEL),
        )
        return store.as_retriever(search_kwargs={"k": k})
    if mode == "hybrid":
        return get_vector_store().as_retriever(search_kwargs={"k": k})
    if mode == "rerank":
        return build_reranking_retriever(fetch_k=20, top_n=20)
    raise ValueError(f"unknown mode: {mode}")


def evaluate(retriever, questions: list[dict], k_values: list[int]) -> dict:
    per_cat = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overall = defaultdict(lambda: defaultdict(list))

    for i, q in enumerate(questions, start=1):
        expected = set(q["articles"])
        ranked = retrieved_ranking(retriever.invoke(q["input"]))
        for k in k_values:
            h, m, r = (hit_at_k(ranked, expected, k),
                       mrr_at_k(ranked, expected, k),
                       recall_at_k(ranked, expected, k))
            for name, val in (("hit", h), ("mrr", m), ("recall", r)):
                per_cat[q["category"]][k][name].append(val)
                overall[k][name].append(val)
        if i % 25 == 0:
            print(f"    ...{i}/{len(questions)}", flush=True)

    return {"per_category": per_cat, "overall": overall}


def report(results: dict, k_values: list[int], label: str) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(f"\n{'k':<6}{'Hit@k':<12}{'MRR@k':<12}{'Recall@k':<12}")
    print("-" * 42)
    for k in k_values:
        o = results["overall"][k]
        print(f"{k:<6}{_mean(o['hit']):<12.3f}{_mean(o['mrr']):<12.3f}"
              f"{_mean(o['recall']):<12.3f}")

    print("\n--- Per category (Hit@5 / MRR@5 / Recall@5) ---")
    for cat, ks in sorted(results["per_category"].items()):
        s = ks[5]
        print(f"  {cat:<20} n={len(s['hit']):<4} "
              f"Hit={_mean(s['hit']):.3f}  MRR={_mean(s['mrr']):.3f}  "
              f"Recall={_mean(s['recall']):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dense", "hybrid", "rerank", "all"],
                        default="all")
    args = parser.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} generated questions")

    k_values = [1, 3, 5, 10]
    modes = ["dense", "hybrid", "rerank"] if args.mode == "all" else [args.mode]
    labels = {
        "dense": "DENSE ONLY — sparse vectors ignored",
        "hybrid": "HYBRID — dense + sparse, fused by Qdrant (RRF)",
        "rerank": "HYBRID + cross-encoder reranker",
    }

    for mode in modes:
        print(f"\nRunning {mode}...")
        r = evaluate(build_retriever(mode, max(k_values)), questions, k_values)
        report(r, k_values, labels[mode])


if __name__ == "__main__":
    main()
