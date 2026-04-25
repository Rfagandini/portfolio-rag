"""
Retrieval-only evaluation for the hybrid retriever.

This measures how good the RETRIEVER is at surfacing the right articles,
independent of whether the LLM answers correctly. It's a cheaper and more
focused signal than end-to-end evaluation — no LLM calls per query.

Metrics (all in [0, 1], the higher the better):

    Hit@k     — 1 if ANY expected article appears in the top-k retrieved,
                else 0. Averaged across queries.
                Question it answers: "Did we find SOMETHING relevant?"

    MRR@k     — Reciprocal of the rank of the FIRST relevant article
                (1/rank), or 0 if no relevant article in top-k.
                Averaged across queries.
                Question it answers: "How high up did we rank the first
                relevant result?" Values closer to 1 mean better ranking.

    Recall@k  — (# expected articles found in top-k) / (# expected articles).
                Averaged across queries.
                Question it answers: "Did we find ALL the relevant articles?"
                Most useful on comparative/multi-article questions.

Usage:
    python evaluate_retrieval.py
"""

import re
from collections import defaultdict

from langchain_classic.retrievers import EnsembleRetriever
import pickle

from rag_chain import get_vector_store, build_reranking_retriever
from test_questions import test_questions

def sanitize_filename(title: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE)
    safe = re.sub(r"\s+", "_", safe).strip("_")
    return safe[:120]


def article_from_source(source: str) -> str:
    """
    Extract the sanitized article title from a chunk's source path.

    Source looks like:  docs/players_big3__Roger_Federer.txt
    We want:            Roger_Federer

    Filenames are {category}__{sanitized_title}.txt, so we strip the path,
    the .txt extension, and the category prefix before '__'.
    """
    filename = source.replace("\\", "/").split("/")[-1]
    stem = filename[:-4] if filename.endswith(".txt") else filename
    return stem.split("__", 1)[1] if "__" in stem else stem


def expected_article_keys(expected_titles: list[str]) -> set[str]:
    """Normalize expected titles the same way filenames are sanitized."""
    return {sanitize_filename(t) for t in expected_titles}


def retrieved_article_ranking(docs) -> list[str]:
    """
    Convert a list of retrieved chunks (possibly with repeats) into the
    ordered list of UNIQUE article keys, preserving first-appearance order.

    Why: multiple chunks can come from the same article. For ranking,
    'the article is at rank r' means r = position of its FIRST chunk.
    """
    seen = []
    for doc in docs:
        key = article_from_source(doc.metadata.get("source", ""))
        if key not in seen:
            seen.append(key)
    return seen


#METRICS

def hit_at_k(ranked_articles: list[str], expected: set[str], k: int) -> float:
    
    for article in expected:
        
        if article in ranked_articles[:k]:
            return 1
    
    return 0
    
    
def mrr_at_k(ranked_articles: list[str], expected: set[str], k: int) -> float:
    
    for i, found_article in enumerate(ranked_articles[:k], start=1):
        
        if found_article in expected:
            return 1/i
    
    return 0


def recall_at_k(ranked_articles: list[str], expected: set[str], k: int) -> float:

    checked_articles = set()
    for found_article in ranked_articles[:k]:
        
        if found_article in expected:
            checked_articles.add(found_article)
    
    return len(checked_articles)/len(expected) if expected else 0

#DRIVER

def build_eval_retriever(k: int = 10) -> EnsembleRetriever:
    """
    Build the same hybrid retriever used in production, but with a larger
    k so we can compute metrics at multiple cutoffs (1, 3, 5, 10).
    """
    dense = get_vector_store().as_retriever(search_kwargs={"k": k})
    with open("bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    bm25.k = k
    return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.5, 0.5])


def evaluate(retriever, k_values: list[int] = [1, 3, 5, 10]) -> dict:
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overall = defaultdict(lambda: defaultdict(list))

    for q in test_questions:
        expected = expected_article_keys(q["articles"])
        docs = retriever.invoke(q["input"])
        ranked = retrieved_article_ranking(docs)

        for k in k_values:
            h = hit_at_k(ranked, expected, k)
            m = mrr_at_k(ranked, expected, k)
            r = recall_at_k(ranked, expected, k)

            scores[q["category"]][k]["hit"].append(h)
            scores[q["category"]][k]["mrr"].append(m)
            scores[q["category"]][k]["recall"].append(r)
            overall[k]["hit"].append(h)
            overall[k]["mrr"].append(m)
            overall[k]["recall"].append(r)

    return {"per_category": scores, "overall": overall}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def print_report(results: dict, k_values: list[int], label: str = "RETRIEVAL EVALUATION") -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    print("\n--- Overall ---")
    header = f"{'k':<6}{'Hit@k':<12}{'MRR@k':<12}{'Recall@k':<12}"
    print(header)
    print("-" * len(header))
    for k in k_values:
        o = results["overall"][k]
        print(f"{k:<6}{_mean(o['hit']):<12.3f}{_mean(o['mrr']):<12.3f}{_mean(o['recall']):<12.3f}")

    print("\n--- Per category (Hit@5 / MRR@5 / Recall@5) ---")
    for cat, ks in results["per_category"].items():
        s = ks[5]
        n = len(s["hit"])
        print(f"  {cat:<18} n={n:<4} "
              f"Hit={_mean(s['hit']):.3f}  "
              f"MRR={_mean(s['mrr']):.3f}  "
              f"Recall={_mean(s['recall']):.3f}")


if __name__ == "__main__":
    k_values = [1, 3, 5, 10]

    # Baseline: hybrid retriever only (BM25 + dense, no reranking)
    base_retriever = build_eval_retriever(k=max(k_values))
    base_results = evaluate(base_retriever, k_values=k_values)
    print_report(base_results, k_values, label="BASELINE — hybrid only")

    # Reranked: hybrid + cross-encoder reranker
    # fetch_k=20 gives the reranker a candidate pool to choose from;
    # top_n=20 keeps enough chunks to dedup down to ~10 unique articles.
    rerank_retriever = build_reranking_retriever(fetch_k=20, top_n=20)
    rerank_results = evaluate(rerank_retriever, k_values=k_values)
    print_report(rerank_results, k_values, label="RERANKED — hybrid + cross-encoder")
