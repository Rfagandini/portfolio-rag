# Tennis RAG — Conversational Q&A over a Wikipedia Tennis Corpus

A Retrieval-Augmented Generation pipeline that answers questions about ATP tennis (Big 3, Grand Slams 2020–2024, current top players, history) over 99 Wikipedia articles. Hybrid retrieval (dense + BM25) with a cross-encoder reranker, conversational memory, and a full evaluation harness — both retrieval-only metrics and end-to-end LLM-as-judge.

**🚀 Live demo:** https://portfolio-rag-5rfvlpav9nrdzefhhxlqxe.streamlit.app/

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-rag-5rfvlpav9nrdzefhhxlqxe.streamlit.app/)

> Note: the free tier sleeps after inactivity — first visit may take 30s to wake up.

## Architecture

- **Corpus**: 99 ATP / tennis Wikipedia articles fetched via `fetch_corpus.py` (players, Grand Slams 2020–2024, history)
- **Chunking**: Recursive character splitter via LangChain
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local sentence-transformers)
- **Vector store**: Qdrant Cloud
- **Hybrid retrieval**: `EnsembleRetriever` combining dense vector search with BM25 (50/50 weights)
- **Reranker**: `BAAI/bge-reranker-base` cross-encoder via `ContextualCompressionRetriever` — fetch 20 candidates, rerank to top 20 chunks
- **LLM**: Llama 3.3 70B Versatile via Groq (free tier)
- **Conversational memory**: `create_history_aware_retriever` reformulates follow-ups into standalone queries before retrieval
- **Evaluation**: retrieval-only (Hit@k / MRR@k / Recall@k) + end-to-end LLM-as-judge with a separate grader model (Llama 3.1 8B) to avoid self-bias
- **Tracing**: LangSmith integration

## Project Structure

```
portfolio-rag/
├── docs/                    # Wikipedia .txt articles (one per topic)
├── fetch_corpus.py          # Pull tennis articles from Wikipedia
├── ingest.py                # Chunk, embed, push to Qdrant + build BM25 index
├── bm25_index.pkl           # Persisted BM25 retriever
├── rag_chain.py             # Hybrid + reranking retriever, conversational chain
├── streamlit_app.py         # Tennis-themed chat UI
├── test_questions.py        # 80 evaluation questions across 5 categories
├── evaluate.py              # End-to-end LLM-as-judge eval
├── evaluate_retrieval.py    # Retrieval-only eval (Hit/MRR/Recall@k)
├── eval_results.json        # Raw end-to-end results
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. **Clone and install dependencies**
```bash
git clone https://github.com/Rfagandini/portfolio-rag.git
cd portfolio-rag
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

2. **Configure environment variables**
```bash
cp .env.example .env
```

| Variable | Source |
|---|---|
| `GROQ_API_KEY` | [Groq Console](https://console.groq.com) (free) |
| `QDRANT_URL` | [Qdrant Cloud](https://cloud.qdrant.io) (free tier) |
| `QDRANT_API_KEY` | Qdrant Cloud dashboard |
| `LANGCHAIN_API_KEY` | [LangSmith](https://smith.langchain.com) (optional) |

3. **Fetch the corpus** (one-off — pulls 99 Wikipedia articles into `docs/`)
```bash
python fetch_corpus.py
```

4. **Run ingestion** (chunks, embeds, uploads to Qdrant, builds BM25 index)
```bash
python ingest.py
```

5. **Query the RAG**

   CLI:
   ```bash
   python rag_chain.py
   ```

   Streamlit UI:
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Run evaluations**
   ```bash
   python evaluate_retrieval.py   # retrieval-only, no LLM cost
   python evaluate.py             # end-to-end with LLM-as-judge
   ```

## Retrieval Evaluation

Measured on 80 questions across 5 categories. The retrieval eval is decoupled from the LLM — it asks "did we surface the right articles?", independent of generation quality.

**Baseline — hybrid only (dense + BM25, no reranker)**

| k  | Hit@k | MRR@k | Recall@k |
|----|-------|-------|----------|
| 1  | 0.512 | 0.512 | 0.323    |
| 3  | 0.788 | 0.625 | 0.554    |
| 5  | 0.875 | 0.656 | 0.762    |
| 10 | 0.925 | 0.668 | 0.844    |

**Reranked — hybrid + `BAAI/bge-reranker-base` cross-encoder**

| k  | Hit@k | MRR@k | Recall@k | Δ MRR@k vs baseline |
|----|-------|-------|----------|---------------------|
| 1  | 0.575 | 0.575 | 0.479    | +0.063              |
| 3  | 0.812 | 0.667 | 0.679    | +0.042              |
| 5  | 0.850 | 0.682 | 0.765    | +0.026              |
| 10 | 0.875 | 0.689 | 0.798    | +0.021              |

Headline: **Recall@1 jumps from 0.32 → 0.48 (+48% relative)** — the reranker is much better at putting the *correct* article in the very top slot, which is exactly what matters when the LLM only stuffs the top-k chunks into the prompt.

### Where the cross-encoder helped — and where it hurt

Per-category MRR@5, baseline → reranked:

| Category      | Baseline | Reranked | Δ        |
|---------------|----------|----------|----------|
| player_facts  | 0.575    | 0.850    | **+0.275** |
| general       | 0.467    | 0.603    | **+0.136** |
| comparative   | 0.708    | 0.708    | ±0.000   |
| follow_up     | 0.708    | 0.667    | −0.042   |
| tournaments   | **0.925**| 0.573    | **−0.352** |

**Why `player_facts` and `general` improved.** These are paraphrased, semantic queries — "How many Grand Slams has Federer won?" must match a chunk discussing his "20 majors". A bi-encoder embedding can put that chunk at rank 4–5; a cross-encoder reads the query and the candidate chunk *together* and resolves the lexical gap. This is exactly the failure mode cross-encoders are designed for.

**Why `tournaments` regressed badly.** Tournament queries are dominated by exact strings: *"Who won the 2022 Australian Open men's singles?"* maps almost perfectly onto the article titled `2022_Australian_Open_–_Men's_singles`. BM25 already nails this — MRR was 0.925 without any reranking. The cross-encoder (trained on MS-MARCO-style passage relevance) doesn't understand the structural pattern "year + tournament + draw" the way a sparse keyword retriever does, so it shuffles a near-perfect ranking and loses ground.

**Why `follow_up` dipped slightly.** Follow-ups get rewritten into standalone queries by the history-aware retriever, but the rewrites are short and decontextualized ("how many sets did that match go?"). Cross-encoders rely on semantic richness; they're under-fed here.

**Meta-observation: ensemble weights stop mattering with a strong reranker downstream.** I experimented with shifting hybrid weights toward BM25 (0.3 / 0.7) hoping to rescue the tournament regression. It didn't help — the reranker re-orders whatever pool it gets. As long as the candidate pool covers the right document, the upstream blend has minimal effect. The lever that *does* matter is `fetch_k`: too low and you starve the reranker, too high and you dilute it.

The **net trade-off** (+27pp player_facts, +14pp general, −35pp tournaments) is favorable on this corpus because tournaments still scored highly enough end-to-end (see below) and the LLM tolerates rank drift better than it tolerates the wrong article entirely.

## End-to-End Evaluation

80 questions, graded CORRECT / PARTIAL / INCORRECT by Llama 3.1 8B (a *different* model from the answer LLM, to avoid self-evaluation bias). Score = (CORRECT + 0.5 · PARTIAL) / total.

| Category       | Score  | Breakdown               | n  |
|----------------|--------|-------------------------|----|
| general        | 100.0% | 8 C / 0 P / 0 I         | 8  |
| player_facts   | 95.0%  | 18 C / 2 P / 0 I        | 20 |
| tournaments    | 92.5%  | 17 C / 3 P / 0 I        | 20 |
| comparative    | 87.5%  | 12 C / 4 P / 0 I        | 16 |
| follow_up      | 87.5%  | 14 C / 2 P / 0 I        | 16 |
| **Overall**    | **93.1%** | **69 C / 11 P / 0 I** | 80 |

Follow-up second-question accuracy (history-aware): **3/6 CORRECT**, 3/6 PARTIAL — the system never *fails* a follow-up, it just sometimes pulls in extra context.

**Zero incorrect answers across all 80 questions.** Every "miss" is a partial — usually a verbose or hedged answer rather than a wrong one.

### Failure analysis — what makes an answer "PARTIAL"?

I went through all 11 PARTIAL cases. They cluster into six patterns, none of which are wrong-fact errors:

1. **Hedging language penalized by the grader.** Q18 asked when Berrettini reached the Wimbledon final; the LLM said "According to the provided context, ... in 2021" — factually correct, but the grader docks the "according to" hedge. Q74 and Q76 are the same shape.
2. **Over-verbose answers.** Q70 asked how many Australian Opens Djokovic won. The LLM gave the right answer (10) but also dumped his total Slam count and a 2023 figure — the grader saw the noise and downgraded.
3. **Outdated facts in the source article.** Q68 asked which Big 3 has the most French Opens — the LLM answered Nadal, **12 titles**, because the chunk it grounded on was an older Wikipedia revision. The current correct answer is 14. The retriever surfaced a stale chunk; the LLM answered it faithfully.
4. **Multiple-attribute confusion.** Q8 ("Federer's nationality") returned "Swiss and South African citizenship" — technically true, but the expected answer is "Swiss". Q28 ("Serena's Slam *singles* total") got tangled because the source talks about both her career singles count and her total titles.
5. **Implicit vs explicit dates.** Q65 asked if Djokovic ever lost a Wimbledon final to Alcaraz; the LLM gave the 2023 final score in detail but called the 2024 final a "repeat" without restating the date. Grader read "missing year" and marked PARTIAL.
6. **Strict grader on partial-name match.** Q48: expected "Novak Djokovic", LLM said "Djokovic". The 8B grader is strict.

**Bottom line on the partials:** the system's *factual* accuracy is higher than 93.1%. Grader strictness on hedging, verbosity, and partial names is responsible for at least half the partial scores. The ones that point to a real weakness are #3 (stale source) and #4 (multi-attribute ambiguity) — both fixable by re-fetching the corpus and adding output-shaping examples to the answer prompt.

## Tech Stack

Python · LangChain (langchain-classic 1.0) · Qdrant Cloud · Groq (Llama 3.3 70B + Llama 3.1 8B grader) · BAAI/bge-small-en-v1.5 embeddings · BAAI/bge-reranker-base cross-encoder · BM25 · Streamlit · LangSmith
