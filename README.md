# Tennis RAG: Conversational Q&A over an ATP Tennis Corpus

A Retrieval-Augmented Generation pipeline that answers questions about ATP tennis from 1990 to today. It covers every player who reached the top 50 and every tour-level tournament, using hybrid retrieval (dense embeddings plus BM25, fused inside Qdrant), conversational memory, and an evaluation harness with both retrieval metrics and ground-truth question sets.

**Live demo:** https://portfolio-rag-5rfvlpav9nrdzefhhxlqxe.streamlit.app/

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-rag-5rfvlpav9nrdzefhhxlqxe.streamlit.app/)

> The free tier sleeps after inactivity, so the first visit may take around 30 seconds to wake up.

## Example output

```
Q: What is the head-to-head record between Roger Federer and Rafael Nadal?
A: Rafael Nadal leads 24-16. They played 40 completed matches at tour level
   between 2004 and 2019, and also met 1 further time that ended in a
   walkover, which does not count towards their head-to-head record.

Q: Who won the 1995 Wimbledon final and what was the score?
A: Pete Sampras won, beating Boris Becker 6-7(5) 6-2 6-4 6-2.
```

## Where the data comes from

The corpus combines two sources, because neither one is enough on its own.

| Source | Supplies | Why |
|---|---|---|
| Wikipedia | 539 articles, including 485 player biographies | Prose about careers, playing styles, rivalries |
| [TML-Database](https://github.com/Tennismylife/TML-Database) | 115,243 matches, 1990 to 2026 | Tournament results, scores, head-to-heads |

**Why I generate tournament documents instead of scraping them.** I tried scraping Wikipedia tournament pages first and the results were unusable. The `wikipedia-api` library strips wiki tables, and tournament draws *are* tables. Measuring the extracted text made it obvious:

| Page | Characters extracted |
|---|---|
| Pete Sampras | 37,897 |
| 2023 Wimbledon, Men's singles | 2,392 |
| 1995 Wimbledon, Men's singles | 1,116 |

Scraping 5,000 tournament pages would have produced 5,000 near-empty stubs. Building those documents from the match CSVs instead gives complete results with every score and round.

**Writing chunk-safe documents.** Chunks get split at around 800 characters and a chunk carries no memory of the document it came from. So instead of a header followed by bare rows, every generated line repeats its own context:

```
At the 1995 Wimbledon (Grass, Grand Slam), in the Final,
Pete Sampras beat Boris Becker 6-7(5) 6-2 6-4 6-2.
```

This costs disk space and buys retrieval correctness. Without it, a chunk starting mid-document would read "in the Round of 64, X beat Y 6-4 6-3" with no year or tournament, which is unretrievable.

## Architecture

- **Corpus:** 10,260 documents and 54,488 chunks. 539 Wikipedia articles, 5,357 tournament-edition results, 485 career summaries, 3,880 head-to-head documents.
- **Chunking:** recursive character splitter, tuned per document type. Wikipedia prose at 500/100, generated match documents at 800 with newline separators so match lines survive whole.
- **Embeddings:** `BAAI/bge-small-en-v1.5` run locally through sentence-transformers.
- **Vector store:** Qdrant Cloud, storing a dense and a sparse vector per chunk.
- **Hybrid retrieval:** Qdrant fuses dense and sparse results server-side with Reciprocal Rank Fusion (`RetrievalMode.HYBRID`).
- **Reranker:** `BAAI/bge-reranker-base` cross-encoder, available but see the evaluation section below for why I would not use it on lookup queries.
- **LLM:** Llama 3.3 70B Versatile through Groq (free tier).
- **Conversational memory:** `create_history_aware_retriever` rewrites follow-ups into standalone queries before retrieval.
- **Tracing:** LangSmith.

### Why BM25 moved into Qdrant

The earlier version pickled a `BM25Retriever` to disk and combined it with the dense retriever using an `EnsembleRetriever`. That worked at 98 documents. It does not scale.

The pickle was 8.7 MB for 9,179 chunks, which is 948 bytes per chunk. At 54,488 chunks it would be roughly 50 MB, rewritten on every ingest, committed to git, and cloned by Streamlit Cloud on every boot.

Qdrant can store a sparse BM25 vector next to the dense one and fuse them itself. The retrieval idea is the same, but there is no pickle, no large file in git, and no BM25 index sitting in the app's memory. As a side effect `rag_chain.py` got smaller.

One thing worth knowing if you build this: the dense vector is named `""` in both DENSE and HYBRID mode, so reading a hybrid collection without passing `retrieval_mode=HYBRID` does **not** raise an error. It silently ignores the sparse vectors and returns dense-only results. The evaluation below shows exactly how much that costs.

## Project structure

```
portfolio-rag/
├── docs/                    # Corpus: Wikipedia articles + generated match docs
├── data/                    # Raw match CSVs + derived player list (gitignored)
├── fetch_match_data.py      # Download ATP match CSVs 1990-2026
├── build_player_list.py     # Derive the 485 top-50 players from match data
├── fetch_corpus.py          # Pull player biographies from Wikipedia
├── generate_match_docs.py   # Turn match data into results/career/h2h docs
├── ingest.py                # Chunk, embed (dense + sparse), push to Qdrant
├── rag_chain.py             # Retriever + conversational chain
├── build_eval_questions.py  # Derive 150 ground-truth questions from match data
├── evaluate_generated.py    # Ablation: dense vs hybrid vs reranked
├── evaluate_retrieval.py    # Retrieval-only eval (Hit/MRR/Recall@k)
├── evaluate.py              # End-to-end LLM-as-judge eval
├── test_questions.py        # 80 hand-written evaluation questions
├── streamlit_app.py         # Tennis-themed chat UI
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. **Clone and install**
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

3. **Build the corpus.** Each step is idempotent, so it is safe to re-run after a failure.
```bash
python fetch_match_data.py     # ~24 MB of match CSVs into data/matches/
python build_player_list.py    # derives the 485 top-50 players
python fetch_corpus.py         # Wikipedia biographies into docs/
python generate_match_docs.py  # results / career / h2h docs into docs/
```

4. **Ingest**
```bash
python ingest.py
```
> This takes around 75 minutes on CPU. Every chunk is embedded twice, dense and sparse, and uploaded in batches.

5. **Run it**
```bash
python rag_chain.py             # CLI
streamlit run streamlit_app.py  # UI
```

6. **Evaluate**
```bash
python build_eval_questions.py  # regenerate the ground-truth question set
python evaluate_generated.py    # dense vs hybrid vs reranked
python evaluate_retrieval.py    # original 80 questions
```

## Data sources and licensing

- **Wikipedia** text is under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **[TML-Database](https://github.com/Tennismylife/TML-Database)** by Tennismylife is under **CC BY-NC-SA 4.0**. Attribution is required and use must be non-commercial. This project is a non-commercial portfolio piece.

> The `JeffSackmann/tennis_atp` repository that this data ecosystem grew out of is no longer available on GitHub. TML-Database uses the same column format and is actively maintained.

## Evaluation

### Did scaling the corpus 105x hurt retrieval?

The expansion deliberately left all 98 original documents and all 80 original questions untouched, so re-running the same eval isolates one variable: what a much larger corpus does to retrieval on identical queries.

**Reranked retrieval, same 80 questions, 98 documents vs 10,260 documents:**

| k | Hit@k | MRR@k | Recall@k |
|---|---|---|---|
| 1 | 0.575 → 0.550 | 0.575 → 0.550 | 0.479 → 0.460 |
| 3 | 0.812 → 0.812 | 0.667 → 0.679 | 0.679 → 0.679 |
| 5 | 0.850 → 0.875 | 0.682 → **0.692** | 0.765 → 0.748 |
| 10 | 0.875 → **0.900** | 0.689 → **0.695** | 0.798 → **0.810** |

Essentially no cost. Every movement is within about 0.025 in either direction, which is noise on an 80-question set, and half of them are improvements. I expected a clear drop, so this surprised me. My reading is that the added documents are relevant rather than random noise, so they do not act as distractors the way I assumed they would.

### Ground-truth evaluation on the expanded corpus

The 80 hand-written questions only cover the original corpus, so they cannot test the 9,722 generated documents. Rather than hand-label another 150 questions or ask an LLM to invent them, `build_eval_questions.py` derives them straight from the match CSVs. That means the gold answers are exact ground truth and the expected document is known by construction.

The set is 150 questions, 30 each of: tournament winner, final scoreline, head-to-head, career titles, career best ranking.

| k | Dense only | Hybrid | Hybrid + reranker |
|---|---|---|---|
| Hit@1 | 0.780 | **0.853** | 0.687 |
| Hit@5 | 0.927 | 0.980 | **1.000** |
| Hit@10 | 0.933 | **1.000** | **1.000** |
| MRR@10 | 0.848 | **0.912** | 0.828 |
| Recall@10 | 0.933 | **1.000** | **1.000** |

**Finding 1: hybrid retrieval is worth 7.3 points of Hit@1 over dense-only, and the gain sits almost entirely in one category.**

| Category | Dense Hit@5 | Hybrid Hit@5 |
|---|---|---|
| `career_rank` | 0.667 | **0.933** |
| `career_titles` | 0.967 | 0.967 |
| `head_to_head` | 1.000 | 1.000 |
| `tournament_score` | 1.000 | 1.000 |
| `tournament_winner` | 1.000 | 1.000 |

`career_rank` is the category whose answers depend on exact tokens, a player name plus a specific rank number, which is what BM25 is good at and embeddings are not. The other categories are semantically distinctive enough that dense retrieval already handles them. This is also the concrete cost of the silent-failure mode mentioned earlier: forgetting `retrieval_mode=HYBRID` would not error, it would just quietly hand back the left column.

**Finding 2: the cross-encoder reranker hurts these queries.** It guarantees the right document lands in the top 5 (Hit@5 of 1.000), but it drops Hit@1 from 0.853 to 0.687 and MRR from 0.912 to 0.828. `career_titles` MRR@5 falls from 0.894 to 0.547.

This matches what I saw in the earlier version at smaller scale: the reranker helps paraphrased, semantic questions and hurts exact-match lookups. The generated corpus is overwhelmingly lookups, so on this traffic hybrid retrieval on its own beats hybrid plus reranker. The reranker still earns its place on paraphrased player questions, where `player_facts` MRR@5 reached 0.867.

### Data quality issues I found

Spot-checking generated documents against published records caught three problems. All three would have made the system state wrong facts confidently, and none of them were visible from retrieval metrics alone.

| Issue | Cause | Before | After |
|---|---|---|---|
| Head-to-heads inflated | Walkovers counted as wins. 542 in the dataset. | Federer vs Nadal 24-17 | **24-16**, matching the ATP |
| Title counts inflated | Team events (World Team Cup, Davis Cup) counted as singles titles | Sampras 65 titles | **64**, matching the ATP |
| Career stats misleading | Data starts in 1990, so players who peaked earlier looked worse | Wilander "best rank 10" | Window now stated explicitly |

Walkovers are still listed in the documents, but phrased as "advanced when X withdrew" rather than "beat", and excluded from win/loss records. The third fix is in the code and applies on the next ingest.

The lesson I took from this: when you generate a corpus from structured data, the generation code can be correct and the output still wrong, because the source encodes things differently than you assume. Checking a handful of outputs against known facts found all three in about ten minutes.

## Tech stack

Python, LangChain (langchain-classic 1.0), Qdrant Cloud with hybrid dense and sparse vectors, Groq (Llama 3.3 70B), BAAI/bge-small-en-v1.5 embeddings, BAAI/bge-reranker-base cross-encoder, FastEmbed BM25, Streamlit, LangSmith.
