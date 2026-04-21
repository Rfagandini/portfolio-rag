# Multi-Document RAG with Conversational Memory

A Retrieval-Augmented Generation (RAG) pipeline that answers questions across multiple PDF documents while maintaining conversational context.

**🚀 Live demo:** https://portfolio-rag-5rfvlpav9nrdzefhhxlqxe.streamlit.app/

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-rag-5rfvlpav9nrdzefhhxlqxe.streamlit.app/)

> Note: the free tier sleeps after inactivity — first visit may take 30s to wake up.

## Architecture

- **Document ingestion**: PDF loading + recursive chunking with LangChain
- **Embeddings**: Local sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store**: Qdrant Cloud for persistent storage and similarity search
- **LLM**: Llama 3.1 8B via Groq (free tier)
- **Retrieval**: Hybrid search combining semantic (dense embeddings) + keyword matching (BM25) via EnsembleRetriever
- **Conversational memory**: History-aware retriever that reformulates follow-up questions into standalone queries
- **Evaluation**: 100-question test suite with LLM-as-judge auto-grading
- **Tracing**: LangSmith integration for monitoring and debugging

## Project Structure

```
portfolio-rag/
├── docs/               # PDF documents to index
├── ingest.py           # Load, chunk, embed, and store documents in Qdrant
├── rag_chain.py        # Conversational RAG chain with session history
├── streamlit_app.py    # Chat UI for interactive querying
├── test_questions.py   # 100 evaluation questions with expected answers
├── evaluate.py         # Auto-grading script (LLM-as-judge)
├── eval_results.json   # Raw evaluation output
├── .env.example        # Required environment variables
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
# Fill in your API keys in .env
```

| Variable | Source |
|---|---|
| `GROQ_API_KEY` | [Groq Console](https://console.groq.com) (free) |
| `QDRANT_URL` | [Qdrant Cloud](https://cloud.qdrant.io) (free tier) |
| `QDRANT_API_KEY` | Qdrant Cloud dashboard |
| `LANGCHAIN_API_KEY` | [LangSmith](https://smith.langchain.com) (optional, for tracing) |

3. **Add PDFs** to the `docs/` folder

4. **Run ingestion** (only needed once, or when adding new documents)
```bash
python ingest.py
```

5. **Query the RAG**

   Option A — CLI:
   ```bash
   python rag_chain.py
   ```

   Option B — Streamlit chat UI:
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Run the evaluation suite** (optional)
   ```bash
   python evaluate.py
   ```
   Results are printed to console and saved to `eval_results.json`.

## Evaluation Results

Evaluated on 100 questions (factual, conceptual, and follow-up) with LLM-as-judge auto-grading:

| Document | Score | Breakdown |
|---|---|---|
| AlexNet paper | 98.0% | 24 correct, 1 partial |
| Attention Is All You Need | 93.3% | 26 correct, 4 partial |
| IPCC Climate Report | 90.0% | 20 correct, 5 partial |
| NASA Artemis Plan | 97.5% | 19 correct, 1 partial |
| **Overall** | **94.5%** | **89 correct, 11 partial, 0 incorrect** |

Follow-up accuracy (conversational memory): 4/5

## Sample Documents

- AlexNet paper (2012)
- Attention Is All You Need (2017)
- IPCC Climate Report
- NASA Artemis Plan

## Tech Stack

Python | LangChain | Qdrant Cloud | Groq | Sentence-Transformers | BM25 | Streamlit | LangSmith
