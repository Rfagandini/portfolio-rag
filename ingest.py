"""
Build the Qdrant index from docs/.

WHAT CHANGED AND WHY
--------------------
Previously this script pickled a BM25Retriever to bm25_index.pkl and
rag_chain.py combined it with the dense retriever using an EnsembleRetriever.
That worked at 98 documents. It does not work at ~6,500.

Measured: the pickle is 8.7 MB for 9,179 chunks -- 948 bytes per chunk. The
expanded corpus is ~80,000 chunks, which makes it ~75 MB, rewritten in full
on every ingest, committed to git, and cloned by Streamlit Cloud on every
boot. That is a dead end.

Qdrant can do BM25 itself. Storing a SPARSE vector next to each dense vector
lets the server do hybrid search natively:

    dense vector   -> semantic similarity  ("who dominated on grass")
    sparse vector  -> exact term matching  ("Kafelnikov", "6-7(5)")

Qdrant fuses the two rankings server-side with Reciprocal Rank Fusion. The
retrieval quality is the same idea as the old EnsembleRetriever, but there is
no pickle, no 75 MB file, no BM25 index held in Streamlit's memory, and no
need to ship docs/ to production at all.

Usage:
    pip install fastembed
    python ingest.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --- CONFIG ---

COLLECTION_NAME = "portfolio-rag"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"  # a real BM25, computed by fastembed

# Wikipedia prose keeps the original settings -- they were tuned in Phase 4.
PROSE_CHUNK_SIZE, PROSE_OVERLAP = 500, 100

# Generated docs are line-oriented: one self-contained sentence per match.
# Bigger chunks with newline separators keep whole match lines intact instead
# of guillotining them mid-score.
GENERATED_CHUNK_SIZE, GENERATED_OVERLAP = 800, 100

# Filename prefixes written by generate_match_docs.py.
GENERATED_PREFIXES = ("results__", "career__", "h2h__")

UPLOAD_BATCH_SIZE = 64


# --- DOCUMENTS LOADING ---

def load_docs(docs_dir: str = "docs") -> list:
    path = Path(f"{docs_dir}")
    loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    return loader.load()


def metadata_from_filename(source: str) -> dict:
    """
    Derive filterable metadata from a document's filename.

    Every file is named "{category}__{key}.txt", so the category prefix is
    free structure we would otherwise throw away. Putting it in the payload
    lets Qdrant filter before searching -- "only 1995", "only career docs" --
    which is a retrieval-quality lever we get for nothing.

    Example:
        "docs/results__1995_Wimbledon.txt"
            -> {"category": "results", "doc_type": "generated",
                "year": 1995, "tournament": "Wimbledon"}
    """
    stem = Path(source.replace("\\", "/")).stem
    category, _, key = stem.partition("__")
    if not key:  # no "__" in the name; treat the whole stem as the key
        category, key = "unknown", stem

    meta: dict = {
        "category": category,
        "doc_type": "generated"
            if f"{category}__" in GENERATED_PREFIXES else "wikipedia",
    }

    if category == "results":
        year, _, tournament = key.partition("_")
        if year.isdigit():
            meta["year"] = int(year)
            meta["tournament"] = tournament.replace("_", " ")
        else:
            meta["tournament"] = key.replace("_", " ")
    elif category == "career" or category.startswith("players"):
        meta["player"] = key.replace("_", " ")

    return meta


# --- CHUNKS CREATION ---

def is_generated(doc) -> bool:
    """True if the document was written by generate_match_docs.py."""
    name = Path(doc.metadata.get("source", "").replace("\\", "/")).name
    return name.startswith(GENERATED_PREFIXES)


def get_chunks(docs: list) -> list:
    """
    Split documents, using a different strategy per document type.

    Why two splitters: a 500-char window across a results doc cuts match lines
    in half, so a chunk can end "...Pete Sampras beat Boris Beck". The default
    separator list also splits on " ", which makes that worse. Generated docs
    get newline-first separators and a bigger window so whole lines survive.
    """
    prose_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PROSE_CHUNK_SIZE,
        chunk_overlap=PROSE_OVERLAP,
        length_function=len,
    )
    generated_splitter = RecursiveCharacterTextSplitter(
        chunk_size=GENERATED_CHUNK_SIZE,
        chunk_overlap=GENERATED_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n"],
    )

    prose_docs = [d for d in docs if not is_generated(d)]
    generated_docs = [d for d in docs if is_generated(d)]

    chunks = prose_splitter.split_documents(prose_docs)
    chunks += generated_splitter.split_documents(generated_docs)

    # Chunks inherit the parent document's metadata, so enrich once, here.
    for chunk in chunks:
        chunk.metadata.update(metadata_from_filename(chunk.metadata.get("source", "")))

    return chunks


# --- EMBEDDING AND STORAGE ---

def embed_and_store(chunks: list) -> QdrantVectorStore:
    """
    Embed chunks densely AND sparsely, then upload to Qdrant.

    The API shape is not guessable, so here is the call you need:

        dense = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
        sparse = FastEmbedSparse(model_name=SPARSE_MODEL)

        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=COLLECTION_NAME,
            batch_size=UPLOAD_BATCH_SIZE,
        )

    RetrievalMode.HYBRID is what makes Qdrant create BOTH a dense and a sparse
    vector config on the collection. Get this wrong on the write side and the
    read side in rag_chain.py will fail with a missing-sparse-vector error.

    RetrievalMode.HYBRID is what makes Qdrant create BOTH a dense and a sparse
    vector config on the collection. Note that the dense vector name is "" in
    both DENSE and HYBRID mode, so a DENSE-mode reader will NOT error against a
    hybrid collection -- it will silently ignore the sparse vectors and return
    dense-only results. rag_chain.py must therefore ask for HYBRID explicitly.
    """
    dense = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
    sparse = FastEmbedSparse(model_name=SPARSE_MODEL)

    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME,
        batch_size=UPLOAD_BATCH_SIZE,
    )


if __name__ == "__main__":

    # DELETE THE ALREADY EXISTING VECTORS
    # Still required: switching to hybrid changes the collection schema, so an
    # old dense-only collection cannot be reused and must be dropped first.
    try:
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'")
    except Exception:
        print("No existing collection to drop (fine on a first run)")

    docs = load_docs()
    print(f"Loaded {len(docs)} documents")

    chunks = get_chunks(docs)
    print(f"Got {len(chunks)} chunks")

    print("Embedding (dense + sparse) and uploading to Qdrant...")
    vector_store = embed_and_store(chunks)
    print(f"Done! Collection '{COLLECTION_NAME}' created in Qdrant Cloud.")

    # Quick test: a query that only the expanded corpus can answer.
    results = vector_store.similarity_search("Who won Wimbledon in 1995?", k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ({doc.metadata.get('source')}) ---")
        print(doc.page_content[:150])
