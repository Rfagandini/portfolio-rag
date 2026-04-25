from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
import os
import pickle


load_dotenv()




# DOCUMENTS LOADING
def load_docs(docs_dir: str = "docs") -> list:
    
    path = Path(f"{docs_dir}")
    loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    return loader.load()


# CHUNKS CREATION
def get_chunks(docs: list) -> list:

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                              chunk_overlap= 100,
                                              length_function = len)
    return splitter.split_documents(docs)


def embed_and_store(chunks: list) -> QdrantVectorStore:

    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    vectorstore = QdrantVectorStore.from_documents(documents=chunks,
                                                   embedding=embeddings,
                                                   url = os.getenv("QDRANT_URL"),
                                                   api_key = os.getenv("QDRANT_API_KEY"),
                                                   collection_name = "portfolio-rag")
    
    return vectorstore
    
    
if __name__ == "__main__":
    
    #DELETE THE ALREADY EXISTING VECTORS
    try:
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        client.delete_collection("portfolio-rag")
    except Exception as e:
        print("Error during the first run, collection prob. does not exist")
    
    docs = load_docs()
    print(f"Loaded {len(docs)} pages")
    
    chunks = get_chunks(docs)
    print(f"Got {len(chunks)} chunks")
    
    bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    print("Dumping the bm25 retriever")
    
    print("Embedding and uploading to Qdrant...")
    vector_store = embed_and_store(chunks)
    print(f"Done! Collection 'portfolio-rag' created in Qdrant Cloud.")
    
    # Quick test: similarity search
    results = vector_store.similarity_search("What is attention in transformers?", k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ({doc.metadata['source']}) ---")
        print(doc.page_content[:150])