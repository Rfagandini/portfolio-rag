from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()




# DOCUMENTS LOADING
def load_pdfs(docs_dir: str = "docs") -> list:
    
    path = Path(f"{docs_dir}")
    loader = PyPDFDirectoryLoader(path, glob="*.pdf", mode="page")
    return loader.load()


# CHUNKS CREATION
def get_chunks(docs: list) -> list:

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                              chunk_overlap= 100,
                                              length_function = len)
    return splitter.split_documents(docs)


def embed_and_store(chunks: list) -> QdrantVectorStore:

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = QdrantVectorStore.from_documents(documents=chunks,
                                                   embedding=embeddings,
                                                   url = os.getenv("QDRANT_URL"),
                                                   api_key = os.getenv("QDRANT_API_KEY"),
                                                   collection_name = "portfolio-rag")
    
    return vectorstore
    
    
if __name__ == "__main__":
    
    #DELETE THE ALREADY EXISTING VECTORS
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    client.delete_collection("portfolio-rag")
    
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages")
    
    chunks = get_chunks(docs)
    print(f"Got {len(chunks)} chunks")
    
    print("Embedding and uploading to Qdrant...")
    vector_store = embed_and_store(chunks)
    print(f"Done! Collection 'portfolio-rag' created in Qdrant Cloud.")
    
    # Quick test: similarity search
    results = vector_store.similarity_search("What is attention in transformers?", k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ({doc.metadata['source']}, p.{doc.metadata['page']}) ---")
        print(doc.page_content[:150])