from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


import pickle
import os

from dotenv import load_dotenv
load_dotenv()

#SESSION HISTORY STORE
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_vector_store() -> QdrantVectorStore:

    embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
    client = QdrantClient(url = os.getenv("QDRANT_URL"),
                          api_key = os.getenv("QDRANT_API_KEY"))
    
    return QdrantVectorStore(client = client,
                             embedding = embeddings,
                             collection_name = "portfolio-rag")

def build_reranking_retriever(fetch_k = 10, top_n = 5):
    
    #RETRIEVERS
    retriever = get_vector_store().as_retriever(search_kwargs = {"k": fetch_k})
    
    with open("bm25_index.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)
    
    hybrid_retriever = EnsembleRetriever(
    retrievers=[retriever, bm25_retriever],
    weights=[0.5, 0.5]
    )
    
    #RERANKER
    
    crossEncoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    
    reranker = CrossEncoderReranker(model = crossEncoder, top_n = top_n)
    
    compressionRetriever = ContextualCompressionRetriever(base_retriever = hybrid_retriever, base_compressor = reranker)
    
    return compressionRetriever
    


def build_chain():
    
    #RETRIEVER
    hybrid_retriever = build_reranking_retriever()
    
    #LLM
    llm = ChatOpenAI(
    base_url='https://api.groq.com/openai/v1',
    openai_api_key=os.environ.get('GROQ_API_KEY'),
    model='llama-3.3-70b-versatile',
    temperature=0.2,
    )
    
    #PROMPTS
    answer_prompt = ChatPromptTemplate.from_messages([
        ('system', 'Answer the question only making use of the available context. Obtain the required data from the specific document if the user has indicated so.'
         'If the provided context is not enough information, just say so. '),
        ("placeholder",'{chat_history}'),
        ('human', 'Context:\n{context}\n\nQuestion: {input}')
    ])
    
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ('system', 'Given chat history, reformulate the question to be standalone. Do not and never by any means, try to answer it. That is not your purpose.'),
        ("placeholder", '{chat_history}'),
        ('human', '{input}')
    ])
    
    #HISTORY AWARE RETRIEVER
    history_aware_retriever = create_history_aware_retriever(llm, hybrid_retriever, contextualize_prompt)
    
    #CHAIN
    
    query_answer_chain = create_stuff_documents_chain(
    llm, answer_prompt
    )
    
    return create_retrieval_chain(history_aware_retriever, query_answer_chain)
    

if __name__ == "__main__":
    rag_chain = build_chain()
    
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    config = {"configurable": {"session_id": "test-session"}}
    
    print("RAG ready! Type 'quit' to exit.\n")
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        response = conversational_chain.invoke({"input": question}, config=config)
        print(f"\nAssistant: {response['answer']}\n")

