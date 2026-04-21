import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from rag_chain import build_chain, get_session_history

# --- PAGE CONFIG ---
# Sets the browser tab title and the layout.
st.set_page_config(page_title="Multi-Document RAG", page_icon="📄")
st.title("Multi-Document RAG")
st.caption("Ask questions about: AlexNet, Attention Is All You Need, IPCC Climate Report, NASA Artemis Plan")


# --- CHAIN SETUP (cached so it only loads once) ---
# @st.cache_resource tells Streamlit: "run this function once, cache the result,
# and reuse it on every rerun." Without this, the chain would rebuild every time
# the user sends a message (slow — loads embeddings + connects to Qdrant each time).
@st.cache_resource
def get_chain():
    rag_chain = build_chain()
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

chain = get_chain()


# --- SESSION STATE ---
# st.session_state is a dict that persists between reruns.
# We use it to store the chat messages so they stay visible on screen.
# "messages" is a list of dicts: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- DISPLAY CHAT HISTORY ---
# On every rerun, Streamlit re-renders the page from scratch.
# This loop re-draws all previous messages so the conversation stays visible.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- HANDLE USER INPUT ---
# st.chat_input() renders the text box at the bottom of the page.
# It returns the user's message when they press Enter, or None if they haven't typed anything.
if prompt := st.chat_input("Ask a question about the documents..."):

    # 1. Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": st.session_state.session_id}}
            response = chain.invoke({"input": prompt}, config=config)
            answer = response["answer"]

        st.markdown(answer)

    # 3. Save assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
