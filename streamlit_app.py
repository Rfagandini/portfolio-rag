import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from rag_chain import build_chain, get_session_history

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tennis RAG", page_icon="🎾", layout="centered")


# --- TENNIS COURT THEME (custom CSS + animated SVG header) ---
# Grass-court inspired: alternating green stripes for the page background,
# a Wimbledon-flavoured banner, and a translucent chat surface so the
# RAG conversation remains the focal point.

st.markdown("""
<style>
    .stApp {
        background:
            repeating-linear-gradient(
                0deg,
                #3a8a3a 0px, #3a8a3a 28px,
                #4ea14e 28px, #4ea14e 56px
            );
    }

    [data-testid="stHeader"] { background: transparent; }

    .block-container {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 14px;
        padding: 1.5rem 2rem 2rem 2rem;
        margin-top: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 60, 0, 0.25);
    }

    .court-banner {
        background: linear-gradient(180deg, #2e6b2e 0%, #1f4d1f 100%);
        border: 3px solid #f5d04a;
        border-radius: 10px;
        padding: 6px;
        margin-bottom: 1rem;
    }

    h1 { color: #1f4d1f !important; }

    /* Chat bubbles — slight tint */
    [data-testid="stChatMessage"] {
        background: rgba(245, 250, 245, 0.95);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Pixel-style match scene: side view of grass court, two players
# (Nadal in red on the left, Djokovic in white/blue on the right), animated ball.
# Built from rectangles with crispEdges so it reads as pixel art.
COURT_SVG = """
<div class="court-banner">
<svg viewBox="0 0 600 180" style="width:100%; display:block;" shape-rendering="crispEdges" xmlns="http://www.w3.org/2000/svg">
    <!-- Grass with mowed stripes -->
    <rect x="0" y="0" width="600" height="180" fill="#3a8a3a"/>
    <rect x="0" y="0"   width="600" height="22" fill="#4ea14e"/>
    <rect x="0" y="44"  width="600" height="22" fill="#4ea14e"/>
    <rect x="0" y="88"  width="600" height="22" fill="#4ea14e"/>
    <rect x="0" y="132" width="600" height="22" fill="#4ea14e"/>

    <!-- Court rectangle + center line -->
    <rect x="40" y="30" width="520" height="120" fill="none" stroke="#ffffff" stroke-width="3"/>
    <line x1="300" y1="30" x2="300" y2="150" stroke="#ffffff" stroke-width="2"/>
    <!-- Service boxes -->
    <line x1="140" y1="60" x2="460" y2="60" stroke="#ffffff" stroke-width="2"/>
    <line x1="140" y1="120" x2="460" y2="120" stroke="#ffffff" stroke-width="2"/>

    <!-- Net (post + mesh) -->
    <line x1="300" y1="20" x2="300" y2="160" stroke="#1c1c1c" stroke-width="3"/>
    <line x1="280" y1="60" x2="320" y2="60" stroke="#1c1c1c" stroke-width="2"/>

    <!-- LEFT PLAYER: Nadal (red shirt, sleeveless, headband) -->
    <g>
        <rect x="118" y="56" width="16" height="14" fill="#f4c896"/>          <!-- head -->
        <rect x="116" y="52" width="20" height="6"  fill="#3a2210"/>          <!-- hair -->
        <rect x="116" y="56" width="20" height="3"  fill="#cc2222"/>          <!-- headband -->
        <rect x="112" y="70" width="28" height="32" fill="#cc2222"/>          <!-- shirt -->
        <rect x="104" y="74" width="10" height="6"  fill="#f4c896"/>          <!-- left arm -->
        <rect x="138" y="74" width="22" height="6"  fill="#f4c896"/>          <!-- right arm holding racket -->
        <rect x="158" y="64" width="3"  height="22" fill="#444"/>             <!-- racket handle -->
        <ellipse cx="159" cy="60" rx="7" ry="9" fill="none" stroke="#444" stroke-width="2"/>
        <rect x="112" y="102" width="12" height="22" fill="#ffffff"/>          <!-- left leg shorts -->
        <rect x="128" y="102" width="12" height="22" fill="#ffffff"/>          <!-- right leg shorts -->
        <rect x="112" y="124" width="12" height="6"  fill="#f4c896"/>          <!-- left calf -->
        <rect x="128" y="124" width="12" height="6"  fill="#f4c896"/>          <!-- right calf -->
        <rect x="110" y="130" width="16" height="4"  fill="#ffffff"/>          <!-- left shoe -->
        <rect x="126" y="130" width="16" height="4"  fill="#ffffff"/>          <!-- right shoe -->
    </g>

    <!-- RIGHT PLAYER: Djokovic (white shirt, dark shorts) -->
    <g>
        <rect x="466" y="56" width="16" height="14" fill="#f4c896"/>           <!-- head -->
        <rect x="464" y="52" width="20" height="6"  fill="#5a3220"/>           <!-- hair -->
        <rect x="460" y="70" width="28" height="32" fill="#ffffff"/>           <!-- white shirt -->
        <rect x="460" y="70" width="28" height="4"  fill="#1f4d8c"/>           <!-- shirt collar accent -->
        <rect x="440" y="74" width="20" height="6"  fill="#f4c896"/>           <!-- left arm holding racket -->
        <rect x="486" y="74" width="10" height="6"  fill="#f4c896"/>           <!-- right arm -->
        <rect x="439" y="64" width="3"  height="22" fill="#444"/>              <!-- racket handle -->
        <ellipse cx="440" cy="60" rx="7" ry="9" fill="none" stroke="#444" stroke-width="2"/>
        <rect x="460" y="102" width="12" height="22" fill="#1c1c1c"/>          <!-- left shorts -->
        <rect x="476" y="102" width="12" height="22" fill="#1c1c1c"/>          <!-- right shorts -->
        <rect x="460" y="124" width="12" height="6"  fill="#f4c896"/>
        <rect x="476" y="124" width="12" height="6"  fill="#f4c896"/>
        <rect x="458" y="130" width="16" height="4"  fill="#ffffff"/>
        <rect x="474" y="130" width="16" height="4"  fill="#ffffff"/>
    </g>

    <!-- Tennis ball, bouncing across the net -->
    <circle cx="180" cy="80" r="6" fill="#dfff3a" stroke="#222" stroke-width="1">
        <animate attributeName="cx"
                 values="170;300;430;300;170"
                 keyTimes="0;0.25;0.5;0.75;1"
                 dur="2.6s" repeatCount="indefinite"/>
        <animate attributeName="cy"
                 values="78;42;78;42;78"
                 keyTimes="0;0.25;0.5;0.75;1"
                 dur="2.6s" repeatCount="indefinite"/>
    </circle>
</svg>
</div>
"""


# --- HEADER ---
# st.markdown strips <svg> tags even with unsafe_allow_html=True. Use the
# components iframe renderer (stable across all Streamlit versions) so the
# inline SVG actually paints.
import streamlit.components.v1 as components
components.html(COURT_SVG, height=210)
st.title("🎾 Tennis RAG")
st.caption(
    "Ask anything about ATP tennis — Big 3, Grand Slams 2020–2024, current top players, history. "
    "Knowledge base: 99 Wikipedia articles, hybrid retrieval + cross-encoder reranker."
)


# --- CHAIN SETUP (cached so it only loads once) ---
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
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- HANDLE USER INPUT ---
if prompt := st.chat_input("Ask about a player, tournament, or rule..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Rallying..."):
            config = {"configurable": {"session_id": st.session_state.session_id}}
            response = chain.invoke({"input": prompt}, config=config)
            answer = response["answer"]

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
