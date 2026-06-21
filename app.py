"""
app.py - Main Streamlit Web UI for the Persona-Adaptive Customer Support Agent.

Run with:
    streamlit run app.py
"""

import json
import time
import random
import streamlit as st

from src.config import APP_TITLE, APP_ICON, DATA_DIR, CHROMA_DB_DIR
from src.classifier import classify_customer_persona
from src.rag_pipeline import LocalRAGPipeline
from src.generator import generate_adaptive_response

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background - clean white */
    .stApp { background-color: #f8fafc; color: #1a202c; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 2px solid #334155;
    }
    section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] .stTextInput input {
        background: #0f172a;
        color: #f1f5f9 !important;
        border: 1px solid #475569;
    }

    /* Chat input */
    .stChatInput input {
        background: #ffffff;
        color: #1a202c;
        border: 2px solid #cbd5e0;
        border-radius: 10px;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
        color: #1a202c;
    }

    /* All text in main area */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    p, li, span, div { color: #1a202c; }

    /* Persona badges */
    .persona-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    .badge-technical { background: #dbeafe; color: #1d4ed8; border: 2px solid #3b82f6; }
    .badge-frustrated { background: #fee2e2; color: #b91c1c; border: 2px solid #ef4444; }
    .badge-executive  { background: #dcfce7; color: #15803d; border: 2px solid #22c55e; }
    .badge-escalated  { background: #fef3c7; color: #b45309; border: 2px solid #f59e0b; }

    /* Confidence bar */
    .conf-bar-wrap {
        background: #e2e8f0;
        border-radius: 4px;
        height: 8px;
        margin: 4px 0 12px 0;
        width: 100%;
    }
    .conf-bar-fill {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #3b82f6, #22c55e);
    }
    .conf-label { color: #64748b !important; font-size: 12px; font-weight: 600; }

    /* Handoff block */
    .handoff-block {
        background: #fffbeb;
        border: 1px solid #f59e0b;
        border-left: 4px solid #d97706;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
    }

    /* Metric cards */
    .metric-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-label { color: #94a3b8 !important; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; }
    .metric-value { color: #f1f5f9 !important; font-size: 24px; font-weight: 700; margin-top: 2px; }

    /* Status dot */
    .status-dot {
        display: inline-block;
        width: 9px; height: 9px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-green { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
    .dot-red   { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

    /* Buttons */
    .stButton button {
        background: #1e293b;
        color: #f1f5f9 !important;
        border: 1px solid #475569;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton button:hover {
        background: #334155;
        border-color: #64748b;
    }
    /* API Key input placeholder and eye icon */
    .stTextInput input::placeholder { 
        color: #94a3b8 !important; 
        opacity: 1; 
    }
    .stTextInput input {
        background: #0f172a !important;
        color: #f1f5f9 !important;
        border: 1px solid #475569 !important;
        border-radius: 8px;
    }
    /* Eye icon / input suffix button */
    .stTextInput button {
        background: transparent !important;
        border: none !important;
        color: #94a3b8 !important;
    }
    .stTextInput button svg {
        fill: #94a3b8 !important;
        stroke: #94a3b8 !important;
    }
    .stTextInput button:hover svg {
        fill: #f1f5f9 !important;
        stroke: #f1f5f9 !important;
    }

    /* Info/success boxes */
    .stAlert { border-radius: 10px; color: #1a202c; }

    /* Code blocks */
    code { background: #f1f5f9; color: #0f172a; border-radius: 4px; padding: 2px 6px; }
    pre  { background: #0f172a; color: #e2e8f0; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────

def init_session():
    defaults = {
        "messages": [],            # Chat history: [{role, content, meta}]
        "rag_ready": False,        # Whether knowledge base is indexed
        "persona_counts": {"Technical Expert": 0, "Frustrated User": 0, "Business Executive": 0},
        "escalation_count": 0,
        "turn_count": 0,
        "frustration_streak": 0,   # Consecutive frustrated-persona turns
        "rag_pipeline": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ── RAG Pipeline Initialization ───────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    pipeline = LocalRAGPipeline(db_dir=CHROMA_DB_DIR)
    return pipeline


def ensure_indexed(pipeline: LocalRAGPipeline):
    """Index documents if not already done."""
    if not pipeline.is_indexed():
        with st.spinner("📚 Indexing knowledge base... (one-time setup)"):
            summary = pipeline.ingest_all_documents(data_dir=DATA_DIR)
        st.success(f"✅ Knowledge base ready: {sum(summary.values())} chunks across {len(summary)} documents.")
    return True


# ── LLM Call with Exponential Backoff ─────────────────────────────────────────

def call_with_backoff(func, *args, max_retries=4, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.markdown("---")

    # API Key Input
    api_key_input = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here...",
        help="Your key is stored only in this session."
    )
    if api_key_input:
        import os
        os.environ["GEMINI_API_KEY"] = api_key_input
        from src.config import GEMINI_API_KEY
        import src.config as cfg
        cfg.GEMINI_API_KEY = api_key_input

    st.markdown("---")

    # Index Knowledge Base Button
    if st.button("📂 Load & Index Knowledge Base", use_container_width=True):
        pipeline = get_rag_pipeline()
        ensure_indexed(pipeline)
        st.session_state["rag_ready"] = True
        st.session_state["rag_pipeline"] = pipeline

    # Status indicator
    status_html = (
        '<span class="status-dot dot-green"></span> <b>Knowledge Base Ready</b>'
        if st.session_state["rag_ready"]
        else '<span class="status-dot dot-red"></span> Knowledge Base Not Loaded'
    )
    st.markdown(status_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Session Analytics")

    # Metrics
    counts = st.session_state["persona_counts"]
    for persona, emoji in [("Technical Expert", "🔧"), ("Frustrated User", "😤"), ("Business Executive", "💼")]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{emoji} {persona}</div>
            <div class="metric-value">{counts[persona]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🚨 Escalations</div>
        <div class="metric-value">{st.session_state["escalation_count"]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Quick Test Prompts")
    test_prompts = {
        "🔧 Tech": "What are the header parameter requirements for bearer token auth? Include the exact format.",
        "😤 Frustrated": "I've been waiting over an hour and your interface is STILL not loading! This is unacceptable!",
        "💼 Executive": "Our operational uptime is decreasing. What is the resolution timeline for billing disputes?",
        "🚨 Escalate": "I have duplicate charges on my statement and I demand an immediate refund or I'll dispute with my bank!",
    }
    for label, prompt in test_prompts.items():
        if st.button(label, use_container_width=True, key=f"test_{label}"):
            st.session_state["_inject_prompt"] = prompt

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["turn_count"] = 0
        st.session_state["frustration_streak"] = 0
        st.rerun()


# ── Main Chat Area ────────────────────────────────────────────────────────────

st.markdown(f"## {APP_ICON} {APP_TITLE}")
st.markdown("*An intelligent support agent that adapts its response style to your communication persona.*")
st.markdown("---")

# Welcome message
if not st.session_state["messages"]:
    st.info(
        "👋 Welcome! To get started:\n"
        "1. Enter your **Gemini API Key** in the sidebar.\n"
        "2. Click **Load & Index Knowledge Base**.\n"
        "3. Then type your support question below.\n\n"
        "The agent automatically detects whether you're a **Technical Expert**, **Frustrated User**, or **Business Executive** — and adapts its response accordingly!"
    )

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            persona = meta.get("persona", "")

            # Persona badge
            badge_class = {
                "Technical Expert": "badge-technical",
                "Frustrated User": "badge-frustrated",
                "Business Executive": "badge-executive",
            }.get(persona, "badge-escalated")

            if meta.get("escalated"):
                st.markdown('<span class="persona-badge badge-escalated">🚨 ESCALATED TO HUMAN</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="persona-badge {badge_class}">Persona: {persona}</span>', unsafe_allow_html=True)

            # Confidence bar
            score = meta.get("confidence", 0.0)
            bar_width = int(score * 100)
            st.markdown(f"""
            <div style="font-size:11px; color:#475569;">Retrieval Confidence: {score:.0%}</div>
            <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{bar_width}%"></div></div>
            """, unsafe_allow_html=True)

        st.markdown(msg["content"])

        # Show handoff JSON if escalated
        if msg["role"] == "assistant" and msg.get("meta", {}).get("handoff_summary"):
            with st.expander("📋 View Human Agent Handoff Report", expanded=False):
                st.markdown('<div class="handoff-block">', unsafe_allow_html=True)
                st.code(msg["meta"]["handoff_summary"], language="json")
                st.markdown('</div>', unsafe_allow_html=True)

    # Render user message
    if msg["role"] == "user":
        pass  # Already rendered above


# ── Handle Injected Prompts from Sidebar Buttons ─────────────────────────────

injected = st.session_state.pop("_inject_prompt", None)

# ── Chat Input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Type your support question here...") or injected

if user_input:
    # Guard: ensure setup is complete
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("⚠️ Please enter your Gemini API Key in the sidebar first.")
        st.stop()

    if not st.session_state["rag_ready"]:
        st.error("⚠️ Please click **Load & Index Knowledge Base** in the sidebar first.")
        st.stop()

    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["turn_count"] += 1

    with st.chat_message("user"):
        st.markdown(user_input)

    # Process message
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your message..."):
            try:
                pipeline = st.session_state["rag_pipeline"] or get_rag_pipeline()

                # Step 1: Classify Persona
                classification = call_with_backoff(classify_customer_persona, user_input)
                persona = classification["persona"]
                cls_confidence = classification["confidence"]

                # Track frustration streak
                if persona == "Frustrated User":
                    st.session_state["frustration_streak"] += 1
                else:
                    st.session_state["frustration_streak"] = 0

                # Step 2: Retrieve Context
                context_chunks = pipeline.retrieve_context(user_input, top_k=3)

                # Step 3: Generate Response
                result = call_with_backoff(
                    generate_adaptive_response,
                    user_query=user_input,
                    persona=persona,
                    context_chunks=context_chunks,
                    frustration_turns=st.session_state["frustration_streak"],
                )

                # Update analytics
                if persona in st.session_state["persona_counts"]:
                    st.session_state["persona_counts"][persona] += 1
                if result["escalated"]:
                    st.session_state["escalation_count"] += 1

                # Build response metadata
                meta = {
                    "persona": persona,
                    "confidence": result["confidence"],
                    "escalated": result["escalated"],
                    "handoff_summary": result.get("handoff_summary"),
                }

                # Render response
                badge_class = {
                    "Technical Expert": "badge-technical",
                    "Frustrated User": "badge-frustrated",
                    "Business Executive": "badge-executive",
                }.get(persona, "badge-escalated")

                if result["escalated"]:
                    st.markdown('<span class="persona-badge badge-escalated">🚨 ESCALATED TO HUMAN</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="persona-badge {badge_class}">Persona: {persona}</span>', unsafe_allow_html=True)

                score = result["confidence"]
                bar_width = int(score * 100)
                st.markdown(f"""
                <div style="font-size:11px; color:#475569;">Retrieval Confidence: {score:.0%}</div>
                <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{bar_width}%"></div></div>
                """, unsafe_allow_html=True)

                st.markdown(result["response"])

                if result.get("handoff_summary"):
                    with st.expander("📋 View Human Agent Handoff Report", expanded=True):
                        st.markdown('<div class="handoff-block">', unsafe_allow_html=True)
                        st.code(result["handoff_summary"], language="json")
                        st.markdown('</div>', unsafe_allow_html=True)

                # Store in session history
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": result["response"],
                    "meta": meta,
                })

            except Exception as e:
                error_msg = f"⚠️ An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_msg,
                })

    st.rerun()
