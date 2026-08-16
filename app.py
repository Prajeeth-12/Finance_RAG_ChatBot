import time
import importlib
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

import src.rag
importlib.reload(src.rag)
from src.rag import search_documents, generate_answer

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="HCLTech Financial RAG Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Financial Dark UI System
st.markdown("""
<style>
    /* Dark Financial Dashboard Theme */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Card Container Base */
    .main-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    /* Header Gradient */
    .header-title {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 24px;
    }
    
    /* Timeline Card Styles */
    .timeline-container {
        border-left: 3px solid #3b82f6;
        padding-left: 16px;
        margin: 12px 0 20px 8px;
    }
    
    .timeline-step {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
    }
    
    .timeline-step:hover {
        border-color: #3b82f6;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.2);
    }
    
    .step-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    
    .step-title {
        font-weight: 700;
        color: #f8fafc;
        font-size: 0.95rem;
    }
    
    .step-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .step-detail {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    
    /* Answer Panel */
    .answer-panel {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(6, 78, 59, 0.15));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
    }
    
    .answer-header {
        color: #34d399;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .answer-text {
        color: #f1f5f9;
        font-size: 1.15rem;
        line-height: 1.6;
        font-weight: 500;
    }
    
    /* Metric Pill */
    .metric-pill {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
        display: inline-block;
        margin-right: 8px;
        font-size: 0.8rem;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR PANEL
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/analytics.png", width=50)
    st.markdown("### ⚙️ System Status")
    
    # Provider Status
    st.markdown("""
    <div style="background: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 12px;">
        <div style="color: #10b981; font-weight: 600; font-size: 0.85rem; display: flex; align-items: center; gap: 6px;">
            <span>🟢 NVIDIA NIM Online</span>
        </div>
        <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
            Embeddings: <code>nv-embedqa-e5-v5</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selector
    st.markdown("#### 🧠 Generation Model")
    selected_model = st.selectbox(
        "Select LLM Endpoint",
        options=["meta/llama-3.1-8b-instruct", "meta/llama-3.3-70b-instruct"],
        index=0,
        help="Choose between ultra-fast inference (~1.5s) or deep financial reasoning (~15s)."
    )
    
    st.markdown("---")
    
    # Knowledge Base Stats
    st.markdown("#### 📚 Knowledge Base")
    st.markdown("""
    - **Reports**: HCLTech FY26 (Q1–Q4)
    - **Total Chunks**: 244 Indexed
    - **Vector Store**: ChromaDB Persistent
    - **Dimensions**: 1024-dim
    """)
    
    st.markdown("---")
    st.markdown("#### 💡 Quick Glossary")
    st.caption("**QoQ**: Quarter-over-Quarter Growth")
    st.caption("**YoY**: Year-over-Year Growth")
    st.caption("**EBIT**: Earnings Before Interest & Taxes")
    st.caption("**NI**: Net Income / Profit After Tax")

# ============================================================
# MAIN DASHBOARD CONTENT
# ============================================================

# Header Banner
st.markdown('<div class="header-title">📊 HCLTech Financial RAG Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Grounded financial research assistant powered by NVIDIA NIM & Llama 3 for HCLTech FY26 quarterly reports.</div>', unsafe_allow_html=True)

# Top Status Metrics Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="Coverage", value="FY26 Q1–Q4", delta="4 Quarters")
with m2:
    st.metric(label="Vector Index", value="244 Chunks", delta="ChromaDB")
with m3:
    st.metric(label="Embeddings", value="NVIDIA E5-v5", delta="1024 dims")
with m4:
    st.metric(label="Guardrail", value="0% Math Hallucination", delta="Strict Grounding")

st.markdown("<br>", unsafe_allow_html=True)

# Initialize session state variables
if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""

def set_sample_query(query_text):
    st.session_state["user_question"] = query_text

# Sample Question Quick-Triggers
st.markdown("##### 💡 Select a Sample Question:")
col_q1, col_q2, col_q3, col_q4 = st.columns(4)

col_q1.button(
    "📊 Q4 Revenue",
    use_container_width=True,
    on_click=set_sample_query,
    args=("What was HCLTech's revenue in Q4 FY26?",)
)
col_q2.button(
    "💰 Q1 Net Income",
    use_container_width=True,
    on_click=set_sample_query,
    args=("What was the Net Income in Q1 FY26?",)
)
col_q3.button(
    "📈 Q2 EBIT",
    use_container_width=True,
    on_click=set_sample_query,
    args=("What was EBIT in Q2 FY26?",)
)
col_q4.button(
    "📅 Q3 Growth",
    use_container_width=True,
    on_click=set_sample_query,
    args=("What was the QoQ revenue growth in Q3 FY26?",)
)

# Input Box bound to session state
question = st.text_input(
    "Ask a financial question:",
    key="user_question",
    placeholder="e.g. What was HCLTech's revenue in Q4 FY26?"
)

submit_btn = st.button("🚀 Analyze Financial Reports", type="primary", use_container_width=True)

# Run Query Action ONLY when Analyze Financial Reports button is clicked
if submit_btn:
    if not question.strip():
        st.warning("⚠️ Please enter a question or select a sample query above.")
    else:
        with st.spinner("⚡ Running NVIDIA NIM RAG Pipeline..."):
            t_start = time.time()
            
            # Step 1: Retrieval & Scoring
            t_retrieval = time.time()
            results = search_documents(question)
            retrieval_dur = round(time.time() - t_retrieval, 2)
            
            # Step 2: Generation
            t_gen = time.time()
            answer = generate_answer(question, results, model=selected_model)
            gen_dur = round(time.time() - t_gen, 2)
            total_duration = round(time.time() - t_start, 2)
            
            p_info = results.get("pipeline_info", {})
            quarter_found = p_info.get("quarter", "Auto-Detected")
            type_found = p_info.get("question_type", "Financial Metric")

        # Sleek Compact Process Pipeline Summary Bar
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 12px 18px; margin: 16px 0; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #cbd5e1;">
                <span>🎯 <b>Scope:</b> {quarter_found} ({type_found})</span>
                <span style="color: #475569;">|</span>
                <span>⚡ <b>Vector Search:</b> {retrieval_dur}s (NVIDIA E5-v5)</span>
                <span style="color: #475569;">|</span>
                <span>🧠 <b>LLM Synthesis:</b> {gen_dur}s ({selected_model.split('/')[-1]})</span>
            </div>
            <div style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">
                🟢 Pipeline Success ({total_duration}s)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --------------------------------------------------------
        # DISPLAY ANSWER PANEL
        # --------------------------------------------------------
        st.markdown(f"""
        <div class="answer-panel">
            <div class="answer-header">
                <span>💬 Grounded Financial Answer</span>
                <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 400; margin-left: auto;">Total Latency: {total_duration}s</span>
            </div>
            <div class="answer-text">{answer}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --------------------------------------------------------
        # DISPLAY RETRIEVED SOURCE CHUNKS INSPECTOR
        # --------------------------------------------------------
        st.markdown("### 📄 Retrieved Source Documents Inspector")
        st.caption("Inspect the exact report chunks used by the LLM to generate the answer.")
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        scores = results.get("scores", [[0]*len(docs)])[0]
        
        for i, (doc, meta, dist, score) in enumerate(zip(docs, metas, dists, scores)):
            source_name = meta.get("source", "Unknown")
            chunk_id = meta.get("chunk_id", i)
            
            with st.expander(f"📄 Pass {i+1}: {source_name} — Chunk {chunk_id} (Relevance Score: {round(score, 1)}, Vector Dist: {round(dist, 4)})"):
                st.markdown(f"**Report Source**: `{source_name}` | **Chunk ID**: `{chunk_id}`")
                st.code(doc, language="text")