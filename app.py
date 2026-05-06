# ── Silence HuggingFace / transformers log spam before any other import ───────
import os, logging, warnings

# Force pure-Python protobuf: avoids _CheckCalledFromGeneratedFile crash
# that chromadb/opentelemetry triggers on Python 3.14 with protobuf>=4.21
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

os.environ["TRANSFORMERS_VERBOSITY"]       = "error"
os.environ["TOKENIZERS_PARALLELISM"]       = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]         = ""   # force CPU — avoids NVML/CUDA assert crash
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Suppress via Python logging (what transformers actually uses internally)
for _noisy in (
    "transformers",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "sentence_transformers",
    "langchain",
    "langchain_core",
    "langchain_community",
    "chromadb",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

warnings.filterwarnings("ignore")          # catch any remaining UserWarnings
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import re
from rag_pipeline import create_vector_store, process_query, translate_text, PROVIDERS

st.set_page_config(
    page_title="Ayushman Bharat AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Base CSS (dark default, uses CSS variables) ─────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* ── 1. CSS Variables: Dark Theme (default) ── */
:root {
    --bg-base:         #0a0e1a;
    --bg-card:         linear-gradient(135deg, rgba(15,22,41,0.97) 0%, rgba(17,24,39,0.93) 100%);
    --bg-input:        #000000;
    --bg-sidebar:      linear-gradient(180deg,#0f1629 0%,#0d1117 100%);
    --bg-chip:         rgba(99,179,237,0.09);
    --bg-source:       rgba(255,255,255,0.03);
    --txt-primary:     #e2e8f0;
    --txt-secondary:   #94a3b8;
    --txt-muted:       #64748b;
    --txt-placeholder: #475569;
    --accent:          #63b3ed;
    --purple:          #a78bfa;
    --teal:            #34d399;
    --pink:            #f472b6;
    --border-accent:   rgba(99,179,237,0.20);
    --border-accent-h: rgba(99,179,237,0.55);
    --border-card:     rgba(99,179,237,0.18);
    --border-subtle:   rgba(255,255,255,0.06);
    --shadow-card:     0 4px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    --shadow-input:    0 8px 32px rgba(0,0,0,0.55);
    --alert-bg:        rgba(245,158,11,0.09);
    --alert-border:    rgba(245,158,11,0.25);
    --alert-txt:       #fbbf24;
    --scrollbar:       rgba(99,179,237,0.25);
    --code-bg:         rgba(0,0,0,0.45);
    --top-shimmer:     linear-gradient(90deg,transparent,rgba(99,179,237,0.5),rgba(167,139,250,0.4),transparent);
    --title-grad:      linear-gradient(135deg,#e2e8f0 0%,#63b3ed 45%,#a78bfa 100%);
    --expander-bg:     rgba(255,255,255,0.02);
    --expander-cnt:    rgba(10,14,26,0.65);
    --user-msg-bg:     rgba(99,179,237,0.07);
}

/* ── 2. Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body { font-family: 'Inter', sans-serif; background: var(--bg-base); color: var(--txt-primary); }
.stApp { background: var(--bg-base) !important; color: var(--txt-primary) !important; font-family: 'Inter', sans-serif !important; }

/* ── 3. Hide chrome ── */
#MainMenu, footer { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; border: none !important; }
.block-container { max-width: 1100px !important; margin: 0 auto !important; padding: 2rem 1.5rem 7rem !important; }

/* ── 4. Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-accent) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important; font-size: 0.72rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    font-weight: 700 !important; margin-bottom: 0.8rem !important;
}
[data-testid="stSidebar"] label {
    color: var(--txt-secondary) !important; font-size: 0.78rem !important; font-weight: 500 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 8px !important; color: var(--txt-primary) !important; font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:hover,
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--border-accent-h) !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.12) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border-subtle) !important; margin: 0.8rem 0 !important; }
.sidebar-badge {
    background: var(--bg-chip); border: 1px solid var(--border-accent);
    border-radius: 10px; padding: 10px 14px; font-size: 0.78rem;
    color: var(--txt-secondary); line-height: 1.8;
}
.sidebar-badge b { color: var(--accent); }

/* ── 5. Header ── */
.chat-header { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.chat-header .badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--bg-chip); border: 1px solid var(--border-accent);
    border-radius: 100px; padding: 5px 16px; font-size: 0.71rem;
    color: var(--accent); letter-spacing: 0.08em;
    text-transform: uppercase; font-weight: 600; margin-bottom: 1rem;
}
.chat-header h1 {
    font-size: 2.4rem !important; font-weight: 800 !important;
    background: var(--title-grad) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 0.5rem !important; line-height: 1.15 !important;
}
.chat-header p { color: var(--txt-muted); font-size: 0.92rem; font-weight: 400; }

/* ── 6. Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 0.4rem 0 !important; animation: slideUp 0.3s ease;
}

/* ── 7. Answer card ── */
.answer-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 0.5rem;
    backdrop-filter: blur(16px); position: relative; overflow: visible;
    box-shadow: var(--shadow-card); animation: slideUp 0.35s ease;
    color: var(--txt-primary) !important;
    font-size: 1.25rem; line-height: 2.0;
}
.answer-card p, .answer-card li, .answer-card span, .answer-card div {
    color: var(--txt-primary) !important;
}
.answer-card strong, .answer-card b { color: var(--accent) !important; font-weight: 600; }
.answer-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--top-shimmer);
}

/* ── 8. User message ── */
[data-testid="stChatMessageContent"] {
    color: var(--txt-primary) !important;
}

/* ── 9. Citations ── */
.citations-bar { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.5rem 0; }
.citation-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--bg-chip); border: 1px solid var(--border-accent);
    border-radius: 100px; padding: 3px 12px; font-size: 0.71rem;
    color: var(--accent); font-weight: 600; white-space: nowrap;
}

/* ── 10. Sources ── */
.sources-section {
    margin-top: 0.5rem; padding: 0.75rem 1rem;
    background: var(--bg-source); border: 1px solid var(--border-subtle);
    border-radius: 10px; animation: slideUp 0.4s ease;
}
.sources-title {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--txt-muted); margin-bottom: 0.4rem;
}
.source-item {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.78rem; color: var(--txt-secondary); padding: 2px 0;
}
.source-item::before { content: '›'; color: var(--accent); font-weight: 700; }

/* ── 11. Expander ── */
.streamlit-expanderHeader {
    background: var(--expander-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important; color: var(--txt-muted) !important; font-size: 0.8rem !important;
}
.streamlit-expanderContent {
    background: var(--expander-cnt) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important;
}

/* ── 12. Buttons ── */
.stButton > button {
    background: var(--bg-chip) !important; border: 1px solid var(--border-accent) !important;
    border-radius: 100px !important; color: var(--accent) !important;
    font-size: 0.75rem !important; font-weight: 600 !important;
    padding: 4px 14px !important; letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important; min-height: unset !important;
    height: auto !important; line-height: 1.6 !important;
}
.stButton > button:hover {
    background: rgba(99,179,237,0.18) !important;
    border-color: var(--border-accent-h) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(99,179,237,0.25) !important;
}

/* ── 13. Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-input) !important; border: 1px solid var(--border-accent) !important;
    border-radius: 14px !important; backdrop-filter: blur(12px) !important;
    box-shadow: var(--shadow-input) !important;
}
[data-testid="stChatInput"] textarea {
    color: #1e293b !important; font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--txt-placeholder) !important; }
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,179,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.10), var(--shadow-input) !important;
}

/* ── 14. Avatars ── */
[data-testid="stChatMessageAvatarUser"], .stChatMessageAvatarUser {
    background: linear-gradient(135deg,#3b82f6,#8b5cf6) !important; border-radius: 50% !important;
}
[data-testid="stChatMessageAvatarAssistant"], .stChatMessageAvatarAssistant {
    background: linear-gradient(135deg,#06b6d4,#3b82f6) !important; border-radius: 50% !important;
}

/* ── 15. Alert, Spinner ── */
.stAlert {
    background: var(--alert-bg) !important; border: 1px solid var(--alert-border) !important;
    border-radius: 10px !important; color: var(--alert-txt) !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── 16. Animations ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

/* ── 17. Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--scrollbar); border-radius: 4px; }

/* ── 18. Code ── */
.stCode { background: var(--code-bg) !important; border-radius: 8px !important; }

/* ── 19. Responsive ── */
@media (max-width: 640px) {
    .chat-header h1 { font-size: 1.7rem !important; }
    .block-container { padding: 1rem 0.75rem 6rem !important; }
}
</style>
""")

# ─── Theme state ────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Inject light-mode CSS variable overrides when needed
if st.session_state.theme == "light":
    st.html("""
    <style>
    :root {
        --bg-base:         #f0f5ff;
        --bg-card:         linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(240,245,255,0.95) 100%);
        --bg-input:        #000000;
        --bg-sidebar:      linear-gradient(180deg,#e8eef8 0%,#dde6f5 100%);
        --bg-chip:         rgba(37,99,235,0.08);
        --bg-source:       rgba(37,99,235,0.04);
        --txt-primary:     #1e293b;
        --txt-secondary:   #475569;
        --txt-muted:       #64748b;
        --txt-placeholder: #94a3b8;
        --accent:          #2563eb;
        --purple:          #7c3aed;
        --teal:            #0d9488;
        --pink:            #db2777;
        --border-accent:   rgba(37,99,235,0.22);
        --border-accent-h: rgba(37,99,235,0.55);
        --border-card:     rgba(37,99,235,0.18);
        --border-subtle:   rgba(0,0,0,0.08);
        --shadow-card:     0 4px 24px rgba(0,0,0,0.09), inset 0 1px 0 rgba(255,255,255,0.8);
        --shadow-input:    0 4px 16px rgba(0,0,0,0.09);
        --alert-bg:        rgba(245,158,11,0.09);
        --alert-border:    rgba(245,158,11,0.30);
        --alert-txt:       #b45309;
        --scrollbar:       rgba(37,99,235,0.25);
        --code-bg:         rgba(0,0,0,0.05);
        --top-shimmer:     linear-gradient(90deg,transparent,rgba(37,99,235,0.5),rgba(124,58,237,0.4),transparent);
        --title-grad:      linear-gradient(135deg,#1e293b 0%,#2563eb 45%,#7c3aed 100%);
        --expander-bg:     rgba(37,99,235,0.04);
        --expander-cnt:    rgba(240,245,255,0.90);
    }
    html, body, .stApp { background: #f0f5ff !important; color: #1e293b !important; }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(37,99,235,0.05) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #1e293b !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }
    </style>
    """)





# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Theme toggle ─────────────────────────────
    _is_dark = st.session_state.get("theme", "dark") == "dark"
    if st.button("☀️ Light Mode" if _is_dark else "🌙 Dark Mode", key="theme_toggle"):
        st.session_state.theme = "light" if _is_dark else "dark"
        st.rerun()
    st.markdown("---")
    # ── Clear Chat ───────────────────────────────
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")


    provider = st.selectbox("LLM Provider", options=list(PROVIDERS.keys()),
                            index=0, key="provider")
    provider_cfg = PROVIDERS[provider]

    model = st.selectbox("Model", options=provider_cfg["models"],
                         index=0, key="model")

    key_label = f"{provider_cfg['env_key']}"
    api_key = st.text_input(f"🔑 API Key  ({key_label})", type="password", key="api_key")
    if api_key:
        os.environ[provider_cfg["env_key"]] = api_key

    st.markdown("---")
    st.markdown(
        f'<div class="sidebar-badge">'
        f'Provider: <b>{provider}</b><br>'
        f'Model: <b>{model}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    language = st.selectbox("🌐 Reply language", options=["English", "Hindi"],
                            index=0, key="language")

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.72rem;color:#334155;text-align:center;">'
        'Powered by official PM-JAY documents<br>'
        '<span style="color:#1e3a5f;">© Ayushman Bharat AI</span>'
        '</p>',
        unsafe_allow_html=True,
    )

# ─── Key warning ────────────────────────────────────────────────────────────
if not os.environ.get(provider_cfg["env_key"]):
    st.warning(f"⚠️  Enter your **{provider_cfg['env_key']}** in the sidebar to start chatting.")

# ─── Session state ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def _extract_answer_text(content: str) -> str:
    match = re.search(r'<div class="answer-card">\s*(.*?)\s*</div>', content, re.S)
    if match:
        return match.group(1).strip()
    return content.strip()


def _normalize_message_history():
    for message in st.session_state.messages:
        if message.get("role") == "assistant" and "answer_text" not in message:
            content = message.get("content", "")
            message["answer_text"] = _extract_answer_text(content)
            message["generated_language"] = message.get("generated_language", "English")
            message["display_language"] = message.get("display_language",
                                                       message["generated_language"])
            message["sources"] = message.get("sources", [])


def _render_assistant_message(message: dict, index: int):
    provider_name = st.session_state.get("provider", list(PROVIDERS.keys())[0])
    model_name    = st.session_state.get("model")
    generated_language = message.get("generated_language", "English")
    display_language   = message.get("display_language", generated_language)
    answer_text = message.get("answer_text", "")
    sources     = message.get("sources", [])

    if display_language != generated_language:
        # Cache translation in the message dict to avoid re-calling LLM on every rerun
        # (e.g. theme toggle triggers st.rerun which would otherwise re-translate and truncate)
        cache_key = f"translated_{display_language}"
        if cache_key not in message:
            message[cache_key] = translate_text(
                answer_text,
                target_language=display_language,
                source_language=generated_language,
                provider=provider_name,
                model=model_name,
            )
        rendered_answer = message[cache_key]
    else:
        rendered_answer = answer_text

    # Language toggle button
    toggle_target = "Hindi" if display_language == "English" else "English"
    st.button(f"🌐 {toggle_target}", key=f"toggle_lang_{index}",
              on_click=lambda: message.update({"display_language": toggle_target}))

    # Answer card
    st.markdown(
        f'<div class="answer-card">{rendered_answer}</div>',
        unsafe_allow_html=True,
    )

    # Citations chips
    citations = message.get("debug_info", {}).get("citations", [])
    if citations:
        chips = "".join(
            f'<span class="citation-chip">📄 {c}</span>'
            for c in citations
        )
        st.markdown(
            f'<div class="citations-bar">{chips}</div>',
            unsafe_allow_html=True,
        )

    # Sources section
    if sources:
        items = "".join(
            f'<div class="source-item">{s}</div>'
            for s in sorted(set(sources))
        )
        st.markdown(
            f'<div class="sources-section">'
            f'<div class="sources-title">Sources</div>'
            f'{items}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─── Vector store ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading knowledge base...")
def load_db():
    return create_vector_store()

try:
    vector_store = load_db()
except Exception as e:
    st.error(f"Error loading vector store: {e}")
    st.stop()

_normalize_message_history()

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <div class="badge">🏥 &nbsp; Official PM-JAY Intelligence</div>
    <h1>Ayushman Bharat AI</h1>
    <p>Ask questions grounded in official PM-JAY scheme documents</p>
</div>
""", unsafe_allow_html=True)

# ─── Chat history ─────────────────────────────────────────────────────────────
for index, message in enumerate(st.session_state.messages):
    role   = message["role"]
    avatar = "👤" if role == "user" else "🏥"
    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            _render_assistant_message(message, index)
        else:
            _q_color = "#1e293b" if st.session_state.get("theme", "dark") == "light" else "#e2e8f0"
            st.markdown(
                f'<div style="color:{_q_color};font-size:1.2rem;font-weight:500;">{message["content"]}</div>',
                unsafe_allow_html=True,
            )

# ─── Input ───────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Ayushman Bharat PM-JAY…"):
    with st.chat_message("user", avatar="👤"):
        _q_color = "#1e293b" if st.session_state.get("theme", "dark") == "light" else "#e2e8f0"
        st.markdown(
            f'<div style="color:{_q_color};font-size:1.2rem;font-weight:500;">{prompt}</div>',
            unsafe_allow_html=True,
        )
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner(""):
            response, sources, debug_info = process_query(
                prompt, vector_store,
                provider=st.session_state.get("provider", list(PROVIDERS.keys())[0]),
                model=st.session_state.get("model"),
                chat_history=st.session_state.messages,
                language=st.session_state.get("language", "English"),
            )

        assistant_message = {
            "role": "assistant",
            "content": response,
            "answer_text": response,
            "generated_language": st.session_state.get("language", "English"),
            "display_language":   st.session_state.get("language", "English"),
            "sources":    sources,
            "debug_info": debug_info,
        }
        st.session_state.messages.append(assistant_message)
        _render_assistant_message(assistant_message, len(st.session_state.messages) - 1)


