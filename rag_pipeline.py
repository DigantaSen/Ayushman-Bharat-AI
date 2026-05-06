import os
import glob
import logging
import warnings

# ── Silence HuggingFace / transformers noise (env vars + logging API) ─────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")
for _logger_name in ("transformers", "sentence_transformers",
                      "langchain", "langchain_core", "chromadb"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
# ─────────────────────────────────────────────────────────────────────────────

import chromadb
import fitz  # pymupdf — better PDF extraction
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Provider config — free tiers only
# ---------------------------------------------------------------------------
PROVIDERS = {
    "Groq (Llama-3 — Free)": {
        "env_key": "GROQ_API_KEY",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        "default_model": "llama-3.3-70b-versatile",
    },
    "Gemini (Google AI Studio — Free)": {
        "env_key": "GOOGLE_API_KEY",
        "models": [
            "gemini-1.5-flash-latest",
        ],
        "default_model": "gemini-1.5-flash-latest",
    },
}


def get_llm(provider: str = None, model: str = None):
    """Return the correct LangChain LLM object for the chosen provider."""
    _default_key = next(iter(PROVIDERS))
    cfg = PROVIDERS.get(provider, PROVIDERS[_default_key])
    chosen_model = model or cfg["default_model"]
    api_key = os.environ.get(cfg["env_key"], "")

    # Fail fast with a clear error when API key is missing
    if not api_key:
        raise ValueError(
            f"Missing API key for provider '{provider or _default_key}'. "
            f"Set environment variable '{cfg['env_key']}' or provide the key in the app sidebar."
        )

    if provider and provider.startswith("Gemini"):
        return ChatGoogleGenerativeAI(
            model=chosen_model,
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=2048,
        )
    else:
        return ChatGroq(
            model=chosen_model,
            api_key=api_key,
            temperature=0,
            max_tokens=2048,
        )


def translate_text(text: str, target_language: str, provider: str = None,
                   model: str = None, source_language: str = None) -> str:
    """Translate a response using Google Translate (no token limits, always complete).

    Uses deep-translator which calls Google Translate under the hood.
    Falls back to original text on any error.
    """
    if not text.strip():
        return text
    nt = (target_language or "English").strip().lower()
    ns = (source_language or "").strip().lower()
    # No translation needed if already in target language
    if nt.startswith("en") and ns.startswith("en"):
        return text
    if nt.startswith("hi") and ns.startswith("hi"):
        return text

    lang_map = {"en": "en", "hi": "hi"}
    src_code = "hi" if ns.startswith("hi") else "en"
    tgt_code = "hi" if nt.startswith("hi") else "en"

    try:
        from deep_translator import GoogleTranslator
        # GoogleTranslator handles long texts up to 5000 chars per chunk;
        # split on double-newlines to preserve bullet formatting, then rejoin.
        paragraphs = text.split("\n\n")
        translated_parts = []
        translator = GoogleTranslator(source=src_code, target=tgt_code)
        for para in paragraphs:
            if para.strip():
                # Handle lines individually to preserve bullets/numbering
                lines = para.split("\n")
                translated_lines = []
                for line in lines:
                    if line.strip():
                        t = translator.translate(line.strip())
                        translated_lines.append(t if t else line)
                    else:
                        translated_lines.append(line)
                translated_parts.append("\n".join(translated_lines))
            else:
                translated_parts.append(para)
        result = "\n\n".join(translated_parts).strip()
        print(f"[Translation] {src_code}→{tgt_code}: {len(text)} chars → {len(result)} chars", flush=True)
        return result or text
    except Exception as e:
        print(f"\n[Translation ERROR] {e}", flush=True)
        return text


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, "data")
CHROMA_PATH = os.path.join(_BASE_DIR, "chroma_db")
_COLLECTION_NAME = "pmjay_mpnet"   # new name: signals mpnet embeddings

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
_RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder = None   # lazy-loaded


def get_embeddings():
    """768-dim mpnet embeddings — higher quality than MiniLM (CPU-only, no CUDA)."""
    return HuggingFaceEmbeddings(
        model_name=_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker (loaded once per session)."""
    global _reranker
    if _reranker is None:
        print("[Reranker] Loading cross-encoder...", flush=True)
        _reranker = CrossEncoder(_RERANKER_MODEL)
        print("[Reranker] Ready.", flush=True)
    return _reranker


def rerank_docs(query: str, docs: list, top_k: int = 4) -> list:
    """Re-rank candidates by cross-encoder relevance; return top_k."""
    if not docs:
        return docs
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    print(f"[Reranker] Scores: {[round(float(s), 3) for s, _ in scored[:top_k]]}", flush=True)
    return [doc for _, doc in scored[:top_k]]


# ---------------------------------------------------------------------------
# PDF loading — pymupdf (cleaner than PyPDFLoader)
# ---------------------------------------------------------------------------
def load_documents() -> list:
    """Load PDFs with pymupdf for clean, structure-preserving extraction."""
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    if not pdf_files:
        print(f"[PDFLoader] Warning: no PDFs found in {DATA_PATH}")
    documents = []
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "filename": os.path.basename(pdf_path),
                            "page": page_num + 1,
                        },
                    ))
            doc.close()
        except Exception as e:
            print(f"[PDFLoader] Error: {pdf_path}: {e}", flush=True)
    print(f"[PDFLoader] {len(documents)} pages from {len(pdf_files)} PDFs", flush=True)
    return documents


def split_documents(documents: list) -> list:
    """Chunk at 800 chars / 200 overlap for richer, less-fragmented chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Chunk quality filter
# ---------------------------------------------------------------------------
_PDF_LIGATURES = "\uFB00\uFB01\uFB02\uFB03\uFB04\uFB05\uFB06"


def _sanitize_text(text: str) -> str:
    """Replace PDF ligature characters with ASCII equivalents."""
    for lig, rep in {"\uFB00": "ff", "\uFB01": "fi", "\uFB02": "fl",
                     "\uFB03": "ffi", "\uFB04": "ffl", "\uFB05": "st", "\uFB06": "st"}.items():
        text = text.replace(lig, rep)
    return text


def _is_garbage_chunk(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[0] in _PDF_LIGATURES:
        return True
    sanitized = _sanitize_text(stripped)
    non_space = sanitized.replace(" ", "").replace("\n", "")
    if not non_space:
        return True
    if sum(c.isdigit() for c in non_space) / len(non_space) > 0.35:
        return True
    words = [w.strip(".,;:()") for w in sanitized.split() if any(c.isalpha() for c in w)]
    if len(words) < 25:
        return True
    if sum(len(w) for w in words) / len(words) < 3.5:
        return True
    if sum(1 for w in words if len(w) <= 3) / len(words) > 0.60:
        return True
    # Page headers/footers from pymupdf (e.g. "Page 5 of 91\nFor latest information...")
    lower = sanitized.lower()
    if "for latest information please refer" in lower:
        return True
    if "page " in lower and " of " in lower and len(words) < 40:
        # Short chunk dominated by a page reference
        return True

    return False


def filter_chunks(chunks: list) -> list:
    before = len(chunks)
    good = [c for c in chunks if not _is_garbage_chunk(c.page_content)]
    print(f"[ChunkFilter] Kept {len(good)}/{before} chunks.", flush=True)
    return good


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def create_vector_store():
    """Build or load ChromaDB with all-mpnet-base-v2 embeddings."""
    embeddings = get_embeddings()
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        col = client.get_collection(_COLLECTION_NAME)
        if col.count() > 0:
            print(f"[VectorStore] Loaded {col.count()} chunks from disk.", flush=True)
            return Chroma(client=client, collection_name=_COLLECTION_NAME,
                          embedding_function=embeddings)
    except Exception:
        pass

    print("[VectorStore] Building new index...", flush=True)
    docs = load_documents()
    if not docs:
        raise ValueError("No PDFs in ./data/")
    chunks = filter_chunks(split_documents(docs))
    print(f"[VectorStore] Indexing {len(chunks)} chunks...", flush=True)
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings,
                               client=client, collection_name=_COLLECTION_NAME)
    print(f"[VectorStore] Done. Path: {CHROMA_PATH}", flush=True)
    return vs


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------
def resolve_query(query: str, chat_history: list) -> str:
    """Expand vague queries using chat history context."""
    vague_triggers = ["it", "this", "that", "they", "them", "the process",
                      "the scheme", "the criteria", "the benefit", "explain more",
                      "tell me more", "and what", "also"]
    words = query.lower().split()
    is_vague = len(words) <= 6 or any(t in query.lower() for t in vague_triggers)
    if is_vague and chat_history:
        prior = [m["content"] for m in chat_history if m["role"] == "user"][-3:]
        if prior:
            resolved = " | ".join(prior) + " -> " + query
            print(f"[QueryResolver] Resolved: '{resolved}'", flush=True)
            return resolved
    return query


def detect_question_type(query: str) -> str:
    """Return 'process' | 'eligibility' | 'comparison' | 'factual'."""
    q = query.lower()
    if any(w in q for w in ["how", "steps", "process", "apply", "register",
                             "procedure", "enroll", "empanel"]):
        return "process"
    if any(w in q for w in ["who", "eligible", "qualify", "criteria",
                             "entitled", "benefit from"]):
        return "eligibility"
    if any(w in q for w in ["difference", " vs ", "versus", "compare"]):
        return "comparison"
    return "factual"


def expand_query(query: str, provider: str = None, model: str = None) -> str:
    """Rewrite query to PM-JAY domain terms for better retrieval."""
    prompt = PromptTemplate.from_template(
        """You are a search query optimizer for PM-JAY (Ayushman Bharat) documents.

Rewrite the user question into a concise search query using specific PM-JAY terminology.
Return ONLY the rewritten query — no explanation, no quotes, one line.

IMPORTANT: Distinguish between two different "application" meanings:
- "apply for the scheme / enroll / get Ayushman card" → use enrollment/registration terms
- "apply for treatment / hospital admission / claims" → use hospital/claims terms

Examples:
- "what is the process of application?" -> beneficiary enrollment registration Ayushman card SECC eligibility verification
- "how to apply for ayushman bharat?" -> beneficiary enrollment registration Ayushman card how to join PM-JAY
- "how do I register for PM-JAY?" -> beneficiary registration enrollment Ayushman card Common Service Centre
- "how to enroll?" -> beneficiary enrollment registration SECC eligibility PM-JAY Ayushman card
- "who are eligible?" -> eligible beneficiaries SECC database criteria families PM-JAY
- "what benefits are covered?" -> medical benefits health packages treatment coverage AB PM-JAY
- "what is the hospital empanelment process?" -> hospital empanelment HEM registration procedure steps SHA
- "how are claims settled?" -> claim adjudication settlement reimbursement process TMS hospital
- "what is PM-JAY?" -> Ayushman Bharat PM-JAY scheme overview health insurance coverage
- "what is the treatment process at hospital?" -> beneficiary E-card BIS verification TMS cashless treatment EHCP

User question: {question}

Rewritten search query:"""
    )
    try:
        llm = get_llm(provider=provider, model=model)
        expanded = (prompt | llm | StrOutputParser()).invoke(
            {"question": query}).strip().splitlines()[0]
        print(f"[QueryExpansion] '{query}' -> '{expanded}'", flush=True)
        return expanded
    except Exception as e:
        print(f"[QueryExpansion ERROR] {e}", flush=True)
        return query


def keyword_search(query: str, vector_store, k: int = 4) -> list:
    """Keyword fallback using ChromaDB $contains."""
    stop = {"what", "is", "the", "are", "how", "does", "can", "for",
            "and", "of", "in", "a", "an"}
    keywords = [w.strip("?.,!").lower() for w in query.split()
                if len(w.strip("?.,!")) > 3 and w.lower().strip("?.,!") not in stop]
    seen, docs = set(), []
    for kw in keywords[:3]:
        try:
            res = vector_store._collection.get(
                where_document={"$contains": kw}, limit=k,
                include=["documents", "metadatas"])
            for text, meta in zip(res["documents"], res["metadatas"]):
                if text not in seen:
                    seen.add(text)
                    docs.append(Document(page_content=text, metadata=meta or {}))
        except Exception as e:
            print(f"[KeywordSearch] '{kw}' error: {e}", flush=True)
    print(f"[KeywordSearch] {len(docs)} chunks for {keywords[:3]}", flush=True)
    return docs[:k]


_SCORE_THRESHOLD = 0.80   # L2 distance; lower = more similar
_MAX_CONTEXT_CHARS = 3500


def retrieve_context(query, vector_store, k=10, provider=None, model=None):
    """Expand → semantic search → filter → rerank → optional keyword fallback."""
    expanded = expand_query(query, provider=provider, model=model)
    search_q = f"Ayushman Bharat PM-JAY {expanded}"

    results = vector_store.similarity_search_with_score(search_q, k=k)
    print(f"[Retrieval] Query: {search_q}", flush=True)

    filtered = [(doc, s) for doc, s in results if s <= _SCORE_THRESHOLD]
    print(f"[Retrieval] {len(filtered)}/{len(results)} passed threshold", flush=True)

    sem_docs = [doc for doc, _ in filtered]
    sem_set = {doc.page_content for doc in sem_docs}

    extra = []
    if len(sem_docs) < 3:
        kw = keyword_search(query, vector_store, k=4)
        extra = [d for d in kw if d.page_content not in sem_set]

    candidates = sem_docs + extra

    # Fallback: if expansion gave bad results, retry with original query
    if not candidates:
        print("[Retrieval] Expanded query gave 0 results — retrying with original query", flush=True)
        raw_results = vector_store.similarity_search_with_score(query, k=k)
        candidates = [doc for doc, s in raw_results if s <= _SCORE_THRESHOLD]
        # If still nothing, take top-3 regardless of threshold
        if not candidates:
            candidates = [doc for doc, _ in raw_results[:3]]
            print("[Retrieval] Threshold override — using top-3 by score", flush=True)

    reranked = rerank_docs(query, candidates, top_k=4) if candidates else []

    # Debug info
    debug_chunks = []
    for i, (doc, s) in enumerate(filtered):
        safe = doc.page_content[:300].encode("ascii", errors="replace").decode("ascii")
        print(f"\n--- Chunk {i} (score={s:.4f}) ---\n{safe}", flush=True)
        debug_chunks.append({"score": round(s, 4), "preview": doc.page_content[:200]})
    for i, doc in enumerate(extra):
        safe = doc.page_content[:300].encode("ascii", errors="replace").decode("ascii")
        print(f"\n--- KW Chunk {i} ---\n{safe}", flush=True)
        debug_chunks.append({"score": "kw", "preview": doc.page_content[:200]})

    if not reranked:
        return "", [], debug_chunks, expanded, []

    # Cap context at chunk boundaries (don't cut mid-chunk)
    chunks_for_context = []
    total = 0
    for d in reranked:
        if total + len(d.page_content) <= _MAX_CONTEXT_CHARS:
            chunks_for_context.append(d.page_content)
            total += len(d.page_content)
        else:
            break
    if not chunks_for_context:  # at least include the top chunk even if oversized
        chunks_for_context = [reranked[0].page_content[:_MAX_CONTEXT_CHARS]]

    raw_ctx = _sanitize_text("\n\n".join(chunks_for_context))
    context = raw_ctx
    print(f"[Retrieval] Context: {len(context)} chars from {len(chunks_for_context)} chunks", flush=True)

    sources = list({os.path.basename(d.metadata.get("source", ""))
                    for d in reranked if d.metadata.get("source")})

    # Citations: top-2 reranked chunks with page numbers
    citations = []
    for doc in reranked[:2]:
        fname = os.path.basename(doc.metadata.get("source", ""))
        page = doc.metadata.get("page", "?")
        if fname:
            citations.append(f"{fname}, p.{page}")

    return context, sources, debug_chunks, expanded, citations


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
_PROMPT_RULES = """\
You are a precise assistant for Ayushman Bharat PM-JAY documents.
Answer ONLY from the context below. Rules:
1. No repeated points.
2. Stay strictly on-topic.
3. No headings like "Key Features" or "Challenges".
4. No generic padding.
5. You may INFER challenges, issues, or difficulties from process descriptions in the context
   (e.g. if a process requires recovery of excess payments, that IS a challenge for hospitals).
   You do NOT need the exact word to appear — but every point must be grounded in the context.
6. If context has absolutely zero relevance → output only: [NO_INFO]
"""

_FORMAT_HINTS = {
    "process":     "Format: numbered steps (1. 2. 3. …), max 5 steps.",
    "eligibility": "Format: 3-5 bullet points on who qualifies and the criteria.",
    "comparison":  "Format: 2-4 paired bullet points comparing the two.",
    "factual":     "Format: 2-4 concise bullet points directly answering the question.",
}


def generate_answer(query, context, provider=None, model=None,
                    language="English", chat_history=None):
    """Type-aware answer with conversation memory. Returns (answer, raw)."""
    if not context.strip():
        return "Not available in documents.", ""

    lang_note = ""
    if language and language.lower().startswith("hi"):
        lang_note = ("\n\nIMPORTANT: Answer in Hindi. "
                     "Use 'दस्तावेज़ों में उपलब्ध नहीं है.' as the [NO_INFO] fallback.")

    q_type = detect_question_type(query)
    fmt_hint = _FORMAT_HINTS.get(q_type, _FORMAT_HINTS["factual"])
    print(f"[QuestionType] {q_type}", flush=True)

    # Conversation memory — last 2 Q&A pairs
    history_block = ""
    if chat_history:
        msgs = [m for m in chat_history if m["role"] in ("user", "assistant")]
        pairs = []
        for i in range(len(msgs) - 1):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                pairs.append((msgs[i]["content"][:150],
                               msgs[i + 1].get("answer_text", "")[:150]))
        if pairs:
            lines = ["Previous Q&A (for context only):"]
            for q, a in pairs[-2:]:
                lines += [f"Q: {q}", f"A: {a}", ""]
            history_block = "\n".join(lines) + "\n"

    template = (
        _PROMPT_RULES + "\n" + fmt_hint + "\n\n"
        "{history}"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:" + lang_note
    )

    prompt = PromptTemplate.from_template(template)
    llm = get_llm(provider=provider, model=model)
    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({"context": context, "question": query,
                            "history": history_block}).strip()
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "quota" in err or "resource_exhausted" in err:
            return "⚠️ API quota exceeded. Try again in a minute or switch model.", str(e)
        if "401" in err or "403" in err or "api_key" in err or "invalid_argument" in err:
            return "⚠️ Invalid API key. Check the sidebar.", str(e)
        if "404" in err or "not_found" in err:
            return "⚠️ Model not found. Try another model.", str(e)
        return f"⚠️ LLM error: {str(e)[:200]}", str(e)

    safe = raw.encode("ascii", errors="replace").decode("ascii")
    print(f"[LLM Raw]\n{safe}", flush=True)

    # Strip lines containing system error strings or hallucination markers
    _HALLUCINATION_MARKERS = [
        "api quota", "api key", "switch to a different model", "try again in a minute",
        "not available in document", "language model", "as an ai", "i cannot",
        "i don't have", "i do not have",
    ]
    lines = raw.splitlines()
    clean_lines = [
        l for l in lines
        if not any(m in l.lower() for m in _HALLUCINATION_MARKERS)
    ]
    if len(clean_lines) < len(lines):
        removed = len(lines) - len(clean_lines)
        print(f"[HallucinationFilter] Removed {removed} line(s)", flush=True)
    raw = "\n".join(clean_lines).strip()
    if not raw:
        fb = "Not available in documents."
        if language and language.lower().startswith("hi"):
            fb = "दस्तावेज़ों में उपलब्ध नहीं है."
        return fb, raw

    # Re-number bullet points so they always start at 1 (no gaps after filtering)
    import re as _re
    renumbered, counter = [], 1
    for line in raw.splitlines():
        if _re.match(r"^\d+\.", line.strip()):
            line = _re.sub(r"^\d+\.", f"{counter}.", line.strip())
            counter += 1
        renumbered.append(line)
    raw = "\n".join(renumbered)

    # Repetition loop guard
    words = raw.split()
    for s in range(min(len(words) - 6, 30)):
        phrase = " ".join(words[s:s + 6])
        if raw.count(phrase) > 3:
            print(f"[RepetitionDetected] '{phrase}'", flush=True)
            fb = "Not available in documents."
            if language and language.lower().startswith("hi"):
                fb = "दस्तावेज़ों में उपलब्ध नहीं है."
            return fb, raw

    # [NO_INFO] anywhere
    if "[NO_INFO]" in raw:
        before = raw.split("[NO_INFO]")[0].strip()
        mw = [w for w in before.split() if w.startswith(("*", "-")) or len(w) > 3]
        if len(mw) >= 5:
            lines = before.splitlines()
            bullets = [l for l in lines if l.strip().startswith(
                ("*", "-", "•", "1", "2", "3", "4", "5"))]
            if bullets:
                return "\n".join(bullets), raw
        fb = "Not available in documents."
        if language and language.lower().startswith("hi"):
            fb = "दस्तावेज़ों में उपलब्ध नहीं है."
        return fb, raw

    if any(p in raw.lower() for p in ["<|", "start_header"]):
        return "Not available in documents.", raw

    return raw, raw


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def process_query(query, vector_store, provider=None, model=None,
                  chat_history=None, language="English"):
    """Full pipeline: resolve → retrieve → generate."""
    resolved = resolve_query(query, chat_history or [])

    context, sources, debug_chunks, expanded, citations = retrieve_context(
        resolved, vector_store, provider=provider, model=model)

    debug_info = {
        "resolved_query": resolved,
        "expanded_query": expanded,
        "chunks": debug_chunks,
        "llm_raw": None,
        "citations": citations,
    }

    if not context:
        fb = "Not available in documents."
        if language and language.lower().startswith("hi"):
            fb = "दस्तावेज़ों में उपलब्ध नहीं है."
        return fb, [], debug_info

    answer, llm_raw = generate_answer(
        query, context,
        provider=provider, model=model,
        language=language, chat_history=chat_history,
    )
    debug_info["llm_raw"] = llm_raw

    if answer.strip() == "[NO_INFO]" or "not available in documents" in answer.lower():
        fb = "Not available in documents."
        if language and language.lower().startswith("hi"):
            fb = "दस्तावेज़ों में उपलब्ध नहीं है."
        return fb, [], debug_info

    return answer, sources, debug_info