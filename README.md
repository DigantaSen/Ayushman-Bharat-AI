# Ayushman Bharat AI — PM-JAY RAG Chatbot

> **Group:** BongoAI &nbsp;|&nbsp; **Course:** CS6690 — Statistical Methods in AI &nbsp;|&nbsp; **Assignment 3**
>
> **GitHub:** https://github.com/DigantaSen/Ayushman-Bharat-AI

| Name | Roll No. | Email |
|---|---|---|
| Nilkanta Karak | 2025201031 | nilkanta.karak@students.iiit.ac.in |
| Diganta Sen | 2025201050 | diganta.sen@students.iiit.ac.in |
| Surjit Mandal | 2025201057 | surjit.mandal@students.iiit.ac.in |

> **Dataset:** [Google Drive Link](https://drive.google.com/drive/folders/142-w3AWPS5FhVqoC5l_BoH6uSpl9KSnu)
> **Demo Video:** [Google Drive Link](https://drive.google.com/file/d/1CuQl6hAgYufhl-tfxXz2A5KMpbVStwHt)

---

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about India's PM-JAY (Pradhan Mantri Jan Arogya Yojana) health scheme strictly from official NHA documents — no hallucination, no external knowledge.

---

## Project Structure

```
Assignment3_2/
├── app.py              # Streamlit frontend (chat UI, sidebar, theme, translation)
├── rag_pipeline.py     # Full RAG backend (loading, chunking, retrieval, generation)
├── eval_metrics.py     # Retrieval evaluation script (no API key required)
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── data/               # Place official PM-JAY PDF documents here
└── chroma_db/          # Auto-created: persisted ChromaDB vector index
```

---

## Prerequisites

- Python **3.9+**
- A valid API key from **Groq** (free) or **Google AI Studio** (free)
  - Groq: https://console.groq.com → free Llama-3 access
  - Google: https://aistudio.google.com → free Gemini 1.5 Flash access
- PDF documents in `./data/` (official NHA PM-JAY documents)

---

## Setup

### 1. Clone and enter directory
```bash
git clone https://github.com/DigantaSen/Ayushman-Bharat-AI
cd Ayushman-Bharat-AI
```

### 2. Create a virtual environment (recommended)
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First install may take a few minutes. It downloads sentence-transformer models (~400 MB) and the cross-encoder reranker.

### 4. Add PM-JAY PDF documents
Place official NHA PDF files inside the `data/` folder:
```
data/
├── pmjay_guidelines.pdf
├── claims_adjudication_manual.pdf
└── ...
```
Any PDF placed here will be indexed automatically on first launch.

### 5. Set your API key

**Option A — Environment variable (recommended):**
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_..."

# Linux / Mac
export GROQ_API_KEY="gsk_..."
```

**Option B — Enter directly in the app sidebar** (no env var needed).

---

## Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

> **First launch** builds the ChromaDB vector index from your PDFs (~2–5 minutes on CPU). Subsequent launches load the cached index in under 3 seconds.

---

## How It Works

### Pipeline Overview

```
User Query
    │
    ▼
Query Resolution      ← resolves anaphoric/vague follow-ups using chat history
    │
    ▼
LLM Query Expansion   ← rewrites query into PM-JAY terminology (needs API key)
    │
    ▼
Semantic Search       ← ChromaDB similarity search, k=10, L2 metric
    │
    ▼
L2 Threshold Filter   ← keeps only chunks with L2 ≤ 0.80
    │
    ▼
Keyword Fallback      ← triggered if < 3 chunks pass (no API needed)
    │
    ▼
Cross-Encoder Rerank  ← ms-marco-MiniLM-L-6-v2, selects top-4 chunks
    │
    ▼
LLM Answer Generation ← type-aware prompt (process/eligibility/factual)
    │
    ▼
Hallucination Filter  ← 3 layers: marker strip → repetition guard → [NO_INFO]
    │
    ▼
[Optional Translation] ← Hindi ↔ English via deep-translator
```

### Key Components

| Component | Model / Library | Purpose |
|---|---|---|
| PDF extraction | PyMuPDF (`fitz`) | Clean page-by-page text extraction |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | 800 chars / 200 overlap |
| Embeddings | `all-mpnet-base-v2` (768-dim) | Dense semantic vectors |
| Vector store | ChromaDB (local, persistent) | Fast similarity search |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder precision ranking |
| LLM | Groq (Llama-3.3-70B) or Gemini 1.5 Flash | Answer generation |
| Translation | `deep-translator` (Google Translate) | Hindi ↔ English |
| Frontend | Streamlit | Chat UI with dark/light themes |

---

## App Features

- 💬 **Conversational interface** — multi-turn chat with context memory
- 📄 **Source citations** — every answer shows the source PDF and page number
- 🌐 **Bilingual** — toggle any answer between English and Hindi
- 🌙 **Dark / Light theme** — toggle in sidebar
- 🔄 **Pluggable LLM** — switch between Groq and Gemini from the sidebar
- 🚫 **Zero hallucination** — refuses out-of-scope questions with "Not available in documents."

---

## Running Evaluation

To compute real retrieval metrics (no API key required):

```bash
python eval_metrics.py
```

This runs 12 PM-JAY test queries across 4 pipeline configurations and outputs:
- **Context Precision** — fraction of retrieved chunks with tight L2 score
- **Keyword Recall** — fraction of ground-truth keywords found in context
- **NDCG@4** — ranking quality of top-4 retrieved chunks
- **Answer Relevance** — cosine similarity between query and context embeddings
- **Faithfulness** — cosine similarity between ideal answer and context embeddings

Results are printed to console and saved to `eval_results.json`.

---

## Hallucination Prevention

Three layers of defence:

1. **Retrieval-level** — L2 distance threshold (≤ 0.80) filters weakly-matched chunks; cross-encoder reranking promotes only genuinely relevant passages.
2. **Prompt-level** — System prompt explicitly forbids using external knowledge or guessing. LLM is instructed to output `[NO_INFO]` when context is insufficient.
3. **Output-level** — Post-generation regex validation strips AI-refusal markers, detects 6-gram repetition loops, and maps `[NO_INFO]` to a safe fallback message.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "⚠️ Enter your API key" | Paste key in sidebar or set env var |
| First launch very slow | Normal — building ChromaDB index from PDFs (~2–5 min) |
| "Not available in documents" for all queries | Check that PDFs are in `./data/` and non-empty |
| Hindi translation broken | Run `pip install deep-translator` |
| CUDA/GPU error on startup | Ignored automatically — app forces CPU mode |

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11+ |
| RAM | 4 GB | 8 GB |
| Disk | 2 GB (models + index) | 4 GB |
| GPU | Not required | Optional (speeds up embedding) |
| Internet | For LLM API calls | — |
