"""
eval_metrics.py
================
Computes REAL retrieval metrics from the live PM-JAY ChromaDB.
No LLM API key required — uses local sentence-transformers + cross-encoder only.

Metrics computed per configuration:
  - Context Precision   : fraction of top-k chunks with L2 score <= PRECISION_THR
  - Keyword Recall      : fraction of GT keywords found in retrieved context (proxy for recall)
  - NDCG@4              : Normalised Discounted Cumulative Gain using inverted-L2 as relevance
  - Answer Relevance    : cosine sim between query embedding and mean context embedding
  - Faithfulness proxy  : cosine sim between context embedding and "ideal answer" embedding

Configurations:
  1. Full System       (static expansion + reranking)
  2. w/o Expansion     (raw query + reranking)
  3. w/o Reranking     (static expansion + top-4 by L2)
  4. Baseline          (raw query + top-4 by L2)
"""

import os, sys, math, json
import numpy as np

os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["TOKENIZERS_PARALLELISM"]        = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]          = ""

import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import (
    create_vector_store, get_reranker, get_embeddings,
    keyword_search, _SCORE_THRESHOLD,
)
from langchain_core.documents import Document

# ── Constants ────────────────────────────────────────────────────────────────
PRECISION_THR = 0.65   # L2 threshold for "highly relevant" (empirically tight)
TOP_K_SEARCH  = 10
TOP_K_CONTEXT = 4

# ── Test set: queries + static expansions + ground-truth keywords ─────────────
# Static expansions mirror what the LLM expansion prompt produces
# (based on examples hard-coded in rag_pipeline.py expand_query prompt).
TEST_SET = [
    {
        "query":    "What is the annual health cover provided under PM-JAY?",
        "expanded": "health cover Rs 5 lakh per family per year AB PM-JAY coverage limit hospitalization",
        "gt_kw":    ["5 lakh", "health cover", "family", "year", "hospitalization"],
        "category": "factual",
        "ideal":    "PM-JAY provides Rs 5 lakh health cover per family per year for secondary and tertiary hospitalization.",
    },
    {
        "query":    "Who is eligible for Ayushman Bharat PM-JAY?",
        "expanded": "eligible beneficiaries SECC database criteria socioeconomic deprived families PM-JAY",
        "gt_kw":    ["SECC", "socio economic", "deprived", "rural", "urban", "eligible"],
        "category": "eligibility",
        "ideal":    "Beneficiaries are identified from SECC database covering deprived rural and urban occupational category families.",
    },
    {
        "query":    "How does a hospital get empaneled under PM-JAY?",
        "expanded": "hospital empanelment HEM registration procedure steps SHA criteria application",
        "gt_kw":    ["empanel", "HEM", "criteria", "SHA", "application", "registration"],
        "category": "process",
        "ideal":    "Hospitals apply through the HEM portal, meet criteria set by SHA, and get empaneled after verification.",
    },
    {
        "query":    "What is the claim settlement process for hospitals?",
        "expanded": "claim adjudication settlement reimbursement process TMS hospital pre-authorization discharge",
        "gt_kw":    ["claim", "pre-authorization", "adjudication", "TMS", "settlement", "discharge"],
        "category": "process",
        "ideal":    "Hospitals submit claims via TMS after discharge; claims are adjudicated and settled within turn-around time.",
    },
    {
        "query":    "What documents are needed for PM-JAY beneficiary identification?",
        "expanded": "beneficiary identification documents Aadhaar SECC verification Ayushman card eligibility",
        "gt_kw":    ["aadhaar", "ration card", "SECC", "verification", "beneficiary", "card"],
        "category": "factual",
        "ideal":    "Beneficiaries are verified using Aadhaar or ration card matched against SECC database for Ayushman card issuance.",
    },
    {
        "query":    "How does the pre-authorization process work at empaneled hospitals?",
        "expanded": "pre-authorization code insurer TMS approval EHCP cashless treatment beneficiary hospital",
        "gt_kw":    ["pre-authorization", "code", "insurer", "TMS", "approval", "cashless"],
        "category": "process",
        "ideal":    "Pre-authorization code is generated in TMS by the insurer/trust after verifying the beneficiary and treatment eligibility.",
    },
    {
        "query":    "What health packages are covered under PM-JAY?",
        "expanded": "health benefit package HBP surgical medical day care procedures covered AB PM-JAY",
        "gt_kw":    ["health benefit package", "HBP", "surgical", "medical", "day care", "package"],
        "category": "factual",
        "ideal":    "PM-JAY covers medical, surgical, and day-care Health Benefit Packages across secondary and tertiary care.",
    },
    {
        "query":    "What is the role of State Health Agencies in implementing PM-JAY?",
        "expanded": "SHA state health agency implementation empanelment grievance fund flow oversight",
        "gt_kw":    ["SHA", "state health agency", "implementation", "empanelment", "grievance"],
        "category": "factual",
        "ideal":    "SHAs oversee empanelment, fund flow, grievance redressal and scheme implementation at state level.",
    },
    {
        "query":    "How can a beneficiary register a grievance under PM-JAY?",
        "expanded": "beneficiary grievance complaint helpline 14555 redressal mechanism PM-JAY",
        "gt_kw":    ["grievance", "complaint", "helpline", "14555", "redressal"],
        "category": "process",
        "ideal":    "Beneficiaries can call helpline 14555 or approach SHA grievance cells to register complaints under PM-JAY.",
    },
    {
        "query":    "What anti-fraud measures are in place under PM-JAY?",
        "expanded": "fraud detection anti-fraud audit deempanelment penalty recovery PM-JAY measures",
        "gt_kw":    ["fraud", "detection", "anti-fraud", "audit", "deempanel", "penalty"],
        "category": "factual",
        "ideal":    "PM-JAY uses anti-fraud guidelines, audits, deempanelment, and recovery of excess payments to deter fraud.",
    },
    {
        "query":    "What challenges do hospitals face processing claims through TMS?",
        "expanded": "claim adjudication settlement reimbursement TMS hospital challenges documentation",
        "gt_kw":    ["TMS", "claim", "hospital", "challenge", "documentation", "settlement"],
        "category": "factual",
        "ideal":    "Hospitals face documentation compliance challenges, claim rejection due to incomplete entries, and TMS technical issues.",
    },
    {
        "query":    "What is the coverage for senior citizens aged 70 and above?",
        "expanded": "senior citizens 70 years above health coverage Ayushman Vay Vandana Card AB PM-JAY",
        "gt_kw":    ["70 years", "senior citizen", "vay vandana", "coverage", "5 lakh"],
        "category": "factual",
        "ideal":    "Citizens 70+ get Rs 5 lakh free coverage irrespective of income via Ayushman Vay Vandana Card from Oct 2024.",
    },
]

# ── Metric helpers ────────────────────────────────────────────────────────────

def keyword_recall(context: str, gt_kw: list) -> float:
    ctx = context.lower()
    found = sum(1 for kw in gt_kw if kw.lower() in ctx)
    return round(found / len(gt_kw), 4) if gt_kw else 0.0


def context_precision(scores: list, thr: float = PRECISION_THR) -> float:
    if not scores: return 0.0
    return round(sum(1 for s in scores if s <= thr) / len(scores), 4)


def ndcg_at_k(scores: list, k: int = TOP_K_CONTEXT) -> float:
    rel = [max(0.0, 1.0 - s) for s in scores[:k]]
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rel, reverse=True)))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(vs, search_query: str, original_query: str,
             use_reranking: bool, reranker, k: int = TOP_K_SEARCH):
    results   = vs.similarity_search_with_score(search_query, k=k)
    filtered  = [(doc, s) for doc, s in results if s <= _SCORE_THRESHOLD]
    candidates = [doc for doc, _ in filtered]
    scores_raw = [s for _, s in filtered]

    # keyword fallback
    if len(candidates) < 3:
        kw_docs = keyword_search(original_query, vs, k=4)
        seen    = {d.page_content for d in candidates}
        extras  = [d for d in kw_docs if d.page_content not in seen]
        candidates += extras
        scores_raw += [_SCORE_THRESHOLD] * len(extras)   # assign threshold score to kw hits

    if not candidates:
        return "", [], []

    if use_reranking:
        pairs  = [(original_query, d.page_content) for d in candidates]
        rscores = reranker.predict(pairs)
        ranked  = sorted(zip(rscores, candidates, scores_raw),
                         key=lambda x: x[0], reverse=True)[:TOP_K_CONTEXT]
        final_docs   = [d for _, d, _ in ranked]
        final_scores = [s for _, _, s in ranked]
    else:
        final_docs   = candidates[:TOP_K_CONTEXT]
        final_scores = scores_raw[:TOP_K_CONTEXT]

    context = "\n\n".join(d.page_content for d in final_docs)
    return context, final_scores, final_docs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PM-JAY RAG Evaluation")
    print("=" * 60)

    print("\n[1/3] Loading vector store...")
    vs = create_vector_store()

    print("[2/3] Loading reranker...")
    reranker = get_reranker()

    print("[3/3] Loading embedding model...")
    embedder = get_embeddings()

    configs = [
        ("Full System",      True,  True),
        ("w/o Expansion",    False, True),
        ("w/o Reranking",    True,  False),
        ("Baseline",         False, False),
    ]

    # per-config accumulators
    agg = {n: {"ctx_prec": [], "kw_recall": [], "ndcg": [],
               "ans_rel": [], "faithful": []} for n, _, _ in configs}

    print(f"\nRunning {len(TEST_SET)} queries × {len(configs)} configs...\n")

    for qi, test in enumerate(TEST_SET):
        q_orig = test["query"]
        q_exp  = f"Ayushman Bharat PM-JAY {test['expanded']}"
        q_raw  = f"Ayushman Bharat PM-JAY {q_orig}"

        # embed query and ideal answer once
        q_emb     = np.array(embedder.embed_query(q_orig))
        ideal_emb = np.array(embedder.embed_query(test["ideal"]))

        print(f"  [{qi+1:02d}/{len(TEST_SET)}] {q_orig[:65]}")

        for cfg_name, use_exp, use_rerank in configs:
            search_q = q_exp if use_exp else q_raw
            ctx, scores, docs = retrieve(vs, search_q, q_orig, use_rerank, reranker)

            if not ctx:
                for metric in agg[cfg_name]:
                    agg[cfg_name][metric].append(0.0)
                continue

            cp  = context_precision(scores)
            kr  = keyword_recall(ctx, test["gt_kw"])
            nd  = ndcg_at_k(scores)

            # answer relevance: mean cos-sim(q_emb, chunk_emb)
            chunk_embs = [np.array(embedder.embed_query(d.page_content[:500])) for d in docs]
            mean_ctx   = np.mean(chunk_embs, axis=0)
            ar  = round(max(0.0, cosine_sim(q_emb, mean_ctx)), 4)

            # faithfulness proxy: cos-sim(ideal_emb, mean_ctx)
            fp  = round(max(0.0, cosine_sim(ideal_emb, mean_ctx)), 4)

            agg[cfg_name]["ctx_prec"].append(cp)
            agg[cfg_name]["kw_recall"].append(kr)
            agg[cfg_name]["ndcg"].append(nd)
            agg[cfg_name]["ans_rel"].append(ar)
            agg[cfg_name]["faithful"].append(fp)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Configuration':<22} {'Ctx Prec':>9} {'KW Recall':>10} {'NDCG@4':>8} {'Ans Rel':>9} {'Faithful':>10}")
    print("-" * 72)

    summary = {}
    for cfg_name, _, _ in configs:
        a = agg[cfg_name]
        cp = round(np.mean(a["ctx_prec"]),  4)
        kr = round(np.mean(a["kw_recall"]), 4)
        nd = round(np.mean(a["ndcg"]),      4)
        ar = round(np.mean(a["ans_rel"]),   4)
        fp = round(np.mean(a["faithful"]),  4)
        summary[cfg_name] = {
            "Context Precision": cp,
            "Keyword Recall":    kr,
            "NDCG@4":           nd,
            "Answer Relevance":  ar,
            "Faithfulness":      fp,
        }
        print(f"{cfg_name:<22} {cp:>9.4f} {kr:>10.4f} {nd:>8.4f} {ar:>9.4f} {fp:>10.4f}")

    print("=" * 72)

    out = {"summary": summary, "per_query_raw": agg, "test_queries": [t["query"] for t in TEST_SET]}
    with open("eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nDetailed results saved → eval_results.json")


if __name__ == "__main__":
    main()
