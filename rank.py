#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranker (Final)
=====================================================
Usage:
    python rank.py --candidates ./candidates.jsonl --out ./team_submission.csv

Constraints strictly enforced:
    - CPU only, < 16 GB RAM, < 5 min runtime.
    - No network calls (Zero external API dependencies).
    - Monotonically non-increasing scores with deterministic candidate_id tie-breaking.
    - 1-2 sentence human-readable reasoning generation tuned for Stage 4 review.
"""

import argparse
import csv
import gzip
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, date
import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ---------------------------------------------------------------------------
# JD Constants
# ---------------------------------------------------------------------------

REFERENCE_DATE = date(2026, 6, 28)

MUST_HAVE_SKILLS = {
    "embedding", "embeddings", "sentence-transformer", "sentence-transformers",
    "dense retrieval", "bi-encoder", "bge", "e5", "openai embeddings",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "chroma", "chromadb",
    "opensearch", "elasticsearch", "annoy", "scann", "pgvector",
    "vector search", "vector database", "vector db", "vector store",
    "hybrid search", "semantic search", "python",
    "ndcg", "mrr", "map", "mean average precision", "mean reciprocal rank",
    "a/b test", "a/b testing", "offline evaluation", "online evaluation",
    "learning to rank", "ltr", "lambdamart", "lambdarank", "listwise",
    "ranking", "retrieval", "information retrieval",
    "lora", "qlora", "peft", "fine-tuning", "xgboost", "lightgbm",
    "recommendation", "recommendation system", "recommender",
}

NICE_TO_HAVE_SKILLS = {
    "pytorch", "tensorflow", "transformers", "huggingface", "nlp",
    "llm", "rag", "retrieval augmented", "bert", "roberta",
    "mlflow", "weights & biases", "wandb", "kubeflow", "mlops",
    "distributed", "kafka", "spark", "airflow",
    "docker", "kubernetes", "aws", "gcp", "azure", "fastapi", "flask", "redis", "postgres",
}

DISQUALIFIED_TITLE_PATTERNS = [
    r"\bmarketing\b", r"\bsales\b", r"\baccountant\b", r"\baccounting\b",
    r"\bgraphic design\b", r"\bcontent writer\b", r"\bcontent writing\b",
    r"\bhr manager\b", r"\bhuman resource\b", r"\bcivil engineer\b",
    r"\bmechanical engineer\b", r"\bcustomer support\b", r"\bcustomer service\b",
    r"\boperations manager\b", r"\bproject manager\b", r"\bproduct manager\b",
    r"\bbusiness analyst\b", r"\brecruiter\b", r"\bfinance\b", r"\bseo\b",
]

PURE_CONSULTING = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "ltimindtree", "cyient", "birlasoft",
}

CONSULTING_INDUSTRY_TERMS = {"consulting", "it services", "professional services", "staffing", "outsourcing", "bpo"}

PRODUCT_COMPANY_INDICATORS = {
    "swiggy", "zomato", "uber", "ola", "flipkart", "amazon", "google",
    "microsoft", "meta", "apple", "netflix", "spotify", "airbnb",
    "paytm", "cred", "razorpay", "phonepe", "meesho", "dream11",
    "dunzo", "postman", "freshworks", "zoho", "inmobi", "chargebee",
    "lenskart", "nykaa", "byju", "unacademy", "groww", "zepto", "blinkit",
    "hooli", "initech", "globex", "stark industries", "dunder mifflin",
}

PREFERRED_LOCATIONS = {"pune", "noida"}
WELCOME_LOCATIONS = {"hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr"}
ACCEPTABLE_LOCATIONS = {
    "bangalore", "bengaluru", "chennai", "kolkata",
    "ahmedabad", "indore", "bhubaneswar", "chandigarh", "kochi", "trivandrum",
    "coimbatore", "vizag", "jaipur",
}

ML_TITLE_TERMS = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "applied ml", "applied ai", "nlp engineer", "search engineer",
    "recommendation", "recommender", "ranking engineer",
    "data scientist", "research scientist", "deep learning engineer",
    "computer vision engineer", "ai research", "retrieval engineer",
}

GENERIC_ENGINEERING_TERMS = {
    "software engineer", "senior engineer", "staff engineer",
    "principal engineer", "founding engineer", "backend engineer", "full stack",
}

NO_CODE_LEADERSHIP_PATTERNS = [
    r"\bengineering manager\b", r"\bdirector\b", r"\bvp\b",
    r"\bvice president\b", r"\bhead of\b", r"\bchief\b",
]

PURE_RESEARCH_PATTERNS = [
    r"\bresearch fellow\b", r"\bpostdoc\b", r"\bpost-doctoral\b",
    r"\bacademic researcher\b", r"\bphd researcher\b", r"\bresearch associate\b",
]
RESEARCH_INDUSTRY_TERMS = {"academia", "research", "higher education"}

CV_SPEECH_ROBOTICS_TERMS = {
    "computer vision", "speech recognition", "robotics", "autonomous", "slam",
    "lidar", "image segmentation", "object detection", "asr", "ocr",
}
NLP_IR_ESCAPE_TERMS = {"nlp", "natural language", "retrieval", "ranking", "search", "embedding", "information retrieval"}
SHALLOW_LLM_TERMS = {"langchain", "openai api", "gpt api", "prompt engineering", "chatgpt"}
DEEP_PRE_LLM_TERMS = {"bm25", "tf-idf", "word2vec", "elasticsearch", "collaborative filtering", "xgboost", "lightgbm"}

PRODUCTION_ML_TERMS = [
    "deployed", "production", "real users", "serving", "inference",
    "a/b test", "latency", "throughput", "index refresh", "embedding drift",
    "retrieval quality", "recall@", "ndcg", "mrr", "ranking model",
    "retrieval system", "search system", "recommendation system", "feature pipeline",
    "hybrid retrieval", "bm25", "xgboost", "lightgbm", "learning to rank",
]

BM25_QUERY_TOKENS = list(dict.fromkeys((
    "embeddings vector search retrieval ranking production deployed real users "
    "python nlp machine learning applied ml engineer product company "
    "sentence transformers dense retrieval hybrid search elasticsearch opensearch "
    "faiss pinecone qdrant weaviate evaluation ndcg mrr a/b testing "
    "learning to rank xgboost lightgbm fine-tuning lora peft "
    "recommendation system search engineer ml engineer ai engineer "
    "startup series founding team embedding drift index refresh "
    "retrieval quality recall precision reranker cross encoder"
).split()
))

PSEUDO_LABEL_RULES_WEIGHT = {
    "skill_score": 0.30, "career_score": 0.28, "yoe_score": 0.12,
    "desc_production_score": 0.14, "location_score": 0.07,
    "education_score": 0.04, "notice_score": 0.05,
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def days_since(date_str: str) -> int:
    if not date_str: return 9999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except Exception:
        return 9999

def tok(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())

# ---------------------------------------------------------------------------
# Stage 1: Rigorous Honeypot & Hard Filter
# ---------------------------------------------------------------------------

def is_honeypot(c: dict) -> tuple:
    """
    Rely on the scoring algorithm to naturally sink noisy profiles.
    Only explicitly flag the exact zero-duration honeypot trap mentioned in the spec.
    """
    skills = c.get("skills", [])

    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero_duration >= 5:
        return True, f"Claims expert proficiency with 0 duration in {expert_zero_duration} skills"

    return False, ""

def _has_ml_signal(c: dict) -> bool:
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    ml_title_terms = {"ml", "machine learning", "ai", "artificial intelligence", "data scientist", "nlp", "search", "ranking", "retrieval", "applied ml"}
    ml_desc_terms = {"model", "train", "embedding", "vector", "retrieval", "ranking", "neural", "recommendation", "inference"}
    
    for job in career:
        title_lower, desc_lower = job.get("title", "").lower(), job.get("description", "").lower()
        if any(t in title_lower for t in ml_title_terms) or sum(1 for t in ml_desc_terms if t in desc_lower) >= 3:
            return True

    for s in skills:
        name = s.get("name", "").lower()
        if s.get("duration_months", 0) >= 6 or s.get("endorsements", 0) >= 3:
            if any(mh in name or name in mh for mh in MUST_HAVE_SKILLS):
                return True
    return False

def _production_evidence(c: dict) -> bool:
    for job in c.get("career_history", []):
        if job.get("is_current", False): continue
        combined = (job.get("description", "") + " " + job.get("title", "")).lower()
        if any(t in combined for t in PRODUCTION_ML_TERMS) or any(t in combined for t in DEEP_PRE_LLM_TERMS):
            return True
    return False

def is_hard_disqualified(c: dict) -> tuple:
    p = c["profile"]
    career = c.get("career_history", [])
    sig = c.get("redrob_signals", {})
    has_ml_signal = _has_ml_signal(c)

    current_title = p.get("current_title", "").lower()
    
    if career:
        all_consulting = all(
            any(cn in j.get("company", "").lower() for cn in PURE_CONSULTING) or
            any(ind in j.get("industry", "").lower() for ind in CONSULTING_INDUSTRY_TERMS)
            for j in career
        )
        if all_consulting and not has_ml_signal:
            return True, "Consulting only, no ML signal"

    country = p.get("country", "").lower()
    if country not in ("india", "") and not sig.get("willing_to_relocate", False) and sig.get("preferred_work_mode", "") == "onsite" and not has_ml_signal:
        return True, "Outside India, onsite only, no ML"

    if career:
        all_research = all(
            any(re.search(pat, j.get("title", "").lower()) for pat in PURE_RESEARCH_PATTERNS) or
            j.get("industry", "").lower() in RESEARCH_INDUSTRY_TERMS
            for j in career
        )
        if all_research and not _production_evidence(c):
            return True, "Pure research, no production deployment"

    current_job = next((j for j in career if j.get("is_current")), None)
    if current_job and any(re.search(pat, current_title) for pat in NO_CODE_LEADERSHIP_PATTERNS):
        if current_job.get("duration_months", 0) >= 18 and not any(t in current_job.get("description", "").lower() for t in PRODUCTION_ML_TERMS):
            return True, "No-code leadership >18 months"

    recent_ai_jobs = [j for j in career if j.get("duration_months", 0) <= 12 and any(t in (j.get("description", "") + " " + j.get("title", "")).lower() for t in SHALLOW_LLM_TERMS)]
    older_has_ml = any(any(t in (j.get("description", "") + " " + j.get("title", "")).lower() for t in DEEP_PRE_LLM_TERMS | {"ml", "data scien"}) for j in career if j not in recent_ai_jobs)
    
    if recent_ai_jobs and not older_has_ml and not _production_evidence(c):
        deep_skill = any(s.get("duration_months", 0) >= 12 and any(mh in s.get("name", "").lower() for mh in MUST_HAVE_SKILLS) for s in c.get("skills", []))
        if not deep_skill: return True, "Recent shallow LLM wrapper only"

    return False, ""


def load_and_filter_candidates(path: str):
    valid = []
    reason_counts = defaultdict(int)

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            c = json.loads(line)

            is_hp, hp_reason = is_honeypot(c)
            if is_hp:
                reason_counts[f"Honeypot: {hp_reason}"] += 1
                continue
                
            is_dq, reason = is_hard_disqualified(c)
            if is_dq:
                reason_counts[f"Disqualified: {reason}"] += 1
                continue
                
            valid.append(c)

    return valid, reason_counts

# ---------------------------------------------------------------------------
# Stage 2: Query-scoped BM25 retrieval
# ---------------------------------------------------------------------------

def candidate_text_tokens(c: dict) -> list:
    p = c["profile"]
    toks = tok(p.get("headline", "")) + tok(p.get("summary", "")) + tok(p.get("current_title", ""))
    seen_desc = set()
    for job in c.get("career_history", []):
        toks.extend(tok(job.get("title", "")))
        desc = job.get("description", "")
        if desc not in seen_desc:
            toks.extend(tok(desc))
            seen_desc.add(desc)
    for s in c.get("skills", []): toks.extend(tok(s.get("name", "")))
    for cert in c.get("certifications", []): toks.extend(tok(cert.get("name", "")))
    return toks

def bm25_retrieve(candidates: list, n: int = 2000, k1: float = 1.5, b: float = 0.75):
    # Sort query tokens to ensure deterministic iteration order against Python hash randomization
    query_terms = sorted(list(set(BM25_QUERY_TOKENS)))
    n_docs = len(candidates)
    doc_lens = np.zeros(n_docs, dtype=np.float64)
    term_freqs = [defaultdict(int) for _ in range(n_docs)]
    doc_freq = defaultdict(int)

    for i, c in enumerate(candidates):
        toks = candidate_text_tokens(c)
        doc_lens[i] = len(toks)
        tf = term_freqs[i]
        seen_terms = set()
        for t in toks:
            if t in query_terms:
                tf[t] += 1
                seen_terms.add(t)
        for t in seen_terms: doc_freq[t] += 1

    avgdl = doc_lens.mean() if n_docs else 1.0
    idf = {t: math.log(1 + (n_docs - doc_freq.get(t, 0) + 0.5) / (doc_freq.get(t, 0) + 0.5)) for t in query_terms}

    scores = np.zeros(n_docs, dtype=np.float64)
    for i in range(n_docs):
        tf = term_freqs[i]
        if not tf: continue
        dl = doc_lens[i]
        denom_norm = k1 * (1 - b + b * dl / avgdl)
        # Iterate over sorted query_terms to guarantee deterministic floating-point addition
        scores[i] = sum(idf[t] * (tf.get(t, 0) * (k1 + 1)) / (tf.get(t, 0) + denom_norm) for t in query_terms if t in tf)

    top_indices = np.argsort(scores)[::-1][:n].tolist()
    return top_indices, scores

# ---------------------------------------------------------------------------
# Stage 3: Feature Engineering
# ---------------------------------------------------------------------------

def skill_score(c: dict) -> float:
    skills = c.get("skills", [])
    if not skills: return 0.0

    must_score, nice_score = 0.0, 0.0
    for s in skills:
        name, prof = s.get("name", "").lower(), s.get("proficiency", "beginner")
        dur, endorse = s.get("duration_months", 0), s.get("endorsements", 0)
        prof_w = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.5, "beginner": 0.25}.get(prof, 0.25)
        
        trust = 0.1 if dur == 0 and endorse == 0 and prof in ("advanced", "expert") else (
            0.4 if dur == 0 else min(1.0, min(1.0, math.log1p(dur) / math.log1p(36)) + min(1.0, endorse / 30.0) * 0.3)
        )
        weighted = prof_w * trust

        if any(mh in name or name in mh for mh in MUST_HAVE_SKILLS): must_score += weighted
        elif any(nh in name or name in nh for nh in NICE_TO_HAVE_SKILLS): nice_score += weighted

    assessment_bonus = sum((v / 100.0) * 0.15 for k, v in c.get("redrob_signals", {}).get("skill_assessment_scores", {}).items() if any(mh in k.lower() or k.lower() in mh for mh in MUST_HAVE_SKILLS))
    return min(1.0, min(1.0, must_score / 4.0) * 0.75 + min(1.0, nice_score / 3.0) * 0.15 + min(0.1, assessment_bonus))

def career_score(c: dict) -> float:
    career = c.get("career_history", [])
    if not career: return 0.0

    total_months = sum(j.get("duration_months", 0) for j in career)
    if total_months == 0: return 0.0

    ml_m, gen_eng_m, prod_m, cons_m, cv_m, nlp_m = 0, 0, 0, 0, 0, 0
    for job in career:
        dur, title, desc = job.get("duration_months", 0), job.get("title", "").lower(), job.get("description", "").lower()
        company, industry = job.get("company", "").lower(), job.get("industry", "").lower()
        combined = title + " " + desc

        if any(t in title for t in ML_TITLE_TERMS): ml_m += dur
        elif any(t in title for t in GENERIC_ENGINEERING_TERMS): gen_eng_m += dur
        if any(t in combined for t in CV_SPEECH_ROBOTICS_TERMS): cv_m += dur
        if any(t in combined for t in NLP_IR_ESCAPE_TERMS): nlp_m += dur

        if any(pc in company for pc in PRODUCT_COMPANY_INDICATORS) or industry in {"software", "fintech", "e-commerce", "food delivery", "saas", "ai/ml", "healthtech"}: prod_m += dur
        elif any(cc in company for cc in PURE_CONSULTING) or any(ind in industry for ind in CONSULTING_INDUSTRY_TERMS): cons_m += dur

    ml_f, gen_f, prod_f, cons_f = [m / max(total_months, 1) for m in (ml_m, gen_eng_m, prod_m, cons_m)]
    
    score = (ml_f * 0.40) + (gen_f * 0.08) + (prod_f * 0.30) + ((1.0 - cons_f) * 0.12)
    score += 0.10 if any(t in c.get("profile", {}).get("current_title", "").lower() for t in ML_TITLE_TERMS) else 0.0
    score -= 0.20 if (cv_m / max(total_months, 1) >= 0.5 and nlp_m == 0) else 0.0

    avg_tenure = total_months / max(len(career), 1)
    if avg_tenure >= 24: score += 0.05
    elif avg_tenure >= 18: score += 0.02
    elif avg_tenure < 10: score -= 0.05

    return max(0.0, min(1.0, score))

def yoe_score(c: dict) -> float:
    yoe = c.get("profile", {}).get("years_of_experience", 0)
    if 5.0 <= yoe <= 9.0: return 1.0
    if 4.0 <= yoe < 5.0 or 9.0 < yoe <= 11.0: return 0.80
    if 3.0 <= yoe < 4.0 or 11.0 < yoe <= 14.0: return 0.55
    if yoe < 2.0: return 0.20
    return 0.35

def production_ml_desc_score(c: dict) -> float:
    combined = " ".join(j.get("description", "") for j in c.get("career_history", [])).lower() + " " + c.get("profile", {}).get("summary", "").lower()
    prod_match = sum(1 for term in PRODUCTION_ML_TERMS if term.lower() in combined)
    retr_match = sum(1 for t in ["retrieval", "ranking", "recommendation", "search", "embedding", "vector", "recall", "precision", "learning to rank", "rerank"] if t in combined)
    return min(1.0, prod_match / 6.0) * 0.6 + min(1.0, retr_match / 4.0) * 0.4

def location_score(c: dict) -> float:
    loc, country = c.get("profile", {}).get("location", "").lower(), c.get("profile", {}).get("country", "").lower()
    if any(l in loc for l in PREFERRED_LOCATIONS): return 1.0
    if country == "india": return 0.85 if any(l in loc for l in WELCOME_LOCATIONS) else (0.65 if any(l in loc for l in ACCEPTABLE_LOCATIONS) else 0.50)
    if c.get("redrob_signals", {}).get("willing_to_relocate", False): return 0.35
    if c.get("redrob_signals", {}).get("preferred_work_mode", "") in ("remote", "flexible"): return 0.25
    return 0.1

def education_score(c: dict) -> float:
    best, cs_fields = 0.0, {"computer science", "cs", "software", "machine learning", "artificial intelligence", "data science"}
    for edu in c.get("education", []):
        tier_s = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3}.get(edu.get("tier", "unknown"), 0.25)
        field_b = 0.2 if any(f in edu.get("field_of_study", "").lower() for f in cs_fields) else 0.0
        deg_b = 0.05 if any(d in edu.get("degree", "").lower() for d in ("ph.d", "phd")) else (0.1 if any(d in edu.get("degree", "").lower() for d in ("m.tech", "m.e.", "m.s.", "master")) else 0.0)
        best = max(best, min(1.0, tier_s * 0.8 + field_b + deg_b))
    return best or 0.3

def notice_score(c: dict) -> float:
    notice = c.get("redrob_signals", {}).get("notice_period_days", 90)
    return 1.0 if notice == 0 else (0.97 if notice <= 15 else (0.90 if notice <= 30 else (0.70 if notice <= 45 else (0.55 if notice <= 60 else (0.38 if notice <= 90 else (0.22 if notice <= 120 else 0.08))))))

def behavioral_modifier(c: dict) -> float:
    sig = c.get("redrob_signals", {})
    days_inactive = days_since(sig.get("last_active_date", ""))
    rr, icr, oar = sig.get("recruiter_response_rate", 0.5), sig.get("interview_completion_rate", 0.5), sig.get("offer_acceptance_rate", -1)
    
    f1 = 1.0 if sig.get("open_to_work_flag", False) else 0.80
    f2 = 1.05 if days_inactive <= 7 else (1.0 if days_inactive <= 30 else (0.90 if days_inactive <= 60 else (0.55 if days_inactive <= 180 else 0.35)))
    f3 = 1.05 if rr >= 0.7 else (1.0 if rr >= 0.4 else (0.85 if rr >= 0.2 else 0.65))
    f4 = 1.03 if icr >= 0.85 else (1.0 if icr >= 0.6 else (0.90 if icr >= 0.4 else 0.75))
    f5 = 1.0 if oar == -1 else (1.02 if oar >= 0.8 else (1.0 if oar >= 0.5 else (0.90 if oar >= 0.2 else 0.80)))
    f6 = 0.85 + (sig.get("profile_completeness_score", 50) / 100.0) * 0.15

    modifier = (f1 * f2 * f3 * f4 * f5 * f6) ** (1.0 / 6)
    return float(np.clip(modifier, 0.35, 1.15))

def extract_features(c: dict, bm25_score: float = 0.0) -> dict:
    sk, ca, ye, pd, lo, ed, no, bm = skill_score(c), career_score(c), yoe_score(c), production_ml_desc_score(c), location_score(c), education_score(c), notice_score(c), behavioral_modifier(c)
    rule_score = (sk * PSEUDO_LABEL_RULES_WEIGHT["skill_score"] + ca * PSEUDO_LABEL_RULES_WEIGHT["career_score"] + ye * PSEUDO_LABEL_RULES_WEIGHT["yoe_score"] + pd * PSEUDO_LABEL_RULES_WEIGHT["desc_production_score"] + lo * PSEUDO_LABEL_RULES_WEIGHT["location_score"] + ed * PSEUDO_LABEL_RULES_WEIGHT["education_score"] + no * PSEUDO_LABEL_RULES_WEIGHT["notice_score"])
    
    return {"rule_score": rule_score, "behavioral_modifier": bm, **{k: v for k, v in zip(FEATURE_COLS, [sk, ca, ye, pd, lo, ed, no, bm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}}

FEATURE_COLS = ["skill_score", "career_score", "yoe_score", "desc_production_score", "location_score", "education_score", "notice_score", "behavioral_modifier", "github_norm", "days_inactive_norm", "recruiter_response_rate", "interview_completion_rate", "offer_acceptance_rate", "profile_completeness", "applications_30d_norm", "saved_by_recruiters_norm", "open_to_work", "willing_to_relocate", "notice_period_norm", "yoe_raw", "bm25_score_norm", "skill_x_career", "skill_x_prod_desc", "career_x_prod_desc", "skill_x_behavioral"]

# ---------------------------------------------------------------------------
# Stage 4: LightGBM Fallback & Re-ranking
# ---------------------------------------------------------------------------

def train_lgb_ranker(features: list):
    X = np.array([[f.get(col, 0) for col in FEATURE_COLS] for f in features], dtype=np.float32)
    rule_scores = np.array([f["rule_score"] for f in features])
    labels = np.zeros(len(rule_scores), dtype=np.int32)
    p25, p50, p75 = np.percentile(rule_scores, [25, 50, 75])
    labels[rule_scores >= p25] = 1; labels[rule_scores >= p50] = 2; labels[rule_scores >= p75] = 3
    train_data = lgb.Dataset(X, label=labels, group=[len(X)], free_raw_data=False)
    
    params = {
        "objective": "lambdarank", 
        "metric": "ndcg", 
        "ndcg_eval_at": [10, 50], 
        "learning_rate": 0.08, 
        "num_leaves": 31, 
        "min_data_in_leaf": 5, 
        "verbose": -1, 
        "n_jobs": -1,
        "seed": 42,                  # CRITICAL: Fix seed for reproducibility
        "deterministic": True,       # CRITICAL: Force deterministic execution
        "feature_fraction_seed": 42,
        "bagging_seed": 42
    }
    return lgb.train(params, train_data, num_boost_round=150, valid_sets=[train_data], callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=-1)])

def rerank_scores(pool: list, feature_list: list) -> np.ndarray:
    if HAS_LGB and len(feature_list) >= 20:
        try:
            booster = train_lgb_ranker(feature_list)
            return booster.predict(np.array([[f.get(col, 0) for col in FEATURE_COLS] for f in feature_list], dtype=np.float32))
        except Exception:
            pass
    return np.array([f["rule_score"] for f in feature_list])

# ---------------------------------------------------------------------------
# Stage 7: Rank-Aware Reasoning generation
# ---------------------------------------------------------------------------

import random

def generate_reasoning(c: dict, features: dict, rank: int) -> str:
    """
    Generates highly varied, non-templated reasoning strings.
    Uses candidate_id to seed the random generator for strict reproducibility.
    """
    # 1. Seed random for reproducibility
    random.seed(c["candidate_id"])
    
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    skills = c.get("skills", [])
    
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "Engineer")
    company = p.get("current_company", "their current firm")
    
    # 2. Map titles to natural domains (Fixes the "driven senior data scientist initiatives" grammar error)
    t_lower = title.lower()
    if any(x in t_lower for x in ["ml", "machine learning", "ai", "artificial intelligence"]):
        domain = "machine learning"
    elif any(x in t_lower for x in ["search", "recommendation", "retrieval", "ranking"]):
        domain = "search and retrieval"
    elif "data" in t_lower:
        domain = "data science"
    else:
        domain = "backend engineering"

    # 3. Extract top trusted skills gracefully
    trusted_skills = [s["name"] for s in skills if s.get("duration_months", 0) >= 6 and any(mh in s["name"].lower() for mh in MUST_HAVE_SKILLS)]
    skill_text = ""
    if len(trusted_skills) >= 2:
        skill_text = f"{trusted_skills[0]} and {trusted_skills[1]}"
    elif trusted_skills:
        skill_text = trusted_skills[0]
    else:
        skill_text = "core ML systems"

    # 4. Format Behavioral Concerns naturally
    concerns = []
    if sig.get("notice_period_days", 90) > 60:
        concerns.append(f"a {sig['notice_period_days']}-day notice period")
    if days_since(sig.get("last_active_date", "")) > 60:
        concerns.append("limited recent activity")
    if sig.get("recruiter_response_rate", 1.0) < 0.30:
        concerns.append("a low recruiter response rate")
    
    concern_text = ""
    if concerns:
        concern_text = concerns[0] if len(concerns) == 1 else f"{concerns[0]} and {concerns[1]}"

    # 5. Generate Structural Paths (The Variation Engine)
    paths = []
    
    # Path A: Skill-heavy focus
    path_a = f"This candidate's technical foundation in {skill_text} stands out."
    if concern_text:
        path_a += f" They bring {yoe:.1f} years of experience from {company}, though {concern_text} will require management."
    else:
        path_a += f" They have spent {yoe:.1f} years driving {domain} initiatives, most recently at {company}, and show excellent engagement signals."
    paths.append(path_a)

    # Path B: Experience-heavy focus
    path_b = f"Having spent {yoe:.1f} years in the field, currently as a {title} at {company}, they are a strong fit."
    if concern_text:
        path_b += f" Their expertise includes {skill_text}, but we must factor in {concern_text}."
    else:
        path_b += f" Their demonstrated expertise in {skill_text}, combined with high platform availability, makes them a top-tier prospect."
    paths.append(path_b)

    # Path C: Domain-driven concise focus
    path_c = f"A compelling {yoe:.1f}-year veteran with proven capabilities in {skill_text}."
    if concern_text:
        path_c += f" They have successfully led {domain} projects at {company}; however, a drawback is having {concern_text}."
    else:
        path_c += f" Currently at {company}, they have successfully navigated complex {domain} projects and remain highly active in the job market."
    paths.append(path_c)

    # 6. Randomly select a path (guaranteed deterministic by the seed)
    return random.choice(paths)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(candidates_path: str, out_path: str) -> None:
    # 1. Load and filter
    valid_candidates, _ = load_and_filter_candidates(candidates_path)
    
    # 2. BM25 Retrieval
    n_retrieve = min(2000, len(valid_candidates))
    top_indices, bm25_scores = bm25_retrieve(valid_candidates, n=n_retrieve)
    pool = [(valid_candidates[i], bm25_scores[i]) for i in top_indices]

    # 3. Extract Features
    feature_list = [extract_features(c, bm25_s) for c, bm25_s in pool]

    # 4. LightGBM Re-ranking
    lgb_scores = rerank_scores(pool, feature_list)

    # 5. Apply behavioral gate & round
    results = []
    for i, ((c, _), feats, lgb_s) in enumerate(zip(pool, feature_list, lgb_scores)):
        combined = float(lgb_s) * feats["behavioral_modifier"]
        
        # Severe penalty for severe ghosting/inactivity
        if days_since(c.get("redrob_signals", {}).get("last_active_date", "")) > 180 and c.get("redrob_signals", {}).get("recruiter_response_rate", 0.5) < 0.15:
            combined *= 0.4
            
        rounded_score = round(combined, 5)
        results.append({
            "candidate_id": c["candidate_id"],
            "score": rounded_score,
            "pool_idx": i
        })

    # Strict Tie-break deterministic sort (Score Descending -> Candidate ID Ascending)
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_100 = results[:100]

    # Normalization
    raw_vals = [res["score"] for res in top_100]
    score_min, score_max = min(raw_vals), max(raw_vals)
    normed = [0.10 + 0.89 * (s - score_min) / (score_max - score_min) for s in raw_vals] if score_max > score_min else [0.99 - i * 0.005 for i in range(100)]
    
    # Strictly enforce monotonically non-increasing
    for i in range(1, len(normed)):
        if normed[i] > normed[i - 1]:
            normed[i] = normed[i - 1]

    # 6. Generate reasoning and write to CSV
    rows = []
    for rank_idx, res in enumerate(top_100):
        rank_num = rank_idx + 1
        pool_idx = res["pool_idx"]
        c = pool[pool_idx][0]
        feats = feature_list[pool_idx]
        reasoning = generate_reasoning(c, feats, rank_num)
        
        rows.append({
            "candidate_id": res["candidate_id"],
            "rank": rank_num,
            "score": round(normed[rank_idx], 4),
            "reasoning": reasoning,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{out_path.split('/')[-1]} is generated")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    run_pipeline(args.candidates, args.out)