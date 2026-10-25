"""
service.py - FastAPI Inference Service (< 80 lines of logic)
=============================================================
Exposes TWO endpoints:
  GET  /health   - Is the service alive? Returns {"status": "ok"}
  POST /predict  - Takes job fields, returns fraud prediction

The frontend (Express) calls this service and saves results to MongoDB.

Run with:
    cd fake-job-detector
    source ml/venv/bin/activate
    uvicorn ml.service:app --port 8000 --reload

Design choice: We keep this minimal (< 60 lines of logic) so a MERN
developer can understand it without knowing FastAPI internals.
"""

import os, sys, json, re
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Paths (relative to where uvicorn is run, i.e., fake-job-detector/) ──────
MODELS_DIR = "ml/models"

# ── Load best model + vectorizers at startup ─────────────────────────────────
try:
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf.joblib"))
    ohe   = joblib.load(os.path.join(MODELS_DIR, "ohe.joblib"))
    with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()
    print(f"✅ Loaded model: {model_name}")
    MODEL_READY = True
except Exception as e:
    print(f"⚠️  Model not found: {e}. Run train.py first.")
    MODEL_READY = False

# Scam keyword list (same as preprocess.py)
SCAM_KEYWORDS = [
    "work from home", "make money fast", "no experience required",
    "earn $", "wire transfer", "western union", "money order",
    "processing fee", "training fee", "investment required", "guaranteed",
    "unlimited earning", "be your own boss", "multi-level", "mlm",
    "upfront fee", "registration fee", "starter kit",
]

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Fake Job Detector API", version="1.0")

# Allow requests from the React frontend and Express server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema (Pydantic validates input automatically) ──────────────────
class JobInput(BaseModel):
    title:              Optional[str] = ""
    company_profile:    Optional[str] = ""
    description:        Optional[str] = ""
    requirements:       Optional[str] = ""
    benefits:           Optional[str] = ""
    salary_range:       Optional[str] = "Unknown"
    employment_type:    Optional[str] = "Unknown"
    required_education: Optional[str] = "Unknown"
    required_experience:Optional[str] = "Unknown"
    telecommuting:      Optional[int] = 0
    has_company_logo:   Optional[int] = 1
    has_questions:      Optional[int] = 0


# ── Helper: build a one-row feature matrix from input ────────────────────────
def build_input_features(job: JobInput):
    """Convert a JobInput → the same feature matrix used during training."""
    import sys, os
    sys.path.insert(0, "ml/src")

    # Import the same cleaning functions used at training time
    from preprocess import clean_text, SCAM_KEYWORDS as SK, EMAIL_RE, PHONE_RE
    from features import CATEGORICAL_COLS, NUMERIC_COLS
    from scipy.sparse import hstack, csr_matrix

    # Combine text (raw, before cleaning)
    raw_text = " ".join([
        job.title or "", job.company_profile or "",
        job.description or "", job.requirements or "", job.benefits or "",
    ])
    combined_clean = clean_text(raw_text)

    # TF-IDF transform (fitted on training data)
    tfidf_vec = tfidf.transform([combined_clean])

    # One-hot encode metadata
    row_df = pd.DataFrame([{
        "employment_type":     job.employment_type or "Unknown",
        "required_experience": job.required_experience or "Unknown",
        "required_education":  job.required_education or "Unknown",
        "salary_range":        job.salary_range or "Unknown",
    }])
    ohe_vec = ohe.transform(row_df)

    # Numeric features
    scam_count = sum(1 for kw in SK if kw in raw_text.lower())
    num_row = csr_matrix([[
        job.telecommuting,
        job.has_company_logo,
        job.has_questions,
        int(bool(EMAIL_RE.search(raw_text))),
        int(bool(PHONE_RE.search(raw_text))),
        scam_count,
        1 if (not job.salary_range or job.salary_range == "Unknown") else 0,
        len(job.title or ""),
        len(job.description or ""),
        len(job.requirements or ""),
    ]])

    return hstack([tfidf_vec, ohe_vec, num_row]), raw_text, combined_clean, scam_count


# ── Helper: extract top words pushing toward fraud ────────────────────────────
def extract_top_fraud_words(combined_clean: str, top_n: int = 10) -> list:
    """
    Find which words in the input have the highest TF-IDF scores.
    These are the tokens most responsible for the fraud prediction.
    """
    tfidf_vec = tfidf.transform([combined_clean])
    vocab = tfidf.get_feature_names_out()
    scores = tfidf_vec.toarray()[0]
    top_idx = scores.argsort()[::-1][:top_n]
    return [
        {"word": str(vocab[i]), "score": round(float(scores[i]), 4)}
        for i in top_idx if scores[i] > 0
    ]


# ── Helper: detect red flags ─────────────────────────────────────────────────
def detect_red_flags(job: JobInput, raw_text: str, scam_count: int) -> list:
    flags = []
    if not job.has_company_logo:
        flags.append("No company logo")
    if not job.salary_range or job.salary_range == "Unknown":
        flags.append("Salary not disclosed")
    if scam_count > 0:
        flags.append(f"Contains {scam_count} scam keyword(s)")
    if re.search(r"fee|pay.*registration|send.*money", raw_text, re.I):
        flags.append("Possible fee/payment request detected")
    if re.search(r"\b[A-Za-z0-9._%+-]+@(gmail|yahoo|hotmail|outlook)\.", raw_text, re.I):
        flags.append("Personal email address detected (not corporate)")
    if re.search(r"\b\d[\d\s\-\(\)]{7,}\d\b", raw_text):
        flags.append("Phone number present in posting")
    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Simple health check. Express server pings this on startup."""
    return {"status": "ok", "model": model_name if MODEL_READY else "not loaded"}


@app.post("/predict")
def predict(job: JobInput):
    """
    Main prediction endpoint.
    Returns:
      label              : "Fraudulent" | "Real"
      fraud_probability  : float 0-1
      top_words          : list of {word, score}
      red_flags          : list of warning strings
    """
    if not MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python ml/src/train.py first.",
        )

    X, raw_text, combined_clean, scam_count = build_input_features(job)

    # Predict probability (all our saved models support predict_proba)
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    else:
        prob = float(model.predict(X)[0])

    label = "Fraudulent" if prob >= 0.5 else "Real"
    top_words = extract_top_fraud_words(combined_clean)
    red_flags  = detect_red_flags(job, raw_text, scam_count)

    return {
        "label":             label,
        "fraud_probability": round(prob, 4),
        "top_words":         top_words,
        "red_flags":         red_flags,
    }
