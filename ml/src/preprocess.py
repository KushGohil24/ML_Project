"""
preprocess.py - Data cleaning & feature engineering for Fake Job Detection
==========================================================================
This module handles everything BEFORE training:
  1. Loading the raw CSV
  2. Filling missing values
  3. Merging all text fields into one "combined_text" column
  4. Cleaning that text (HTML, URLs, stopwords, lemmatisation)
  5. Engineering extra numeric features
  6. Returning train/test splits (stratified, no data leakage)

A beginner tip: NEVER fit any transformer (TF-IDF, scaler) on test data.
Fit only on train, then *transform* both. This avoids "data leakage".
"""

import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---------- NLTK (optional, falls back gracefully) ----------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    # Download required NLTK data quietly
    for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        nltk.download(pkg, quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except Exception:
    STOP_WORDS = set()
    lemmatizer = None
    NLTK_AVAILABLE = False

# Scam keywords often found in fraudulent postings
SCAM_KEYWORDS = [
    "work from home", "make money fast", "no experience required",
    "earn $", "wire transfer", "western union", "money order",
    "processing fee", "training fee", "investment required", "guaranteed",
    "unlimited earning", "be your own boss", "multi-level", "mlm",
    "upfront fee", "registration fee", "starter kit",
]

# Regex patterns
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d[\d\s\-\(\)]{7,}\d\b")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & first-pass cleaning
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Read CSV, handle obvious issues, return cleaned DataFrame."""
    df = pd.read_csv(csv_path)

    # Columns that should be text
    text_cols = [
        "title", "location", "department", "company_profile",
        "description", "requirements", "benefits",
        "required_qualifications",  # may not exist
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Binary / categorical columns
    cat_cols = [
        "employment_type", "required_experience",
        "required_education", "industry", "function",
        "salary_range",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Numeric / boolean columns
    bool_cols = ["telecommuting", "has_company_logo", "has_questions"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hand-crafted numeric features before vectorisation."""

    # --- Merge all free-text columns into one string ---
    text_parts = ["title", "company_profile", "description", "requirements", "benefits"]
    existing_parts = [c for c in text_parts if c in df.columns]
    df["combined_text"] = df[existing_parts].apply(
        lambda row: " ".join(str(v) for v in row.values), axis=1
    )

    # --- Boolean / email / phone features (raw text) ---
    df["has_email_in_text"] = df["combined_text"].apply(
        lambda t: int(bool(EMAIL_RE.search(t)))
    )
    df["has_phone_in_text"] = df["combined_text"].apply(
        lambda t: int(bool(PHONE_RE.search(t)))
    )

    # --- Scam keyword count (raw, before cleaning) ---
    def count_scam_kw(text: str) -> int:
        tl = text.lower()
        return sum(1 for kw in SCAM_KEYWORDS if kw in tl)

    df["scam_keyword_count"] = df["combined_text"].apply(count_scam_kw)

    # --- Missing salary flag ---
    if "salary_range" in df.columns:
        df["missing_salary"] = (df["salary_range"] == "Unknown").astype(int)
    else:
        df["missing_salary"] = 1

    # --- Text length features ---
    df["title_len"] = df["title"].apply(len) if "title" in df.columns else 0
    df["desc_len"] = df["description"].apply(len) if "description" in df.columns else 0
    df["req_len"] = df["requirements"].apply(len) if "requirements" in df.columns else 0

    # --- Clean combined text AFTER extracting raw features ---
    df["combined_text"] = df["combined_text"].apply(clean_text)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Text cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strip HTML, URLs, punctuation, stopwords; lemmatise."""
    text = HTML_RE.sub(" ", text)          # remove HTML tags
    text = URL_RE.sub(" ", text)           # remove URLs
    text = text.lower()                    # lowercase
    text = re.sub(r"\d+", " ", text)       # remove numbers
    text = text.translate(                  # remove punctuation
        str.maketrans("", "", string.punctuation)
    )
    tokens = text.split()
    if NLTK_AVAILABLE:
        tokens = [t for t in tokens if t not in STOP_WORDS]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stratified train / test split
# ─────────────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame, target: str = "fraudulent", test_size: float = 0.2):
    """
    Stratified 80/20 split. Returns (X_train, X_test, y_train, y_test).
    Stratify ensures both splits have the same ~5% fraud ratio.
    """
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
