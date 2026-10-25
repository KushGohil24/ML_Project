"""
features.py - Vectorization & feature matrix construction
==========================================================
Converts cleaned text + metadata into a numeric matrix that ML models
can understand.

Key concepts for a beginner:
  - TF-IDF: Turns words into numbers based on how often they appear
    in a doc vs. the whole corpus. Words like "the" get low scores.
  - One-Hot Encoding: Turns categories (e.g., "Full-time") into 0/1 columns.
  - scipy.sparse: Efficient matrix format when most values are 0.

IMPORTANT: All transformers are FIT on training data only, then applied
to test data. This is how you prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
import joblib
import os

# Metadata columns that will be one-hot encoded
CATEGORICAL_COLS = [
    "employment_type",
    "required_experience",
    "required_education",
    "salary_range",   # already has "Unknown" for missing
]

# Numeric columns engineered in preprocess.py
NUMERIC_COLS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "has_email_in_text",
    "has_phone_in_text",
    "scam_keyword_count",
    "missing_salary",
    "title_len",
    "desc_len",
    "req_len",
]


def build_feature_matrix(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    models_dir: str = "ml/models",
    fit: bool = True,
):
    """
    Build TF-IDF + one-hot + numeric feature matrices.

    Parameters
    ----------
    X_train, X_test : DataFrames (output of split_data)
    models_dir      : Where to save/load fitted vectorizers
    fit             : True = fit on X_train (training phase)
                      False = load saved vectorizers (inference phase)

    Returns
    -------
    X_train_feat, X_test_feat : scipy sparse matrices
    tfidf, ohe                : fitted transformers
    """
    os.makedirs(models_dir, exist_ok=True)
    tfidf_path = os.path.join(models_dir, "tfidf.joblib")
    ohe_path   = os.path.join(models_dir, "ohe.joblib")

    # ── 1. TF-IDF on combined_text ────────────────────────────────────────
    if fit:
        tfidf = TfidfVectorizer(
            max_features=10_000,   # keep top 10k terms
            ngram_range=(1, 2),    # unigrams and bigrams
            sublinear_tf=True,     # log(tf) dampens very frequent words
            min_df=2,              # ignore terms appearing < 2 times
        )
        tfidf.fit(X_train["combined_text"].fillna(""))
        joblib.dump(tfidf, tfidf_path)
    else:
        tfidf = joblib.load(tfidf_path)

    train_tfidf = tfidf.transform(X_train["combined_text"].fillna(""))
    test_tfidf  = tfidf.transform(X_test["combined_text"].fillna(""))

    # ── 2. One-Hot Encode categorical metadata ────────────────────────────
    # Only keep columns that actually exist in the dataframe
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

    if fit:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        ohe.fit(X_train[cat_cols])
        joblib.dump(ohe, ohe_path)
    else:
        ohe = joblib.load(ohe_path)

    train_cat = ohe.transform(X_train[cat_cols])
    test_cat  = ohe.transform(X_test[cat_cols])

    # ── 3. Numeric features ───────────────────────────────────────────────
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    train_num = csr_matrix(X_train[num_cols].fillna(0).values)
    test_num  = csr_matrix(X_test[num_cols].fillna(0).values)

    # ── 4. Horizontally stack all feature blocks ──────────────────────────
    X_train_feat = hstack([train_tfidf, train_cat, train_num])
    X_test_feat  = hstack([test_tfidf,  test_cat,  test_num])

    return X_train_feat, X_test_feat, tfidf, ohe


def get_feature_names(tfidf, ohe, num_cols):
    """Return a flat list of all feature names (for importance plots)."""
    cat_names = list(ohe.get_feature_names_out())
    return list(tfidf.get_feature_names_out()) + cat_names + num_cols
