"""
interpret.py - Model Interpretability
=======================================
Answers the question: "WHY did the model predict fraud?"

Tools:
  1. Feature Importance from Random Forest / XGBoost (global)
  2. Top words for fraud vs real (TF-IDF + RF coefficients)
  3. SHAP values for individual predictions (local explanations)

Why bother?
  A model that's a black box is hard to trust and hard to debug.
  Feature importance shows which signals matter most (e.g., "work from home").
  SHAP explains a single prediction: "word X pushed toward fraud by 0.12".
"""

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

EXPORTS = "ml/exports"
os.makedirs(EXPORTS, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(EXPORTS, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. RF / XGBoost feature importance
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 30):
    """
    Tree-based models expose .feature_importances_ after training.
    Importance = how much each feature reduces impurity (Gini / entropy).
    Higher = more important for splitting decisions.
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        print(f"  {model_name} has no feature_importances_, skipping.")
        return []

    # Sort descending
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [str(feature_names[i]) for i in idx]
    top_scores   = [float(importances[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colours = ["#F44336" if "scam" in f or "fee" in f or "earn" in f
               else "#1565C0" for f in top_features]
    ax.barh(top_features[::-1], top_scores[::-1], color=colours[::-1])
    ax.set_title(f"Top {top_n} Feature Importances – {model_name}")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")

    return [{"feature": f, "importance": round(s, 6)}
            for f, s in zip(top_features, top_scores)]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Top words for fraud vs real
# ─────────────────────────────────────────────────────────────────────────────
def top_words_by_class(tfidf, X_tfidf, y, top_n: int = 20):
    """
    For each class (fraud / real), sum TF-IDF scores across all documents.
    The words with the highest total score are most 'characteristic'.
    """
    # X_tfidf may be a sparse slice of the full feature matrix
    # We need just the TF-IDF portion (first len(vocab) columns)
    from scipy.sparse import issparse
    n_tfidf = len(tfidf.vocabulary_)

    if issparse(X_tfidf):
        X_dense_tfidf = X_tfidf[:, :n_tfidf]
    else:
        X_dense_tfidf = X_tfidf[:, :n_tfidf]

    y_arr = np.array(y)
    vocab = np.array(tfidf.get_feature_names_out())

    results = {}
    for label, lname in [(0, "real"), (1, "fraud")]:
        mask = y_arr == label
        if issparse(X_dense_tfidf):
            mean_scores = np.asarray(X_dense_tfidf[mask].mean(axis=0)).ravel()
        else:
            mean_scores = X_dense_tfidf[mask].mean(axis=0)

        top_idx  = np.argsort(mean_scores)[::-1][:top_n]
        results[lname] = [
            {"word": str(vocab[i]), "score": round(float(mean_scores[i]), 6)}
            for i in top_idx
        ]

    # Bar chart comparison
    fraud_words  = [r["word"]  for r in results["fraud"]][:15]
    fraud_scores = [r["score"] for r in results["fraud"]][:15]
    real_words   = [r["word"]  for r in results["real"]][:15]
    real_scores  = [r["score"] for r in results["real"]][:15]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.barh(fraud_words[::-1], fraud_scores[::-1], color="#EF5350")
    ax1.set_title("Top Words in FRAUD Postings")
    ax1.set_xlabel("Mean TF-IDF Score")

    ax2.barh(real_words[::-1], real_scores[::-1], color="#43A047")
    ax2.set_title("Top Words in REAL Postings")
    ax2.set_xlabel("Mean TF-IDF Score")

    plt.tight_layout()
    _save(fig, "top_words_fraud_vs_real.png")

    with open(os.path.join(EXPORTS, "top_words.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP explanations
# ─────────────────────────────────────────────────────────────────────────────
def run_shap(model, X_train_feat, X_test_feat, feature_names: list,
             model_name: str = "XGBoost", n_samples: int = 200):
    """
    SHAP (SHapley Additive exPlanations):
      - Assigns each feature a 'contribution' value for a specific prediction.
      - Positive SHAP = pushed toward fraud; Negative = pushed toward real.
      - We use a background sample to speed up computation.
    """
    try:
        import shap
        from scipy.sparse import issparse

        X_bg = X_train_feat[:100]
        X_explain = X_test_feat[:n_samples]

        if issparse(X_bg):
            X_bg      = X_bg.toarray().astype(np.float32)
            X_explain = X_explain.toarray().astype(np.float32)

        explainer = shap.TreeExplainer(model, X_bg)
        shap_vals = explainer.shap_values(X_explain)

        # For binary classification XGBoost returns a 2D array
        if isinstance(shap_vals, list):
            sv = shap_vals[1]  # class 1 (fraud)
        else:
            sv = shap_vals

        # Mean |SHAP| per feature
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:20]
        top_feat = [str(feature_names[i]) for i in top_idx]
        top_shap = [float(mean_abs[i]) for i in top_idx]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top_feat[::-1], top_shap[::-1], color="#7B1FA2")
        ax.set_title(f"SHAP Feature Importance – {model_name}")
        ax.set_xlabel("Mean |SHAP value|")
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, f"shap_{model_name.lower().replace(' ', '_')}.png")

        shap_out = [{"feature": f, "mean_abs_shap": round(s, 6)}
                    for f, s in zip(top_feat, top_shap)]

        with open(os.path.join(EXPORTS, "shap_importance.json"), "w") as f:
            json.dump(shap_out, f, indent=2)

        return shap_out

    except Exception as e:
        print(f"  ⚠️  SHAP failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Save combined feature importance JSON
# ─────────────────────────────────────────────────────────────────────────────
def save_feature_importance(rf_imp: list, xgb_imp: list):
    out = {
        "random_forest": rf_imp[:20],
        "xgboost": xgb_imp[:20],
    }
    with open(os.path.join(EXPORTS, "feature_importance.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved ml/exports/feature_importance.json")
