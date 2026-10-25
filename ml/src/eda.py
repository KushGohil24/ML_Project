"""
eda.py - Exploratory Data Analysis for Fake Job Detection
==========================================================
Generates charts and stats BEFORE any modelling.

Charts saved to ml/exports/ as PNG.
Summary stats saved to ml/exports/eda_stats.json.

Why EDA first?
  - Reveals class imbalance (~5% fraud) → we need SMOTE or class_weight.
  - Shows which features have missing values → we fill them in preprocess.py.
  - Word clouds show what language scam vs real jobs use.
"""

import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
warnings.filterwarnings("ignore")

try:
    from wordcloud import WordCloud
    WC_AVAILABLE = True
except ImportError:
    WC_AVAILABLE = False

EXPORTS = "ml/exports"
os.makedirs(EXPORTS, exist_ok=True)

# ── Colour palette (consistent across charts) ─────────────────────────────
PALETTE = {"real": "#4CAF50", "fraud": "#F44336"}


def _save(fig, name: str):
    path = os.path.join(EXPORTS, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Class imbalance bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame):
    counts = df["fraudulent"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Real (0)", "Fraudulent (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=[PALETTE["real"], PALETTE["fraud"]],
        edgecolor="black", linewidth=0.5,
    )
    for bar, cnt in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{cnt}\n({cnt / len(df) * 100:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Class Distribution (Fraudulent vs Real)", fontsize=13)
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "class_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Missing value heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame):
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values found.")
        return
    fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.35)))
    missing.plot.barh(ax=ax, color="#FF9800", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Missing Fraction")
    ax.set_title("Missing Values per Column")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(missing):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
    _save(fig, "missing_values.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fraud rate by categorical feature
# ─────────────────────────────────────────────────────────────────────────────
def plot_fraud_rate_by_category(df: pd.DataFrame):
    cat_cols = [c for c in ["employment_type", "required_experience",
                             "required_education", "telecommuting",
                             "has_company_logo", "has_questions"]
                if c in df.columns]

    n = len(cat_cols)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
    axes = axes.flatten()

    for idx, col in enumerate(cat_cols):
        ax = axes[idx]
        rates = df.groupby(col)["fraudulent"].mean().sort_values(ascending=False)
        bars = ax.bar(
            [str(x)[:20] for x in rates.index],
            rates.values,
            color="#EF5350", edgecolor="black", linewidth=0.5,
        )
        ax.set_title(f"Fraud Rate by {col}", fontsize=10)
        ax.set_ylabel("Fraud Rate")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        for b, v in zip(bars, rates.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=7)

    # hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Fraud Rate by Categorical Feature", fontsize=13)
    plt.tight_layout()
    _save(fig, "fraud_rate_by_category.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Word clouds
# ─────────────────────────────────────────────────────────────────────────────
def plot_word_clouds(df: pd.DataFrame):
    if not WC_AVAILABLE:
        print("  wordcloud not installed, skipping.")
        return

    for label, name in [(0, "real"), (1, "fraud")]:
        subset = df[df["fraudulent"] == label]
        text_cols = [c for c in ["title", "description", "requirements"] if c in df.columns]
        text = " ".join(
            subset[text_cols].fillna("").apply(
                lambda r: " ".join(r.values), axis=1
            )
        )
        wc = WordCloud(
            width=800, height=400,
            background_color="white" if name == "real" else "#1a1a1a",
            colormap="Greens" if name == "real" else "Reds",
            max_words=150,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud – {name.capitalize()} Postings", fontsize=13)
        _save(fig, f"wordcloud_{name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Text length distributions
# ─────────────────────────────────────────────────────────────────────────────
def plot_text_lengths(df: pd.DataFrame):
    for col in ["description", "requirements"]:
        if col not in df.columns:
            continue
        df2 = df.copy()
        df2["_len"] = df2[col].fillna("").apply(len)
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, colour, lname in [(0, PALETTE["real"], "Real"),
                                      (1, PALETTE["fraud"], "Fraud")]:
            data = df2[df2["fraudulent"] == label]["_len"]
            ax.hist(data, bins=60, alpha=0.6, color=colour, label=lname, density=True)
        ax.set_title(f"Text Length Distribution: {col}")
        ax.set_xlabel("Character count")
        ax.set_ylabel("Density")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, f"text_length_{col}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Numeric EDA summary → JSON
# ─────────────────────────────────────────────────────────────────────────────
def save_eda_stats(df: pd.DataFrame):
    total = len(df)
    n_fraud = int(df["fraudulent"].sum())
    n_real = total - n_fraud

    # Per-category fraud rates
    cat_rates = {}
    for col in ["employment_type", "required_experience", "required_education"]:
        if col in df.columns:
            rates = df.groupby(col)["fraudulent"].agg(
                ["mean", "count"]
            ).reset_index()
            cat_rates[col] = [
                {
                    "category": str(row[col]),
                    "fraud_rate": round(float(row["mean"]), 4),
                    "count": int(row["count"]),
                }
                for _, row in rates.iterrows()
            ]

    # Top missing-value columns
    missing_pct = {
        col: round(float(pct), 4)
        for col, pct in df.isnull().mean().items()
        if pct > 0
    }

    stats = {
        "total_rows": total,
        "n_real": n_real,
        "n_fraud": n_fraud,
        "fraud_rate": round(n_fraud / total, 4),
        "missing_pct": missing_pct,
        "fraud_by_category": cat_rates,
        "has_logo_fraud_rate": round(
            float(df[df["has_company_logo"] == 0]["fraudulent"].mean()), 4
        ) if "has_company_logo" in df.columns else None,
        "no_logo_fraud_rate": round(
            float(df[df["has_company_logo"] == 0]["fraudulent"].mean()), 4
        ) if "has_company_logo" in df.columns else None,
    }

    out_path = os.path.join(EXPORTS, "eda_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved {out_path}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    print("\n📊 Running EDA …")
    plot_class_distribution(df)
    plot_missing_values(df)
    plot_fraud_rate_by_category(df)
    plot_word_clouds(df)
    plot_text_lengths(df)
    stats = save_eda_stats(df)
    print("✅ EDA complete.")
    return stats
