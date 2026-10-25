"""
unsupervised.py - Clustering & Dimensionality Reduction
========================================================
Covers:
  - PCA / TruncatedSVD: reduce thousands of features to 2D for plotting
  - K-Means: automatically group job postings into k clusters
  - Elbow Method + Silhouette Score: find the right k
  - Self-Organizing Map (MiniSom): neighbourhood-preserving neural network

Key insight:
  Clustering is *unsupervised* (no labels). After clustering we colour
  data points by their true fraud labels to see if clusters align with
  fraud vs real. This is exploratory, not a replacement for classification.
"""

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from minisom import MiniSom
    MINISOM_OK = True
except ImportError:
    MINISOM_OK = False

warnings.filterwarnings("ignore")

EXPORTS = "ml/exports"
os.makedirs(EXPORTS, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(EXPORTS, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. TruncatedSVD → 2D plot (PCA equivalent for sparse matrices)
# ─────────────────────────────────────────────────────────────────────────────
def plot_svd_2d(X, y, sample_size: int = 3000):
    """
    TruncatedSVD is like PCA but works directly on sparse TF-IDF matrices.
    We project from ~10k dimensions → 2 dimensions for visualisation.
    """
    print("  Running TruncatedSVD …")
    # Sample for speed
    if X.shape[0] > sample_size:
        idx = np.random.RandomState(42).choice(X.shape[0], sample_size, replace=False)
        X_s = X[idx] if not issparse(X) else X[idx]
        y_s = y[idx] if hasattr(y, "__getitem__") else np.array(y)[idx]
    else:
        X_s, y_s = X, np.array(y)

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_s)

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["#4CAF50" if lab == 0 else "#F44336" for lab in y_s]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colours, alpha=0.4, s=8)

    # Legend patches
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#4CAF50", label="Real"),
        mpatches.Patch(color="#F44336", label="Fraud"),
    ])
    ax.set_title("TruncatedSVD (PCA) – 2D projection coloured by label")
    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")
    _save(fig, "svd_2d.png")

    return X_2d, y_s


# ─────────────────────────────────────────────────────────────────────────────
# 2. K-Means with Elbow and Silhouette
# ─────────────────────────────────────────────────────────────────────────────
def run_kmeans(X_2d, y_s, k_range=range(2, 9)):
    """
    Elbow Method: as k increases, within-cluster inertia drops.
      The 'elbow' point is the optimal k.
    Silhouette Score: measures how similar a point is to its own cluster
      vs other clusters. Higher is better (-1 to 1).
    """
    print("  Running K-Means elbow + silhouette …")
    inertias, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_2d)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_2d, labels) if len(set(labels)) > 1 else 0
        sil_scores.append(sil)

    # ── Elbow plot ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(list(k_range), inertias, "o-", color="#1976D2")
    ax1.set_title("K-Means Elbow Method")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-cluster SSE)")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(list(k_range), sil_scores, "s-", color="#7B1FA2")
    ax2.set_title("Silhouette Score vs k")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "kmeans_elbow_silhouette.png")

    # ── Best k cluster plot ───────────────────────────────────────────────
    best_k = list(k_range)[np.argmax(sil_scores)]
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_best.fit_predict(X_2d)

    # Fraud rate per cluster
    cluster_fraud = {}
    for cl in range(best_k):
        mask = cluster_labels == cl
        fr = float(y_s[mask].mean()) if mask.sum() > 0 else 0.0
        cluster_fraud[f"cluster_{cl}"] = {
            "size": int(mask.sum()),
            "fraud_rate": round(fr, 4),
        }

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10", best_k)
    for cl in range(best_k):
        mask = cluster_labels == cl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap(cl)], alpha=0.5, s=8, label=f"Cluster {cl}")
    ax.set_title(f"K-Means Clusters (k={best_k}) on 2D SVD")
    ax.legend(markerscale=3, loc="best")
    _save(fig, "kmeans_clusters.png")

    # Save cluster stats as JSON
    out = {
        "best_k": best_k,
        "silhouette_scores": {str(k): round(s, 4)
                              for k, s in zip(k_range, sil_scores)},
        "cluster_fraud_rates": cluster_fraud,
        "note": (
            "Clustering is unsupervised—no labels used during training. "
            "Some clusters correlate with fraud labels (high fraud rate) "
            "but clustering is exploratory, not a replacement for classification."
        ),
    }
    with open(os.path.join(EXPORTS, "clustering_stats.json"), "w") as f:
        json.dump(out, f, indent=2)

    return cluster_labels, out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Self-Organizing Map (MiniSom)
# ─────────────────────────────────────────────────────────────────────────────
def run_som(X_2d, y_s, grid_size: int = 10):
    """
    SOM: unsupervised neural network that maps high-dim data onto a 2D grid.
    Neurons organise themselves so similar inputs land near each other.
    We then colour the map by average fraud rate to spot fraud-prone regions.
    """
    if not MINISOM_OK:
        print("  MiniSom not available, skipping SOM.")
        return

    print("  Training SOM …")
    som = MiniSom(
        grid_size, grid_size,
        input_len=X_2d.shape[1],
        sigma=1.5,
        learning_rate=0.5,
        random_seed=42,
    )
    som.train_batch(X_2d, num_iteration=2000, verbose=False)

    # Build fraud-rate map
    fraud_map = np.zeros((grid_size, grid_size))
    count_map = np.zeros((grid_size, grid_size))
    for xi, label in zip(X_2d, y_s):
        bmu = som.winner(xi)
        fraud_map[bmu] += label
        count_map[bmu] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_map = np.where(count_map > 0, fraud_map / count_map, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rate_map, cmap="RdYlGn_r", vmin=0, vmax=rate_map.max())
    plt.colorbar(im, ax=ax, label="Fraud Rate")
    ax.set_title("Self-Organizing Map – Fraud Rate per Neuron")
    ax.set_xlabel("SOM Column")
    ax.set_ylabel("SOM Row")
    _save(fig, "som_fraud_map.png")

    print("  SOM note: Red neurons = high fraud region on the map.")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────
def run_unsupervised(X_feat, y):
    print("\n🔷 Running unsupervised analysis …")
    X_2d, y_s = plot_svd_2d(X_feat, y)
    cluster_labels, stats = run_kmeans(X_2d, y_s)
    run_som(X_2d, y_s)
    print("✅ Unsupervised analysis complete.")
    return stats
