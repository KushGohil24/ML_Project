"""
models.py - All ML model definitions and training
=================================================
Covers:
  - Baselines: Logistic Regression, KNN, Decision Tree, SVM
  - Ensembles: Random Forest, Gradient Boosting, XGBoost
  - Deep Learning: Keras MLP and 1D-CNN
  - Overfitting demo: train score vs test score comparison
  - Regularisation and hyperparameter tuning with GridSearchCV

Key evaluation metrics:
  - Precision, Recall, F1 (macro), ROC-AUC, PR-AUC
  - We deliberately avoid accuracy because with ~5% fraud,
    a model predicting everything as "real" gets 95% accuracy — useless!
"""

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

warnings.filterwarnings("ignore")

EXPORTS = "ml/exports"
MODELS  = "ml/models"
os.makedirs(EXPORTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaluate any sklearn-compatible classifier
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, name: str, train_score: float = None):
    """
    Compute all metrics on TEST data (never on SMOTE-resampled training data).
    Returns a dict and also prints a summary.
    """
    y_pred = model.predict(X_test)

    # For probabilistic metrics we need predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Calibrate so we get probabilities (e.g., SVM)
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
    else:
        y_prob = y_pred.astype(float)

    prec   = precision_score(y_test, y_pred, zero_division=0)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    roc    = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    # Train score (for overfitting demo)
    test_score = f1

    result = {
        "name": name,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "roc_auc":   round(roc, 4),
        "pr_auc":    round(pr_auc, 4),
        "train_score": round(train_score, 4) if train_score is not None else None,
        "test_score":  round(test_score, 4),
        "isBest": False,
    }

    print(f"  [{name:30s}] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
          f"ROC={roc:.3f}  PR-AUC={pr_auc:.3f}")
    return result, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save confusion matrix PNG + JSON
# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Real", "Pred Fraud"])
    ax.set_yticklabels(["True Real", "True Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_title(f"Confusion Matrix – {name}")
    slug = name.lower().replace(" ", "_")
    _save(fig, f"cm_{slug}.png")

    # Also save as JSON for the frontend
    cm_json = {"tn": int(cm[0,0]), "fp": int(cm[0,1]),
               "fn": int(cm[1,0]), "tp": int(cm[1,1])}
    with open(os.path.join(EXPORTS, f"cm_{slug}.json"), "w") as f:
        json.dump(cm_json, f)


def _save(fig, name: str):
    path = os.path.join(EXPORTS, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save ROC and PR curves as JSON (for Recharts in the frontend)
# ─────────────────────────────────────────────────────────────────────────────
def save_curves(y_test, y_prob, name: str):
    slug = name.lower().replace(" ", "_")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
                for f, t in zip(fpr, tpr)]

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    pr_data = [{"precision": round(float(p), 4), "recall": round(float(r), 4)}
               for p, r in zip(prec_arr, rec_arr)]

    with open(os.path.join(EXPORTS, f"roc_{slug}.json"), "w") as f:
        json.dump(roc_data, f)
    with open(os.path.join(EXPORTS, f"pr_{slug}.json"), "w") as f:
        json.dump(pr_data, f)


# ─────────────────────────────────────────────────────────────────────────────
# Convert sparse to dense if needed (for Keras / KNN)
# ─────────────────────────────────────────────────────────────────────────────
def to_dense(X):
    return X.toarray() if issparse(X) else X


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────
def train_baselines(X_train, y_train, X_test, y_test):
    """
    Baselines: Logistic Regression, KNN, Decision Tree, SVM.
    class_weight='balanced' compensates for class imbalance.
    """
    print("\n🔵 Training baseline models …")
    results = []

    configs = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, class_weight="balanced",
                            C=1.0, random_state=42)),
        ("Decision Tree (overfit)",   # intentionally no max_depth to show overfitting
         DecisionTreeClassifier(random_state=42)),
        ("Decision Tree (tuned)",
         DecisionTreeClassifier(max_depth=10, min_samples_leaf=5,
                                class_weight="balanced", random_state=42)),
        ("KNN",
         KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]

    for name, clf in configs:
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").strip()
        model_path = os.path.join(MODELS, f"{slug}.joblib")
        if os.path.exists(model_path):
            print(f"  Loading cached {name} …")
            clf = joblib.load(model_path)
        else:
            print(f"  Training {name} …")
            clf.fit(X_train, y_train)
            joblib.dump(clf, model_path)

        if hasattr(clf, "predict_proba"):
            train_prob = clf.predict_proba(X_train)[:, 1]
        else:
            train_prob = clf.predict(X_train).astype(float)
        train_pr_auc = average_precision_score(y_train, train_prob)

        res, y_pred, y_prob = evaluate(clf, X_test, y_test, name, train_score=train_pr_auc)
        results.append(res)
        save_confusion_matrix(y_test, y_pred, name)
        save_curves(y_test, y_prob, name)

    # SVM needs calibration for predict_proba
    svm_path = os.path.join(MODELS, "svm.joblib")
    if os.path.exists(svm_path):
        print("  Loading cached SVM …")
        svm = joblib.load(svm_path)
    else:
        print("  Training SVM (this may take a moment) …")
        svm_base = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
        svm = CalibratedClassifierCV(svm_base, cv=3)
        svm.fit(X_train, y_train)
        joblib.dump(svm, svm_path)

    train_prob = svm.predict_proba(X_train)[:, 1]
    train_pr_auc = average_precision_score(y_train, train_prob)
    res, y_pred, y_prob = evaluate(svm, X_test, y_test, "SVM", train_score=train_pr_auc)
    results.append(res)
    save_confusion_matrix(y_test, y_pred, "SVM")
    save_curves(y_test, y_prob, "SVM")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE MODELS
# ─────────────────────────────────────────────────────────────────────────────
def train_ensembles(X_train, y_train, X_test, y_test):
    """Random Forest, Gradient Boosting, XGBoost."""
    print("\n🟢 Training ensemble models …")
    results = []
    scale_pos = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)

    # ── Random Forest ──────────────────────────────────────────────────────
    rf_path = os.path.join(MODELS, "random_forest.joblib")
    if os.path.exists(rf_path):
        print("  Loading cached Random Forest …")
        best_rf = joblib.load(rf_path)
    else:
        print("  Training Random Forest …")
        best_rf = RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=4
        )
        best_rf.fit(X_train, y_train)
        joblib.dump(best_rf, rf_path)

    train_prob = best_rf.predict_proba(X_train)[:, 1]
    train_pr_auc = average_precision_score(y_train, train_prob)
    res, y_pred, y_prob = evaluate(best_rf, X_test, y_test,
                                   "Random Forest", train_score=train_pr_auc)
    results.append(res)
    save_confusion_matrix(y_test, y_pred, "Random Forest")
    save_curves(y_test, y_prob, "Random Forest")

    # ── Gradient Boosting ─────────────────────────────────────────────────
    gb_path = os.path.join(MODELS, "gradient_boosting.joblib")
    if os.path.exists(gb_path):
        print("  Loading cached Gradient Boosting …")
        gb = joblib.load(gb_path)
    else:
        print("  Training Gradient Boosting …")
        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        gb.fit(X_train, y_train)
        joblib.dump(gb, gb_path)

    train_prob = gb.predict_proba(X_train)[:, 1]
    train_pr_auc = average_precision_score(y_train, train_prob)
    res, y_pred, y_prob = evaluate(gb, X_test, y_test,
                                   "Gradient Boosting", train_score=train_pr_auc)
    results.append(res)
    save_confusion_matrix(y_test, y_pred, "Gradient Boosting")
    save_curves(y_test, y_prob, "Gradient Boosting")

    # ── XGBoost ────────────────────────────────────────────────────────────
    xgb_path = os.path.join(MODELS, "xgboost.joblib")
    if os.path.exists(xgb_path):
        print("  Loading cached XGBoost …")
        best_xgb = joblib.load(xgb_path)
    else:
        print("  Training XGBoost …")
        best_xgb = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
        best_xgb.fit(X_train, y_train)
        joblib.dump(best_xgb, xgb_path)

    train_prob = best_xgb.predict_proba(X_train)[:, 1]
    train_pr_auc = average_precision_score(y_train, train_prob)
    res, y_pred, y_prob = evaluate(best_xgb, X_test, y_test,
                                   "XGBoost", train_score=train_pr_auc)
    results.append(res)
    save_confusion_matrix(y_test, y_pred, "XGBoost")
    save_curves(y_test, y_prob, "XGBoost")

    return results, best_rf, best_xgb


# ─────────────────────────────────────────────────────────────────────────────
# DEEP LEARNING: Keras MLP
# ─────────────────────────────────────────────────────────────────────────────
def train_keras_mlp(X_train, y_train, X_test, y_test):
    """
    Feed-forward Neural Network:
      Input → Dense(256, ReLU) → Dropout(0.4) →
              Dense(128, ReLU) → Dropout(0.3) →
              Dense(64,  ReLU) →
              Dense(1, Sigmoid)  ← outputs fraud probability

    Early stopping: stop training when validation loss stops improving.
    """
    print("\n🔴 Training Keras MLP …")
    try:
        import tensorflow as tf
        from tensorflow import keras

        X_tr = to_dense(X_train).astype(np.float32)
        X_te = to_dense(X_test).astype(np.float32)

        n_features = X_tr.shape[1]
        class_ratio = float((y_train == 0).sum()) / float((y_train == 1).sum())

        model = keras.Sequential([
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ], name="MLP")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )

        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = model.fit(
            X_tr, y_train,
            validation_split=0.1,
            epochs=30,
            batch_size=256,
            class_weight={0: 1.0, 1: class_ratio},
            callbacks=[es],
            verbose=0,
        )

        # Save model
        model.save(os.path.join(MODELS, "mlp.keras"))

        y_prob = model.predict(X_te, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        # Train score
        y_tr_prob = model.predict(X_tr, verbose=0).ravel()
        train_pr_auc = average_precision_score(y_train, y_tr_prob)

        res, _, _ = evaluate(
            type("Wrap", (), {"predict": lambda s, X: (model.predict(X, verbose=0).ravel() >= 0.5).astype(int),
                               "predict_proba": lambda s, X: np.column_stack([1 - model.predict(X, verbose=0).ravel(), model.predict(X, verbose=0).ravel()])})(),
            X_te, y_test, "Keras MLP", train_score=train_pr_auc,
        )
        # Override with actual values
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        roc  = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        res = {
            "name": "Keras MLP",
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "roc_auc":   round(roc, 4),
            "pr_auc":    round(pr_auc, 4),
            "train_score": round(train_pr_auc, 4),
            "test_score":  round(f1, 4),
            "isBest": False,
        }
        save_confusion_matrix(y_test, y_pred, "Keras MLP")
        save_curves(y_test, y_prob, "Keras MLP")

        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history["loss"], label="Train Loss")
        ax1.plot(history.history["val_loss"], label="Val Loss")
        ax1.set_title("MLP: Loss over Epochs"); ax1.legend()
        ax2.plot(history.history["AUC"], label="Train AUC")
        ax2.plot(history.history["val_AUC"], label="Val AUC")
        ax2.set_title("MLP: AUC over Epochs"); ax2.legend()
        _save(fig, "mlp_training_history.png")

        print(f"  [Keras MLP] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
              f"ROC={roc:.3f}  PR-AUC={pr_auc:.3f}")
        return res, model

    except Exception as e:
        print(f"  ⚠️  Keras MLP failed: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# DEEP LEARNING: 1D-CNN
# ─────────────────────────────────────────────────────────────────────────────
def train_keras_cnn(X_train, y_train, X_test, y_test):
    """
    1D-CNN treats each row as a 'sequence' of feature values.
    Conv1D learns local patterns (e.g., co-occurring TF-IDF peaks).
    """
    print("\n🔴 Training Keras 1D-CNN …")
    try:
        import tensorflow as tf
        from tensorflow import keras

        X_tr = to_dense(X_train).astype(np.float32)
        X_te = to_dense(X_test).astype(np.float32)

        # Reshape for Conv1D: (samples, steps, channels)
        X_tr_cnn = X_tr[:, :, np.newaxis]
        X_te_cnn = X_te[:, :, np.newaxis]

        n_features = X_tr.shape[1]
        class_ratio = float((y_train == 0).sum()) / float((y_train == 1).sum())

        model = keras.Sequential([
            keras.layers.Input(shape=(n_features, 1)),
            keras.layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
            keras.layers.MaxPooling1D(pool_size=4),
            keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ], name="CNN1D")

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["AUC"],
        )

        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )

        model.fit(
            X_tr_cnn, y_train,
            validation_split=0.1,
            epochs=20,
            batch_size=256,
            class_weight={0: 1.0, 1: class_ratio},
            callbacks=[es],
            verbose=0,
        )

        model.save(os.path.join(MODELS, "cnn1d.keras"))

        y_prob = model.predict(X_te_cnn, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        y_tr_prob = model.predict(X_tr_cnn, verbose=0).ravel()
        train_pr_auc = average_precision_score(y_train, y_tr_prob)

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        roc  = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        res = {
            "name": "Keras 1D-CNN",
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "roc_auc":   round(roc, 4),
            "pr_auc":    round(pr_auc, 4),
            "train_score": round(train_pr_auc, 4),
            "test_score":  round(f1, 4),
            "isBest": False,
        }
        save_confusion_matrix(y_test, y_pred, "Keras 1D-CNN")
        save_curves(y_test, y_prob, "Keras 1D-CNN")
        print(f"  [Keras 1D-CNN] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
              f"ROC={roc:.3f}  PR-AUC={pr_auc:.3f}")
        return res, model

    except Exception as e:
        print(f"  ⚠️  Keras CNN failed: {e}")
        return None, None
