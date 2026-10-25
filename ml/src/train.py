"""
train.py - Main ML Pipeline Orchestrator
=========================================
Run this script to train ALL models end-to-end:
  1. Load & clean data
  2. EDA (charts + stats)
  3. Feature engineering
  4. Build feature matrices (TF-IDF + one-hot + numeric)
  5. Apply SMOTE on training set only
  6. Train all models (baselines + ensembles + deep learning)
  7. Evaluate all models → metrics.json
  8. Unsupervised analysis (SVD, K-Means, SOM)
  9. Interpretability (feature importance, SHAP, top words)
  10. Save best model + vectorizers

Usage (from project root):
    cd fake-job-detector/ml
    source venv/bin/activate
    cd ..  # back to fake-job-detector/
    python ml/src/train.py

Note: The script uses relative paths from the fake-job-detector/ directory.
      Keep that as your working directory.
"""

import os, sys, json, warnings, time
import numpy as np
import joblib

# Make sure we can import from ml/src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to fake-job-detector/)
# ─────────────────────────────────────────────────────────────────────────────
DATA_CSV   = "ml/data/fake_job_postings.csv"
MODELS_DIR = "ml/models"
EXPORTS    = "ml/exports"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPORTS, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    start = time.time()
    print("=" * 60)
    print("  FAKE JOB DETECTION – ML PIPELINE")
    print("=" * 60)

    # ── Step 1: Load & clean ──────────────────────────────────────────────
    print("\n📂 Step 1: Loading data …")
    from preprocess import load_and_clean, engineer_features, split_data

    df = load_and_clean(DATA_CSV)
    print(f"  Rows: {len(df):,}  |  Fraud: {df['fraudulent'].sum():,}  "
          f"({df['fraudulent'].mean()*100:.1f}%)")

    # ── Step 2: EDA ───────────────────────────────────────────────────────
    print("\n📊 Step 2: EDA …")
    from eda import run_eda
    eda_stats = run_eda(df)

    # ── Step 3: Feature engineering ───────────────────────────────────────
    print("\n🔧 Step 3: Feature engineering …")
    df = engineer_features(df)
    X_train_df, X_test_df, y_train, y_test = split_data(df)
    print(f"  Train: {len(X_train_df):,}  |  Test: {len(X_test_df):,}")

    # ── Step 4: Build feature matrix ──────────────────────────────────────
    print("\n🏗️  Step 4: Building TF-IDF + metadata feature matrix …")
    from features import build_feature_matrix, get_feature_names, NUMERIC_COLS

    X_train_feat, X_test_feat, tfidf, ohe = build_feature_matrix(
        X_train_df, X_test_df, models_dir=MODELS_DIR, fit=True
    )
    print(f"  Feature matrix shape: {X_train_feat.shape}")

    # Feature names (for importance charts)
    num_cols = [c for c in NUMERIC_COLS if c in X_train_df.columns]
    feature_names = get_feature_names(tfidf, ohe, num_cols)

    # ── Step 5: SMOTE (only on training data!) ────────────────────────────
    print("\n⚖️  Step 5: Applying SMOTE to training set …")
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train_sm, y_train_sm = sm.fit_resample(X_train_feat, y_train)
        print(f"  After SMOTE – Train: {X_train_sm.shape[0]:,}  "
              f"Fraud: {int(y_train_sm.sum()):,}")
        # NOTE: We evaluate on ORIGINAL test set, never on resampled data
    except Exception as e:
        print(f"  ⚠️  SMOTE failed ({e}), using class_weight instead.")
        X_train_sm, y_train_sm = X_train_feat, y_train

    # ── Step 6: Train models ──────────────────────────────────────────────
    all_results = []

    from models import train_baselines, train_ensembles, train_keras_mlp, train_keras_cnn

    # Baselines (use SMOTE-resampled train, original test)
    baseline_results = train_baselines(X_train_sm, y_train_sm, X_test_feat, y_test)
    all_results.extend(baseline_results)

    # Ensembles
    ensemble_results, best_rf, best_xgb = train_ensembles(
        X_train_sm, y_train_sm, X_test_feat, y_test
    )
    all_results.extend(ensemble_results)

    # Keras MLP
    mlp_result, mlp_model = train_keras_mlp(
        X_train_sm, y_train_sm, X_test_feat, y_test
    )
    if mlp_result:
        all_results.append(mlp_result)

    # Keras 1D-CNN
    cnn_result, cnn_model = train_keras_cnn(
        X_train_sm, y_train_sm, X_test_feat, y_test
    )
    if cnn_result:
        all_results.append(cnn_result)

    # ── Step 7: Choose best model by PR-AUC ───────────────────────────────
    print("\n🏆 Step 7: Selecting best model …")
    best_idx = int(np.argmax([r["pr_auc"] for r in all_results]))
    all_results[best_idx]["isBest"] = True
    best_name = all_results[best_idx]["name"]
    print(f"  Best model: {best_name}  (PR-AUC = {all_results[best_idx]['pr_auc']})")

    # Save metrics
    with open(os.path.join(EXPORTS, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    # Also save to models dir for seeding
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved ml/exports/metrics.json")

    # ── Step 8: Unsupervised ──────────────────────────────────────────────
    from unsupervised import run_unsupervised
    y_test_arr = np.array(y_test)
    run_unsupervised(X_test_feat, y_test_arr)

    # ── Step 9: Interpretability ──────────────────────────────────────────
    print("\n🔍 Step 9: Interpretability …")
    from interpret import (
        plot_feature_importance, top_words_by_class,
        run_shap, save_feature_importance,
    )

    rf_imp  = plot_feature_importance(best_rf,  feature_names, "Random Forest")
    xgb_imp = plot_feature_importance(best_xgb, feature_names, "XGBoost")
    save_feature_importance(rf_imp, xgb_imp)

    top_words_by_class(tfidf, X_test_feat, y_test_arr)

    # SHAP (use XGBoost as it's fastest)
    run_shap(best_xgb, X_train_feat, X_test_feat, feature_names, "XGBoost")

    # ── Step 10: Save best model ──────────────────────────────────────────
    print("\n💾 Step 10: Saving best model …")
    # Map name to joblib file
    name_to_file = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree (overfit)": "decision_tree_overfit.joblib",
        "Decision Tree (tuned)": "decision_tree_tuned.joblib",
        "KNN": "knn.joblib",
        "SVM": "svm.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "XGBoost": "xgboost.joblib",
    }
    if best_name in name_to_file:
        best_model = joblib.load(os.path.join(MODELS_DIR, name_to_file[best_name]))
        joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
        # Save model name so service.py knows which file to load
        with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
            f.write(best_name)
        print(f"  Saved ml/models/best_model.joblib  ({best_name})")
    elif "Keras MLP" in best_name and mlp_model is not None:
        import shutil
        shutil.copy(os.path.join(MODELS_DIR, "mlp.keras"),
                    os.path.join(MODELS_DIR, "best_model_keras.keras"))
        with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
            f.write("Keras MLP")
        print("  Saved ml/models/best_model_keras.keras")
    elif "CNN" in best_name and cnn_model is not None:
        import shutil
        shutil.copy(os.path.join(MODELS_DIR, "cnn1d.keras"),
                    os.path.join(MODELS_DIR, "best_model_keras.keras"))
        with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
            f.write("Keras 1D-CNN")
        print("  Saved ml/models/best_model_keras.keras")

    elapsed = time.time() - start
    print(f"\n✅ Pipeline complete in {elapsed/60:.1f} minutes.")
    print("  Next: run `python ml/service.py` to start the prediction API.")


if __name__ == "__main__":
    main()
