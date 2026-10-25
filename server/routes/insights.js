// routes/insights.js - GET /api/insights
// =========================================
// Returns feature importance, SHAP, top-words, and clustering stats.
// Powers the Insights page (confusion matrices, ROC/PR curves, SHAP).

const express = require("express");
const fs = require("fs");
const path = require("path");
const router = express.Router();

const EXPORTS = path.join(__dirname, "../../ml/exports");

// Helper: safely read a JSON file
function readJSON(filename) {
  const fp = path.join(EXPORTS, filename);
  if (!fs.existsSync(fp)) return null;
  try { return JSON.parse(fs.readFileSync(fp, "utf-8")); }
  catch { return null; }
}

router.get("/", (req, res) => {
  // Collect all available insights
  const featureImportance = readJSON("feature_importance.json");
  const shapImportance    = readJSON("shap_importance.json");
  const topWords          = readJSON("top_words.json");
  const clusteringStats   = readJSON("clustering_stats.json");

  // Confusion matrices for each model
  const cmFiles = fs.existsSync(EXPORTS)
    ? fs.readdirSync(EXPORTS).filter(f => f.startsWith("cm_") && f.endsWith(".json"))
    : [];
  const confusionMatrices = {};
  for (const f of cmFiles) {
    const modelSlug = f.replace("cm_", "").replace(".json", "");
    confusionMatrices[modelSlug] = readJSON(f);
  }

  // ROC curve data for each model
  const rocFiles = fs.existsSync(EXPORTS)
    ? fs.readdirSync(EXPORTS).filter(f => f.startsWith("roc_") && f.endsWith(".json"))
    : [];
  const rocCurves = {};
  for (const f of rocFiles) {
    const modelSlug = f.replace("roc_", "").replace(".json", "");
    // Downsample to 100 points to keep response small
    const data = readJSON(f);
    if (data) {
      const step = Math.max(1, Math.floor(data.length / 100));
      rocCurves[modelSlug] = data.filter((_, i) => i % step === 0);
    }
  }

  // PR curve data for each model
  const prFiles = fs.existsSync(EXPORTS)
    ? fs.readdirSync(EXPORTS).filter(f => f.startsWith("pr_") && f.endsWith(".json"))
    : [];
  const prCurves = {};
  for (const f of prFiles) {
    const modelSlug = f.replace("pr_", "").replace(".json", "");
    const data = readJSON(f);
    if (data) {
      const step = Math.max(1, Math.floor(data.length / 100));
      prCurves[modelSlug] = data.filter((_, i) => i % step === 0);
    }
  }

  // List all PNGs for display (SVD, SOM, training history, etc.)
  const pngs = fs.existsSync(EXPORTS)
    ? fs.readdirSync(EXPORTS)
        .filter(f => f.endsWith(".png"))
        .map(f => `/static/${f}`)
    : [];

  res.json({
    featureImportance,
    shapImportance,
    topWords,
    clusteringStats,
    confusionMatrices,
    rocCurves,
    prCurves,
    charts: pngs,
  });
});

module.exports = router;
