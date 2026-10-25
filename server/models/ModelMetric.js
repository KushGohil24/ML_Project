// models/ModelMetric.js - Mongoose Schema for ML model performance metrics
// =========================================================================
// Seeded from ml/models/metrics.json during `node seed.js`.
// Powers the /metrics endpoint and the Model Comparison page.

const mongoose = require("mongoose");

const modelMetricSchema = new mongoose.Schema(
  {
    name:        { type: String, required: true }, // e.g. "Random Forest"
    precision:   { type: Number },                 // Precision on test set
    recall:      { type: Number },                 // Recall on test set
    f1:          { type: Number },                 // F1 Score (main metric)
    rocAuc:      { type: Number },                 // ROC-AUC
    prAuc:       { type: Number },                 // PR-AUC (key for imbalanced data)
    trainScore:  { type: Number },                 // Train PR-AUC (shows overfitting)
    testScore:   { type: Number },                 // Test F1 (same as f1)
    isBest:      { type: Boolean, default: false },// Best model flag
  },
  { timestamps: true }
);

module.exports = mongoose.model("ModelMetric", modelMetricSchema);
