// seed.js - Import ML results into MongoDB
// ==========================================
// Run this ONCE after training to populate the database with:
//   - Model metrics (from ml/models/metrics.json)
//   - EDA stats (from ml/exports/eda_stats.json)
//
// Usage:
//   cd fake-job-detector/server
//   node seed.js
//
// Safe to re-run: it clears old metric records before inserting new ones.

const mongoose = require("mongoose");
const dotenv = require("dotenv");
const fs = require("fs");
const path = require("path");

dotenv.config();

const ModelMetric = require("./models/ModelMetric");

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/fake_job_detector";
const METRICS_JSON = path.join(__dirname, "../ml/models/metrics.json");

async function seed() {
  try {
    // ── Connect to MongoDB ─────────────────────────────────────────────────
    await mongoose.connect(MONGO_URI);
    console.log("✅ Connected to MongoDB");

    // ── Load metrics.json ──────────────────────────────────────────────────
    if (!fs.existsSync(METRICS_JSON)) {
      console.error(
        "❌ ml/models/metrics.json not found.\n" +
        "   Run the training pipeline first:\n" +
        "   cd fake-job-detector && source ml/venv/bin/activate && python ml/src/train.py"
      );
      process.exit(1);
    }

    const metrics = JSON.parse(fs.readFileSync(METRICS_JSON, "utf-8"));

    // ── Clear old records ──────────────────────────────────────────────────
    await ModelMetric.deleteMany({});
    console.log("🗑️  Cleared old model metrics");

    // ── Insert new records ─────────────────────────────────────────────────
    const docs = metrics.map((m) => ({
      name:       m.name,
      precision:  m.precision,
      recall:     m.recall,
      f1:         m.f1,
      rocAuc:     m.roc_auc,
      prAuc:      m.pr_auc,
      trainScore: m.train_score,
      testScore:  m.test_score,
      isBest:     m.isBest || false,
    }));

    await ModelMetric.insertMany(docs);
    console.log(`✅ Inserted ${docs.length} model metrics`);

    // ── Print summary ──────────────────────────────────────────────────────
    const best = docs.find((d) => d.isBest);
    if (best) {
      console.log(`🏆 Best model: ${best.name}  PR-AUC=${best.prAuc}`);
    }

    console.log("\n✅ Seeding complete. Start the server with: npm start");
    process.exit(0);
  } catch (err) {
    console.error("❌ Seeding error:", err.message);
    process.exit(1);
  }
}

seed();
