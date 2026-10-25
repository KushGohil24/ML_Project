// server.js - Express App Entry Point
// =====================================
// This is the main backend server.
// It connects to MongoDB, registers routes, and serves static ML exports.
//
// Architecture:
//   Browser (React) → Express (port 5000) → MongoDB
//                                         → FastAPI (port 8000) [for /predict]

const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const dotenv = require("dotenv");
const path = require("path");

dotenv.config();

const app = express();

// ── Middleware ──────────────────────────────────────────────────────────────
app.use(cors());                          // allow cross-origin requests from React
app.use(express.json({ limit: "10mb" })); // parse JSON bodies

// ── Static files (ML exports: PNGs, JSONs) ─────────────────────────────────
// The React frontend can fetch images like: GET /static/class_distribution.png
app.use(
  "/static",
  express.static(path.join(__dirname, "../ml/exports"))
);

// ── Routes ──────────────────────────────────────────────────────────────────
app.use("/api/predict",  require("./routes/predict"));
app.use("/api/history",  require("./routes/history"));
app.use("/api/metrics",  require("./routes/metrics"));
app.use("/api/eda",      require("./routes/eda"));
app.use("/api/insights", require("./routes/insights"));

// ── Health check ────────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", db: mongoose.connection.readyState === 1 ? "connected" : "disconnected" });
});

// ── Connect to MongoDB ───────────────────────────────────────────────────────
const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/fake_job_detector";

mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log("✅ MongoDB connected:", MONGO_URI);
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => {
      console.log(`✅ Server running on http://localhost:${PORT}`);
      console.log(`   Static ML exports: http://localhost:${PORT}/static/`);
    });
  })
  .catch((err) => {
    console.error("❌ MongoDB connection error:", err.message);
    process.exit(1);
  });
