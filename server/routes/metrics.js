// routes/metrics.js - GET /api/metrics
// ======================================
// Returns all model metrics from MongoDB (seeded from metrics.json).
// The React Model Comparison page uses this to render the table + charts.

const express = require("express");
const ModelMetric = require("../models/ModelMetric");
const router = express.Router();

router.get("/", async (req, res) => {
  try {
    const metrics = await ModelMetric.find().sort({ prAuc: -1 }); // best first
    res.json(metrics);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
