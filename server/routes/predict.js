// routes/predict.js - POST /api/predict
// ========================================
// 1. Validates input
// 2. Forwards the job data to the Python FastAPI service
// 3. Saves result to MongoDB
// 4. Returns result to the React frontend

const express = require("express");
const axios = require("axios");
const Prediction = require("../models/Prediction");

const router = express.Router();

// Python FastAPI service URL (configurable via .env)
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

router.post("/", async (req, res) => {
  try {
    const {
      title, company_profile, description, requirements, benefits,
      salary_range, employment_type, required_education, required_experience,
      telecommuting, has_company_logo, has_questions,
    } = req.body;

    // ── Basic validation ─────────────────────────────────────────────────
    if (!title && !description) {
      return res.status(400).json({
        error: "At least a job title or description is required.",
      });
    }

    // ── Call Python ML service ────────────────────────────────────────────
    let mlResult;
    try {
      const response = await axios.post(`${ML_SERVICE_URL}/predict`, {
        title:               title || "",
        company_profile:     company_profile || "",
        description:         description || "",
        requirements:        requirements || "",
        benefits:            benefits || "",
        salary_range:        salary_range || "Unknown",
        employment_type:     employment_type || "Unknown",
        required_education:  required_education || "Unknown",
        required_experience: required_experience || "Unknown",
        telecommuting:       Number(telecommuting) || 0,
        has_company_logo:    Number(has_company_logo) ?? 1,
        has_questions:       Number(has_questions) || 0,
      }, { timeout: 30000 });
      mlResult = response.data;
    } catch (mlErr) {
      // Give a helpful error if the Python service is not running
      const hint = mlErr.code === "ECONNREFUSED"
        ? "Python ML service is not running. Run: cd fake-job-detector && source ml/venv/bin/activate && uvicorn ml.service:app --port 8000"
        : mlErr.message;
      return res.status(503).json({ error: `ML service unreachable: ${hint}` });
    }

    // ── Save prediction to MongoDB ────────────────────────────────────────
    const prediction = await Prediction.create({
      title, company_profile, description, requirements, benefits,
      salary_range, employment_type, required_education, required_experience,
      telecommuting: Number(telecommuting) || 0,
      has_company_logo: Number(has_company_logo) ?? 1,
      has_questions: Number(has_questions) || 0,
      label:             mlResult.label,
      fraud_probability: mlResult.fraud_probability,
      top_words:         mlResult.top_words || [],
      red_flags:         mlResult.red_flags || [],
    });

    // ── Return result ─────────────────────────────────────────────────────
    res.json({
      id:                prediction._id,
      label:             mlResult.label,
      fraud_probability: mlResult.fraud_probability,
      top_words:         mlResult.top_words,
      red_flags:         mlResult.red_flags,
      createdAt:         prediction.createdAt,
    });

  } catch (err) {
    console.error("Predict error:", err);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
