// models/Prediction.js - Mongoose Schema for saved predictions
// =============================================================
// Every time a user submits a job posting for analysis, we save the
// inputs and outputs here. This powers the /history page.

const mongoose = require("mongoose");

const predictionSchema = new mongoose.Schema(
  {
    // ── Input fields (what the user submitted) ──────────────────────────
    title:               { type: String, default: "" },
    company_profile:     { type: String, default: "" },
    description:         { type: String, default: "" },
    requirements:        { type: String, default: "" },
    benefits:            { type: String, default: "" },
    salary_range:        { type: String, default: "Unknown" },
    employment_type:     { type: String, default: "Unknown" },
    required_education:  { type: String, default: "Unknown" },
    required_experience: { type: String, default: "Unknown" },
    telecommuting:       { type: Number, default: 0 },
    has_company_logo:    { type: Number, default: 1 },
    has_questions:       { type: Number, default: 0 },

    // ── Output fields (what the ML service returned) ─────────────────────
    label:             { type: String, enum: ["Real", "Fraudulent"], required: true },
    fraud_probability: { type: Number, required: true, min: 0, max: 1 },
    top_words:         [{ word: String, score: Number }], // words pushing toward fraud
    red_flags:         [String],                          // human-readable warnings
  },
  {
    timestamps: true, // adds createdAt and updatedAt automatically
  }
);

module.exports = mongoose.model("Prediction", predictionSchema);
