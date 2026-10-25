// routes/eda.js - GET /api/eda
// ==============================
// Returns EDA statistics and a list of available chart URLs.
// The Dataset & EDA page uses this to display charts and summaries.

const express = require("express");
const fs = require("fs");
const path = require("path");
const router = express.Router();

const EDA_JSON = path.join(__dirname, "../../ml/exports/eda_stats.json");
const EXPORTS  = path.join(__dirname, "../../ml/exports");

router.get("/", async (req, res) => {
  try {
    // Read eda_stats.json
    if (!fs.existsSync(EDA_JSON)) {
      return res.status(404).json({
        error: "EDA not generated yet. Run python ml/src/train.py first.",
      });
    }

    const stats = JSON.parse(fs.readFileSync(EDA_JSON, "utf-8"));

    // List all PNG files in exports/ so frontend knows what to show
    const pngs = fs.readdirSync(EXPORTS)
      .filter(f => f.endsWith(".png"))
      .map(f => `/static/${f}`); // path that Express serves statically

    res.json({ stats, charts: pngs });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
