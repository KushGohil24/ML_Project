// routes/history.js - GET /api/history, DELETE /api/history/:id
// ==============================================================

const express = require("express");
const Prediction = require("../models/Prediction");
const router = express.Router();

// GET /api/history - return last 100 predictions, newest first
router.get("/", async (req, res) => {
  try {
    const history = await Prediction.find()
      .sort({ createdAt: -1 })
      .limit(100)
      .select("title label fraud_probability red_flags createdAt"); // only needed fields
    res.json(history);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// DELETE /api/history/:id - remove a single prediction record
router.delete("/:id", async (req, res) => {
  try {
    const deleted = await Prediction.findByIdAndDelete(req.params.id);
    if (!deleted) {
      return res.status(404).json({ error: "Prediction not found." });
    }
    res.json({ success: true, id: req.params.id });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
