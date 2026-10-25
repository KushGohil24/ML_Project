// src/api/index.js - Centralized API calls
// ==========================================
// All HTTP requests to the Express backend go through here.
// This keeps components clean — they just call functions, not axios directly.

import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:5000/api",
  timeout: 40000,
});

// POST a job posting for fraud prediction
export const predictJob = (jobData) => API.post("/predict", jobData);

// GET prediction history
export const getHistory = () => API.get("/history");

// DELETE a prediction by ID
export const deleteHistory = (id) => API.delete(`/history/${id}`);

// GET all model metrics
export const getMetrics = () => API.get("/metrics");

// GET EDA stats + chart list
export const getEDA = () => API.get("/eda");

// GET insights (feature importance, ROC curves, etc.)
export const getInsights = () => API.get("/insights");
