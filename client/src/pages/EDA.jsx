// pages/EDA.jsx - Dataset & Exploratory Data Analysis page
import { useEffect, useState } from "react";
import { getEDA } from "../api";
import {
  FiDatabase,
  FiPieChart,
  FiBarChart2,
  FiAlertCircle,
  FiLayers,
  FiImage,
} from "react-icons/fi";

const API_BASE = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace(/\/api$/, "")
  : "http://localhost:5001";

// Clean chart filenames to student-friendly titles
function chartLabel(url) {
  const name = url.split("/").pop().replace(".png", "");
  const labels = {
    class_distribution: "Class Distribution (Real vs Fake)",
    fraud_rate_by_category: "Fraud Rate by Job Categories",
    text_length_description: "Job Description Length Distribution",
    text_length_requirements: "Requirements Text Length Distribution",
  };
  return labels[name] || name.replace(/_/g, " ");
}

// Student-friendly core distribution charts (no word clouds)
const EDA_CHART_ORDER = [
  "class_distribution",
  "fraud_rate_by_category",
  "text_length_description",
  "text_length_requirements",
];

export default function EDA() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    getEDA()
      .then(({ data }) => setData(data))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading-center"><div className="spinner" /><p>Loading EDA insights…</p></div>;
  if (error)   return <div className="alert alert-error">{error}</div>;

  const { stats, charts } = data;

  // Filter charts into EDA vs Unsupervised/Other
  const edaCharts = charts.filter((c) =>
    EDA_CHART_ORDER.some((k) => c.includes(k))
  );

  return (
    <div>
      <div className="page-header">
        <h1>
          <FiDatabase size={24} color="var(--accent-blue)" />
          <span>Dataset & Exploratory Analysis</span>
        </h1>
        <p>
          Statistical exploration of the EMSCAD dataset (17,880 online job listings) prior to feature extraction.
        </p>
      </div>

      {/* ── Key stats row ──────────────────────────────────────────────── */}
      <section className="section">
        <div className="grid-4">
          <div className="stat-card">
            <div className="stat-value">{(stats.total_rows || 17880).toLocaleString()}</div>
            <div className="stat-label">Total Job Postings</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{color:"var(--real-color)"}}>
              {(stats.n_real || 17014).toLocaleString()}
            </div>
            <div className="stat-label">Legitimate Listings (95.2%)</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{color:"var(--fraud-color)"}}>
              {(stats.n_fraud || 866).toLocaleString()}
            </div>
            <div className="stat-label">Fraudulent Listings (4.8%)</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">
              {stats.fraud_rate ? `${(stats.fraud_rate * 100).toFixed(1)}%` : "4.8%"}
            </div>
            <div className="stat-label">Positive Class Ratio</div>
          </div>
        </div>
      </section>

      {/* ── Imbalance explanation alert ────────────────────────────────── */}
      <div className="alert alert-info">
        <FiAlertCircle size={20} style={{minWidth:20}} />
        <div>
          <strong>Severe Class Imbalance Context:</strong> Only 4.8% of postings are fraudulent (~1 fraud case per 20 genuine jobs).
          A trivial baseline predicting "Real" for every job achieves 95.2% accuracy while failing to detect a single scam.
          Hence, <strong>Precision-Recall AUC (PR-AUC)</strong> and <strong>F1-score</strong> are our primary evaluation criteria.
        </div>
      </div>

      {/* ── Fraud rate by category ─────────────────────────────────────── */}
      {stats.fraud_by_category && (
        <section className="section">
          <h2 className="section-title">
            <FiPieChart size={18} color="var(--accent-blue)" />
            <span>Fraud Distribution by Categorical Metadata</span>
          </h2>
          <div className="grid-2">
            {Object.entries(stats.fraud_by_category).slice(0, 4).map(([col, rows]) => (
              <div className="card" key={col}>
                <div className="card-title">
                  <span>{col.replace(/_/g, " ").toUpperCase()}</span>
                </div>
                <div style={{display:"flex", flexDirection:"column", gap:6}}>
                  {rows.slice(0, 5).map(({ category, total, fraud, fraud_rate }) => (
                    <div className="metric-bar-row" key={category}>
                      <span className="metric-bar-label" title={category} style={{textOverflow:"ellipsis", overflow:"hidden", whiteSpace:"nowrap"}}>
                        {category}
                      </span>
                      <div className="metric-bar-bg">
                        <div
                          className="metric-bar-fill"
                          style={{
                            width: `${Math.min(fraud_rate * 100 * 3, 100)}%`,
                            background: fraud_rate > 0.1 ? "var(--fraud-color)" : "var(--accent-blue)",
                          }}
                        />
                      </div>
                      <span className="metric-bar-value" style={{color: fraud_rate > 0.1 ? "var(--fraud-color)" : "inherit"}}>
                        {(fraud_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Chart gallery ──────────────────────────────────────────────── */}
      <section className="section">
        <h2 className="section-title">
          <FiImage size={18} color="var(--accent-blue)" />
          <span>Visual Distribution Plots</span>
        </h2>
        <div className="chart-grid">
          {edaCharts.map((url) => (
            <div className="chart-card" key={url}>
              <div className="chart-card-img-wrap">
                <img
                  src={`${API_BASE}${url}`}
                  alt={chartLabel(url)}
                  loading="lazy"
                  onError={(e) => { e.target.style.display = "none"; }}
                />
              </div>
              <div className="chart-card-title">{chartLabel(url)}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
