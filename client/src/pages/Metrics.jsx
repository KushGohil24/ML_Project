// pages/Metrics.jsx - Model Comparison Table + Bar Charts
import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import {
  FiBarChart2,
  FiAward,
  FiTrendingUp,
  FiSliders,
  FiCheckCircle,
  FiAlertTriangle,
  FiInfo,
} from "react-icons/fi";
import { getMetrics } from "../api";

// Format a 0-1 score as percentage string
const pct = (v) => (v == null ? "N/A" : `${(v * 100).toFixed(1)}%`);

// Color a metric cell by value
function MetricCell({ value }) {
  if (value == null) return <td>N/A</td>;
  const v = value * 100;
  const color = v >= 80 ? "var(--real-color)"
               : v >= 60 ? "var(--accent-yellow)"
               : v >= 40 ? "var(--accent-orange)"
               : "var(--fraud-color)";
  return (
    <td>
      <span style={{ color, fontWeight: 600, fontFamily: "var(--font-mono)" }}>
        {pct(value)}
      </span>
    </td>
  );
}

// Overfitting indicator
function OverfitCell({ train, test }) {
  if (train == null || test == null) return <td>N/A</td>;
  const gap = ((train - test) * 100).toFixed(1);
  const color = gap > 20 ? "var(--fraud-color)"
               : gap > 10 ? "var(--accent-orange)"
               : "var(--real-color)";
  return (
    <td>
      <span style={{ color, fontFamily: "var(--font-mono)", fontSize: "0.85rem", fontWeight: 500 }}>
        {gap > 0 ? "+" : ""}{gap}%
      </span>
    </td>
  );
}

const LIGHT_TOOLTIP_STYLE = {
  backgroundColor: "#ffffff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  color: "#0f172a",
  boxShadow: "0 4px 6px -1px rgba(0,0,0,0.08)",
  fontSize: "0.85rem",
};

const CHART_METRICS = [
  { key: "f1",       name: "F1 Score",  color: "#2563eb" },
  { key: "prAuc",    name: "PR-AUC",    color: "#7c3aed" },
  { key: "rocAuc",   name: "ROC-AUC",   color: "#16a34a" },
  { key: "precision",name: "Precision", color: "#d97706" },
  { key: "recall",   name: "Recall",    color: "#dc2626" },
];

export default function Metrics() {
  const [data, setData]     = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);
  const [activeMetric, setActiveMetric] = useState("prAuc");

  useEffect(() => {
    getMetrics()
      .then(({ data }) => setData(data))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading-center"><div className="spinner" /><p>Loading model evaluation metrics…</p></div>;
  if (error)   return <div className="alert alert-error">{error}</div>;

  const chartData = data.map((m) => ({
    name: m.name.replace(" (tuned)", " (Tuned)").replace(" (overfit)", " (Overfit)"),
    f1:       +(m.f1 * 100).toFixed(1),
    prAuc:    +(m.prAuc * 100).toFixed(1),
    rocAuc:   +(m.rocAuc * 100).toFixed(1),
    precision:+(m.precision * 100).toFixed(1),
    recall:   +(m.recall * 100).toFixed(1),
    isBest:   m.isBest,
  }));

  const selected = CHART_METRICS.find((m) => m.key === activeMetric);

  return (
    <div>
      <div className="page-header">
        <h1>
          <FiBarChart2 size={24} color="var(--accent-blue)" />
          <span>Model Benchmark Comparison</span>
        </h1>
        <p>
          Performance metrics across all 8 evaluated models on the held-out test set (3,576 samples). Evaluated on original class ratios without synthetic test data.
        </p>
      </div>

      {/* ── Metrics Table ───────────────────────────────────────────────── */}
      <section className="section">
        <h2 className="section-title">
          <FiAward size={18} color="var(--accent-blue)" />
          <span>Evaluation Summary Table</span>
        </h2>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
                <th title="Train PR-AUC minus Test F1 gap. Higher gap indicates overfitting.">Overfit Gap</th>
                <th>Performance</th>
              </tr>
            </thead>
            <tbody>
              {data.map((m) => (
                <tr key={m._id} className={m.isBest ? "best-row" : ""}>
                  <td>
                    <div style={{display:"flex", alignItems:"center", gap:8}}>
                      <span style={{fontWeight: m.isBest ? 700 : 500}}>{m.name}</span>
                      {m.isBest && (
                        <span className="badge badge-best" style={{display:"inline-flex", alignItems:"center", gap:4}}>
                          <FiAward size={11} />
                          <span>CHAMPION</span>
                        </span>
                      )}
                    </div>
                  </td>
                  <MetricCell value={m.precision} />
                  <MetricCell value={m.recall} />
                  <MetricCell value={m.f1} />
                  <MetricCell value={m.rocAuc} />
                  <MetricCell value={m.prAuc} />
                  <OverfitCell train={m.trainScore} test={m.testScore} />
                  <td>
                    <div style={{width:80}}>
                      <div className="progress-bar-wrap">
                        <div
                          className="progress-bar-fill"
                          style={{
                            width: `${(m.prAuc || 0) * 100}%`,
                            background: m.isBest ? "var(--accent-blue)" : "#94a3b8"
                          }}
                        />
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Bar chart ──────────────────────────────────────────────────── */}
      <section className="section">
        <h2 className="section-title">
          <FiTrendingUp size={18} color="var(--accent-blue)" />
          <span>Metric Comparison Chart</span>
        </h2>
        <div className="card">
          {/* Metric selector buttons */}
          <div style={{display:"flex", gap:8, marginBottom:"var(--space-lg)", flexWrap:"wrap"}}>
            {CHART_METRICS.map((m) => (
              <button
                key={m.key}
                className={`btn btn-sm ${activeMetric === m.key ? "btn-primary" : "btn-ghost"}`}
                onClick={() => setActiveMetric(m.key)}
              >
                {m.name}
              </button>
            ))}
          </div>

          <div style={{width:"100%", height:340}}>
            <ResponsiveContainer>
              <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#64748b", fontSize: 11 }}
                  angle={-25}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: "#64748b", fontSize: 12 }}
                  unit="%"
                />
                <Tooltip
                  contentStyle={LIGHT_TOOLTIP_STYLE}
                  formatter={(val) => [`${val}%`, selected?.name]}
                />
                <Bar
                  dataKey={activeMetric}
                  fill={selected?.color}
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </div>
  );
}
