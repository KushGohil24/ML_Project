// pages/Insights.jsx - Feature Importance, ROC/PR Curves, Confusion Matrices
import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar,
} from "recharts";
import {
  FiTrendingUp,
  FiCrosshair,
  FiSliders,
  FiTag,
  FiGrid,
  FiImage,
} from "react-icons/fi";
import { getInsights } from "../api";

const API_BASE = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace(/\/api$/, "")
  : "http://localhost:5001";

const LIGHT_TOOLTIP = {
  backgroundColor: "#ffffff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  color: "#0f172a",
  boxShadow: "0 4px 6px -1px rgba(0,0,0,0.08)",
  fontSize: "0.85rem",
};

export default function Insights() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [activeModel, setActiveModel] = useState("svm");

  useEffect(() => {
    getInsights()
      .then(({ data }) => setData(data))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading-center"><div className="spinner" /><p>Loading diagnostic insights…</p></div>;
  if (error)   return <div className="alert alert-error">{error}</div>;

  const {
    featureImportance,
    topWords,
    confusionMatrices = {},
    rocCurves = {},
    prCurves  = {},
    charts    = [],
  } = data;

  const modelsWithCurves = Object.keys(rocCurves);
  const curModel = modelsWithCurves.includes(activeModel) ? activeModel : modelsWithCurves[0] || "svm";
  const rocData  = rocCurves[curModel] || [];
  const prData   = prCurves[curModel]  || [];
  const cmData   = confusionMatrices[curModel];

  // RF feature importance top 15
  const rfFeatures = (featureImportance?.random_forest || []).slice(0, 15).map((f) => ({
    name: f.feature,
    importance: +(f.importance * 100).toFixed(2),
  }));

  // XGBoost feature importance top 15
  const xgbFeatures = (featureImportance?.xgboost || []).slice(0, 15).map((f) => ({
    name: f.feature,
    importance: +(f.importance * 100).toFixed(2),
  }));

  // Filter charts that relate to clustering & unsupervised
  const unsuperCharts = charts.filter((c) =>
    ["svd", "kmeans", "som"].some((k) => c.includes(k))
  );

  return (
    <div>
      <div className="page-header">
        <h1>
          <FiTrendingUp size={24} color="var(--accent-blue)" />
          <span>Diagnostic Insights & Interpretability</span>
        </h1>
        <p>
          Model evaluation curves, confusion matrices, distinctive class tokens, and unsupervised embeddings.
        </p>
      </div>

      {/* ── Model selector ─────────────────────────────────────────────── */}
      <div style={{display:"flex", gap:8, marginBottom:"var(--space-xl)", flexWrap:"wrap", alignItems:"center"}}>
        <span style={{fontSize:"0.85rem", color:"var(--text-secondary)", fontWeight:600, marginRight:6}}>Select Model:</span>
        {modelsWithCurves.map((m) => (
          <button
            key={m}
            className={`btn btn-sm ${curModel === m ? "btn-primary" : "btn-ghost"}`}
            onClick={() => setActiveModel(m)}
          >
            {m.replace(/_/g, " ").toUpperCase()}
          </button>
        ))}
      </div>

      {/* ── Confusion matrix + curves ──────────────────────────────────── */}
      <section className="section">
        <h2 className="section-title">
          <FiCrosshair size={18} color="var(--accent-blue)" />
          <span>Evaluation Curves & Confusion Matrix — {curModel.replace(/_/g, " ").toUpperCase()}</span>
        </h2>
        <div className="grid-3">
          {/* Confusion matrix */}
          <div className="card" style={{display:"flex", flexDirection:"column", justifyContent:"space-between"}}>
            <div>
              <div className="card-title">Confusion Matrix</div>
              {cmData ? (
                <div>
                  <div style={{display:"grid", gridTemplateColumns:"60px 1fr 1fr", gap:4, marginBottom:8, fontSize:"0.75rem", color:"var(--text-secondary)", textAlign:"center", fontWeight:600}}>
                    <div />
                    <div>PRED REAL</div>
                    <div>PRED FRAUD</div>
                  </div>
                  <div style={{display:"grid", gridTemplateColumns:"60px 1fr 1fr", gap:4, alignItems:"center"}}>
                    <div style={{fontSize:"0.75rem", color:"var(--text-secondary)", textAlign:"right", paddingRight:6, fontWeight:600}}>ACT REAL</div>
                    <div className="cm-cell cm-tn" title="True Negative (Real correctly predicted)">
                      <div>{cmData.tn?.toLocaleString()}</div>
                      <div style={{fontSize:"0.7rem", fontWeight:400, opacity:0.8}}>TN</div>
                    </div>
                    <div className="cm-cell cm-fp" title="False Positive (Real flagged as fraud)">
                      <div>{cmData.fp?.toLocaleString()}</div>
                      <div style={{fontSize:"0.7rem", fontWeight:400, opacity:0.8}}>FP</div>
                    </div>
                  </div>
                  <div style={{display:"grid", gridTemplateColumns:"60px 1fr 1fr", gap:4, alignItems:"center", marginTop:4}}>
                    <div style={{fontSize:"0.75rem", color:"var(--text-secondary)", textAlign:"right", paddingRight:6, fontWeight:600}}>ACT FRAUD</div>
                    <div className="cm-cell cm-fn" title="False Negative (Fraud missed)">
                      <div>{cmData.fn?.toLocaleString()}</div>
                      <div style={{fontSize:"0.7rem", fontWeight:400, opacity:0.8}}>FN</div>
                    </div>
                    <div className="cm-cell cm-tp" title="True Positive (Fraud correctly caught)">
                      <div>{cmData.tp?.toLocaleString()}</div>
                      <div style={{fontSize:"0.7rem", fontWeight:400, opacity:0.8}}>TP</div>
                    </div>
                  </div>
                </div>
              ) : (
                <p style={{color:"var(--text-secondary)", fontSize:"0.875rem"}}>No matrix data available.</p>
              )}
            </div>
            <div style={{fontSize:"0.75rem", color:"var(--text-secondary)", marginTop:"var(--space-md)", lineHeight:1.4}}>
              Evaluated on the un-resampled test set (3,576 postings).
            </div>
          </div>

          {/* ROC Curve */}
          <div className="card">
            <div className="card-title">ROC Curve (TPR vs FPR)</div>
            {rocData.length > 0 ? (
              <div style={{width:"100%", height:240}}>
                <ResponsiveContainer>
                  <LineChart data={rocData} margin={{top:5, right:10, left:0, bottom:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="fpr" tick={{fill:"#64748b", fontSize:11}} domain={[0,1]} />
                    <YAxis tick={{fill:"#64748b", fontSize:11}} domain={[0,1]} />
                    <Tooltip contentStyle={LIGHT_TOOLTIP} />
                    <Line type="monotone" dataKey="tpr" stroke="#2563eb" dot={false} strokeWidth={2} name="True Positive Rate" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : <p style={{color:"var(--text-secondary)", fontSize:"0.875rem"}}>No ROC curve data.</p>}
          </div>

          {/* PR Curve */}
          <div className="card">
            <div className="card-title">Precision-Recall Curve</div>
            {prData.length > 0 ? (
              <div style={{width:"100%", height:240}}>
                <ResponsiveContainer>
                  <LineChart data={prData} margin={{top:5, right:10, left:0, bottom:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="recall" tick={{fill:"#64748b", fontSize:11}} domain={[0,1]} />
                    <YAxis tick={{fill:"#64748b", fontSize:11}} domain={[0,1]} />
                    <Tooltip contentStyle={LIGHT_TOOLTIP} />
                    <Line type="monotone" dataKey="precision" stroke="#7c3aed" dot={false} strokeWidth={2} name="Precision" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : <p style={{color:"var(--text-secondary)", fontSize:"0.875rem"}}>No PR curve data.</p>}
          </div>
        </div>
      </section>

      {/* ── Feature Importance (RF & XGBoost) ──────────────────────────── */}
      {(rfFeatures.length > 0 || xgbFeatures.length > 0) && (
        <section className="section">
          <h2 className="section-title">
            <FiSliders size={18} color="var(--accent-blue)" />
            <span>Feature Importance Comparison</span>
          </h2>
          <div className="grid-2">
            {rfFeatures.length > 0 && (
              <div className="card">
                <div className="card-title">Random Forest — Top Splitting Features</div>
                <div style={{width:"100%", height:300}}>
                  <ResponsiveContainer>
                    <BarChart layout="vertical" data={rfFeatures} margin={{left:40, right:10, top:5, bottom:5}}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                      <XAxis type="number" tick={{fill:"#64748b", fontSize:11}} unit="%" />
                      <YAxis dataKey="name" type="category" tick={{fill:"#64748b", fontSize:11}} width={120} />
                      <Tooltip contentStyle={LIGHT_TOOLTIP} />
                      <Bar dataKey="importance" fill="#2563eb" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {xgbFeatures.length > 0 && (
              <div className="card">
                <div className="card-title">XGBoost — Top Splitting Features</div>
                <div style={{width:"100%", height:300}}>
                  <ResponsiveContainer>
                    <BarChart layout="vertical" data={xgbFeatures} margin={{left:40, right:10, top:5, bottom:5}}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                      <XAxis type="number" tick={{fill:"#64748b", fontSize:11}} unit="%" />
                      <YAxis dataKey="name" type="category" tick={{fill:"#64748b", fontSize:11}} width={120} />
                      <Tooltip contentStyle={LIGHT_TOOLTIP} />
                      <Bar dataKey="importance" fill="#16a34a" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {/* ── Top words per class ────────────────────────────────────────── */}
      {topWords && (
        <section className="section">
          <h2 className="section-title">
            <FiTag size={18} color="var(--accent-blue)" />
            <span>Discriminative Vocabulary (Mean TF-IDF)</span>
          </h2>
          <div className="grid-2">
            <div className="card">
              <div className="card-title" style={{color:"var(--fraud-color)"}}>
                Top Tokens in Fraudulent Postings
              </div>
              <div className="chips-wrap">
                {(topWords.fraud || []).slice(0, 18).map(({ word, score }) => (
                  <span className="chip chip-flag" key={word}>
                    {word} <span style={{opacity:0.65, fontSize:"0.72rem"}}>({score})</span>
                  </span>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="card-title" style={{color:"var(--real-color)"}}>
                Top Tokens in Genuine Postings
              </div>
              <div className="chips-wrap">
                {(topWords.real || []).slice(0, 18).map(({ word, score }) => (
                  <span className="chip chip-word" key={word}>
                    {word} <span style={{opacity:0.65, fontSize:"0.72rem"}}>({score})</span>
                  </span>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ── Unsupervised Visualizations ─────────────────────────────────── */}
      {unsuperCharts.length > 0 && (
        <section className="section">
          <h2 className="section-title">
            <FiGrid size={18} color="var(--accent-blue)" />
            <span>Unsupervised Embeddings & Topology</span>
          </h2>
          <div className="chart-grid">
            {unsuperCharts.map((url) => (
              <div className="chart-card" key={url}>
                <div className="chart-card-img-wrap">
                  <img
                    src={`${API_BASE}${url}`}
                    alt={url}
                    loading="lazy"
                    onError={(e) => { e.target.style.display = "none"; }}
                  />
                </div>
                <div className="chart-card-title">{url.split("/").pop().replace(".png","").replace(/_/g," ")}</div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
