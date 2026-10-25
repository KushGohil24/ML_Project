// pages/Home.jsx - Landing page
import { Link } from "react-router-dom";
import {
  FiCpu,
  FiPieChart,
  FiSliders,
  FiSearch,
  FiGrid,
  FiBarChart2,
  FiArrowRight,
  FiCheckCircle,
  FiShield,
  FiFileText,
  FiActivity,
  FiLayers,
} from "react-icons/fi";

const FEATURES = [
  { Icon: FiCpu,        title: "9 Machine Learning Models", desc: "Logistic Regression, Support Vector Machine, KNN, Decision Trees, Random Forest, Gradient Boosting, XGBoost, and Keras Deep Learning." },
  { Icon: FiPieChart,   title: "Exploratory Data Analysis", desc: "Detailed class balance analysis, missing field distributions, fraud rates across industries, and token length insights." },
  { Icon: FiSliders,    title: "SMOTE Imbalance Correction", desc: "Corrects 95:5 class imbalance using Synthetic Minority Oversampling strictly on training data without leakage." },
  { Icon: FiSearch,     title: "Model Explainability", desc: "Tree feature importance, top discriminative tokens per class, and actionable structural red flag detections." },
  { Icon: FiGrid,       title: "Unsupervised Clustering", desc: "TruncatedSVD 2D dimensionality reduction, K-Means elbow/silhouette metrics, and Self-Organizing Maps (MiniSom)." },
  { Icon: FiBarChart2,  title: "Honest Evaluation Metrics", desc: "Evaluated using Precision-Recall AUC (PR-AUC), ROC-AUC, and F1-Score rather than deceptively high overall accuracy." },
];

const TECH = [
  { label: "Benchmark Dataset", value: "EMSCAD (17,880 Postings)" },
  { label: "Feature Matrix",    value: "10,810 (TF-IDF + Metadata)" },
  { label: "Optimal Architecture", value: "Linear SVM (PR-AUC 0.936)" },
  { label: "System Stack",      value: "Python · Express · React · Mongo" },
];

export default function Home() {
  return (
    <div>
      {/* ── Hero ────────────────────────────────────────────────── */}
      <section className="hero">
        <h1 className="hero-title">
          Enterprise Detection of <br />
          <span className="highlight">Fraudulent Job Postings</span>
        </h1>
        <p className="hero-sub">
          An end-to-end Machine Learning platform engineered to classify online recruitment listings as legitimate or fraudulent using Natural Language Processing and supervised classification.
        </p>
        <div className="hero-actions">
          <Link to="/predict" className="btn btn-primary" style={{padding:"12px 28px", fontSize:"0.95rem"}}>
            <FiSearch size={16} />
            <span>Analyze a Job Posting</span>
          </Link>
        </div>
      </section>

      {/* ── Key Metrics Overview ────────────────────────────────── */}
      <section className="section">
        <div className="grid-4">
          {TECH.map(({ label, value }) => (
            <div className="stat-card" key={label}>
              <div className="stat-value" style={{fontSize:"1.1rem", fontWeight:700}}>{value}</div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ──────────────────────────────────────────── */}
      <section className="section">
        <h2 className="section-title">
          <FiLayers size={18} color="var(--accent-blue)" />
          <span>Core Capabilities & ML Pipeline</span>
        </h2>
        <div className="grid-3">
          {FEATURES.map(({ Icon, title, desc }) => (
            <div className="card" key={title} style={{display:"flex", flexDirection:"column", gap:"var(--space-sm)"}}>
              <div style={{
                width: 36, height: 36, borderRadius: "var(--radius-sm)",
                background: "var(--accent-blue-subtle)", color: "var(--accent-blue)",
                display: "flex", alignItems: "center", justifyContent: "center"
              }}>
                <Icon size={18} />
              </div>
              <div style={{fontWeight:600, fontSize:"0.95rem", color:"var(--text-primary)"}}>{title}</div>
              <div style={{color:"var(--text-secondary)", fontSize:"0.875rem", lineHeight:1.5}}>{desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── How It Works ──────────────────────────────────────── */}
      <section className="section">
        <h2 className="section-title">
          <FiActivity size={18} color="var(--accent-blue)" />
          <span>Inference Pipeline</span>
        </h2>
        <div className="card">
          <div style={{display:"flex", gap:"var(--space-lg)", flexWrap:"wrap", alignItems:"flex-start"}}>
            {[
              ["1", "Posting Submission", "Job title, description, requirements, and metadata are entered."],
              ["2", "Vectorization & Signals", "Text is processed with TF-IDF; domain features (flags, emails) are engineered."],
              ["3", "Inference Evaluation", "Trained SVM evaluates probability scores against the optimal decision threshold."],
              ["4", "Diagnostic Breakdown", "Verdict, fraud probability score, suspicious tokens, and red flags are surfaced."],
            ].map(([num, title, desc]) => (
              <div key={num} style={{flex:"1", minWidth:200, display:"flex", gap:"var(--space-md)", alignItems:"flex-start"}}>
                <div style={{
                  minWidth:30, height:30, borderRadius:"var(--radius-sm)",
                  background:"var(--accent-blue-subtle)", color:"var(--accent-blue)", display:"flex",
                  alignItems:"center", justifyContent:"center",
                  fontWeight:700, fontSize:"0.85rem", border:"1px solid #bfdbfe"
                }}>{num}</div>
                <div>
                  <div style={{fontWeight:600, fontSize:"0.9rem", marginBottom:4, color:"var(--text-primary)"}}>{title}</div>
                  <div style={{color:"var(--text-secondary)", fontSize:"0.825rem", lineHeight:1.45}}>{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Call to action ─────────────────────────────────────── */}
      <section style={{textAlign:"center", padding:"var(--space-xl) 0"}}>
        <h2 style={{fontSize:"1.35rem", fontWeight:700, marginBottom:"var(--space-sm)", color:"var(--text-primary)"}}>
          Ready to verify job listing authenticity?
        </h2>
        <p style={{color:"var(--text-secondary)", fontSize:"0.95rem", marginBottom:"var(--space-lg)"}}>
          Test arbitrary job postings or evaluate pre-configured fraudulent and genuine listings.
        </p>
        <Link to="/predict" className="btn btn-primary" style={{padding:"12px 32px", fontSize:"0.95rem"}}>
          <span>Launch Classifier</span>
          <FiArrowRight size={16} />
        </Link>
      </section>
    </div>
  );
}
