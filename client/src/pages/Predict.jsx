// pages/Predict.jsx - Job fraud prediction form with smooth scroll & professional icons
import { useState, useRef } from "react";
import { predictJob } from "../api";
import {
  FiSearch,
  FiAlertTriangle,
  FiCheckCircle,
  FiRotateCcw,
  FiFileText,
  FiSliders,
  FiAlertOctagon,
  FiCheck,
  FiFlag,
  FiTag,
  FiLoader,
  FiArrowUp,
  FiShield,
} from "react-icons/fi";

// ── Sample data for quick testing ───────────────────────────────────────────
const SCAM_SAMPLE = {
  title: "Work From Home Data Entry - Earn $500/day - No Experience Required!",
  company_profile: "We are a fast-growing company offering unlimited earning potential. Be your own boss!",
  description: "Make money fast from home! Guaranteed income. Multi-level marketing opportunity. No experience needed. Just send a small registration fee of $50 to get your starter kit and begin earning immediately. Wire transfer or Western union accepted.",
  requirements: "No experience required. Must be willing to pay upfront fee. Must have internet access.",
  benefits: "Unlimited earning potential. Work from home. Be your own boss.",
  salary_range: "",
  employment_type: "Part-time",
  required_education: "Unspecified",
  required_experience: "Not Applicable",
  telecommuting: 1,
  has_company_logo: 0,
  has_questions: 0,
};

const REAL_SAMPLE = {
  title: "Senior Software Engineer - React / Node.js",
  company_profile: "Acme Corp is a publicly listed technology company headquartered in San Francisco. We build enterprise SaaS products serving 10,000+ businesses worldwide.",
  description: "We are seeking an experienced Senior Software Engineer to join our product team. You will design, build, and maintain scalable web applications. You'll work closely with product managers and designers in an Agile environment.",
  requirements: "5+ years of experience with React and Node.js. Experience with MongoDB and REST APIs. BS or MS in Computer Science or related field. Strong understanding of software engineering principles.",
  benefits: "Competitive salary ($120k-$160k). Health, dental, vision insurance. 401k with matching. Remote-friendly work environment. Annual learning budget.",
  salary_range: "120000-160000",
  employment_type: "Full-time",
  required_education: "Bachelor's Degree",
  required_experience: "Mid-Senior level",
  telecommuting: 0,
  has_company_logo: 1,
  has_questions: 1,
};

const EMPTY = {
  title: "", company_profile: "", description: "",
  requirements: "", benefits: "",
  salary_range: "", employment_type: "Unknown",
  required_education: "Unknown", required_experience: "Unknown",
  telecommuting: 0, has_company_logo: 1, has_questions: 0,
};

// ── Probability bar ──────────────────────────────────────────────────────────
function ProbabilityBar({ prob }) {
  const pct = Math.round(prob * 100);
  const color = prob >= 0.7 ? "var(--fraud-color)"
              : prob >= 0.4 ? "var(--accent-orange)"
              : "var(--real-color)";
  return (
    <div>
      <div style={{display:"flex", justifyContent:"space-between", marginBottom:6}}>
        <span style={{fontSize:"0.85rem", color:"var(--text-secondary)", fontWeight:500}}>Calculated Risk Score</span>
        <span style={{fontWeight:700, color, fontFamily:"var(--font-mono)"}}>{pct}%</span>
      </div>
      <div className="progress-bar-wrap" style={{height:10}}>
        <div className="progress-bar-fill" style={{width:`${pct}%`, background:color}} />
      </div>
    </div>
  );
}

export default function Predict() {
  const [form, setForm] = useState(EMPTY);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError]   = useState(null);

  // Ref to scroll up to where the result is displayed
  const resultRef = useRef(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm(prev => ({
      ...prev,
      [name]: type === "checkbox" ? (checked ? 1 : 0) : value,
    }));
  };

  const scrollToResult = () => {
    if (resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    // Scroll up immediately when user clicks analyze
    scrollToResult();

    try {
      const { data } = await predictJob(form);
      setResult(data);
      // Ensure smooth scroll remains focused on the freshly rendered card
      setTimeout(scrollToResult, 50);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
      setTimeout(scrollToResult, 50);
    } finally {
      setLoading(false);
    }
  };

  const fillSample = (sample) => {
    setForm({ ...EMPTY, ...sample });
    setResult(null);
    setError(null);
  };

  const isFraud = result?.label === "Fraudulent";

  return (
    <div>
      {/* ── Page Header ────────────────────────────────────────────── */}
      <div className="page-header">
        <h1>
          <FiSearch size={24} color="var(--accent-blue)" />
          Job Fraud Classification
        </h1>
        <p>Evaluate job posting text and organizational metadata using our trained Support Vector Machine classifier.</p>
      </div>

      {/* ── Preset Buttons ─────────────────────────────────────────── */}
      <div style={{display:"flex", gap:"var(--space-md)", marginBottom:"var(--space-xl)", flexWrap:"wrap"}}>
        <button type="button" className="btn btn-danger" onClick={() => fillSample(SCAM_SAMPLE)}>
          <FiAlertTriangle size={15} />
          <span>Load Fraudulent Sample</span>
        </button>
        <button type="button" className="btn btn-success" onClick={() => fillSample(REAL_SAMPLE)}>
          <FiCheckCircle size={15} />
          <span>Load Genuine Sample</span>
        </button>
        <button type="button" className="btn btn-ghost" onClick={() => fillSample(EMPTY)}>
          <FiRotateCcw size={15} />
          <span>Reset Form</span>
        </button>
      </div>

      <div className="grid-2" style={{gap:"var(--space-xl)", alignItems:"start"}}>
        {/* ── Input Form (Left Column) ─────────────────────────────── */}
        <form onSubmit={handleSubmit}>
          <div className="card" style={{marginBottom:"var(--space-lg)"}}>
            <div className="card-title">
              <FiFileText size={16} color="var(--accent-blue)" />
              <span>Job Posting Content</span>
            </div>
            {[
              { name: "title",           label: "Job Title *",       rows: 1, req: true },
              { name: "description",     label: "Job Description",   rows: 5 },
              { name: "requirements",    label: "Requirements",      rows: 4 },
              { name: "company_profile", label: "Company Profile",   rows: 3 },
              { name: "benefits",        label: "Benefits",          rows: 2 },
            ].map(({ name, label, rows, req }) => (
              <div className="form-group" key={name}>
                <label className="form-label">{label}</label>
                <textarea
                  className="form-control"
                  name={name}
                  rows={rows}
                  value={form[name]}
                  onChange={handleChange}
                  required={req}
                  placeholder={`Enter ${label.toLowerCase()}...`}
                />
              </div>
            ))}
          </div>

          <div className="card" style={{marginBottom:"var(--space-lg)"}}>
            <div className="card-title">
              <FiSliders size={16} color="var(--accent-blue)" />
              <span>Metadata & Structural Flags</span>
            </div>

            <div className="grid-2" style={{gap:"var(--space-md)"}}>
              <div className="form-group">
                <label className="form-label">Employment Type</label>
                <select className="form-control" name="employment_type" value={form.employment_type} onChange={handleChange}>
                  {["Unknown","Full-time","Part-time","Contract","Temporary","Other"].map(o=><option key={o}>{o}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Required Experience</label>
                <select className="form-control" name="required_experience" value={form.required_experience} onChange={handleChange}>
                  {["Unknown","Entry level","Mid-Senior level","Associate","Director","Executive","Not Applicable"].map(o=><option key={o}>{o}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Required Education</label>
                <select className="form-control" name="required_education" value={form.required_education} onChange={handleChange}>
                  {["Unknown","Bachelor's Degree","High School or equivalent","Master's Degree","Associate Degree","Unspecified"].map(o=><option key={o}>{o}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Salary Range</label>
                <input
                  className="form-control"
                  name="salary_range"
                  value={form.salary_range}
                  onChange={handleChange}
                  placeholder="e.g. 60000-80000 or empty"
                />
              </div>
            </div>

            {/* Checkbox flags */}
            <div style={{display:"flex", gap:"var(--space-lg)", marginTop:"var(--space-sm)", flexWrap:"wrap"}}>
              {[
                { name: "has_company_logo", label: "Company Logo Present" },
                { name: "telecommuting",    label: "Remote / Telecommuting" },
                { name: "has_questions",    label: "Screening Questions" },
              ].map(({ name, label }) => (
                <label key={name} style={{display:"flex", alignItems:"center", gap:8, cursor:"pointer", fontSize:"0.875rem", color:"var(--text-secondary)"}}>
                  <input
                    type="checkbox"
                    name={name}
                    checked={Boolean(form[name])}
                    onChange={handleChange}
                    style={{accentColor:"var(--accent-blue)", width:16, height:16}}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{width:"100%", padding:"13px", fontSize:"0.95rem"}}
          >
            {loading ? (
              <>
                <FiLoader size={16} className="spinner" style={{margin:0, width:16, height:16, borderWidth:2}} />
                <span>Running Machine Learning Inference...</span>
              </>
            ) : (
              <>
                <FiSearch size={16} />
                <span>Analyze Job Posting</span>
              </>
            )}
          </button>
        </form>

        {/* ── Results Panel (Right Column) ─────────────────────────── */}
        <div ref={resultRef} style={{scrollMarginTop: "80px"}}>
          {loading && (
            <div className="card" style={{textAlign:"center", padding:"var(--space-2xl)"}}>
              <div className="spinner" />
              <div style={{fontWeight:600, marginTop:"var(--space-md)", color:"var(--text-primary)"}}>
                Extracting Features & Running Classifier
              </div>
              <p style={{color:"var(--text-secondary)", fontSize:"0.85rem", marginTop:4}}>
                Transforming TF-IDF n-grams and evaluating decision boundary...
              </p>
            </div>
          )}

          {error && (
            <div className="alert alert-error">
              <FiAlertOctagon size={18} />
              <div>
                <strong>Inference Error:</strong> {error}
              </div>
            </div>
          )}

          {result && !loading && (
            <div className={`result-card ${isFraud ? "fraud" : "real"}`}>
              {/* Verdict header */}
              <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:"var(--space-md)"}}>
                <span style={{fontSize:"0.8rem", textTransform:"uppercase", fontWeight:700, letterSpacing:"0.05em", color:"var(--text-secondary)"}}>
                  Inference Verdict
                </span>
                <span className={`badge ${isFraud ? "badge-fraud" : "badge-real"}`}>
                  {isFraud ? "High Risk Alert" : "Verified Legitimate"}
                </span>
              </div>

              <div className={`result-verdict ${isFraud ? "fraud-text" : "real-text"}`}>
                {isFraud ? <FiAlertOctagon size={36} /> : <FiCheckCircle size={36} />}
                <span>{result.label}</span>
              </div>

              {/* Risk bar */}
              <div style={{marginBottom:"var(--space-xl)"}}>
                <ProbabilityBar prob={result.fraud_probability} />
              </div>

              {/* Red flags */}
              {result.red_flags?.length > 0 && (
                <div style={{marginBottom:"var(--space-lg)"}}>
                  <div style={{fontWeight:600, fontSize:"0.85rem", color:"var(--fraud-color)", display:"flex", alignItems:"center", gap:6, marginBottom:6}}>
                    <FiFlag size={14} />
                    <span>Suspicious Structural Indicators ({result.red_flags.length})</span>
                  </div>
                  <div className="chips-wrap">
                    {result.red_flags.map((flag, i) => (
                      <span className="chip chip-flag" key={i}>
                        <FiAlertTriangle size={12} />
                        {flag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Distinctive keywords */}
              {result.top_words?.length > 0 && (
                <div>
                  <div style={{fontWeight:600, fontSize:"0.85rem", color:"var(--text-secondary)", display:"flex", alignItems:"center", gap:6, marginBottom:6}}>
                    <FiTag size={14} />
                    <span>Salient Text Tokens (TF-IDF Weight)</span>
                  </div>
                  <div className="chips-wrap">
                    {result.top_words.map(({ word, score }) => (
                      <span className="chip chip-word" key={word}>
                        {word} <span style={{opacity:0.65, fontSize:"0.72rem"}}>({score})</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {!result && !loading && !error && (
            <div className="card" style={{padding:"var(--space-2xl)", textAlign:"center", color:"var(--text-secondary)"}}>
              <FiShield size={44} style={{color:"var(--border-focus)", opacity:0.6, margin:"0 auto var(--space-md)"}} />
              <div style={{fontWeight:600, fontSize:"1rem", color:"var(--text-primary)", marginBottom:4}}>
                No Job Posting Evaluated Yet
              </div>
              <p style={{fontSize:"0.875rem", maxWidth:320, margin:"0 auto"}}>
                Fill the form or load one of the quick test presets above, then click <strong>Analyze Job Posting</strong> to view classification results here.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
