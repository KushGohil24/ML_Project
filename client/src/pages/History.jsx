// pages/History.jsx - Saved prediction history with delete
import { useEffect, useState } from "react";
import {
  FiClock,
  FiRefreshCw,
  FiInbox,
  FiTrash2,
  FiAlertTriangle,
  FiCheckCircle,
} from "react-icons/fi";
import { getHistory, deleteHistory } from "../api";

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [deleting, setDeleting] = useState(null);

  const load = () => {
    setLoading(true);
    getHistory()
      .then(({ data }) => { setHistory(data); setError(null); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleDelete = async (id) => {
    if (!confirm("Are you sure you want to delete this prediction log?")) return;
    setDeleting(id);
    try {
      await deleteHistory(id);
      setHistory((prev) => prev.filter((h) => h._id !== id));
    } catch (e) {
      alert("Error: " + e.message);
    } finally {
      setDeleting(null);
    }
  };

  if (loading) return <div className="loading-center"><div className="spinner" /><p>Loading prediction logs…</p></div>;

  return (
    <div>
      <div className="page-header" style={{display:"flex", justifyContent:"space-between", alignItems:"flex-start", flexWrap:"wrap", gap:"var(--space-md)"}}>
        <div>
          <h1>
            <FiClock size={24} color="var(--accent-blue)" />
            <span>Prediction Audit Log</span>
          </h1>
          <p>Historical record of evaluated job postings stored in MongoDB.</p>
        </div>
        <button className="btn btn-ghost" onClick={load}>
          <FiRefreshCw size={14} />
          <span>Refresh Data</span>
        </button>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {history.length === 0 ? (
        <div className="card" style={{textAlign:"center", padding:"var(--space-2xl)", color:"var(--text-secondary)"}}>
          <FiInbox size={48} style={{margin:"0 auto var(--space-md)", color:"#94a3b8"}} />
          <div style={{fontWeight:600, color:"var(--text-primary)", marginBottom:4}}>No Predictions Recorded</div>
          <p style={{fontSize:"0.875rem"}}>Submit a job posting on the <strong>Detect</strong> page to populate this audit log.</p>
        </div>
      ) : (
        <>
          {/* ── Summary stats ────────────────────────────────────────────── */}
          <div className="grid-4" style={{marginBottom:"var(--space-xl)"}}>
            <div className="stat-card">
              <div className="stat-value">{history.length}</div>
              <div className="stat-label">Total Predictions Logged</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{color:"var(--fraud-color)"}}>
                {history.filter(h => h.label === "Fraudulent").length}
              </div>
              <div className="stat-label">Flagged as Fraudulent</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{color:"var(--real-color)"}}>
                {history.filter(h => h.label === "Real").length}
              </div>
              <div className="stat-label">Verified Legitimate</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {history.length > 0
                  ? `${((history.filter(h => h.label === "Fraudulent").length / history.length) * 100).toFixed(0)}%`
                  : "0%"}
              </div>
              <div className="stat-label">Flagged Proportion</div>
            </div>
          </div>

          {/* ── Table ────────────────────────────────────────────────────── */}
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Job Title</th>
                  <th>Verdict</th>
                  <th>Fraud Probability</th>
                  <th>Identified Red Flags</th>
                  <th>Timestamp</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => {
                  const isFraud = h.label === "Fraudulent";
                  const pct = Math.round((h.fraud_probability || 0) * 100);
                  const color = pct >= 70 ? "var(--fraud-color)"
                              : pct >= 40 ? "var(--accent-orange)"
                              : "var(--real-color)";
                  const date = h.createdAt
                    ? new Date(h.createdAt).toLocaleString(undefined, {
                        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
                      })
                    : "—";

                  return (
                    <tr key={h._id}>
                      <td style={{fontWeight:500, maxWidth:260}}>
                        <div style={{textOverflow:"ellipsis", overflow:"hidden", whiteSpace:"nowrap"}} title={h.title || "Untitled"}>
                          {h.title || "Untitled"}
                        </div>
                      </td>
                      <td>
                        <span className={`badge ${isFraud ? "badge-fraud" : "badge-real"}`}>
                          {isFraud ? <FiAlertTriangle size={11} /> : <FiCheckCircle size={11} />}
                          <span>{h.label}</span>
                        </span>
                      </td>
                      <td>
                        <div style={{display:"flex", alignItems:"center", gap:8}}>
                          <span style={{color, fontWeight:600, fontFamily:"var(--font-mono)", width:36, fontSize:"0.85rem"}}>
                            {pct}%
                          </span>
                          <div style={{width:60}}>
                            <div className="progress-bar-wrap">
                              <div className="progress-bar-fill" style={{width:`${pct}%`, background:color}} />
                            </div>
                          </div>
                        </div>
                      </td>
                      <td style={{maxWidth:240}}>
                        {h.red_flags?.length > 0 ? (
                          <div style={{display:"flex", flexWrap:"wrap", gap:4}}>
                            {h.red_flags.slice(0, 2).map((f, i) => (
                              <span className="chip chip-flag" key={i} style={{fontSize:"0.72rem", padding:"2px 8px"}}>
                                {f}
                              </span>
                            ))}
                            {h.red_flags.length > 2 && (
                              <span style={{fontSize:"0.75rem", color:"var(--text-secondary)"}}>
                                +{h.red_flags.length - 2} more
                              </span>
                            )}
                          </div>
                        ) : (
                          <span style={{color:"var(--text-muted)", fontSize:"0.8rem"}}>None</span>
                        )}
                      </td>
                      <td style={{color:"var(--text-secondary)", fontSize:"0.8rem", whiteSpace:"nowrap"}}>
                        {date}
                      </td>
                      <td>
                        <button
                          className="btn btn-ghost btn-sm"
                          style={{color:"var(--fraud-color)", borderColor:"transparent", padding:"4px 8px"}}
                          disabled={deleting === h._id}
                          onClick={() => handleDelete(h._id)}
                          title="Delete entry"
                        >
                          <FiTrash2 size={14} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
