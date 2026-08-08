const { useState, useEffect } = React;

// Change this if your backend runs elsewhere (e.g. via docker-compose service name)
const API_BASE = "http://localhost:8000";

const FIELD_OPTIONS = {
  gender: ["Male", "Female"],
  Partner: ["Yes", "No"],
  Dependents: ["Yes", "No"],
  PhoneService: ["Yes", "No"],
  MultipleLines: ["Yes", "No", "No phone service"],
  InternetService: ["DSL", "Fiber optic", "No"],
  OnlineSecurity: ["Yes", "No", "No internet service"],
  TechSupport: ["Yes", "No", "No internet service"],
  StreamingTV: ["Yes", "No", "No internet service"],
  Contract: ["Month-to-month", "One year", "Two year"],
  PaperlessBilling: ["Yes", "No"],
  PaymentMethod: ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
};

const DEFAULT_FORM = {
  gender: "Female", SeniorCitizen: 0, Partner: "Yes", Dependents: "No",
  tenure: 5, PhoneService: "Yes", MultipleLines: "No",
  InternetService: "Fiber optic", OnlineSecurity: "No", TechSupport: "No",
  StreamingTV: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check", MonthlyCharges: 89.5, TotalCharges: 450,
};

function Tabs({ active, setActive }) {
  const tabs = [
    ["single", "Single Prediction"],
    ["batch", "Batch CSV Upload"],
    ["records", "Prediction Records (CRUD)"],
  ];
  return (
    <div className="tabs">
      {tabs.map(([key, label]) => (
        <button
          key={key}
          className={"tab" + (active === key ? " active" : "")}
          onClick={() => setActive(key)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function SinglePrediction() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (key, value) => setForm({ ...form, [key]: value });

  const submit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/predict/single`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          SeniorCitizen: Number(form.SeniorCitizen),
          tenure: Number(form.tenure),
          MonthlyCharges: Number(form.MonthlyCharges),
          TotalCharges: Number(form.TotalCharges),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prediction failed");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h3>Customer Details</h3>
      <div className="grid">
        {Object.entries(FIELD_OPTIONS).map(([key, options]) => (
          <div key={key}>
            <label>{key}</label>
            <select value={form[key]} onChange={(e) => handleChange(key, e.target.value)}>
              {options.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
        ))}
        <div>
          <label>Senior Citizen (0/1)</label>
          <input type="number" min="0" max="1" value={form.SeniorCitizen}
                 onChange={(e) => handleChange("SeniorCitizen", e.target.value)} />
        </div>
        <div>
          <label>Tenure (months)</label>
          <input type="number" value={form.tenure}
                 onChange={(e) => handleChange("tenure", e.target.value)} />
        </div>
        <div>
          <label>Monthly Charges</label>
          <input type="number" step="0.01" value={form.MonthlyCharges}
                 onChange={(e) => handleChange("MonthlyCharges", e.target.value)} />
        </div>
        <div>
          <label>Total Charges</label>
          <input type="number" step="0.01" value={form.TotalCharges}
                 onChange={(e) => handleChange("TotalCharges", e.target.value)} />
        </div>
      </div>

      <button className="primary" onClick={submit} disabled={loading}>
        {loading ? "Predicting..." : "Predict Churn"}
      </button>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="result-box">
          <p><strong>Prediction:</strong> {result.churn_prediction}</p>
          <p><strong>Probability:</strong> {(result.churn_probability * 100).toFixed(1)}%</p>
          <p><strong>Risk Level:</strong> <span className={"badge " + result.risk_level}>{result.risk_level}</span></p>
        </div>
      )}
    </div>
  );
}

function BatchUpload() {
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const upload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setSummary(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/predict/batch`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setSummary(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h3>Upload Customer CSV</h3>
      <p style={{ color: "#6b7280", fontSize: 13 }}>
        CSV must contain the standard churn schema columns (gender, tenure,
        Contract, MonthlyCharges, TotalCharges, etc). Invalid files are
        rejected with a clear error message.
      </p>
      <div className="dropzone" onClick={() => document.getElementById("fileInput").click()}>
        {file ? file.name : "Click to choose a .csv file"}
      </div>
      <input id="fileInput" type="file" accept=".csv" style={{ display: "none" }}
             onChange={(e) => setFile(e.target.files[0])} />

      <button className="primary" onClick={upload} disabled={!file || loading}>
        {loading ? "Processing..." : "Upload & Predict"}
      </button>

      {error && <div className="error-box">{error}</div>}

      {summary && (
        <div className="result-box">
          <p><strong>Rows processed:</strong> {summary.total_rows}</p>
          <p><strong>Predicted Churn = Yes:</strong> {summary.predicted_churn_yes}</p>
          <p><strong>Predicted Churn = No:</strong> {summary.predicted_churn_no}</p>
          <a className="download-btn" href={`${API_BASE}${summary.download_url}`}>
            Download Result CSV
          </a>
        </div>
      )}
    </div>
  );
}

function Records() {
  const [records, setRecords] = useState([]);
  const [error, setError] = useState(null);

  const load = async () => {
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/customers/?limit=20`);
      if (!res.ok) throw new Error("Failed to load records");
      setRecords(await res.json());
    } catch (e) {
      setError(e.message);
    }
  };

  const remove = async (id) => {
    await fetch(`${API_BASE}/customers/${id}`, { method: "DELETE" });
    load();
  };

  useEffect(() => { load(); }, []);

  return (
    <div className="card">
      <h3>Recent Prediction Records</h3>
      <button className="primary" onClick={load}>Refresh</button>
      {error && <div className="error-box">{error}</div>}
      <table>
        <thead>
          <tr><th>ID</th><th>Customer ID</th><th>Prediction</th><th>Probability</th><th>Risk</th><th>Created</th><th></th></tr>
        </thead>
        <tbody>
          {records.map((r) => (
            <tr key={r.id}>
              <td>{r.id}</td>
              <td>{r.customer_id || "-"}</td>
              <td>{r.churn_prediction}</td>
              <td>{(r.churn_probability * 100).toFixed(1)}%</td>
              <td><span className={"badge " + r.risk_level}>{r.risk_level}</span></td>
              <td>{new Date(r.created_at).toLocaleString()}</td>
              <td><button onClick={() => remove(r.id)} style={{ color: "#dc2626", border: "none", background: "none", cursor: "pointer" }}>Delete</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function App() {
  const [active, setActive] = useState("single");
  return (
    <div className="app">
      <header>
        <h1>Telecom Customer Churn Prediction</h1>
        <p>ML-powered risk scoring - single lookup, batch CSV, and full CRUD history.</p>
      </header>
      <Tabs active={active} setActive={setActive} />
      {active === "single" && <SinglePrediction />}
      {active === "batch" && <BatchUpload />}
      {active === "records" && <Records />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
