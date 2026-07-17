// frontend/src/App.jsx
import React, { useState, useEffect } from "react";

function FileUploader({ onUpload }) {
  const [file, setFile] = useState(null);
  const handleChange = (e) => setFile(e.target.files?.[0] ?? null);
  const handleUpload = async () => {
    if (!file) return alert("Choose a file first");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      const data = await res.json();
      onUpload?.(data);
      alert("Upload: " + JSON.stringify(data));
    } catch (err) {
      alert("Upload failed: " + err.message);
    }
  };
  return (
    <div className="p-4 rounded-2xl border border-gray-200 bg-white/5">
      <h3 className="text-xl font-semibold mb-2">Upload dataset / model files</h3>
      <input type="file" onChange={handleChange} className="mb-3" />
      <div>
        <button className="px-4 py-2 rounded bg-blue-600 text-white" onClick={handleUpload}>
          Upload
        </button>
      </div>
    </div>
  );
}

function TrainPanel() {
  const [running, setRunning] = useState(false);
  const [output, setOutput] = useState("");

  const handleTrain = async () => {
    setRunning(true);
    setOutput("");
    try {
      const res = await fetch("/api/train", { method: "POST" });
      if (!res.body) {
        setOutput("No streaming body available.");
        setRunning(false);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      while (!done) {
        const { value, done: d } = await reader.read();
        if (value) setOutput((p) => p + decoder.decode(value));
        done = d;
      }
    } catch (err) {
      setOutput((p) => p + "\n[error] " + err.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="p-4 rounded-2xl border border-gray-200 bg-white/5">
      <h3 className="text-xl font-semibold mb-2">Train / Retrain Model</h3>
      <p className="text-sm text-gray-300">Starts training using your existing pipeline and streams logs.</p>
      <div className="mt-3">
        <button
          disabled={running}
          onClick={handleTrain}
          className={`px-4 py-2 rounded ${running ? "bg-gray-500 text-white" : "bg-green-600 text-white"}`}
        >
          {running ? "Training..." : "Start Training"}
        </button>
      </div>
      <pre className="mt-3 max-h-48 overflow-auto p-2 bg-black text-green-200 text-sm rounded">{output || "No live logs yet..."}</pre>
    </div>
  );
}

function PredictPanel() {
  const [engineId, setEngineId] = useState(10);
  const [cycle, setCycle] = useState(100);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  const handlePredict = async () => {
    setBusy(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine_id: Number(engineId), cycle: Number(cycle) }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="p-4 rounded-2xl border border-gray-200 bg-white/5">
      <h3 className="text-xl font-semibold mb-2">Predict RUL (case study)</h3>
      <div className="grid grid-cols-2 gap-3">
        <label className="text-sm">Engine ID
          <input className="block mt-1 w-32 p-1 rounded border" value={engineId} onChange={(e) => setEngineId(e.target.value)} />
        </label>
        <label className="text-sm">Cycle
          <input className="block mt-1 w-32 p-1 rounded border" value={cycle} onChange={(e) => setCycle(e.target.value)} />
        </label>
      </div>
      <div className="mt-3">
        <button onClick={handlePredict} disabled={busy} className="px-4 py-2 rounded bg-indigo-600 text-white">
          {busy ? "Predicting..." : "Predict"}
        </button>
      </div>
      {result && (
        <div className="mt-3 p-3 rounded border bg-white/3">
          <p><strong>Predicted RUL:</strong> {result.predicted_rul ?? "N/A"}</p>
          <pre className="text-xs whitespace-pre-wrap">{result.raw_output ?? result.error ?? ""}</pre>
        </div>
      )}
    </div>
  );
}

function PlotViewer() {
  const [exists, setExists] = useState(true);
  useEffect(() => {
    fetch("/api/plot", { method: "HEAD" }).then((r) => {
      setExists(r.status === 200);
    }).catch(() => setExists(false));
  }, []);
  return (
    <div className="p-4 rounded-2xl border border-gray-200 bg-white/5">
      <h3 className="text-xl font-semibold mb-2">Evaluation Plot</h3>
      <p className="text-sm text-gray-300">This will show rul_plot_final_evaluation.png if it exists.</p>
      {exists ? <img src="/api/plot" alt="RUL Plot" className="mt-3 w-full rounded border" /> : <div className="mt-3 text-sm text-gray-400">Plot not found</div>}
    </div>
  );
}

function LogsViewer() {
  const [logs, setLogs] = useState("");
  useEffect(() => {
    fetch("/api/logs").then((r) => {
      if (r.status === 200) return r.text();
      return "";
    }).then((t) => setLogs(t)).catch(() => setLogs("No logs found."));
  }, []);
  return (
    <div className="p-4 rounded-2xl border border-gray-200 bg-white/5">
      <h3 className="text-xl font-semibold mb-2">Latest Logs</h3>
      <pre className="max-h-64 overflow-auto bg-black text-green-200 p-2 text-sm rounded">{logs || "No logs found."}</pre>
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen p-8 bg-slate-50 text-slate-900">
      <div className="max-w-6xl mx-auto">
        <header className="mb-6">
          <h1 className="text-4xl font-extrabold">Predictive Maintenance — Web UI</h1>
          <p className="text-gray-600 mt-2">Control training, run case studies and view results. Backend endpoints: <code>/api/*</code></p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-6">
            <FileUploader />
            <TrainPanel />
          </div>

          <div className="md:col-span-2 space-y-6">
            <PredictPanel />
            <PlotViewer />
            <LogsViewer />
          </div>
        </div>

        <footer className="mt-8 text-sm text-gray-500">
          <p>Note: This frontend assumes a Flask backend running on port 5000. Use the Vite proxy in development.</p>
        </footer>
      </div>
    </div>
  );
}
