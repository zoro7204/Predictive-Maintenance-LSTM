# backend.py
# Minimal Flask backend to wrap your training / predict / logs endpoints.
# Run: python -m pip install flask flask-cors
# Then: python backend.py

from flask import Flask, request, send_file, jsonify, stream_with_context, Response
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "."
LOG_FILE = "predictive_maintenance.log"


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)
    return jsonify({"ok": True, "filename": f.filename})


@app.route("/api/train", methods=["POST"])
def train():
    """
    Starts the training script and streams stdout back to client.
    Adjust the command below if your project uses different CLI args.
    """
    cmd = ["python", "model_train.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=".")
    def generate():
        try:
            for line in iter(proc.stdout.readline, b""):
                if not line:
                    break
                yield line.decode("utf-8", errors="replace")
        finally:
            proc.stdout.close()
            proc.wait()
    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Call case_study.py via subprocess and return parsed JSON.
    Modify parsing below to fit your case_study.py output format.
    """
    data = request.get_json(silent=True) or {}
    engine_id = data.get("engine_id", 1)
    # Example: call case_study.py --engine_id 10
    cmd = ["python", "case_study.py", "--engine_id", str(engine_id)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    out = proc.stdout or proc.stderr or ""
    # Naive parse: look for a line like: Predicted RUL: 123.45
    predicted = None
    for line in out.splitlines():
        if "Predicted RUL" in line:
            try:
                predicted = float(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass
    return jsonify({"predicted_rul": predicted, "raw_output": out})


@app.route("/api/logs", methods=["GET"])
def logs():
    if os.path.exists(LOG_FILE):
        return send_file(LOG_FILE, mimetype="text/plain")
    return ("", 204)


@app.route("/api/plot", methods=["GET"])
def plot():
    fn = "rul_plot_final_evaluation.png"
    if os.path.exists(fn):
        return send_file(fn, mimetype="image/png")
    return ("Not found", 404)


if __name__ == "__main__":
    # Run dev server on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
