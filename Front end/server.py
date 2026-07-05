"""
BiasLens — Media Bias Dashboard Server
Run with: python server.py
Opens at: http://localhost:5000

Reads all CSVs live from ./Data/ on every request.
No restart needed when pipeline updates the files.
"""

from flask import Flask, jsonify, send_from_directory, abort
from flask_cors import CORS
import pandas as pd
import os
import json

app = Flask(__name__, static_folder=".")
CORS(app)

DATA_DIR = "../src/Data"

# Map of API route name -> CSV filename in Data/
CSV_MAP = {
    "coverage":     "bias_coverage_statistics.csv",
    "event_matrix": "matrix_event_coverage.csv",
    "overlap":      "matrix_pairwise_overlap.csv",
    "bias":         "event_target_bias_comparison.csv",
    "omission":     "target_omission_matrix.csv",
    "triples":      "explanation_triples.csv",
    "passages":     "passages.csv",
    "articles":     "analysis_articles_master.csv",
}


def read_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        df = pd.read_csv(path, low_memory=False)
        # Replace NaN with None so JSON serializes cleanly
        df = df.fillna("")
        return df, None
    except Exception as e:
        return None, str(e)


@app.route("/api/<key>")
def api_endpoint(key):
    if key not in CSV_MAP:
        abort(404, description=f"Unknown data key: {key}. Available: {list(CSV_MAP.keys())}")
    df, err = read_csv(CSV_MAP[key])
    if err:
        return jsonify({"error": err}), 404
    return jsonify({
        "key": key,
        "file": CSV_MAP[key],
        "rows": len(df),
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
    })


@app.route("/api/meta")
def api_meta():
    """Returns file metadata — row counts and last-modified timestamps."""
    meta = {}
    for key, fname in CSV_MAP.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            stat = os.stat(path)
            try:
                df = pd.read_csv(path, low_memory=False)
                rows = len(df)
            except Exception:
                rows = -1
            meta[key] = {
                "file": fname,
                "rows": rows,
                "last_modified": stat.st_mtime,
                "last_modified_human": pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(stat.st_size / 1024, 1),
            }
        else:
            meta[key] = {"file": fname, "exists": False}
    return jsonify(meta)


@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


if __name__ == "__main__":
    print(f"\n BiasLens Dashboard Server")
    print(f" Data directory: {os.path.abspath(DATA_DIR)}")
    print(f" Dashboard:      http://localhost:5000\n")
    for key, fname in CSV_MAP.items():
        path = os.path.join(DATA_DIR, fname)
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"   /api/{key:<16} → {fname}  {status}")
    print()
    app.run(debug=True, port=5000)
