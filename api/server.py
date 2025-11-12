import os
import io
import base64
import tempfile
import threading
from pathlib import Path

from flask import Flask, request, jsonify
import pandas as pd

csv_path = Path("data/emp.csv")
csv_lock = threading.Lock()

# Ensure data folder and CSV header exist
csv_path.parent.mkdir(parents=True, exist_ok=True)
if not csv_path.exists():
    pd.DataFrame(columns=["EN", "NAME", "IMAGE_BINARY"]).to_csv(csv_path, index=False)

app = Flask(__name__)

def read_df():
    with csv_lock:
        return pd.read_csv(csv_path, dtype=str).fillna("")

def write_df(df: pd.DataFrame):
    # atomic write
    with csv_lock:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, newline="", dir=str(csv_path.parent)) as tf:
            df.to_csv(tf.name, index=False)
            tmpname = tf.name
        os.replace(tmpname, str(csv_path))

def validate_base64(s: str) -> bool:
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Welcome to the Employee Management API"})

@app.route("/employees", methods=["GET"])
def list_employees():
    df = read_df()
    records = df.to_dict(orient="records")
    return jsonify({"status": "ok", "count": len(records), "data": records})

@app.route("/employee/<en>", methods=["GET"])
def get_employee(en):
    df = read_df()
    row = df[df["EN"] == en]
    if row.empty:
        return jsonify({"status": "error", "message": "EN not found"}), 404
    return jsonify({"status": "ok", "data": row.iloc[0].to_dict()})

@app.route("/employee", methods=["POST"])
def add_employee():
    payload = request.get_json(force=True)
    en = str(payload.get("EN", "")).strip()
    name = str(payload.get("NAME", "")).strip()
    img_b64 = payload.get("IMAGE_BINARY", "")

    if not en or not name or not img_b64:
        return jsonify({"status": "error", "message": "EN, NAME and IMAGE_BINARY are required"}), 400
    if not validate_base64(img_b64):
        return jsonify({"status": "error", "message": "IMAGE_BINARY is not valid base64"}), 400

    df = read_df()
    if (df["EN"] == en).any():
        return jsonify({"status": "error", "message": "EN already exists"}), 409

    new = pd.DataFrame([{"EN": en, "NAME": name, "IMAGE_BINARY": img_b64}])
    df = pd.concat([df, new], ignore_index=True)
    write_df(df)
    return jsonify({"status": "ok", "message": "added", "data": {"EN": en, "NAME": name}}), 201

@app.route("/employee/<en>", methods=["PUT", "PATCH"])
def update_employee(en):
    payload = request.get_json(force=True)
    df = read_df()
    idx = df.index[df["EN"] == en].tolist()
    if not idx:
        return jsonify({"status": "error", "message": "EN not found"}), 404
    i = idx[0]
    if "NAME" in payload:
        df.at[i, "NAME"] = str(payload["NAME"])
    if "IMAGE_BINARY" in payload:
        if not validate_base64(payload["IMAGE_BINARY"]):
            return jsonify({"status": "error", "message": "IMAGE_BINARY is not valid base64"}), 400
        df.at[i, "IMAGE_BINARY"] = payload["IMAGE_BINARY"]
    write_df(df)
    return jsonify({"status": "ok", "message": "updated", "data": df.iloc[i].to_dict()})

@app.route("/employee/<en>", methods=["DELETE"])
def delete_employee(en):
    df = read_df()
    if not (df["EN"] == en).any():
        return jsonify({"status": "error", "message": "EN not found"}), 404
    df = df[df["EN"] != en].reset_index(drop=True)
    write_df(df)
    return jsonify({"status": "ok", "message": "deleted", "EN": en})

if __name__ == "__main__":
    # run with venv activated: python api.py
    # app.run(host="0.0.0.0", port=5000)
    app.run(debug=True, host='127.0.0.1', port=5003)