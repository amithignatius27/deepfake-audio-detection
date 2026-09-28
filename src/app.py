import os
import uuid
import subprocess
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="../templates")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files["file"]

    filename = f"{uuid.uuid4()}_{audio.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    audio.save(file_path)

    try:
        result = subprocess.check_output(
            [sys.executable, os.path.join(BASE_DIR, "infer.py"), file_path],
            stderr=subprocess.STDOUT,
            text=True
        )

        label = "UNKNOWN"
        confidence = 0.0

        for line in result.splitlines():
            if "Prediction" in line:
                label = line.split(":")[-1].strip()
            if "Confidence" in line:
                confidence = float(line.split(":")[-1].replace("%", "").strip())

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.output}), 500

if __name__ == "__main__":
    app.run(debug=True)
