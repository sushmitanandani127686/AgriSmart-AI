"""Vercel Serverless API entrypoint for AgriSmart AI.

Exposes RESTful endpoints for:
- /api/predict: ML Crop Recommendation
- /api/diagnose: CV Leaf Disease & Lesion Segmentation
- /api/metrics: Pre-computed model evaluation results
- /api/health: Health check probe
"""

import os
import sys
import json
import base64
import io
from flask import Flask, request, jsonify
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import PredictionEngine
from src.disease_analyzer import segment_and_analyze_leaf

app = Flask(__name__)

# Initialize prediction engine singleton
engine = None
try:
    engine = PredictionEngine(
        model_path=os.path.join(BASE_DIR, "models", "crop_rf_model.joblib"),
        metrics_path=os.path.join(BASE_DIR, "models", "model_metrics.json"),
    )
except Exception as e:
    print(f"[Warning] Engine init delayed: {e}")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check probe."""
    return jsonify({
        "status": "healthy",
        "service": "AgriSmart AI Vercel API",
        "engine_loaded": engine is not None,
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Return measured evaluation metrics and feature importances."""
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Metrics file not found"}), 404


@app.route("/api/predict", methods=["POST"])
def predict_crop():
    """Crop recommendation endpoint."""
    global engine
    if engine is None:
        try:
            engine = PredictionEngine(
                model_path=os.path.join(BASE_DIR, "models", "crop_rf_model.joblib"),
                metrics_path=os.path.join(BASE_DIR, "models", "model_metrics.json"),
            )
        except Exception as e:
            return jsonify({"error": f"Failed to initialize model engine: {str(e)}"}), 500

    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Missing JSON request body"}), 400

        # Extract and parse features
        features = {
            "N": float(payload.get("N", 0)),
            "P": float(payload.get("P", 0)),
            "K": float(payload.get("K", 0)),
            "temperature": float(payload.get("temperature", 0)),
            "humidity": float(payload.get("humidity", 0)),
            "ph": float(payload.get("ph", 7)),
            "rainfall": float(payload.get("rainfall", 0)),
        }

        result = engine.predict(features, top_k=3)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as ex:
        return jsonify({"error": f"Inference error: {str(ex)}"}), 500


@app.route("/api/diagnose", methods=["POST"])
def diagnose_leaf():
    """Foliar disease diagnostic and lesion segmentation endpoint."""
    try:
        image_obj = None

        # Handle multipart file upload
        if "file" in request.files:
            file_storage = request.files["file"]
            image_obj = Image.open(file_storage.stream)

        # Handle JSON base64 upload
        elif request.is_json:
            data = request.get_json()
            base64_str = data.get("image", "")
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            image_bytes = base64.b64decode(base64_str)
            image_obj = Image.open(io.BytesIO(image_bytes))

        if image_obj is None:
            return jsonify({"error": "No valid image provided via 'file' or 'image' base64"}), 400

        # Perform CV analysis
        cv_result = segment_and_analyze_leaf(image_obj)

        # Convert segmented PIL image to base64 data URI for frontend display
        buffered = io.BytesIO()
        cv_result["segmented_image"].save(buffered, format="PNG")
        seg_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response_payload = {
            "condition": cv_result["condition"],
            "severity": cv_result["severity"],
            "pathogen": cv_result["pathogen"],
            "confidence_pct": cv_result["confidence_pct"],
            "symptoms": cv_result["symptoms"],
            "management": cv_result["management"],
            "badge_color": cv_result["badge_color"],
            "metrics": cv_result["metrics"],
            "segmented_image_data_uri": f"data:image/png;base64,{seg_base64}",
            "disclaimer": cv_result["disclaimer"],
        }

        return jsonify(response_payload)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as ex:
        return jsonify({"error": f"Diagnostic processing failed: {str(ex)}"}), 500


# Fallback static file serving for local development
@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_static(path):
    from flask import send_from_directory
    public_dir = os.path.join(BASE_DIR, "public")
    if os.path.exists(os.path.join(public_dir, path)):
        return send_from_directory(public_dir, path)
    return send_from_directory(public_dir, "index.html")


# For local development running python api/index.py
if __name__ == "__main__":
    print("Starting AgriSmart AI Web Server on http://localhost:5001 ...")
    app.run(port=5001, debug=True)

