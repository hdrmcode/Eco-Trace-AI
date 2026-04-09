from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)  # <-- allow React frontend on a different port

# Full path to your trained YOLO model
MODEL_PATH = os.path.join("runs", "retrain_from_last_40epochs", "weights", "best.pt")

# Load the model
model = YOLO(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img = Image.open(request.files['file'].stream)

    # Run YOLO detection
    results = model(img)

    # Parse predictions
    predictions = []
    if results and len(results) > 0:
        for det in results[0].boxes.data.tolist():  # xmin, ymin, xmax, ymax, conf, class
            predictions.append({
                "bbox": det[:4],
                "confidence": float(det[4]),
                "class": int(det[5])
            })

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
