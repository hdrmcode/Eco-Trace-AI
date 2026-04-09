# server.py
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import cv2
import pandas as pd
from ultralytics import YOLO
from collections import Counter
from difflib import get_close_matches
import os
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\Sprinthathon25\Main2\runs\retrain_from_last_40epochs\weights\best.pt"
DATASET_PATH = r"D:\Sprinthathon25\Main2\Filtered_Device_List.xlsx"
CAPTURE_PATH = r"D:\Sprinthathon25\Main2\capture.jpg"  # temporary capture

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# ---------------- LOAD DATASET ----------------
metal_data = pd.read_excel(DATASET_PATH)
metal_data.columns = (metal_data.columns
                      .str.replace('\xa0', '', regex=False)
                      .str.strip()
                      .str.replace(' ', '_')
                      .str.lower())

device_names_list = metal_data['normalized_name'].tolist()

# ---------------- GLOBALS ----------------
camera = None

# ---------------- ROUTES ----------------
@app.route("/capture", methods=["POST"])
def capture_photo():
    global camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return jsonify({"error": "Camera could not be opened"}), 500

    ret, frame = camera.read()
    camera.release()

    if not ret:
        return jsonify({"error": "Failed to capture photo"}), 500

    cv2.imwrite(CAPTURE_PATH, frame)
    return send_file(CAPTURE_PATH, mimetype="image/jpeg")


@app.route("/detect", methods=["POST"])
def detect_device():
    if not os.path.exists(CAPTURE_PATH):
        return jsonify({"error": "No photo captured"}), 400

    frame = cv2.imread(CAPTURE_PATH)

    # ---------------- YOLO DETECTION ----------------
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return jsonify({"name": "No device detected", "metalPercentage": 0})

    cls_id = int(boxes[0].cls[0])
    detected_label = model.names[cls_id]

    # ---------------- MATCH WITH DATASET ----------------
    fixed_label_clean = detected_label.strip().lower()
    matched = metal_data[metal_data['normalized_name'].str.lower() == fixed_label_clean]

    if matched.empty:
        closest_match = get_close_matches(fixed_label_clean, device_names_list, n=1, cutoff=0.5)
        if closest_match:
            fixed_label_clean = closest_match[0]
            matched = metal_data[metal_data['normalized_name'] == fixed_label_clean]
        else:
            return jsonify({"name": "Unknown device", "metalPercentage": 0})

    metals = matched.iloc[0].to_dict()
    total_points = 0
    for k, v in metals.items():
        if k != "normalized_name":
            try:
                total_points += float(str(v).replace('%','').strip())
            except:
                continue

    return jsonify({"name": fixed_label_clean, "metalPercentage": total_points})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
