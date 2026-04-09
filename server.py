# server.py
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import threading
import time
import pandas as pd
from collections import Counter
from difflib import get_close_matches

# -------------------- CONFIG --------------------
MODEL_PATH = r"D:\Sprinthathon25\Main2\runs\retrain_from_last_40epochs\weights\best.pt"
DATASET_PATH = r"D:\Sprinthathon25\Main2\Filtered_Device_List.xlsx"
CONFIDENCE_THRESHOLD = 0.6
MIN_DETECTION_TIME = 2  # seconds required for fixed detection

# -------------------- FLASK APP --------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

# -------------------- GLOBALS --------------------
camera = None
frame_lock = threading.Lock()
output_frame = None
running = False

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load dataset
metal_data = pd.read_excel(DATASET_PATH)
metal_data.columns = (metal_data.columns
                      .str.replace('\xa0', '', regex=False)
                      .str.strip()
                      .str.replace(' ', '_')
                      .str.lower())
if 'normalized_name' not in metal_data.columns:
    raise ValueError("'normalized_name' column not found in dataset")
device_names_list = metal_data['normalized_name'].tolist()

# Detection history
label_history = []
detection_start_time = None
fixed_label = None

# -------------------- CAMERA LOOP --------------------
def camera_loop():
    global camera, running, output_frame, label_history, detection_start_time, fixed_label
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Camera could not be opened")
        running = False
        return
    print("[INFO] Camera started")
    while running:
        ret, frame = camera.read()
        if not ret:
            continue

        # YOLO prediction
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        detected_label = None
        if len(boxes) > 0:
            cls_id = int(boxes[0].cls[0])
            conf = float(boxes[0].conf[0])
            if conf >= CONFIDENCE_THRESHOLD:
                detected_label = model.names[cls_id]
                label_history.append(detected_label)
                if detection_start_time is None:
                    detection_start_time = time.time()
            else:
                label_history.clear()
                detection_start_time = None
        else:
            label_history.clear()
            detection_start_time = None

        # Confirm fixed detection
        if detection_start_time and (time.time() - detection_start_time >= MIN_DETECTION_TIME):
            fixed_label = Counter(label_history).most_common(1)[0][0]
            label_history.clear()
            detection_start_time = None

        # Annotate frame
        display_text = f"Detected: {detected_label}" if detected_label else "Detecting..."
        cv2.putText(frame, display_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with frame_lock:
            output_frame = frame

        time.sleep(0.02)
    camera.release()
    print("[INFO] Camera stopped")

# -------------------- ROUTES --------------------
@app.route("/start-camera", methods=["POST"])
def start_camera():
    global running
    if not running:
        running = True
        t = threading.Thread(target=camera_loop)
        t.daemon = True
        t.start()
        return jsonify({"status": "Camera started"})
    return jsonify({"status": "Camera already running"})

@app.route("/stop-camera", methods=["POST"])
def stop_camera():
    global running
    running = False
    return jsonify({"status": "Camera stopped"})

@app.route("/video-feed")
def video_feed():
    def generate():
        global output_frame
        while running:
            with frame_lock:
                if output_frame is None:
                    continue
                ret, buffer = cv2.imencode(".jpg", output_frame)
                frame = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detect")
def detect():
    global fixed_label
    if fixed_label is None:
        return jsonify({"name": "Dummy Device", "metalPercentage": 0})
    
    fixed_label_clean = fixed_label.strip().lower()
    matched = metal_data[metal_data['normalized_name'].str.lower() == fixed_label_clean]
    if matched.empty:
        closest_match = get_close_matches(fixed_label_clean, device_names_list, n=1, cutoff=0.5)
        if closest_match:
            fixed_label_clean = closest_match[0]
            matched = metal_data[metal_data['normalized_name'] == fixed_label_clean]
        else:
            return jsonify({"name": "Unknown Device", "metalPercentage": 0})

    metals = matched.iloc[0].to_dict()
    total_points = 0
    for metal, percentage in metals.items():
        if metal != "normalized_name":
            try:
                total_points += float(str(percentage).replace('%', '').strip())
            except:
                continue

    result = {
        "name": matched.iloc[0]["normalized_name"],
        "metalPercentage": total_points
    }
    return jsonify(result)

@app.route("/")
def home():
    return "Flask server is running. Visit /video-feed for stream."

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
