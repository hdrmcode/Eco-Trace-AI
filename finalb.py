import cv2
import pandas as pd
import time
from collections import Counter
from difflib import get_close_matches
from ultralytics import YOLO

# -------------------- Load YOLO model --------------------
model = YOLO(r"D:\Sprinthathon25\Main2\runs\retrain_from_last_40epochs\weights\best.pt")

# -------------------- Load dataset --------------------
metal_data = pd.read_excel(r"D:\Sprinthathon25\Main2\Filtered_Device_List.xlsx")

# Robust column cleaning
metal_data.columns = (metal_data.columns
                      .str.replace('\xa0', '', regex=False)
                      .str.strip()
                      .str.replace(' ', '_')
                      .str.lower())

# Check if 'normalized_name' exists
if 'normalized_name' not in metal_data.columns:
    print("⚠️ 'normalized_name' column not found! Columns:", metal_data.columns.tolist())
    exit()

# List of device names for approximate matching
device_names_list = metal_data['normalized_name'].tolist()

# -------------------- Initialize webcam --------------------
cap = cv2.VideoCapture(0)

# -------------------- Detection parameters --------------------
confidence_threshold = 0.6
min_detection_time = 5    # seconds a device must be consistently detected
max_runtime = 60          # total webcam runtime in seconds
frame_interval = 0.1      # seconds between frames (~10 FPS)

label_history = []
detection_start_time = None
fixed_label = None

print("\n🔍 Starting webcam... detecting devices...\n")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------- YOLO prediction --------------------
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    detected_label = None
    if len(boxes) > 0:
        cls_id = int(boxes[0].cls[0])
        conf = float(boxes[0].conf[0])
        if conf >= confidence_threshold:
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

    # -------------------- Confirm device if detected consistently --------------------
    if detection_start_time is not None and (time.time() - detection_start_time >= min_detection_time):
        # Device confirmed
        fixed_label = Counter(label_history).most_common(1)[0][0]

        print(f"\n✅ Fixed Detection: {fixed_label}")
        fixed_label_clean = fixed_label.strip().lower()

        # -------------------- Exact or nearest match --------------------
        matched = metal_data[metal_data['normalized_name'].str.lower() == fixed_label_clean]

        if matched.empty:
            # No exact match, find closest device
            closest_match = get_close_matches(fixed_label_clean, device_names_list, n=1, cutoff=0.5)
            if closest_match:
                closest_name = closest_match[0]
                print(f"⚠️ No exact match found. Using closest device: {closest_name}")
                matched = metal_data[metal_data['normalized_name'] == closest_name]
            else:
                print("⚠️ No similar device found in the dataset!")
                label_history.clear()
                detection_start_time = None
                fixed_label = None
                continue

        # -------------------- Display metal composition --------------------
        metals = matched.iloc[0].to_dict()
        total_points = 0
        print("\n🔩 Metal Composition:")
        for metal, percentage in metals.items():
            if metal != "normalized_name":
                try:
                    percentage_value = float(str(percentage).replace('%', '').strip())
                except ValueError:
                    percentage_value = 0
                print(f"  {metal}: {percentage_value}%")
                total_points += percentage_value
        print(f"\n💰 Total Points: {total_points} points (₹{total_points})\n")

        # Reset history to detect next device
        label_history.clear()
        detection_start_time = None
        fixed_label = None

    # -------------------- Display on frame --------------------
    display_text = f"Detected: {detected_label}" if detected_label else "Detecting..."
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("EcoTrace AI - Device Detection", frame)

    # Break on 'q' key or max runtime
    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > max_runtime):
        break

    time.sleep(frame_interval)

# -------------------- Cleanup --------------------
cap.release()
cv2.destroyAllWindows()
print("\n🔒 Webcam closed. Detection finished.")
