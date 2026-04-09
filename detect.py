from ultralytics import YOLO
import os

# Path to your trained YOLOv11 model
model_path = r"D:\Sprinthathon25\Main2\runs\retrain_from_last_40epochs\weights\best.pt"
model = YOLO(model_path)

# Path to the uploaded image (replace with your actual file path)
image_path = r"D:\Sprinthathon25\Main_Project\test_images\img2.jpg"  # or wherever you saved this image

# Output folder
output_path = r"D:\Sprinthathon25\Main_Project\runs\detect\results"
os.makedirs(output_path, exist_ok=True)

# Run detection
results = model.predict(
    source=image_path,
    conf=0.10,                # lower confidence threshold to catch more detections
    save=True,
    save_txt=True,
    project=output_path,
    name='prediction'
)

# Print all detected device names
print("\n🔍 Detected Devices:\n" + "-"*40)
for result in results:
    boxes = result.boxes
    names = model.names
    detected_labels = set()

    for cls_id in boxes.cls:
        label = names[int(cls_id)]
        detected_labels.add(label)

    if detected_labels:
        for device in detected_labels:
            print(f"📱 {device}")
    else:
        print("❌ No device detected in this image.")

print(f"\n✅ Detection complete! Check results in: {output_path}\\prediction")
