from ultralytics import YOLO

def main():
    # Load YOLOv11 model safely (fixed path)
    model = YOLO(r"D:\Sprinthathon25\Main2\runs\retrain_from_last_40epochs\weights\best.pt")

    # Run validation or prediction to verify it works
    results = model.val(
        data=r"D:\Sprinthathon25\Main2\dataset2\data.yaml",
        imgsz=416,
        device=0,
        batch=8,
        name="yolov11_validation_fixed",
        exist_ok=True
    )

    print("✅ Model loaded and validation completed successfully!")

if __name__ == "__main__":
    main()
