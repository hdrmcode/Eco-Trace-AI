from ultralytics import YOLO

def main():
    # Load your YOLOv11 trained model
    model = YOLO(r"D:\Sprinthathon25\Main2\runs\retrain_from_last_40epochs\weights\best.pt")

    # Path to your dataset YAML file
    dataset_yaml = r"D:\Sprinthathon25\Main2\dataset2\data.yaml"

    # Run validation
    metrics = model.val(
        data=dataset_yaml,                     # Dataset path
        imgsz=416,                             # Image size (same as training)
        batch=8,                               # Batch size
        device=0,                              # GPU ID
        split="val",                           # Use validation split
        project=r"D:\Sprinthathon25\Main2\runs",  # Output folder
        name="yolov11_validation",             # Folder name for validation results
        exist_ok=True                          # Allow overwriting existing folder
    )

    # Print key metrics
    print("\n✅ YOLOv11 Validation Completed Successfully!")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()
