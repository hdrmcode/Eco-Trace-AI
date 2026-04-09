from ultralytics import YOLO

def main():
    # Load your last trained weights
    model = YOLO(r"D:\Sprinthathon25\Main2\runs\backup_model_fast\weights\last.pt")

    # Path to your dataset YAML file
    dataset_yaml = r"D:\Sprinthathon25\Main2\dataset2\data.yaml"

    # Continue training for 40 more epochs
    model.train(
        data=dataset_yaml,                     # Dataset path
        epochs=40,                             # Continue training for 40 epochs
        imgsz=416,                             # Image size
        batch=8,                               # Batch size
        device=0,                              # GPU ID
        workers=4,                             # Reduce if CPU threads are low
        project=r"D:\Sprinthathon25\Main2\runs",  # Output directory
        name="retrain_from_last_40epochs",     # New run name
        exist_ok=True,                         # Allow overwriting
        save_period=5                          # Save every 5 epochs
    )

if __name__ == "__main__":
    main()
