import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO(r"D:\Sprinthathon25\Main2\runs\retrain_from_last_40epochs\weights\best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is default webcam, change if you have multiple cameras

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run detection
    results = model(frame)  # For YOLOv11, passing frame directly works

    # Render results on the frame
    annotated_frame = results[0].plot()  # plot() returns frame with boxes and labels

    # Display the output
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
