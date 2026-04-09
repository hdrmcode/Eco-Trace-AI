# EcoTrace AI — E-Waste Tracker

> Smart Monitoring & Recycling Management System  
> Built for **Sprinthathon '25** — Problem Statement PSW702  
> Team **404 Not.Found** · Alliance University, Bengaluru

---

## What It Does

EcoTrace AI uses a YOLOv11 computer vision model to identify e-waste devices through a webcam feed, looks up each device's metal composition from a local dataset, and instantly calculates its recyclable value. Users are rewarded with points (₹) for recycling, tracked through a gamified EcoTree that grows with every submission.

---

## Features

- **Real-time device detection** via YOLOv11 + OpenCV webcam feed
- **Material composition lookup** from a curated device dataset (`Filtered_Device_List.xlsx`)
- **Recycling value estimation** based on metal percentages
- **Pie chart visualization** of material breakdown
- **EcoTree gamification** — tree grows as you recycle more devices
- **Wallet tracking** — points earned per device recycled
- **React web frontend** connected to a Flask backend API

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | YOLOv11 (Ultralytics) |
| Backend | Python, Flask, Flask-CORS |
| Frontend | React Native for Web |
| Computer Vision | OpenCV |
| Data | pandas, openpyxl (Excel dataset) |
| GUI (desktop) | Tkinter + Matplotlib |

---

## Project Structure

```
Main2/
├── server.py                          # Flask backend server
├── main_gui.py                        # Tkinter desktop GUI
├── final.py                           # Core detection logic
├── requirements.txt                   # Python dependencies
├── Filtered_Device_List.xlsx          # Device → metal composition dataset
└── retrain_from_last_40epochs/
    └── weights/
        └── best.pt                    # Trained YOLO model weights
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- A working webcam

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ecotrace-ai.git
cd ecotrace-ai/Main2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install flask flask-cors opencv-python ultralytics numpy pandas openpyxl
```

### 3. Run the backend server

```bash
python server.py
```

You should see:

```
Starting YOLO Detection Server...
Model path: retrain_from_last_40epochs/weights/best.pt
 * Running on http://0.0.0.0:5000
```

### 4. (Optional) Run the desktop GUI

```bash
python main_gui.py
```

### 5. Open the frontend

Open your React frontend in the browser, then:
- Click **Start Scan** to initialize the camera
- Click **Detect Device** to run YOLO detection on the current frame
- Detection results and metal composition appear instantly

---

## How Detection Works

1. Webcam feed is captured frame-by-frame via OpenCV
2. YOLOv11 runs inference — labels with confidence ≥ 0.6 are tracked
3. After 4 seconds of consistent detection, the device label is confirmed
4. The label is matched against `Filtered_Device_List.xlsx` (fuzzy matched if needed)
5. Metal composition percentages are displayed along with total recycling value
6. Points are added to the user's wallet; the EcoTree grows

---

## API Endpoints (Flask)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check server and model status |
| GET | `/video_feed` | Live annotated camera stream |
| POST | `/detect` | Run detection on current frame |

Test the health check:
```bash
curl http://localhost:5000/health
# {"status":"ok","model_loaded":true}
```

---

## Troubleshooting

**`Failed to connect to backend`** — Make sure `server.py` is running on port 5000 and no firewall is blocking it.

**`Camera not initialized`** — Click "Start Scan" before "Detect Device". Check no other app is using the webcam.

**`No module named 'ultralytics'`** — Run `pip install ultralytics`.

**CORS errors** — Run `pip install flask-cors` and restart the server.

**Port 5000 in use** — Change the port in `server.py` (`app.run(port=XXXX)`) and update `BACKEND_URL` in `src/components/CameraFrame.tsx` to match.

---

## Impact & Relevance

- Reduces harmful e-waste pollution by promoting responsible recycling
- Enables efficient recovery of valuable metals from discarded electronics
- Rewards users economically, creating a circular waste ecosystem
- Supports **UN SDG 12** — Responsible Consumption and Production
- E-waste grows by ~50 million tons per year globally — this tool helps address that

---

## Team

**404 Not.Found** — Alliance University, Bengaluru

- Hariduthram PS
- SD Kowshik Raj
- J Subhi
- Varun Baskaran

---

## References

- [AI-based Electronic Waste Classification System](https://www.sciencedirect.com/science/article/pii/S2352340919312082)
- [Blockchain for Supply Chain Transparency](https://ieeexplore.ieee.org/document/8410953)
- [Global E-Waste Monitor 2020 — UN Report](https://globalewastemonitor.org/wp-content/uploads/2020/11/GEM_2020_def_decision_C.pdf)
- [Statista — Global E-Waste Generation](https://www.statista.com/statistics/748149/global-e-waste-generation/)
- [TensorFlow Lite Image Classification](https://www.tensorflow.org/lite/models/image_classification/overview)

---

*Built at Sprinthathon '25 · St. Joseph's College of Engineering*
