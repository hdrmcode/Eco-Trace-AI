# Backend Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Step-by-Step Setup

### 1. Install Python Dependencies
Open terminal/command prompt in the `Main2` folder and run:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install flask flask-cors opencv-python ultralytics numpy
```

### 2. Verify File Structure
Make sure your files are organized like this:
```
Main2/
├── server.py                    # Flask backend server
├── requirements.txt             # Python dependencies
├── final.py                     # Your original detection script
└── retrain_from_last_40epochs/
    └── weights/
        └── best.pt              # Your YOLO model
```

### 3. Start the Backend Server
In the `Main2` folder, run:

```bash
python server.py
```

You should see:
```
Starting YOLO Detection Server...
Model path: retrain_from_last_40epochs/weights/best.pt
 * Running on http://0.0.0.0:5000
```

### 4. Test Backend (Optional)
Open another terminal and test:

```bash
curl http://localhost:5000/health
```

Should return: `{"status":"ok","model_loaded":true}`

### 5. Use the Frontend
1. Make sure the backend is running (Step 3)
2. Open your Lovable project in browser
3. Click "Start Scan" - this connects to your backend camera
4. Click "Detect Device" - this runs YOLO detection
5. Check browser console (F12) to see detection results

## What Happens:
- **Frontend Camera Display**: Shows live feed from your backend camera with YOLO annotations
- **Detection Results**: When you click "Detect Device", it processes the current frame with your YOLO model
- **Console Output**: All detections are logged to browser console, mimicking terminal output
- **Frontend Display**: Shows the best detection (highest confidence) in the Detection Info panel

## Troubleshooting

### "Failed to connect to backend"
- Make sure `server.py` is running on port 5000
- Check if firewall is blocking port 5000

### "Camera not initialized"
- Click "Start Scan" first before "Detect Device"
- Make sure your camera is not being used by another application

### "No module named 'ultralytics'"
- Run: `pip install ultralytics`

### CORS errors
- Make sure flask-cors is installed: `pip install flask-cors`
- Backend should show CORS is enabled

## Advanced: Using Your Original final.py
If you want to integrate your existing `final.py` logic:
1. Open `server.py`
2. Import functions from `final.py`
3. Replace the detection logic in the `/detect` endpoint with your custom logic

## Port Configuration
If port 5000 is already in use, change it in:
- `server.py`: Last line `app.run(host='0.0.0.0', port=5000, debug=True)`
- `src/components/CameraFrame.tsx`: Line with `const BACKEND_URL = 'http://localhost:5000'`
