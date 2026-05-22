
# Real-Time Face Mask Detector using YOLO, ONNX, FastAPI & WebSocket

## Project Overview
This project implements a real-time face mask detection system using the Ultralytics YOLO object detection framework. 
The system detects whether a person is wearing a face mask or not using a webcam video stream.

The trained YOLO model is exported to ONNX format for optimized CPU inference. 
The backend is built using FastAPI and WebSocket for real-time frame streaming, while the frontend displays live annotated video with bounding boxes and labels.

---

## Features
- Real-time face mask detection
- YOLOv8-based object detection
- ONNX optimized inference for CPU
- FastAPI backend
- WebSocket real-time communication
- HTML/CSS frontend
- Webcam live video streaming
- Bounding box visualization

---

## Technologies Used
| Technology | Purpose |
|------------|---------|
| Ultralytics YOLOv8 | Object Detection |
| ONNX | CPU-Optimized Inference |
| FastAPI | Backend API |
| WebSocket | Real-Time Streaming |
| OpenCV | Webcam Frame Processing |
| HTML/CSS | Frontend UI |
| Python | Core Programming Language |

---

## Project Workflow
1. Train YOLOv8 model on face mask dataset
2. Validate model performance
3. Export trained model to ONNX format
4. Create FastAPI backend
5. Implement WebSocket streaming
6. Capture webcam frames using OpenCV
7. Run YOLO inference on frames
8. Draw bounding boxes and labels
9. Send processed frames to frontend
10. Display live video stream in browser

---

## Project Structure
Real-Time-Face-Mask-Detector/
│
├── app.py
├── best.onnx
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css

---

## Installation

### Clone Repository
git clone https://github.com/HaseebQaisar145/Real-Time-Face-Mask-Detector.git

### Create Virtual Environment
python -m venv yolo

### Activate Environment
Windows:
yolo\Scripts\activate

### Install Dependencies
pip install -r requirements.txt

---

## Run the Project

### Start FastAPI Server
uvicorn app:app --reload

### Open Browser
http://127.0.0.1:8000

---

## Model Details
- Model: YOLOv8 Nano
- Framework: Ultralytics
- Export Format: ONNX
- Inference Device: CPU
- Detection Classes:
  - mask
  - no_mask

---

## Real-Time Pipeline
Webcam → YOLO ONNX Model → FastAPI → WebSocket → HTML Frontend

---

## Future Improvements
- Improve FPS performance
- Docker deployment
- Multi-camera support
- Cloud deployment
- Mobile optimization

---

## Author
Haseeb Qaiser

