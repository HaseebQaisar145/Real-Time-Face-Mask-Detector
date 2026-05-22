import cv2
import base64
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load ONNX model
model = YOLO("model/best.onnx", task="detect")

# Load HTML
with open("templates/index.html", "r") as f:
    html_content = f.read()


@app.get("/")
async def home():
    return HTMLResponse(html_content)


# Webcam detection
def get_webcam(max_index=5):

    for i in range(max_index):

        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

        if cap.isOpened():
            print(f"[INFO] Webcam found at index {i}")
            return cap

        cap.release()

    print("[WARN] Webcam not found. Using fallback video.")

    return cv2.VideoCapture("videos/test_video.mp4")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    cap = get_webcam()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO inference
            results = model(frame)

            result = results[0]

            # Loop detections
            for box in result.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                original_label = result.names[cls]

                # Swap labels manually
                if original_label == "mask":

                    display_label = "no_mask"

                    box_color = (0, 0, 255)  # RED

                    text_color = (0, 0, 255)

                else:

                    display_label = "mask"

                    box_color = (255, 0, 0)  # BLUE

                    text_color = (255, 0, 0)

                # Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3
                )

                # Draw text
                cv2.putText(
                    frame,
                    f"{display_label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    text_color,
                    2
                )

            # Encode image
            _, buffer = cv2.imencode(".jpg", frame)

            frame_bytes = base64.b64encode(buffer).decode()

            # Send frame
            await websocket.send_text(frame_bytes)

            await asyncio.sleep(0.03)

    except Exception as e:

        print("[ERROR]", e)

    finally:

        cap.release()

        try:
            await websocket.close()

        except:
            pass

        print("[INFO] Connection closed")