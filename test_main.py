import asyncio, json, subprocess
import base64
import threading
import cv2
import numpy as np
import websockets
import time
import queue
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from ultralytics import YOLO

_Web = "./web"

# Load YOLO model
YOLO_model = YOLO('models/YOLOv8s(TrafficSignNou)_e10_detect_11-30-24.pt')

bbox_queue = queue.Queue()
frame_queue = queue.Queue()

# Encode image to base64
def encode_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# YOLO detection thread
def yolo_detection():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        results = YOLO_model.predict(frame, conf=0.7, verbose=False)
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                bbox_queue.put((x1, y1, x2, y2))
        frame_queue.task_done()

async def Show_Cam(websocket):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    threading.Thread(target=yolo_detection, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_queue.put(frame.copy())

        # Draw bounding boxes from YOLO thread
        while not bbox_queue.empty():
            x1, y1, x2, y2 = bbox_queue.get()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame_bytes = encode_to_base64(frame)

        message = json.dumps({
            "frame": frame_bytes
        })
        await websocket.send(message)

        await asyncio.sleep(0.05)  # 50 ms delay for smooth streaming

    cap.release()

async def main():
    async with websockets.serve(Show_Cam, "0.0.0.0", 8765):
        try:
            subprocess.Popen(["npm", "run", "dev"], cwd=_Web, shell=True)
        except Exception as e:
            print(f"⚠️ Failed to start React frontend: {e}")

        await asyncio.Future()  # Keep server running

if __name__ == "__main__":
    asyncio.run(main())
