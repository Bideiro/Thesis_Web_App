import asyncio
import json
import subprocess
import base64
import threading
import cv2
import numpy as np
import websockets
import time
import queue
from collections import deque
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from ultralytics import YOLO

_Web = "./web"

class_Name = ['placeholder for nonzero start', 'All traffic must turn left', 'All traffic must turn right', 
              'Keep Left', 'Keep Right', 'No Entry', 'No Overtaking', 'No Right - Left Turn', 'No Right Turn', 
              'No U-Turn', '20', '30', '40', '50', '60', '70', '80', '90', '100', '120', 'Stop Sign', 'Yield']

# Load models
ResNet_model = load_model('models/Resnet50V2(newgen_2_22_25)50e_uf20_adam.keras')
YOLO_model = YOLO('models/YOLOv8s(TrafficSignNou)_e10_detect_11-30-24.pt')

resnet_queue = queue.Queue()
resnet_running = threading.Event()
fps_history = deque(maxlen=30)

resnet_results = []  # Stores latest ResNet predictions

def encode_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# ResNet classification function
def ResNet_Phase():
    global resnet_results
    while True:
        cropped_images = resnet_queue.get()
        if cropped_images is None:
            break

        resnet_running.set()
        results = []

        for i, cropped in enumerate(cropped_images):
            resized = cv2.resize(cropped, (224, 224))
            array = np.expand_dims(resized, axis=0)
            array = preprocess_input(array)

            predictions = ResNet_model(array, training=False).numpy()
            class_id = predictions.argmax()
            confidence = predictions.max() * 100
            class_name = class_Name[class_id]
            
            results.append({
                "prediction": str(i) + "1" + class_name + ": " + str(round(confidence, 2))
            })

            # results.append({
            #     "bounding_box": i + 1,
            #     "prediction": class_name,
            #     "confidence": round(confidence, 2)
            # })

        resnet_results = results  # Store latest results
        resnet_queue.task_done()
        resnet_running.clear()

# Asynchronous video streaming (WebSocket Server 1)
async def Show_Cam(websocket):
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    resnet_thread = threading.Thread(target=ResNet_Phase, daemon=True)
    resnet_thread.start()

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        results = YOLO_model.predict(frame, conf=0.70, verbose=False)
        cropped_images = []

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if cropped.size > 0:
                    cropped_images.append(cropped)

        # Queue for ResNet processing
        if not resnet_running.is_set():
            resnet_queue.put(cropped_images.copy())

        fps = 1 / (time.time() - start_time)
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)

        frame_bytes = encode_to_base64(frame)

        message = json.dumps({
            "frame": frame_bytes,
            "fps": round(avg_fps, 2)
        })
        await websocket.send(message)

        await asyncio.sleep(0.05)

    resnet_queue.put(None)
    resnet_thread.join()

# WebSocket Server 2 - Sends ResNet Predictions
async def ResNet_WebSocket(websocket):
    global resnet_results
    while True:
        if resnet_results:
            message = json.dumps({"resnet_predictions": str(resnet_results)})
            await websocket.send(message)
            resnet_results = []  # Clear after sending
        await asyncio.sleep(1)  # Send updates every second

# Main function to run both WebSocket servers
async def main():
    server1 = websockets.serve(Show_Cam, "0.0.0.0", 8765)
    server2 = websockets.serve(ResNet_WebSocket, "0.0.0.0", 8766)

    print("✅ WebSocket server started on ws://0.0.0.0:8765 for video streaming")
    print("✅ WebSocket server started on ws://0.0.0.0:8766 for ResNet results")

    try:
        print("🚀 Starting React frontend...")
        subprocess.Popen(["npm", "run", "dev"], cwd=_Web, shell=True)
    except Exception as e:
        print(f"⚠️ Failed to start React frontend: {e}")
        
    await asyncio.gather(server1, server2)

    await asyncio.Future()  # Prevents the event loop from exiting


if __name__ == "__main__":
    asyncio.run(main())
