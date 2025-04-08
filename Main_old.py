import asyncio, json, subprocess
import base64
import threading
import cv2
import numpy as np
from datetime import datetime
import websockets
import time
import queue
from collections import deque

# ignored cause its just a bug
from tensorflow.keras.applications.resnet_v2 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore

from ultralytics import YOLO

_Web = "./web"
#push for r

# Class names for ResNet classification
class_Name = [
    "All_traffic_must_turn_left",
    "All_traffic_must_turn_right",
    "Be_Aware_of_Pedestrian_Crossing",
    "Be_Aware_of_School_Children_Crossing",
    "Bike_lane_ahead",
    "Give_Way",
    "Keep_Left",
    "Keep_Right",
    "No_Entry",
    "No_Left_Turn",
    "No_Overtaking",
    "No_Parking",
    "No_Right_Turn",
    "No_U-Turn",
    "Pass_Either_Side",
    "Speed_Limit_20_KMPh",
    "Speed_Limit_30_KMPh",
    "Speed_Limit_40_KMPh",
    "Speed_Limit_50_KMPh",
    "Speed_Limit_60_KMPh",
    "Speed_Limit_70_KMPh",
    "Speed_Limit_80_KMPh",
    "Speed_Limit_90_KMPh",
    "Speed_Limit_100_KMPh",
    "Speed_Limit_110_KMPh",
    "Speed_Limit_120_KMPh",
    "Speed_Limit_Derestriction",
    "Stop"
]


# Load models
ResNet_model = load_model('models/Resnet50V2(15E+10FT)04-01-2025.keras')
YOLO_model = YOLO('runs/detect/YOLOv8s(CCTSDB-20e_tt100k-20e)_e40_2025-04-07/weights/best.pt')

resnet_frame_counter = 0  # Counter to control ResNet processing
no_frame_for_det = 30


resnet_queue = queue.Queue()
resnet_running = threading.Event()

fps_history = deque(maxlen=30)
gl_cropped_images = []
frame_bytes = None

resnet_results = []
stored_images = deque(maxlen=10)# Stores latest ResNet predictions
stored_results = deque(maxlen=10)

def encode_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# ResNet classification function
def ResNet_Phase():
    global stored_images
    global stored_results
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
            confidence =round((predictions.max() * 100), 2)
            # print(predictions)
            # print(f"\nClass Name: {class_Name[class_id]}")
            class_name = class_Name[class_id]
            
            results.append(f"Sign {i + 1}: {class_name} ( {class_id} ) @ {confidence}%")
            dt = datetime.now()
            datetime_now = dt.strftime(' %a - %b %d @ %I:%M %p')
            stored_results.append(f"Sign: {class_name} @ {confidence}% Time: {datetime_now}")
            stored_images.append(encode_to_base64(cropped))
            

        resnet_results = results  # Store latest results
        resnet_queue.task_done()
        resnet_running.clear()

async def ResNet_WebSocket(websocket):
    global resnet_results
    while True:
        if resnet_results:
            message = json.dumps({"resnet_predictions": resnet_results})
            await websocket.send(message)
            resnet_results = []  # Clear after sending
        await asyncio.sleep(0.01)  # Send updates every second

async def Send_logs(websocket):
    global stored_results
    global stored_images
    while True:
        if stored_results and stored_images:
            message = json.dumps({
            "logged_image": list(stored_images),
            "results": list(stored_results)
            })
            await websocket.send(message)
        await asyncio.sleep(15)

# Sending Cam Frames
async def Show_Cam(websocket):
    global fps_history, gl_cropped_images, frame_bytes

    while frame_bytes is None:
        await asyncio.sleep(0.1)

    while True:
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

        message = json.dumps({
            "frame": frame_bytes,
            "cropped_images": gl_cropped_images,
            "fps": round(avg_fps, 2)
        })
        await websocket.send(message)
        await asyncio.sleep(0.05)

async def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Start ResNet thread
    resnet_thread = threading.Thread(target=ResNet_Phase, daemon=True)
    resnet_thread.start()

    # Start WebSocket servers
    server1 = await websockets.serve(Show_Cam, "0.0.0.0", 8765)
    server2 = await websockets.serve(ResNet_WebSocket, "0.0.0.0", 8766)
    server3 = await websockets.serve(Send_logs, "0.0.0.0", 8767)

    print("✅ WebSocket servers started.")
    
    try:
        print("🚀 Starting React frontend...")
        subprocess.Popen("npm run dev", cwd=_Web, shell=True)
    except Exception as e:
        print("!!!IMPORTANT!!")
        print("         Try npm install in the web folder!!! ")
        print(f"⚠️ Failed to start React frontend: {e}")

    async def frame_loop():
        global frame_bytes, gl_cropped_images
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            results = YOLO_model.predict(frame, verbose=False, stream=True, conf=0.65)
            cropped_images = []

            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cropped = frame[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if cropped.size > 0:
                        cropped_images.append(cropped)

            if not resnet_running.is_set():
                resnet_queue.put(cropped_images.copy())

            fps = 1 / (time.time() - start_time)
            fps_history.append(fps)

            gl_cropped_images = [encode_to_base64(cropped) for cropped in cropped_images]
            frame_bytes = encode_to_base64(frame)

            await asyncio.sleep(0.01)  # Yield control to event loop

        resnet_queue.put(None)
        cap.release()
        resnet_thread.join()

    # Run everything concurrently
    await asyncio.gather(frame_loop())

if __name__ == "__main__":
    asyncio.run(main())