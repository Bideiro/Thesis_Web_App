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
ResNet_model = load_model('models/Resnet50V2(newgen_2025-04-01)_50e_adam.keras')
YOLO_model = YOLO('models/YOLOV8s(25E)03-28-25.pt')

resnet_frame_counter = 0  # Counter to control ResNet processing
no_frame_for_det = 30


resnet_queue = queue.Queue()
resnet_running = threading.Event()
fps_history = deque(maxlen=30)

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
            cv2.imshow(f"img now {i}",resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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

# Asynchronous video streaming
async def Show_Cam(websocket):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    resnet_thread = threading.Thread(target=ResNet_Phase, daemon=True)
    resnet_thread.start()

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Run YOLO on the frame to get bounding boxes
        results = YOLO_model.predict(frame, conf=0.70, verbose=False, stream=True)
        cropped_images = []

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = frame[y1:y2, x1:x2]
                # Draw bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if cropped.size > 0:
                    cropped_images.append(cropped)

        # Queue the frame for ResNet processing if ResNet is not busy
        if not resnet_running.is_set():
            resnet_queue.put(cropped_images.copy())
            

        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        fps_history.append(fps)  # Store FPS in deque
        
        # Compute average FPS
        avg_fps = sum(fps_history) / len(fps_history)

        # Step 3: Prepare results for frontend
        cropped_images_base64 = []
        cropped_images_base64 = [encode_to_base64(cropped) for cropped in cropped_images]
        
        # Encode frame to Base64 for streaming
        frame_bytes = encode_to_base64(frame)

        # Send JSON data to frontend
        message = json.dumps({
            "frame": frame_bytes,
            "cropped_images": cropped_images_base64,
            "fps": round(avg_fps, 2)
        })
        await websocket.send(message)

        await asyncio.sleep(0.05)  # 50ms delay for smoother streaming
    resnet_queue.put(None)
    resnet_thread.join()

# WebSocket server
async def main():

    server1 = websockets.serve(Show_Cam, "0.0.0.0", 8765)
    server2 = websockets.serve(ResNet_WebSocket, "0.0.0.0", 8766)
    server3 = websockets.serve(Send_logs, "0.0.0.0", 8767)
    print("✅ WebSocket server started on ws://0.0.0.0:8765 for video streaming")
    print("✅ WebSocket server started on ws://0.0.0.0:8766 for ResNet results")
    print("✅ WebSocket server started on ws://0.0.0.0:8767 for ResNet results")
    try:
        print("🚀 Starting React frontend...")
        subprocess.Popen("npm run dev", cwd=_Web, shell=True)

    except Exception as e:
        print("!!!IMPORTANT!!")
        print("         Try npm install on the web folder!!! ")
        print(f"⚠️ Failed to start React frontend: {e}")
        
    await asyncio.gather(server1, server2, server3)
    await asyncio.Future()  # Prevents the event loop from exiting

if __name__ == "__main__":
    asyncio.run(main())