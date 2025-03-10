import asyncio
import json
import subprocess
import base64
import cv2
import numpy as np
import websockets
from collections import deque
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from ultralytics import YOLO

_Web = "./web"

# Class names for ResNet classification
class_Name = ['placeholder for nonzero start', 'All traffic must turn left', 'All traffic must turn right', 'Keep Left', 'Keep Right', 'No Entry', 
              'No Overtaking', 'No Right - Left Turn', 'No Right Turn', 'No U-Turn', '20', '30', '40', '50', '60', 
              '70', '80', '90', '100', '120', 'Stop Sign', 'Yield']

# Load models
ResNet_model = load_model('models/Resnet50V2(newgen_2_22_25)50e_uf20_adam.keras')
YOLO_model = YOLO('models/YOLOv8s(TrafficSignNou)_e10_detect_11-30-24.pt')

frame_buffer = deque(maxlen=20)
resnet_frame_counter = 0  # Counter to control ResNet processing



# YOLO object detection function
def YOLO_Predict_COut(frame):
    results = YOLO_model.predict(source=frame, save=False, conf=0.25, show=False, stream=True)
    cropped_images = []

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw bounding box
            
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                cropped_images.append(cropped)

    return cropped_images

# ResNet classification function (runs only every 20 frames)
def ResNet_Phase(cropped_images):
    predictions_list = []

    for cropped in cropped_images:
        resized = cv2.resize(cropped, (224, 224))  # Resize for ResNet
        array = np.expand_dims(resized, axis=0)
        array = preprocess_input(array)

        predictions = ResNet_model(array, training=False).numpy()
        class_id = predictions.argmax()
        print(class_id)
        confidence = predictions.max() * 100
        class_name = class_Name[class_id]

        print(f'Predicted: {class_name} - {confidence:.3f}%')

        predictions_list.append((class_name, confidence))

    return predictions_list

# Asynchronous video streaming
async def video_stream(websocket):
    global resnet_frame_counter
    cap = cv2.VideoCapture(0)

    last_predictions = []  # Store last valid predictions

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame!")
            break

        resnet_frame_counter += 1  # Increment frame counter

        # Step 1: YOLO detection on every frame
        cropped_images = YOLO_Predict_COut(frame)

        # Step 2: ResNet classification every 20 frames
        if resnet_frame_counter % 20 == 0 and cropped_images:
            last_predictions = ResNet_Phase(cropped_images)

        # Step 3: Prepare results for frontend
        predicted_behavior = ", ".join([f"{cls} ({conf:.1f}%)" for cls, conf in last_predictions])

        # Encode frame to Base64 for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')

        # Send JSON data to frontend
        message = json.dumps({"frame": frame_bytes, "behavior": predicted_behavior})
        await websocket.send(message)

        await asyncio.sleep(0.05)  # 50ms delay for smoother streaming

# WebSocket server
async def main():
    async with websockets.serve(video_stream, "0.0.0.0", 8765):
        print("✅ WebSocket server started on ws://0.0.0.0:8765")
        
        try:
            print("🚀 Starting React frontend...")
            subprocess.Popen(["npm", "run", "dev"], cwd=_Web, shell=True)
        except Exception as e:
            print(f"⚠️ Failed to start React frontend: {e}")
        
        await asyncio.Future()  # Keep server running

if __name__ == "__main__":
    asyncio.run(main())
