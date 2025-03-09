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

# Class names for ResNet (and other things to classify)
class_Name = ['All traffic must turn left','All traffic must turn right','Keep Left','Keep Right','No Entry','No Overtaking','No Right - Left Turn','No Right Turn','No U-Turn',
                '20','30','40','50','60','70','80','90','100','120','Stop Sign','Yield','extra','extra']

# Load models (YOLO for object detection and ResNet for classification)
ResNet_model = load_model('models/Resnet50V2(newgen_2_22_25)50e_uf20_adam.keras')  # Replace with the actual path to your ResNet model
YOLO_model = YOLO('models/YOLOv8s(TrafficSignNou)_e10_detect_11-30-24.pt')  # Replace with the actual path to your YOLO model

frame_buffer = deque(maxlen=20)

# Function to make predictions with YOLO (detect objects)
def YOLO_Predict_COut(frame):
    results = YOLO_model.predict(source=frame, save=False, conf=0.25, show=False, stream=True)

    cropped_images = []

    # Cropping for ResNet Identification Phase
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw a rectangle on detected object
            
            cropped = frame[y1:y2, x1:x2]  # Crop the object from the frame
            if cropped.size > 0:
                cropped_images.append(cropped)  # Add the cropped image to list

    return cropped_images  # Return list of cropped images for further classification

# Function to perform ResNet classification
def ResNet_Phase(cropped_images):
    predictions_list = []

    for cropped in cropped_images:
        resized = cv2.resize(cropped, (224, 224))  # Resize to match ResNet input size
        array = np.expand_dims(resized, axis=0)  # Add batch dimension
        array = preprocess_input(array)  # Preprocess for ResNet50V2

        # Make prediction with ResNet model
        predictions = ResNet_model(array, training=False).numpy()
        class_id = predictions.argmax()  # Get the class with the highest probability
        confidence = predictions.max() * 100  # Get the confidence percentage
        class_name = class_Name[class_id]  # Get the class name

        print(f'Predicted: {class_name} - {confidence:.3f}%')  # Print prediction details

        predictions_list.append((class_name, confidence))  # Store results

    return predictions_list  # Return a list of predictions

# Asynchronous video streaming and processing
async def video_stream(websocket):
    cap = cv2.VideoCapture(0)  # Start video capture (use 0 for the default webcam)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame!")
            break

        # Step 1: Use YOLO to detect objects and get cropped images
        cropped_images = YOLO_Predict_COut(frame)

        # Step 2: Classify the detected objects using ResNet
        predictions = ResNet_Phase(cropped_images) if cropped_images else []

        # Step 3: Format results for frontend (send prediction results)
        predicted_behavior = ", ".join([f"{cls} ({conf:.1f}%)" for cls, conf in predictions])

        # Encode frame to base64 for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()  # Convert the frame to bytes

        # Send JSON data (frame and predictions) to frontend
        # Encode frame to base64 for JSON serialization
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')  # Convert bytes to Base64 string

        # Send JSON data to frontend
        message = json.dumps({"frame": frame_bytes, "behavior": predicted_behavior})
        await websocket.send(message)  # Send the message to WebSocket

        await asyncio.sleep(0.05)  # 50ms delay for smooth streaming

# Main async function to set up WebSocket server
async def main():
    # Start a WebSocket server that listens on a specific port
    async with websockets.serve(video_stream, "0.0.0.0", 8765):
        print("✅ WebSocket server started on ws://0.0.0.0:8765")
        
        # Optionally, start the React frontend asynchronously (you might have it running on a separate process)
        try:
            print("🚀 Starting React frontend...")
            subprocess.Popen(["npm", "run", "dev"], cwd=_Web, shell=True)
        except Exception as e:
            print(f"⚠️ Failed to start React frontend: {e}")
        
        # Keep the server running
        await asyncio.Future()  # Wait indefinitely until the server is closed

# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())  # Start the WebSocket server and streaming
