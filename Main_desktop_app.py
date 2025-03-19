import numpy as np
import cv2
import threading
import time
import queue
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from ultralytics import YOLO

class_Name = ['placeholder for nonzero start', 'All traffic must turn left', 'All traffic must turn right', 'Keep Left', 'Keep Right', 'No Entry', 
            'No Overtaking', 'No Right - Left Turn', 'No Right Turn', 'No U-Turn', '20', '30', '40', '50', '60',
            '70', '80', '90', '100', '120', 'Stop Sign', 'Yield']

# ✅ Load YOLO once (safe)
YOLO_model = YOLO('models/YOLOv8s(TrafficSignNou)_e10_detect_11-30-24.pt')

# ✅ Load ResNet once (safe for threading)
ResNet_model = load_model('models/Resnet50V2(newgen_2_22_25)50e_uf20_adam.keras')

frames_for_Resnet = 30
cropped_images = []

# ✅ Queue to store images for ResNet processing
resnet_queue = queue.Queue()

# ✅ Flag to check if ResNet thread is running
resnet_running = threading.Event()

def ResNet_Phase():
    """Runs ResNet inference asynchronously in a thread."""
    while True:
        cropped_images = resnet_queue.get()  # Waits for images
        if cropped_images is None:  # Exit condition
            break

        resnet_running.set()  # Mark as running

        print("ResNet thread started")

        predictions_list = []

        for cropped in cropped_images:
            resized = cv2.resize(cropped, (224, 224))
            array = np.expand_dims(resized, axis=0)
            array = preprocess_input(array)

            predictions = ResNet_model(array, training=False).numpy()
            class_id = predictions.argmax()
            confidence = predictions.max() * 100
            class_name = class_Name[class_id]

            print(f'Predicted: {class_name} - {confidence:.3f}%')

            predictions_list.append((class_name, confidence))

        print("ResNet thread finished")
        
        # ✅ Clear queue after processing
        with resnet_queue.mutex:
            resnet_queue.queue.clear()

        resnet_running.clear()  # Mark as not running


# ✅ Start the ResNet thread
resnet_thread = threading.Thread(target=ResNet_Phase, daemon=True)
resnet_thread.start()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    last_resnet_time = 0  # ✅ Track last ResNet run time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        start_time = time.time()

        results = YOLO_model.predict(frame, conf=0.80, verbose=False)
        
        cropped_images = []  # ✅ Reset cropped images for each frame
        object_detected = False  # ✅ Flag to check if YOLO detected anything

        # ✅ Check if YOLO detected objects
        for result in results:
            for box in result.boxes.xyxy:
                object_detected = True  # ✅ At least one object detected
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    cropped_images.append(cropped)

        # ✅ Only run ResNet if:
        # 1️⃣ At least 1 object detected
        # 2️⃣ At least 1 second has passed
        if object_detected and (time.time() - last_resnet_time >= 1):
            if not resnet_running.is_set():  # ✅ Avoid duplicate processing
                print("Queueing images for ResNet processing")

                # ✅ Clear old detections before adding new ones
                with resnet_queue.mutex:
                    resnet_queue.queue.clear()

                # ✅ Queue images for ResNet
                resnet_queue.put(cropped_images[:])

                # ✅ Update the last run time
                last_resnet_time = time.time()

            else:
                print("ResNet is still running, skipping new processing...")

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
                # ✅ Draw Bounding Boxes on Frame
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = result.boxes.conf[0].item() * 100  # Confidence score
                
                # ✅ Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ✅ Put label (confidence + "Traffic Sign")
                label = f"Traffic Sign: {conf:.2f}%"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ✅ Stop the ResNet thread safely
    resnet_queue.put(None)  # This tells the thread to exit
    resnet_thread.join()

    cap.release()
    cv2.destroyAllWindows()
