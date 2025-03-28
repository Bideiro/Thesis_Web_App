import numpy as np
import cv2
import multiprocessing
import time
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from ultralytics import YOLO

class_Name = ['placeholder for nonzero start', 'All traffic must turn left', 'All traffic must turn right', 'Keep Left', 'Keep Right', 'No Entry', 
            'No Overtaking', 'No Right - Left Turn', 'No Right Turn', 'No U-Turn', '20', '30', '40', '50', '60',
            '70', '80', '90', '100', '120', 'Stop Sign', 'Yield']

# models
YOLO_model = YOLO('models/YOLOv8s(TrafficSignNou)_e10_detect_11-30-24.pt')


frames_for_Resnet = 10
cropped_images = []

# 🔹 Shared variable to track ResNet process status
resnet_running = multiprocessing.Value('b', False)  # 'b' means boolean (0 or 1)

def ResNet_Phase(cropped_images, running_flag):
    
    if running_flag.value:  # Check if already running (shouldn't happen)
        return

    running_flag.value = True  # Set flag to running
    ResNet_model = load_model('models/Resnet50V2(newgen_2_22_25)50e_uf20_adam.keras')
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

    running_flag.value = False  # Reset flag when done
    # return predictions_list
        
if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    cnt = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        start_time = time.time()

        results = YOLO_model.predict(frame, conf=0.80)
        # Cropping
        
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    cropped_images.append(cropped)
                    
        # if cropped_images.count() == 30:
        #     cropped_images.clear()
                    
        if len(cropped_images) % frames_for_Resnet == 0 and cropped_images:
            # last_predictions = ResNet_Phase(cropped_images)
            print("doing resnet")
            process = multiprocessing.Process(target=ResNet_Phase, args=(cropped_images,))
            process.start()
            cropped_images.clear()
            
            if not resnet_running.value:  # 🔹 Only start if ResNet isn't already running
                print("Starting ResNet in a separate process")

                process = multiprocessing.Process(target=ResNet_Phase, args=(cropped_images[:], resnet_running))
                process.start()

                cropped_images.clear()  # Clear images after sending to the process
            else:
                print("ResNet is still running, skipping new process...")
        
        annotated_frame = frame.copy()
        for r in results:
            annotated_frame = r.plot()

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
