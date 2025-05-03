from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input # type: ignore
from tensorflow.keras import metrics # type: ignore
from ultralytics import YOLO
import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


dataset_dir = Path('D:\Documents\ZZ_Datasets\Synthetic_Cleaned_FINAL(4-21-25)')
images_dir = dataset_dir / "test" / "images"
labels_dir = dataset_dir / "test" / "labels"

yolo_model = YOLO(r"models/YOLOv8s(Synthetic_Cleaned)_e10__2025-04-22(3).pt")
resnet_model = tf.keras.models.load_model('models/Resnet50V2(NewSyn_2025-04-22)_1e.keras')  # Your fine-tuned ResNet model

class_names = [
    "All_traffic_Must_Turn_Left",
    "All_traffic_Must_Turn_Right",
    "Be_Aware_Of_Pedestrian_Crossing",
    "Be_Aware_Of_School_Children_Crossing",
    "Bike_Lane_Ahead",
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
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit",
    "Speed_Limit_Derestriction",
    "Stop"
]
predicted_labels_resnet = []
true_labels_per_detection = []
detections_count = 0
total_yolo_time = 0
total_time = 0
memory_usage = []

true_labels = []
predicted_labels = []

accuracy_metric = metrics.CategoricalAccuracy()
precision_metric = metrics.Precision()
recall_metric = metrics.Recall()

for img in images_dir.iterdir():
    
    # check if file
    if not img.is_file():
        continue
    
    # image_path = os.path.join(dataset_dir, image_file)
    curr_item = img.stem
    label_path = labels_dir / f"{curr_item}.txt"

    if not label_path.exists():
        continue  # Skip if label doesn't exist
    
    # Load the image
    image = cv2.imread(str(img))
    if image is None:
        continue  # Skip unreadable images
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Read annotations and save
    image_true_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])  # YOLO class ID
            x_center, y_center, width, height = map(float, parts[1:])
            image_true_labels.append((class_id, x_center, y_center, width, height))
        
    start_time = time.time()
    results = yolo_model(image)  # Run YOLO model on the full image
    end_time = time.time()
    detection_time = end_time - start_time
    total_yolo_time += detection_time
        
    # detections = results[0].boxes.xyxy.cpu().numpy()
    detections = results[0].boxes.xywhn.cpu().numpy() # YOLO predicted boxes
    closest_gt = None
    
    for box in detections:
        detections_count += 1
        x1, y1, width, height = map(float, box)
            
        # Find the nearest ground truth label (based on center proximity)
        closest_gt = None
        min_distance = float('inf')
            
            # Iterate over the true labels in the image (which should be a list of tuples)
        for label in image_true_labels:
            class_id, gt_x_min, gt_y_min, gt_width, gt_height = label
            gt_center_x = gt_x_min + (gt_width / 2)
            gt_center_y = gt_y_min + (gt_height / 2)
            detection_center_x = x1 + (width / 2)
            detection_center_y = y1 + (height / 2)
            distance = np.sqrt((gt_center_x - detection_center_x)**2 + (gt_center_y - detection_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_gt = (class_id, gt_x_min, gt_y_min, gt_width, gt_height)
            
        if closest_gt is None:
            continue  # No matching ground truth
        
        # relative to img shape
        img_h, img_w, _ = image.shape
        x_min = int((x1 - width / 2) * img_w)
        y_min = int((y1 - height / 2) * img_h)
        x_max = int((x1 + width / 2) * img_w)
        y_max = int((y1 + height / 2) * img_h)

        # Crop the image based on the detected box
        cropped_img = image[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0:
            continue

        # Resize for ResNet input
        resized_img = cv2.resize(cropped_img, (224, 224))
        # cv2.imshow('Resized Image', resized_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        array = np.expand_dims(resized_img, axis=0)
        array = preprocess_input(array)

        # Prediction from ResNet
        memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

        prediction_start = time.time()
        predictions = resnet_model.predict(array, verbose=0)
        prediction_end = time.time()
        prediction_time = prediction_end - prediction_start
        total_time += prediction_time

        memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_usage.append(memory_after - memory_before)

        # Get the predicted class
        predicted_class_id = int(np.argmax(predictions))
        predicted_labels.append(predicted_class_id)
        true_labels.append(closest_gt[0])


# Final metrics

acc = accuracy_score(true_labels, predicted_labels)
prec = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
rec = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)


yolo_fps = detections_count / total_yolo_time if total_yolo_time > 0 else 0
avg_prediction_time = total_time / detections_count if detections_count > 0 else 0

avg_memory_usage = sum(memory_usage) / len(memory_usage) if memory_usage else 0
total_memory_usage = sum(memory_usage)

print("\nDETECTIONS:")
print(detections_count)

print("\nTRUE")
print(true_labels)

print("\n PREDICTED")
print(predicted_labels)

# Print main metrics
print("\n=== Model Performance Metrics ===")
print(f"ResNet Metrics:")
print(f"  - Accuracy:               {acc*100:.2f}%")
print(f"  - Precision:              {prec*100:.2f}%")
print(f"  - Recall:                 {rec*100:.2f}%")
print(f"  - F1 Score:               {f1*100:.2f}%")
print()
print("Performance Metrics:")
print(f"  - FPS:               {yolo_fps:.2f}")

print()
print("Memory Usage Metrics:")
print(f"  - Average Memory Usage:    {avg_memory_usage:.2f} MB")
print(f"  - Total Memory Usage:      {total_memory_usage:.2f} MB")
print("=================================\n")
