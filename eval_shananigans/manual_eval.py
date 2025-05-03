from pathlib import Path
import cv2
import time
import psutil
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

yolo_model = YOLO(str(YOLO_MODEL_PATH))
resnet_model = load_model(str(RESNET_MODEL_PATH))


true_labels = []
predicted_labels = []
frame_times = []


process = psutil.Process()

image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))

for image_path in image_paths:
    label_path = LABEL_DIR / (image_path.stem + ".txt")
    
    if not label_path.exists():
        print(f"⚠️ Missing label for: {image_path.name}")
        continue
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️ Couldn't load image: {image_path.name}")
        continue
    
    gt_boxes = []
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            
    if not gt_boxes:
        print(f"⚠️ No ground truth objects in: {image_path.name}")
        continue
    
    results = yolo_model.predict(image, verbose=False, conf=CONFIDENCE_THRESHOLD)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print(f"❌ No YOLO detections in: {image_path.name}")
        continue

    if result.boxes is None or len(result.boxes) == 0:
        print(f"❌ No YOLO detections in: {image_path.name}")
        continue

    for box_tensor in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box_tensor.tolist())
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        resized = cv2.resize(cropped, (224, 224))
        input_array = np.expand_dims(resized, axis=0)
        input_array = preprocess_input(input_array)

        prediction = resnet_model.predict(input_array, verbose=0)
        pred_class = int(np.argmax(prediction))


# ---------------- METRICS ----------------

if not true_labels:
    print("❌ No valid predictions collected.")
    exit()


acc = accuracy_score(true_labels, predicted_labels)
prec = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
rec = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
avg_fps = len(frame_times) / sum(frame_times)
memory_mb = process.memory_info().rss / 1024 / 1024



print("\n📊 Hybrid YOLO + ResNet Evaluation")
print(f"Total Detections        : {len(true_labels)}")
print(f"Accuracy                : {acc*100:.2f}%")
print(f"Precision               : {prec:.4f}")
print(f"Recall                  : {rec:.4f}")
print(f"F1 Score                : {f1:.4f}")
print(f"Average FPS             : {avg_fps:.2f}")
print(f"Total Memory Usage      : {memory_mb:.2f} MB")