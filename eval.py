from pathlib import Path
import cv2
import time
import psutil
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ultralytics import YOLO
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import pandas as pd

# ---------------- CONFIG ----------------
IMAGE_DIR = Path(r"D:\Documents\ZZ_Datasets\Synthetic_Cleaned_FINAL(4-21-25)\test\images")
LABEL_DIR = Path(r"D:\Documents\ZZ_Datasets\Synthetic_Cleaned_FINAL(4-21-25)\test\labels")
RESNET_MODEL_PATH = Path("models/Resnet50V2(NewSyn_2025-04-21)_15Fe+10UFe.keras")
YOLO_MODEL_PATH = Path("runs/detect/YOLOv8s(Synthetic_Cleaned)_e10__2025-04-21/weights/best.pt")
RESNET_INPUT_SIZE = 224
CONFIDENCE_THRESHOLD = 0.65
IOU_THRESHOLD = 0.5
# ----------------------------------------

# Load models
yolo_model = YOLO(str(YOLO_MODEL_PATH))
resnet_model = load_model(str(RESNET_MODEL_PATH))

true_labels = []
predicted_labels = []
frame_times = []

process = psutil.Process()

image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))

def denormalize_box(box, img_w, img_h):
    cls_id, x_c, y_c, w, h = map(float, box)
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return int(cls_id), max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

for image_path in image_paths:
    label_path = LABEL_DIR / (image_path.stem + ".txt")

    if not label_path.exists():
        print(f"⚠️ Missing label for: {image_path.name}")
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️ Couldn't load image: {image_path.name}")
        continue

    img_h, img_w = image.shape[:2]

    # Load ground truth labels
    gt_boxes = []
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            gt_boxes.append(denormalize_box(parts, img_w, img_h))

    if not gt_boxes:
        print(f"⚠️ No ground truth objects in: {image_path.name}")
        continue

    start_time = time.time()

    for gt_class, x1, y1, x2, y2 in gt_boxes:
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        resized = cv2.resize(cropped, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))
        input_array = np.expand_dims(resized, axis=0)
        input_array = preprocess_input(input_array)

        prediction = resnet_model.predict(input_array, verbose=0)
        pred_class = int(np.argmax(prediction))

        true_labels.append(gt_class)
        predicted_labels.append(pred_class)

    frame_times.append(time.time() - start_time)


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

print("\nTRUE")
print(true_labels)

print("\n PREDICTED")
print(predicted_labels)

# ---------- PER-CLASS METRICS ----------
print("\n📊 Per-Class Breakdown (Precision, Recall, F1-Score)")

class_report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
class_df = pd.DataFrame(class_report).transpose()

# Filter out only class IDs (exclude 'accuracy', 'macro avg', 'weighted avg')
class_df = class_df[class_df.index.str.isdigit()]
class_df.index = class_df.index.astype(int)
class_df = class_df.sort_index()

print(class_df[['precision', 'recall', 'f1-score', 'support']].round(2))

# ---------- TOP MISCLASSIFICATIONS ----------
print("\n❌ Top Misclassifications")

conf_matrix = confusion_matrix(true_labels, predicted_labels)
misclass_counts = []

for true_idx in range(conf_matrix.shape[0]):
    for pred_idx in range(conf_matrix.shape[1]):
        if true_idx != pred_idx and conf_matrix[true_idx][pred_idx] > 0:
            misclass_counts.append(((true_idx, pred_idx), conf_matrix[true_idx][pred_idx]))

# Sort descending by count
top_misclass = sorted(misclass_counts, key=lambda x: x[1], reverse=True)[:10]

print(f"{'True Class':<12}{'Predicted As':<15}{'Count'}")
for (true_class, pred_class), count in top_misclass:
    print(f"{true_class:<12}{pred_class:<15}{count}")

print("\n📊 Hybrid YOLO + ResNet Evaluation")
print(f"Total Detections        : {len(true_labels)}")
print(f"Accuracy                : {acc*100:.2f}%")
print(f"Precision               : {prec:.4f}")
print(f"Recall                  : {rec:.4f}")
print(f"F1 Score                : {f1:.4f}")
print(f"Average FPS             : {avg_fps:.2f}")
print(f"Total Memory Usage      : {memory_mb:.2f} MB")


