from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input  # type: ignore
from tensorflow.keras import metrics  # type: ignore
from ultralytics import YOLO
import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# =========================
# Load models
# =========================
dataset_dir = Path('D:/Documents/ZZ_Datasets/Synthetic_Cleaned_FINAL(4-21-25)')
images_dir = dataset_dir / "test" / "images"
labels_dir = dataset_dir / "test" / "labels"


yolo_model = YOLO(r"models/YOLOv8s(Synthetic_Cleaned)_e10__2025-04-22(3).pt")
resnet_model = tf.keras.models.load_model('models/Resnet50V2(NewSyn_2025-04-22)_1e.keras')


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

# =========================
# Metrics storage
# =========================
predicted_labels = []
true_labels = []

detections_count = 0
total_yolo_time = 0
total_time = 0
memory_usage = []

# =========================
# Utility function: IoU
# =========================
def compute_iou_matrix(detections, ground_truths):
    N = detections.shape[0]
    M = ground_truths.shape[0]

    dets = np.expand_dims(detections, 1)
    gts = np.expand_dims(ground_truths, 0)

    dets_x1 = dets[..., 0] - dets[..., 2] / 2
    dets_y1 = dets[..., 1] - dets[..., 3] / 2
    dets_x2 = dets[..., 0] + dets[..., 2] / 2
    dets_y2 = dets[..., 1] + dets[..., 3] / 2

    gts_x1 = gts[..., 0] - gts[..., 2] / 2
    gts_y1 = gts[..., 1] - gts[..., 3] / 2
    gts_x2 = gts[..., 0] + gts[..., 2] / 2
    gts_y2 = gts[..., 1] + gts[..., 3] / 2

    inter_x1 = np.maximum(dets_x1, gts_x1)
    inter_y1 = np.maximum(dets_y1, gts_y1)
    inter_x2 = np.minimum(dets_x2, gts_x2)
    inter_y2 = np.minimum(dets_y2, gts_y2)

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    dets_area = (dets_x2 - dets_x1) * (dets_y2 - dets_y1)
    gts_area = (gts_x2 - gts_x1) * (gts_y2 - gts_y1)

    union_area = dets_area + gts_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou

# =========================
# Main evaluation loop
# =========================
for img_path in images_dir.iterdir():
    if not img_path.is_file():
        continue

    curr_item = img_path.stem
    label_path = labels_dir / f"{curr_item}.txt"

    if not label_path.exists():
        continue

    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    # Load GT labels
    image_true_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            image_true_labels.append((class_id, x_center, y_center, width, height))

    if len(image_true_labels) == 0:
        continue

    gt_boxes = np.array([(x, y, w, h) for _, x, y, w, h in image_true_labels])
    gt_labels = [class_id for class_id, _, _, _, _ in image_true_labels]

    # YOLO detection
    start_time = time.time()
    results = yolo_model(image)
    end_time = time.time()
    total_yolo_time += (end_time - start_time)

    detections = results[0].boxes.xywhn.cpu().numpy()
    detections_count += len(detections)

    if len(detections) == 0:
        for class_id in gt_labels:
            true_labels.append(class_id)
            predicted_labels.append(-1)
        continue

    # Compute IoU matching
    iou_matrix = compute_iou_matrix(detections, gt_boxes)
    iou_threshold = 0.5

    gt_matched = set()
    det_matched = set()

    for gt_idx in range(iou_matrix.shape[1]):
        best_det_idx = np.argmax(iou_matrix[:, gt_idx])
        best_iou = iou_matrix[best_det_idx, gt_idx]

        if best_iou >= iou_threshold and best_det_idx not in det_matched:
            det_matched.add(best_det_idx)
            gt_matched.add(gt_idx)

            x1, y1, width, height = detections[best_det_idx]
            img_h, img_w, _ = image.shape
            x_min = int((x1 - width / 2) * img_w)
            y_min = int((y1 - height / 2) * img_h)
            x_max = int((x1 + width / 2) * img_w)
            y_max = int((y1 + height / 2) * img_h)

            cropped_img = image[y_min:y_max, x_min:x_max]
            if cropped_img.size == 0:
                continue

            resized_img = cv2.resize(cropped_img, (224, 224))
            array = np.expand_dims(resized_img, axis=0)
            array = preprocess_input(array)

            memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

            prediction_start = time.time()
            predictions = resnet_model.predict(array, verbose=0)
            prediction_end = time.time()
            total_time += (prediction_end - prediction_start)

            memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_usage.append(memory_after - memory_before)

            predicted_class_id = int(np.argmax(predictions))
            predicted_labels.append(predicted_class_id)
            true_labels.append(gt_labels[gt_idx])

    # Handle missed GTs
    for gt_idx in range(len(gt_boxes)):
        if gt_idx not in gt_matched:
                # This ground truth was NOT detected by YOLO -> ResNet never got to classify
                true_labels.append(gt_labels[gt_idx])
                predicted_labels.append(-1)  # -1 means "missed detection"

# =========================
# Metrics
# =========================
acc = accuracy_score(true_labels, predicted_labels)
prec = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
rec = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

yolo_fps = detections_count / total_yolo_time if total_yolo_time > 0 else 0
avg_prediction_time = total_time / detections_count if detections_count > 0 else 0
avg_memory_usage = sum(memory_usage) / len(memory_usage) if memory_usage else 0
total_memory_usage = sum(memory_usage)

print("\n=== Model Performance Metrics ===")
print(f"ResNet Metrics:")
print(f"  - Accuracy:               {acc*100:.2f}%")
print(f"  - Precision:              {prec*100:.2f}%")
print(f"  - Recall:                 {rec*100:.2f}%")
print(f"  - F1 Score:               {f1*100:.2f}%\n")
print("Performance Metrics (YOLO only):")
print(f"  - YOLO FPS:               {yolo_fps:.2f}")
print(f"  - Average Prediction Time (Including ResNet): {avg_prediction_time:.4f} seconds\n")
print("Memory Usage Metrics:")
print(f"  - Average Memory Usage:    {avg_memory_usage:.2f} MB")
print(f"  - Total Memory Usage:      {total_memory_usage:.2f} MB")
print("=================================\n")


# ---------- Per-Class Breakdown ----------
print("\n📊 Per-Class Breakdown (Precision, Recall, F1-Score)")
class_report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
class_df = pd.DataFrame(class_report).transpose()
class_df = class_df[class_df.index.str.isdigit()]
class_df.index = class_df.index.astype(int)
class_df = class_df.sort_index()
print(class_df[['precision', 'recall', 'f1-score', 'support']].round(2))

# ---------- Top Misclassifications ----------
print("\n❌ Top Misclassifications")
conf_matrix = confusion_matrix(true_labels, predicted_labels)
misclass_counts = []

for true_idx in range(conf_matrix.shape[0]):
    for pred_idx in range(conf_matrix.shape[1]):
        if true_idx != pred_idx and conf_matrix[true_idx][pred_idx] > 0:
            misclass_counts.append(((true_idx, pred_idx), conf_matrix[true_idx][pred_idx]))

top_misclass = sorted(misclass_counts, key=lambda x: x[1], reverse=True)[:10]

print(f"{'True Class':<12}{'Predicted As':<15}{'Count'}")
for (true_class, pred_class), count in top_misclass:
    print(f"{true_class:<12}{pred_class:<15}{count}")
