from ultralytics import YOLO
from datetime import date
# Naming convention for Models
dataset_name = "Synthetic"
# Epoch numbers
Model_epoch = 15

# YOLO Yaml File
Yolo_Yaml = r"d:\Documents\ZZ_Datasets\Synthetic_Cleaned_FINAL(4-21-25)\data.yaml"


Model_name = "YOLOv8s(" + dataset_name + ")_e" + str(Model_epoch) + "__"+ str(date.today())
# Create a new YOLO model from scratch
# model = YOLO("yolov8s.pt")
model = YOLO("runs/detect/YOLOv8s(Synthetic_Cleaned)_e10__2025-04-21/weights/best.pt")

# Display model information (optional)
model.info()

# # Train the model
metrics = model.val(data= Yolo_Yaml)