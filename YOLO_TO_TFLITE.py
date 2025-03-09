from ultralytics import YOLO

# Load a model
model = YOLO("/mnt/d/Documents/ZZZ Thesis/YOLOv8s(TrafficSignNou)_e10_detect_11-30-24.pt")  # load a custom trained model

# Export the model
model.export(format="tflite")
