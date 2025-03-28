from ultralytics import YOLO

model = YOLO("yolov8s.pt")


results = model(source="0", show=True)