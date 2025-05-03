from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/YOLOv8s(Synthetic_Cleaned)_e10_30e_2025-04-22/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(r"d:\Documents\ZZ_Datasets\Synthetic_Cleaned_FINAL(4-21-25)\test-old-eval\13_2.jpg", stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    print(boxes)
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk