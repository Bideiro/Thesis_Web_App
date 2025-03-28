from ultralytics import YOLO
from datetime import date

if __name__ == '__main__':
    
    dataset_name = ""
    model_epoch = 50
    Model_name = "YOLOv8s(" + dataset_name + ")_e" + model_epoch + "_"+ str(date.today())
    # Create a new YOLO model from scratch
    model = YOLO("yolov8s.yaml")
    # model = YOLO("Completed Models/YOLOv5s(TrafficSignNou)_e10_detect_12-2-24/weights/best.pt")

    # Display model information (optional)
    model.info()

    # # Train the model
    results = model.train(data="data.yaml", epochs=model_epoch, device='0', save_period= 1, name='YOLOv8s(TrafficSignNou)_e50_3-20-25')

    # Evaluate the model's performance on the validation set
    # results = model.val(data = "data.yaml", device = "0")
    
    # Export the model to ONNX format
    # success = model.export(format="onnx")
