import torch
from ultralytics import YOLO


if __name__ == '__main__':
    # Create a new YOLO model from scratch
    model = YOLO("yolov8s.yaml")
    # # Load a pretrained YOLO model (recommended for training)
    # model = YOLO("Completed Models/YOLOv5s(TrafficSignNou)_e10_detect_12-2-24/weights/best.pt")

    # Display model information (optional)
    model.info()

    # # Train the model
    results = model.train(data="data.yaml", epochs=10, device='0', save_period= 1, name='YOLOv5s(NewGen - 2-13-25)_e10_detect_11-30-24')

    # Evaluate the model's performance on the validation set
    results = model.val(data = "data.yaml", device = "0")
    # Export the model to ONNX format
    success = model.export(format="onnx")


#  training yolov7
# python train.py --cfg cfg/training/yolov7-tiny.yaml --data data.yaml --epoch 10 --name yolov7_newdataset --weights '' --hyp data/hyp.scratch.p5.yaml --img 640 640 --batch-size 32 --workers 8 --save_period 1 

# testing yolov7
# python test.py --data data.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/YOLOv7-tiny_e10_detect-11-25-24/weights/best.pt --name yolov7_640_val


# testing yolov3
# python train.py --data data.yaml --epochs 10 --weights '' --data data.yaml --cfg models/yolov3-tiny.yaml  --batch-size 128 --name yolov3 