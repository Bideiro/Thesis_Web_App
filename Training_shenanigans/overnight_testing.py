from ultralytics import YOLO

Yolo_Yaml = r"runs/detect/train/args.yaml"

model = YOLO(r"runs/detect/train/weights/last.pt")

model.tune(name="Tune(Edited_Dataset)_10e_10i",
            data=Yolo_Yaml,
            plots=True,
            save=True,
            cache='disk',
            resume = True
            )