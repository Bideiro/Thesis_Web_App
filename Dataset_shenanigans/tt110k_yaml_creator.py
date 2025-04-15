import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# === Configuration ===
INPUT_JSON = "c:/Users/dei/Downloads/tt100k_2021/annotations_all.json"
OUTPUT_DIR = "c:/Users/dei/Downloads/tt100k_2021/Yolo_annotations"
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
IMAGE_DIR = "c:/Users/dei/Downloads/tt100k_2021/images"
SPLITS = ["train", "test"]

# === Helper: Normalize YOLO format ===
def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

# === Main Conversion ===
def convert_annotations():
    # Create label subfolders for train/test
    for split in SPLITS:
        os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # Get category names
    categories = data["types"]
    category_to_id = {cat: idx for idx, cat in enumerate(categories)}

    # Iterate over the image entries in the "imgs" dictionary
    for image_info in tqdm(data["imgs"].values(), desc="Converting annotations"):
        img_path = image_info["path"]
        full_img_path = os.path.join(IMAGE_DIR, img_path)

        # Determine whether this is a train or test image
        split = "train" if "train" in img_path else "test"

        # Get image size
        try:
            with Image.open(full_img_path) as img:
                img_width, img_height = img.size
        except FileNotFoundError:
            print(f"⚠️ Image not found at {full_img_path}. Skipping.")
            continue

        # Write label to correct split folder
        label_file_path = os.path.join(LABELS_DIR, split, f"{image_info['id']}.txt")
        with open(label_file_path, "w") as label_file:
            for obj in image_info["objects"]:
                cat = obj["category"]
                class_id = category_to_id.get(cat)

                # Skip if category is not found
                if class_id is None:
                    print(f"⚠️ Category {cat} not found in types list. Skipping object.")
                    continue

                bbox = obj["bbox"]
                yolo_bbox = convert_bbox_to_yolo(
                    bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"], img_width, img_height
                )
                label_file.write(f"{class_id} " + " ".join(f"{v:.6f}" for v in yolo_bbox) + "\n")

    # Create YAML file
    yaml_content = f"""\
train: {IMAGE_DIR}/train
val: {IMAGE_DIR}/test

nc: {len(category_to_id)}
names: {list(category_to_id.keys())}
"""
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as yaml_file:
        yaml_file.write(yaml_content)

    print(f"\n✅ Labels split into 'train' and 'test'.")
    print(f"📁 Labels: {LABELS_DIR}/[train|test]")
    print(f"📄 YAML: {OUTPUT_DIR}/data.yaml")

if __name__ == "__main__":
    convert_annotations()
