import os
import cv2
import random
from tqdm import tqdm  # Import tqdm for progress bar

# Paths
yolo_images_path = r"D:\Documents\ZZ_Datasets\tt100k_2022_YOLO\train\images"
yolo_labels_path = r"D:\Documents\ZZ_Datasets\tt100k_2022_YOLO\train\labels"
resnet_dataset_path = r"D:\Documents\ZZ_Datasets\Resnet_Combined_FINAL(3-15-25)\train"

output_images_path = r"D:\Documents\ZZ_Datasets\New_synthetic\train\images"
output_labels_path = r"D:\Documents\ZZ_Datasets\New_synthetic\train\labels"
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

def yolo_to_pixel(bbox, img_width, img_height):
    """Convert YOLO format to pixel coordinates."""
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return x1, y1, x2, y2

def get_resnet_class_mapping():
    """Return sorted folder names and a mapping from folder name to class index."""
    class_folders = sorted([
        f for f in os.listdir(resnet_dataset_path)
        if os.path.isdir(os.path.join(resnet_dataset_path, f))
    ])
    mapping = {folder_name: idx for idx, folder_name in enumerate(class_folders)}
    return mapping, class_folders

def get_resnet_images_by_class(class_folders):
    """Return dictionary of resnet images per class by folder name."""
    resnet_images = {}
    for folder in class_folders:
        path = os.path.join(resnet_dataset_path, folder)
        images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]
        resnet_images[folder] = images
    return resnet_images
def replace_signs(img, label_path, target_folder, resnet_images, max_per_class, label_mapping):
    h, w, _ = img.shape
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()
        original_id = int(parts[0])
        mapped_id = label_mapping.get(target_folder)

        if mapped_id != original_id:
            continue

        bbox = list(map(float, parts[1:]))
        x1, y1, x2, y2 = yolo_to_pixel(bbox, w, h)

        class_folder = os.path.join(resnet_dataset_path, target_folder)
        if not os.path.exists(class_folder):
            continue

        available_imgs = resnet_images[target_folder]
        if len(available_imgs) < max_per_class:
            available_imgs = available_imgs * ((max_per_class // len(available_imgs)) + 1)

        selected_img = cv2.imread(os.path.join(class_folder, random.choice(available_imgs)))
        if selected_img is None:
            continue

        try:
            stretched = cv2.resize(selected_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            img[y1:y2, x1:x2] = stretched
        except:
            continue

        new_line = f"{mapped_id} {' '.join(map(str, parts[1:]))}"
        new_lines.append(new_line)

    return img, new_lines

def main():
    label_mapping, sorted_class_folders = get_resnet_class_mapping()
    resnet_images = get_resnet_images_by_class(sorted_class_folders)

    print("\nResNet Classes and Image Counts:")
    for folder_name, idx in label_mapping.items():
        print(f"Class folder '{folder_name}' → Class ID {idx}: {len(resnet_images[folder_name])} images")

    max_per_class = int(input("\nEnter the number of signs per class to include in the output dataset: "))

    # Get all YOLO images
    yolo_images = [f for f in os.listdir(yolo_images_path) if f.lower().endswith(('.jpg', '.png'))]

    used_per_class = {k: 0 for k in label_mapping.keys()}  # original class ID -> count

    for filename in tqdm(yolo_images, desc="Processing images", unit="image"):
        image_path = os.path.join(yolo_images_path, filename)
        label_path = os.path.join(yolo_labels_path, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(image_path)
        final_lines = []

        for class_id in used_per_class:
            if used_per_class[class_id] >= max_per_class:
                continue
            img, new_lines = replace_signs(img, label_path, class_id, resnet_images, max_per_class, label_mapping)
            used_per_class[class_id] += len(new_lines)
            final_lines.extend(new_lines)

        # Save only if we added new lines (labels)
        if final_lines:
            cv2.imwrite(os.path.join(output_images_path, filename), img)
            out_label = os.path.join(output_labels_path, os.path.splitext(filename)[0] + ".txt")
            with open(out_label, "w") as f:
                f.write("\n".join(final_lines) + "\n")

if __name__ == "__main__":
    main()