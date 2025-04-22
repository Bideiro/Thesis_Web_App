from pathlib import Path
import cv2
from tqdm import tqdm
import itertools
from collections import defaultdict
from torchvision import datasets
import random

yolo_dataset = r"D:\Documents\ZZ_Datasets\tt100k_2022_YOLO\train"
resnet_dataset = r"D:\Documents\ZZ_Datasets\Resnet_GTSRB_Cleaned_FINAL(4-20-25)\train"

output = r"D:\Documents\ZZ_Datasets\Synthetic_Cleaned_FINAL(4-21-25)\train"

resnet_dataset = Path(resnet_dataset)
yolo_dataset = Path(yolo_dataset)
output = Path(output)

yolo_images = yolo_dataset / "images"
yolo_labels = yolo_dataset / "labels"

output_images = output / "images"
output_labels = output / "labels"
output_images.mkdir(parents=True, exist_ok=True)
output_labels.mkdir(parents=True, exist_ok=True)


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


def pick_random_lowest_value(d, exclude_value=None):
    # Step 1: Find the minimum value in the dictionary
    min_value = min(d.values())

    # Step 2: Find all keys with the minimum value, excluding those with a value >= exclude_value
    min_keys = [key for key, value in d.items() if value == min_value and (exclude_value is None or value < exclude_value)]

    # Step 3: If no valid keys remain (after excluding), handle the case
    if not min_keys:
        return False
    
    # Step 4: If there are multiple keys with the minimum value, pick one randomly
    return random.choice(min_keys)


if __name__ == "__main__":
    
    # getting resnet labels
    # Assuming resnet_dataset is already defined (path to your ImageFolder dataset)
    dataset = datasets.ImageFolder(resnet_dataset)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Initialize a defaultdict for class counts
    class_counts = defaultdict(int)

    # Initialize a dictionary for class names to image names
    class_images_paths = defaultdict(list)

    # Count the images in each class and collect image names
    for img_path, label in dataset.samples:
        class_name = idx_to_class[label]
        class_counts[class_name] += 1
        
        # Using pathlib to get the image filename (just the basename of the file)
        image_path = Path(img_path).resolve()
        class_images_paths[class_name].append(image_path)

    # Print the class counts
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")

    # Labels list
    Resnet_labels = []
    print("\nPreparing Resnet Classes")
    for thing in tqdm(list(Path(resnet_dataset).iterdir()), desc="Getting Resnet Classes", unit="Folders"):
        if thing.is_dir():
            Resnet_labels.append(thing.name)
    Resnet_labels.sort()
    
    print("\nDone Preparing Resnet Classes")
    
    print("\nProceeding with main process")
    
    max = input("\nEach Class must have?: ")
    max = int(max)
    
    curr_class_counts = defaultdict(int, {key: 0 for key in class_counts})
    total_class_counts = defaultdict(int, {key: 0 for key in class_counts})
    
    # temp = list(yolo_images.iterdir())
    # for image_path in itertools.cycle(temp):
    
    temp = list(yolo_images.iterdir())
    total_needed = len(curr_class_counts) * max
    pbar = tqdm(total=total_needed, desc="Generating synthetic images", unit="image")

    for image_path in itertools.cycle(temp):
        if sum(total_class_counts.values()) >= total_needed:
            break

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        # Get annotation
        name = image_path.stem
        curr_txt_path = yolo_labels / f"{name}.txt"

        if not curr_txt_path.exists():
            print("\nAnnotation does not exist! Skipping!")
            continue

        with open(curr_txt_path, "r") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()
            h, w, _ = image.shape
            bbox = list(map(float, parts[1:]))
            x1, y1, x2, y2 = yolo_to_pixel(bbox, w, h)
            
            
            # Choosing a resnet image
            chosen_class = pick_random_lowest_value(curr_class_counts)
            classNo = Resnet_labels.index(chosen_class)
            curr_class = resnet_dataset / chosen_class
            
            if curr_class_counts[chosen_class] >= class_counts[chosen_class]:
                curr_class = curr_class_counts[chosen_class]%class_counts[chosen_class]
                curr_class_counts[chosen_class] += 1
                total_class_counts[chosen_class] += 1
                Rimage_path = class_images_paths[chosen_class][curr_class - 1]
            else:
                curr_class_counts[chosen_class] += 1
                total_class_counts[chosen_class] += 1
                Rimage_path = class_images_paths[chosen_class][curr_class_counts[chosen_class] - 1]
            
            if sum(total_class_counts.values()) >= total_needed:
                break

            
            # Load the ResNet image
            Rimage = cv2.imread(str(Rimage_path))
            if Rimage is None:
                continue  # Skip broken or non-image files
            # Done
            
            # put in yolo image
            try:
                stretched = cv2.resize(Rimage, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                image[y1:y2, x1:x2] = stretched
            except Exception as e:
                print(f"Resize or paste failed: {e}")
                continue

            new_line = f"{classNo} {' '.join(map(str, parts[1:]))}"
            new_lines.append(new_line)

        # Save modified image
        output_filename = f"{image_path.stem}_{classNo}.jpg"
        output_path = output_images / output_filename
        cv2.imwrite(str(output_path), image)

        # Save modified annotation
        modified_label_path = output_labels / f"{name}_{classNo}.txt"
        with open(modified_label_path, "w") as f:
            for line in new_lines:
                f.write(line + "\n")
            
        pbar.update(1)
        
    pbar.close()
            
    print("\nProcess Done")
    