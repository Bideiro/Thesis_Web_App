import os
import yaml

def find_classes(labels_dir):
    class_ids = set()
    for file in os.listdir(labels_dir):
        if file.endswith(".txt"):
            with open(os.path.join(labels_dir, file)) as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
    return sorted(class_ids)

def create_yaml(images_path, labels_path, save_path="dataset.yaml", names=None):
    dataset = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(images_path, split)
        if os.path.exists(split_path):
            dataset[split] = split_path

    class_ids = find_classes(os.path.join(labels_path, 'train' if 'train' in os.listdir(labels_path) else ''))
    nc = len(class_ids)

    if names is None:
        names = [f"class_{i}" for i in class_ids]

    yaml_dict = {
        "path": images_path,
        "train": dataset.get("train", ""),
        "val": dataset.get("val", ""),
        "test": dataset.get("test", ""),
        "nc": nc,
        "names": names
    }

    with open(save_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"YAML file saved to {save_path}")

# Example usage:
create_yaml(
    images_path=r"C:\Users\dei\Downloads\YOLO\synthetic\train\images",  # adjust this to your images directory
    labels_path=r"C:\Users\dei\Downloads\YOLO\synthetic\train\labels",  # adjust this to your labels directory
)
