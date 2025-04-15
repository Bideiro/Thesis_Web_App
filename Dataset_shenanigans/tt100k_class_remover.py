import os
import shutil
import yaml
from tqdm import tqdm

def load_classes_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        return data['names'] if 'names' in data else data.get('classes', [])

def get_annotation_classes(annotation_path):
    classes = set()
    with open(annotation_path, 'r') as f:
        for line in f:
            if line.strip():
                cls_id = int(line.split()[0])
                classes.add(cls_id)
    return classes

def copy_files(image_path, annotation_path, dest_image_dir, dest_label_dir):
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    shutil.copy(image_path, os.path.join(dest_image_dir, os.path.basename(image_path)))
    shutil.copy(annotation_path, os.path.join(dest_label_dir, os.path.basename(annotation_path)))

def organize_dataset(yaml_file, annotations_dir, images_dir, output_dir='output'):
    class_names = load_classes_from_yaml(yaml_file)
    print(f"Classes found: {class_names}")
    
    keep_input = input("Enter the class names you want to KEEP (comma-separated): ")
    keep_class_names = [name.strip() for name in keep_input.split(',')]
    keep_class_ids = [i for i, name in enumerate(class_names) if name in keep_class_names]

    unkeep_class_ids = [i for i in range(len(class_names)) if i not in keep_class_ids]

    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    
    for ann_file in tqdm(annotation_files, desc="Processing files", unit="file"):
        annotation_path = os.path.join(annotations_dir, ann_file)
        image_name = os.path.splitext(ann_file)[0]
        image_path = os.path.join(images_dir, image_name + '.jpg')  # assuming jpg

        if not os.path.exists(image_path):
            print(f"[!] Image {image_name}.jpg not found, skipping...")
            continue

        classes_in_ann = get_annotation_classes(annotation_path)
        has_keep = any(cls in keep_class_ids for cls in classes_in_ann)
        has_unkeep = any(cls in unkeep_class_ids for cls in classes_in_ann)

        if has_keep and has_unkeep:
            both_img_dir = os.path.join(output_dir, 'both', 'images')
            both_lbl_dir = os.path.join(output_dir, 'both', 'labels')
            copy_files(image_path, annotation_path, both_img_dir, both_lbl_dir)
        elif has_keep:
            for cls_id in classes_in_ann:
                if cls_id in keep_class_ids:
                    cls_name = class_names[cls_id]
                    img_dir = os.path.join(output_dir, 'keep', cls_name, 'images')
                    lbl_dir = os.path.join(output_dir, 'keep', cls_name, 'labels')
                    copy_files(image_path, annotation_path, img_dir, lbl_dir)
        elif has_unkeep:
            for cls_id in classes_in_ann:
                if cls_id in unkeep_class_ids:
                    cls_name = class_names[cls_id]
                    img_dir = os.path.join(output_dir, 'unkeep', cls_name, 'images')
                    lbl_dir = os.path.join(output_dir, 'unkeep', cls_name, 'labels')
                    copy_files(image_path, annotation_path, img_dir, lbl_dir)
# Example usage
organize_dataset(
    yaml_file=r'd:\Documents\ZZ_Datasets\CCTSDB-AUG\data.yaml',
    annotations_dir=r'd:\Documents\ZZ_Datasets\CCTSDB-AUG\train\labels',
    images_dir=r'd:\Documents\ZZ_Datasets\CCTSDB-AUG\train\images'
)
