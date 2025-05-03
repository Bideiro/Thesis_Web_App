import os
import shutil
import random
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

def make_kfold_classification_dataset(input_train_dir, output_dir, k=5, seed=42):
    random.seed(seed)
    image_paths = []
    image_labels = []

    # Step 1: Gather all images and their labels
    for class_name in sorted(os.listdir(input_train_dir)):
        class_path = os.path.join(input_train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_path, img_file))
                image_labels.append(class_name)

    # Step 2: Stratified K-Fold split across full dataset
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, image_labels)):
        print(f"\n🔁 Creating Fold {fold}...")

        fold_dir = os.path.join(output_dir, f'fold{fold}')
        train_txt_path = os.path.join(fold_dir, 'train.txt')
        val_txt_path = os.path.join(fold_dir, 'val.txt')

        train_txt = []
        val_txt = []

        for split_name, indices, txt_file in zip(['train', 'val'], [train_idx, val_idx], [train_txt, val_txt]):
            for idx in indices:
                img_path = image_paths[idx]
                label = image_labels[idx]
                dst_dir = os.path.join(fold_dir, split_name, label)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dst_path)
                txt_file.append(dst_path)

        # Save train.txt and val.txt
        with open(train_txt_path, 'w') as f:
            f.write('\n'.join(train_txt))
        with open(val_txt_path, 'w') as f:
            f.write('\n'.join(val_txt))

        print(f"✅ Fold {fold} done. Images: {len(train_txt)} train, {len(val_txt)} val")

    print(f"\n🎉 All {k} folds created successfully at: {output_dir}")

# Example usage:
make_kfold_classification_dataset(r'D:\Documents\ZZ_Datasets\Resnet_GTSRB_Cleaned_FINAL(4-20-25)\train', r'D:\Documents\ZZ_Datasets\resnet-5-fold', k=5)
