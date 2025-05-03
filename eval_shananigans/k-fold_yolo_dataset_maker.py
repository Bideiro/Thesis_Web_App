import os
import shutil
import random
from sklearn.model_selection import KFold

def make_kfold_dataset(input_train_dir, output_dir, k=5, seed=42):
    img_dir = os.path.join(input_train_dir, 'images')
    lbl_dir = os.path.join(input_train_dir, 'labels')

    assert os.path.exists(img_dir) and os.path.exists(lbl_dir), "Train Image/Label directories not found."

    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    image_files.sort()
    random.seed(seed)
    random.shuffle(image_files)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        print(f"Creating Fold {fold}...")
        fold_path = os.path.join(output_dir, f'fold{fold}')

        for split_name, indices in zip(['train', 'val'], [train_idx, val_idx]):
            split_img_dir = os.path.join(fold_path, split_name, 'images')
            split_lbl_dir = os.path.join(fold_path, split_name, 'labels')
            os.makedirs(split_img_dir, exist_ok=True)
            os.makedirs(split_lbl_dir, exist_ok=True)

            for i in indices:
                img_file = image_files[i]
                base_name = os.path.splitext(img_file)[0]
                lbl_file = base_name + '.txt'

                src_img = os.path.join(img_dir, img_file)
                src_lbl = os.path.join(lbl_dir, lbl_file)

                dst_img = os.path.join(split_img_dir, img_file)
                dst_lbl = os.path.join(split_lbl_dir, lbl_file)

                shutil.copy2(src_img, dst_img)
                if os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, dst_lbl)

    print(f"✅ Done! {k}-Fold dataset saved to: {output_dir}")

# Example usage:
make_kfold_dataset(r'D:\Documents\ZZ_Datasets\Synthetic_Cleaned_FINAL(4-21-25)\train', r'D:\Documents\ZZ_Datasets\yolo-5-fold', k=5)
