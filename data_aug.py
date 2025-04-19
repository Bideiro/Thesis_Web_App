import os
import cv2
import numpy as np
import random
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define ResNet-compatible augmentations
augment = A.Compose([
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50), p=0.3),    # Simulating motion blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Adding noise
    ], p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    ], p=0.5),
    #A.HorizontalFlip(p=0.5),  # Flip horizontally if applicable
])

def augment_images(input_dir, output_dir, num_augments=3):
    """Applies data augmentation to images in a directory and saves them."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip unreadable images
        
        filename = os.path.basename(img_path).split('.')[0]

        # Save the original image as well
        cv2.imwrite(os.path.join(output_dir, f"{filename}_original.jpg"), img)

        for i in range(num_augments):
            augmented = augment(image=img)["image"]
            cv2.imwrite(os.path.join(output_dir, f"{filename}_aug{i+1}.jpg"), augmented)

    print(f"Augmented images saved in: {output_dir}")

# Example usage
input_folder = "C:/Users/Rheiniel F. Damasco/Desktop/Paulit-ulit-na- katarantaduhan/CONT"  # Change to your dataset folder
output_folder = "C:/Users/Rheiniel F. Damasco/Desktop/Paulit-ulit-na- katarantaduhan/aug"  # Change to your output folder

augment_images(input_folder, output_folder, num_augments=3)
