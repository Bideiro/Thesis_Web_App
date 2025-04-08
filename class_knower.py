import os
from tqdm import tqdm
from collections import defaultdict

def extract_classes_with_counts(folder_path):
    class_counts = defaultdict(int)
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    # Progress bar for file processing
    for filename in tqdm(txt_files, desc="Processing annotation files"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

    return dict(sorted(class_counts.items()))

# Example usage:
folder = "path/to/your/annotations"  # change this to your folder
class_counts = extract_classes_with_counts(folder)

print("\nClass ID counts:")
for class_id, count in class_counts.items():
    print(f"Class {class_id}: {count} instances")
