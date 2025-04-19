import os
from tqdm import tqdm

def rename_files_in_folder(parent_folder):
    for foldername in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, foldername)

        if os.path.isdir(folder_path):
            files = sorted(os.listdir(folder_path))
            total_files = len(files)

            with tqdm(total=total_files, desc=f"Renaming in '{foldername}'", unit="file") as pbar:
                for idx, filename in enumerate(files, start=1):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(filename)
                        new_filename = f"{idx}_{foldername}{ext}"
                        new_file_path = os.path.join(folder_path, new_filename)
                        os.rename(file_path, new_file_path)
                    pbar.update(1)

if __name__ == "__main__":
    parent_folder = input("Enter the path to the parent folder: ")
    rename_files_in_folder(parent_folder)
