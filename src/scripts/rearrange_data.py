import os
import shutil
from tqdm import tqdm

def rearrange_dicom_files(exam_folder):
    # Ensure the folder exists
    if not os.path.exists(exam_folder):
        print(f"The folder {exam_folder} does not exist.")
        return
    
    # List all DICOM files in the folder
    dicom_files = [f for f in os.listdir(exam_folder) if f.endswith('.dicom')]
    
    # Create subfolders and move files
    for file in dicom_files:
        # Extract the base name without extension for folder naming
        base_name = os.path.splitext(file)[0]
        subfolder_path = os.path.join(exam_folder, base_name)
        
        # Create a subfolder with the name of the file
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Construct the old and new file paths
        old_file_path = os.path.join(exam_folder, file)
        new_file_path = os.path.join(subfolder_path, '1-1.dcm')
        
        # Move and rename the DICOM file
        shutil.move(old_file_path, new_file_path)


if __name__ == "__main__":
    path = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/images/"
    folders = [os.path.join(path, f) for f in os.listdir(path) if not "." in f]

    for folder in tqdm(folders):
        rearrange_dicom_files(folder)
