import os
import shutil

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
        print(f"Moved {file} to {new_file_path}")

# Replace 'path_to_exam_folder' with the actual path to your exam folder
rearrange_dicom_files('dicom_exams/ff02ce510218e943fba03aa113761fc4')
