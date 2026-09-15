import pydicom
import pandas as pd
import numpy as np
import os, sys
import typing
import logging
import multiprocessing
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

import src.cropping.crop_mammogram as cropping
import src.optimal_centers.get_optimal_centers as centers
import src.data_loading.loading as loading


"""
This script reads data from the CBIS-DDSM dataset and converts it to
the format used in data_loading.dataset.ClassificationImages.

Output format:
- root directory `cbis_ddsm_data`
    - subdirectories `0`, `1`, `2`, `3` for the 4 classes:
        0: Benign Calcification
        1: Malignant Calcification
        2: Benign Mass
        3: Malignant Mass
        Each subdirectory contains PNGs representing scans belonging to
        the class. The PNGs are preprocessed, prepared to be given to the GMIC.
    - CSV file `mapping.csv` which has one line per image with columns:
        - png: file path of the PNG image, relative to cbis_ddsm_data
        - dicom: file path of the original DICOM file
        - label: string representing the class
        - view: mammography view (e.g., "L-CC", "R-MLO")
        - center: optimal center coordinates
"""


Category = typing.Literal[
    'Benign Calcification', 
    'Malignant Calcification', 
    'Benign Mass', 
    'Malignant Mass'
]
Dicom = tuple[str, np.ndarray, Category, str, tuple]


def get_category(abnormality_type: str, pathology: str) -> Category:
    """
    Determine the category based on abnormality type and pathology.
    
    Args:
        abnormality_type: 'calcification' or 'mass'
        pathology: 'BENIGN', 'BENIGN_WITHOUT_CALLBACK', or 'MALIGNANT'
    
    Returns:
        One of the four categories
    """
    abnormality_type = abnormality_type.lower()
    pathology = pathology.upper()
    
    # Treat BENIGN_WITHOUT_CALLBACK as BENIGN
    is_benign = pathology in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']
    
    if abnormality_type == 'calcification':
        return 'Benign Calcification' if is_benign else 'Malignant Calcification'
    else:  # mass
        return 'Benign Mass' if is_benign else 'Malignant Mass'


def get_center(image: np.ndarray, view: str, horizontal_flip: str):
    """Extract optimal center from mammogram image."""
    mode = 'right' if view.startswith('R') else 'left'
    crop = cropping.crop_img_from_largest_connected(image, mode)
    datum = {
        'window_location': crop[0],
        'rightmost_points': crop[1],
        'bottommost_points': crop[2],
        'distance_from_starting_side': crop[3],
        'full_view': view,
        'view': view.split('-')[1] if '-' in view else view,
        'horizontal_flip': horizontal_flip
    }
    return centers.extract_center(datum, image)


def determine_view(laterality: str, view_position: str) -> str:
    """
    Determine standardized view string.
    
    Args:
        laterality: 'LEFT' or 'RIGHT'
        view_position: 'CC', 'MLO', etc.
    
    Returns:
        Standardized view string like 'L-CC' or 'R-MLO'
    """
    if view_position in ['ML', 'LM']:
        view_position = 'MLO'
    
    lat = 'L' if laterality == 'LEFT' else 'R'
    return f"{lat}-{view_position}"


def process_dicom(dicom_path: str, abnormality_type: str, pathology: str, 
                  laterality: str, view_position: str, dst_dir: str) -> Dicom:
    """
    Process a single DICOM file.
    
    Args:
        dicom_path: Full path to DICOM file
        abnormality_type: 'calcification' or 'mass'
        pathology: 'BENIGN', 'BENIGN_WITHOUT_CALLBACK', or 'MALIGNANT'
        laterality: 'LEFT' or 'RIGHT'
        view_position: 'CC', 'MLO', etc.
        dst_dir: Destination directory for processed images
    
    Returns:
        Tuple containing (dicom_path, png_path, category, view, center)
    """
    try:
        with pydicom.dcmread(dicom_path) as file:
            image = file.pixel_array
            image = loading.adjust_brightness(image)
            image = loading.optimize_contrast(image)
            
            # Determine view and horizontal flip
            view = determine_view(laterality, view_position)
            horizontal_flip = getattr(file, 'FieldOfViewHorizontalFlip', 'NO')
            horizontal_flip = 'YES' if horizontal_flip == 'YES' else 'NO'
        
        # Determine category
        category = get_category(abnormality_type, pathology)
        
        # Get optimal center
        center = get_center(image, view, horizontal_flip)
        
        # Create output path
        category_folder = os.path.join(dst_dir, "temporary", category)
        os.makedirs(category_folder, exist_ok=True)
        
        # Generate unique filename from DICOM path
        image_name = dicom_path.replace('/', '_').replace('.dcm', '.png')
        image_path = os.path.join(category_folder, image_name)
        
        if not os.path.exists(image_path):
            image = loading.flip_image(image, view, horizontal_flip)
            image = loading.crop_image(image, view, center)
            loading.write_image(image_path, image)
        
        return (dicom_path, image_path, category, view, center)
    
    except FileNotFoundError:
        logging.error(f'File not found: {dicom_path}')
        return None
    
    except Exception:
        logging.error(f'Error processing DICOM file {dicom_path}', exc_info=True)
        return None


def process_dicom_star(args: tuple) -> Dicom:
    """Wrapper for multiprocessing."""
    return process_dicom(*args)


def find_dicom_file(base_dir: str, case_folder: str) -> str:
    """
    Find the actual DICOM file in the CBIS-DDSM directory structure.
    
    The CSV contains paths like "Mass-Training_P_00001_LEFT_CC/1.3.6.../000000.dcm"
    but the actual structure is "Mass-Training_P_00001_LEFT_CC/07-20-2016-DDSM-XXXXX/1.000000-full mammogram images-XXXXX/1-1.dcm"
    
    Args:
        base_dir: Root directory of CBIS-DDSM dataset
        case_folder: Case folder name (e.g., "Mass-Training_P_00001_LEFT_CC")
    
    Returns:
        Full path to the DICOM file, or None if not found
    """
    case_path = os.path.join(base_dir, case_folder)
    
    if not os.path.exists(case_path):
        return None
    
    try:
        # Get the date folder (e.g., "07-20-2016-DDSM-74994")
        date_folders = [d for d in os.listdir(case_path) if os.path.isdir(os.path.join(case_path, d))]
        
        if not date_folders:
            return None
        
        # Usually there's only one date folder
        date_folder = date_folders[0]
        date_path = os.path.join(case_path, date_folder)
        
        # Get the series folder (e.g., "1.000000-full mammogram images-24515")
        series_folders = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
        
        if not series_folders:
            return None
        
        # Usually there's only one series folder
        series_folder = series_folders[0]
        series_path = os.path.join(date_path, series_folder)
        
        # Find the .dcm file
        dcm_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        
        if not dcm_files:
            return None
        
        # Usually there's only one .dcm file
        dcm_file = dcm_files[0]
        return os.path.join(series_path, dcm_file)
    
    except Exception as e:
        logging.error(f'Error finding DICOM in {case_path}: {e}')
        return None


def load_csv_metadata(csv_path: str, abnormality_type: str, src_dir: str) -> list[tuple]:
    """
    Load metadata from CBIS-DDSM CSV file.
    
    Args:
        csv_path: Path to CSV file (e.g., mass_case_description_train_set.csv)
        abnormality_type: 'calcification' or 'mass'
        src_dir: Root directory of CBIS-DDSM dataset
    
    Returns:
        List of tuples: (dicom_path, abnormality_type, pathology, laterality, view_position)
    """
    df = pd.read_csv(csv_path)
    
    metadata = []
    for _, row in df.iterrows():
        # Extract required fields
        image_path = row['image file path']
        pathology = row['pathology']
        laterality = row['left or right breast']
        view_position = row['image view']
        
        # Find the actual DICOM file from the case folder name (first part of the path)
        dicom_path = find_dicom_file(src_dir, image_path.split('/')[0])
        
        # Only process if the file exists
        if dicom_path is not None and os.path.exists(dicom_path):
            metadata.append((
                dicom_path,
                abnormality_type,
                pathology,
                laterality,
                view_position
            ))
        else:
            logging.warning(f'DICOM file not found: {dicom_path}')
        if len(metadata) > 4:
            break
    
    return metadata


def process_dataset(src_dir: str, dst_dir: str, csv_files: dict):
    """
    Process all CBIS-DDSM data.
    
    Args:
        src_dir: Root directory of CBIS-DDSM dataset
        dst_dir: Destination directory for processed data
        csv_files: Dictionary with keys 'mass_train', 'mass_test', 'calc_train', 'calc_test'
                   mapping to CSV file paths
    """
    all_metadata = []
    
    # Load metadata from all CSV files
    for key, csv_path in csv_files.items():
        if 'mass' in key:
            abnormality_type = 'mass'
        else:
            abnormality_type = 'calcification'
        
        print(f"Loading metadata from {csv_path}...")
        metadata = load_csv_metadata(csv_path, abnormality_type, src_dir)
        all_metadata.extend(metadata)
        print(f"  Found {len(metadata)} cases")
    
    print(f"\nTotal cases to process: {len(all_metadata)}")
    
    # Add dst_dir to each metadata tuple
    process_args = [(path, abn_type, pathology, lat, view, dst_dir) 
                    for path, abn_type, pathology, lat, view in all_metadata]
    
    # Process DICOMs in parallel
    print("\nProcessing DICOM files...")
    with multiprocessing.Pool(multiprocessing.cpu_count() - 3) as pool:
        results = list(tqdm(
            pool.imap(process_dicom_star, process_args),
            total=len(process_args),
            desc="Processing DICOMs"
        ))
    
    # Filter out None results (failed processing)
    dicom_list = [r for r in results if r is not None]
    
    return dicom_list


def sorted_categories(dicoms: list[Dicom]) -> list[Category]:
    """Get categories sorted by frequency."""
    sizes = {}
    print(f"\nTotal number of processed DICOMs: {len(dicoms)}")
    for _, _, category, _, _ in dicoms:
        sizes[category] = sizes.get(category, 0) + 1
    print(f"Category distribution: {sizes}")
    return sorted(sizes.keys(), key=lambda category: -sizes[category])


def group_by_category(dicoms: list[Dicom], categories: list[Category]) -> list[list[Dicom]]:
    """Group DICOMs by category."""
    split = [[] for _ in categories]
    for dicom in dicoms:
        split[categories.index(dicom[2])].append(dicom)
    return split


def save_image(dcm: Dicom, category_index: int, category_name: Category, 
               dst_dir: str, index: int) -> list:
    """Save processed image to final location."""
    dst_name = f'{category_index}/{index}.png'
    dcm_name = dcm[0]  # Original DICOM path
    
    try:
        os.rename(dcm[1], os.path.join(dst_dir, dst_name))
    except FileNotFoundError:
        logging.warning(f'File not found when saving image: {dcm[1]}')
    except Exception as e:
        logging.error(f'Error saving image {dcm[1]} to {dst_name}: {e}', exc_info=True)
    return [dst_name, dcm_name, category_name, dcm[3], dcm[4]]


def save_image_star(args: tuple) -> list:
    """Wrapper for multiprocessing."""
    return save_image(*args)


def main(src_dir: str, dst_dir: str, csv_files: dict):
    """
    Main processing function.
    
    Args:
        src_dir: Root directory of CBIS-DDSM dataset
        dst_dir: Destination directory for processed data
        csv_files: Dictionary mapping dataset splits to CSV file paths
    """
    # Process all DICOMs
    dicom_list = process_dataset(src_dir, dst_dir, csv_files)
    
    # Sort and group by category
    categories = sorted_categories(dicom_list)
    dicoms = group_by_category(dicom_list, categories)
    
    # Create output DataFrame
    mapp = pd.DataFrame({
        'png': [], 
        'dicom': [], 
        'label': [], 
        'view': [], 
        'center': []
    }, dtype=str)
    
    print(f"\nSaving images by category...")
    for category_index, category_name in enumerate(categories):
        os.makedirs(os.path.join(dst_dir, str(category_index)), exist_ok=True)
        
        dicom_list_args = [
            (dcm, category_index, category_name, dst_dir, i) 
            for i, dcm in enumerate(dicoms[category_index])
        ]
        
        with multiprocessing.Pool(multiprocessing.cpu_count() - 3) as pool:
            results = list(tqdm(
                pool.imap(save_image_star, dicom_list_args),
                total=len(dicoms[category_index]),
                desc=f"Saving {category_name}"
            ))
        
        for result in results:
            mapp.loc[len(mapp), :] = result
    
    # Save mapping CSV
    mapp.to_csv(os.path.join(dst_dir, 'mapping.csv'), index=False)
    print(f"\nProcessing complete! Mapping saved to {os.path.join(dst_dir, 'mapping.csv')}")


if __name__ == '__main__':
    # Example usage:
    # Update these paths according to your CBIS-DDSM installation
    
    src_directory = '../sdsHD/sd18a006/DataBaseMammography/CBIS-DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/CBIS-DDSM'
    dst_directory = 'Data_CBIS-DDSM'

    # Set up logging
    logging.basicConfig(filename=os.path.join(dst_directory, 'error_log.txt'), level=logging.ERROR)

    # Paths to the CSV metadata files
    csv_metadata_files = {
        'mass_train': os.path.join(dst_directory, 'mass_case_description_train_set.csv'),
        'mass_test': os.path.join(dst_directory, 'mass_case_description_test_set.csv'),
        'calc_train': os.path.join(dst_directory, 'calc_case_description_train_set.csv'),
        'calc_test': os.path.join(dst_directory, 'calc_case_description_test_set.csv'),
    }
    
    main(src_directory, dst_directory, csv_metadata_files)