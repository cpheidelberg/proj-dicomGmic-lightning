import pydicom
import pandas as pd
import numpy as np
import os, json, sys
import typing
import logging
import multiprocessing
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

import src.cropping.crop_mammogram as cropping
import src.optimal_centers.get_optimal_centers as centers
import src.data_loading.loading as loading


"""
This script reads data from the OMI dataset and converts it to
the format we use in data_loading.dataset.ClassificationImages.

Output format:
- root directory `vindrmammo_data`
    - subdirectories `0`, `1`, ..., `K-1` for the top K classes (here, K = 5)
        Each subdirectory contains PNGs representing scans belonging to
        the class. The PNGs are preprocessed, prepared to be given to the GMIC.
    - CSV file `mapping.csv` which has one line per image with three columns:
        - png: file path of the PNG image, relative to vindrmammo_data (e.g. "2/1357.png")
        - dicom: file path of the original DICOM file
        - label: string representing the class (e.g. "Suspicious Calcification")
"""


Category = typing.Literal['Mass', 'Suspicious Calcification', 'Focal Asymmetry', 'Architectural Distortion', 'No Finding']
Dicom = tuple[str, np.ndarray, Category]

def get_category(data: list[dict]) -> Category:
    for lesion in data:
        if lesion.get('Mass') == 1:
            return 'Mass'
        if lesion.get('SuspiciousCalcifications') is not None:
            return 'Suspicious Calcification'
        if lesion.get('FocalAsymmetry') is not None:
            return 'Focal Asymmetry'
        if lesion.get('ArchitecturalDistortion') is not None:
            return 'Architectural Distortion'
    return 'No Finding'


def get_center(image: np.ndarray, view: str, horizontal_flip: str):
    mode = 'right' if view.startswith('R') else 'left'
    crop = cropping.crop_img_from_largest_connected(image, mode)
    datum = {
        'window_location': crop[0],
        'rightmost_points': crop[1],
        'bottommost_points': crop[2],
        'distance_from_starting_side': crop[3],
        'full_view': view,
        'view': view[2:],
        'horizontal_flip': horizontal_flip
    }
    return centers.extract_center(datum, image)


def process_dicom(path: str, data: dict) -> Dicom:
    try:
        with pydicom.dcmread(path) as file:
            image = file.pixel_array
            view = f'{file.ImageLaterality}-{file.ViewPosition}'
            horizontal_flip = 'YES' if file.FieldOfViewHorizontalFlip == 'YES' else 'NO'

        category = get_category(data)
        center = get_center(image, view, horizontal_flip)

        image = loading.flip_image(image, view, horizontal_flip)
        image = loading.crop_image(image, view, center)

        return (path, image, category)
    
    except FileNotFoundError:
        logging.error(f'File not found: {path}')
        return None
    
    except Exception as e:
        logging.error(f'Error processing DICOM file {path}: {e}')
        return None
    
    
def process_dicoms(inputs: list[tuple[str, dict]]) -> list[Dicom]:
    dicoms = []
    for path, data in inputs:
        dicom = process_dicom(path, data)
        if dicom is not None:
            dicoms.append(dicom)
    return dicoms


def process_patient(root: str, patient: str) -> list[Dicom]:
    """Load patient data from JSON file"""
    scans = []
    try:
        with open(os.path.join(root, 'data', patient, f'IMAGEDB_{patient}.json')) as file:
            loaded = json.loads(file.read())['STUDIES']
    except FileNotFoundError:
        logging.error(f'JSON file not found for patient: {patient}')
        return []
    except json.JSONDecodeError:
        logging.error(f'Error decoding JSON for patient: {patient}')
        return []

    folders = [(folder, val) for folder, it in loaded.items() for val in it.values()]
    dicts = [(folder, val) for folder, val in folders if isinstance(val, dict)]
    scans = [(f'{folder}/{key}.dcm', val) for folder, dic in dicts for key, val in dic.items()]
    findings = {scan: (list(lesions.values()) if lesions else []) for scan, lesions in scans}

    try:
        studies = os.listdir(os.path.join(root, 'images', patient))
    except FileNotFoundError:
        logging.error(f'Image folder not found for patient: {patient}')
        return []

    images = [f'{study}/{img}' for study in studies for img in os.listdir(os.path.join(root, 'images', patient, study))]
    paths = []
    for img in images:
        if img in findings:
            paths.append((os.path.join(root, 'images', patient, img), findings[img]))
        else:
            logging.error(f'KeyError: {img} not found in findings for patient: {patient}')

    return process_dicoms(paths)


def process_patients(root: str):
    try:
        patients = sorted(os.listdir(os.path.join(root, 'images')))[:2]
    except FileNotFoundError:
        logging.error('Root images directory not found')
        return []

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = list(tqdm(
            pool.starmap(process_patient, [(root, patient) for patient in patients]),
            total=len(patients),
            desc="Processing Patients"
        ))
    
    return [dcm for patient_results in results for dcm in patient_results]

    return [dcm for patient in patients for dcm in process_patient(root, patient)]


def sorted_categories(dicoms: list[Dicom]) -> list[Category]:
    sizes = {}
    for _, _, category in dicoms:
        sizes[category] = sizes.get(category, 0) + 1
    return sorted(sizes.keys(), key=lambda category: -sizes[category])


def group_by_category(dicoms: list[Dicom], categories: list[Category]) -> list[list[Dicom]]:
    split = [[] for _ in categories]
    for path, image, category in dicoms:
        split[categories.index(category)].append((path, image, category))
    return split


def save_image(dcm: Dicom, category_index: int, category_name: Category, dst_dir: str, index: int):
    dst_name = f'{category_index}/{index}.png'
    dcm_name = '/'.join(dcm[0].split('/')[-3:])

    loading.write_image(os.path.join(dst_dir, dst_name), dcm[1])
    return [dst_name, dcm_name, category_name]


def main(src_dir: str, dst_dir: str):
    dicom_list = process_patients(src_dir)
    categories = sorted_categories(dicom_list)
    dicoms = group_by_category(dicom_list, categories)

    mapp = pd.DataFrame({'png': [], 'dicom': [], 'label': []}, dtype=str)

    for category_index, category_name in enumerate(categories):
        os.makedirs(os.path.join(dst_dir, str(category_index)), exist_ok=True)

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = list(tqdm(
                pool.starmap(
                    save_image,
                    [(dcm, category_index, category_name, dst_dir, i) for i, dcm in enumerate(dicoms[category_index])]
                ),
                total=len(dicoms[category_index]),
                desc=f"Saving images for category {category_name}"
            ))
            
        for result in results:
            mapp.loc[len(mapp), :] = result

    mapp.to_csv(os.path.join(dst_dir, 'mapping.csv'))


if __name__ == '__main__':
    main(src_dir='/home/pb438/medken/testData', dst_dir='/home/pb438/medken/testData/extracted')
