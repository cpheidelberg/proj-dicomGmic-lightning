import numpy as np
import pydicom as dcm
import os, json, sys
import imageio.v3 as imageio

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

import src.cropping.crop_mammogram as cropping
import src.optimal_centers.get_optimal_centers as optimal_centers


def get_view(file: dcm.FileDataset):
    return f'{file.ImageLaterality}-{file.ViewPosition}'


def get_best_center(image: np.ndarray, view: str, horizontal_flip: str):
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
    return optimal_centers.extract_center(datum, image)


def get_category(file: dcm.FileDataset, data: dict):
    screening = data['SCREENING'][file.ImageLaterality]
    if screening.get('Mass'):
        return 'Mass'
    elif screening.get('Microcalcification') or screening.get('MicrocalcWithMass'):
        return 'Suspicious Calcification'
    else:
        return 'No Finding'


def convert_file(path: str, data: dict):
    with dcm.read_file(path) as file:
        horizontal_flip = file.FieldOfViewHorizontalFlip
        view = get_view(file)

        return {
            'path': path,
            'horizontal_flip': horizontal_flip,
            'view': view,
            'best_center': get_best_center(file.pixel_array, view, horizontal_flip),
            'category': get_category(file, data)
        }


def convert_patient(root: str, patient: str):
    with open(os.path.join(root, 'DATA', patient, f'NBSS_{patient}.json')) as file:
        episodes = [value for value in json.load(file).values() if isinstance(value, dict)]

    studies = [(study, data) for data in episodes for study in data['StudyList'].split(',') if study]
    paths = [(e.path, data) for study, data in studies for e in os.scandir(os.path.join(root, 'IMAGES', patient, study))]

    return [convert_file(path, data) for path, data in paths]


def convert_all(root: str) -> list:
    result = []
    for patient in os.listdir(os.path.join(root, 'IMAGES')):
        result += convert_patient(root, patient)
    return result


the_root = '/home/student/sdsHD/sd21c015/DataBaseMammography/OMI_SAMPLE'

