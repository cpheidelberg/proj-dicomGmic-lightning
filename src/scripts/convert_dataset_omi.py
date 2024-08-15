import pydicom
import pandas as pd, numpy as np
import os, json, sys, dataclasses

import pydicom.errors

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

import src.cropping.crop_mammogram as cropping
import src.optimal_centers.get_optimal_centers as centers
import src.data_loading.loading as loading


@dataclasses.dataclass(frozen=True)
class DICOM:
    path: str
    image: np.ndarray
    view: str
    horizontal_flip: str
    category: str
    center: tuple[int, int]


def _get_category(data: list[dict]):
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


def _get_center(image: np.ndarray, view: str, horizontal_flip: str):
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


def _process_dicom(path: str, data: dict) -> list[DICOM]:
    try:
        with pydicom.read_file(path) as file:
            path = path
            image = file.pixel_array
            view = f'{file.ImageLaterality}-{file.ViewPosition}'
            horizontal_flip = 'YES' if file.FieldOfViewHorizontalFlip == 'YES' else 'NO'
            category = _get_category(data)
            center = _get_center(image, view, horizontal_flip)
        return [DICOM(path, image, view, horizontal_flip, category, center)]
    except:
        return []


def _process_patient(root: str, patient: str):
    scans = []
    with open(os.path.join(root, 'DATA', patient, f'IMAGEDB_{patient}.json')) as file:
        try:
            loaded = json.loads(file.read())['STUDIES']
        except:
            print('Could not load JSON for patient:', patient)
            return []

    folders = [(folder, val) for folder, it in loaded.items() for val in it.values()]
    dicts = [(folder, val) for folder, val in folders if isinstance(val, dict)]
    scans = [(f'{folder}/{key}.dcm', val) for folder, dic in dicts for key, val in dic.items()]
    findings = {scan: (list(lesions.values()) if lesions else []) for scan, lesions in scans}

    studies = os.listdir(os.path.join(root, 'IMAGES', patient))
    images = [f'{study}/{img}' for study in studies for img in os.listdir(os.path.join(root, 'IMAGES', patient, study))]
    paths = [(os.path.join(root, 'IMAGES', patient, img), findings[img]) for img in images]

    return [dcm for path, data in paths for dcm in _process_dicom(path, data)]


def _process_patients(root: str):
    return [dcm for patient in os.listdir(os.path.join(root, 'IMAGES')) for dcm in _process_patient(root, patient)]


def _sorted_categories(dicoms: list[DICOM]):
    sizes = {}
    for dcm in dicoms:
        sizes[dcm.category] = sizes.get(dcm.category, 0) + 1
    return sorted(sizes.keys(), key=lambda category: -sizes[category])


def _divide_dicoms(dicoms: list[DICOM], categories: list[str]) -> list[list[DICOM]]:
    split = [[] for _ in categories]
    for dcm in dicoms:
        split[categories.index(dcm.category)].append(dcm)
    return split


def _save_image(dcm: DICOM, category_index: int, category_name: str, dst_dir: str, index: int):
    dst_name = f'{category_index}/{index}.png'
    dcm_name = '/'.join(dcm.path.split('/')[-3:])

    image = loading.flip_and_crop(dcm.image, dcm.view, dcm.horizontal_flip, dcm.center)
    loading.write_image(os.path.join(dst_dir, dst_name), image)

    return [dst_name, dcm_name, category_name]


def main(src_dir: str, dst_dir: str):
    dicoms = _process_patients(src_dir)
    categories = _sorted_categories(dicoms)
    dicoms = _divide_dicoms(dicoms, categories)

    mapp = pd.DataFrame({'png': [], 'dicom': [], 'label': []}, dtype=str)

    for category_index, category_name in enumerate(categories):
        os.makedirs(os.path.join(dst_dir, str(category_index)), exist_ok=True)

        for i, dcm in enumerate(dicoms[category_index]):
            mapp.loc[len(mapp), :] = _save_image(dcm, category_index, category_name, dst_dir, i)
            print(f'{category_index}/{len(categories)}: {100 * i // len(dicoms[category_index])} %')

    mapp.to_csv(os.path.join(dst_dir, 'mapping.csv'))


if __name__ == '__main__':
    main('/home/ubuntu/sdsHD/sd21c015/DataBaseMammography/OMI_SAMPLE', '/home/ubuntu/gmic/omidb_data')
