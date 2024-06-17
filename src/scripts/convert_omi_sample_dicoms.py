import pydicom
import pandas as pd
import os, json, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

import src.cropping.crop_mammogram as cropping
import src.optimal_centers.get_optimal_centers as centers
import src.data.loading as loading


class _dicom:
    @staticmethod
    def get_category(file: pydicom.FileDataset, data: dict):
        lesions = data.get('LESION', {}).get(file.ImageLaterality, {})
        lesion = ';'.join(l.get('LesionDescription', '') for l in lesions.values()).lower()

        if 'mass' in lesion:
            return 'Mass'
        elif 'calcification' in lesion:
            return 'Suspicious Calcification'
        else:
            return 'No Finding'

    def get_center(self):
        mode = 'right' if self.view.startswith('R') else 'left'
        crop = cropping.crop_img_from_largest_connected(self.image, mode)
        datum = {
            'window_location': crop[0],
            'rightmost_points': crop[1],
            'bottommost_points': crop[2],
            'distance_from_starting_side': crop[3],
            'full_view': self.view,
            'view': self.view[2:],
            'horizontal_flip': self.horizontal_flip
        }
        return centers.extract_center(datum, self.image)

    def __init__(self, path: str, data: dict):
        with pydicom.read_file(path) as file:
            self.path = path
            self.image = file.pixel_array
            self.view = f'{file.ImageLaterality}-{file.ViewPosition}'
            self.horizontal_flip = 'YES' if file.FieldOfViewHorizontalFlip == 'YES' else 'NO'
            self.category = _dicom.get_category(file, data)
            self.center = self.get_center()


def _process_dicom(path: str, data: dict) -> list[_dicom]:
    try:
        return [_dicom(path, data)]
    except RuntimeError:
        return []


def _process_patient(root: str, patient: str):
    with open(os.path.join(root, 'DATA', patient, f'NBSS_{patient}.json')) as file:
        episodes = [value for value in json.load(file).values() if isinstance(value, dict)]

    studies = [(study, data) for data in episodes for study in data['StudyList'].split(',') if study]
    paths = [(e.path, data) for study, data in studies for e in os.scandir(os.path.join(root, 'IMAGES', patient, study))]

    return [dcm for path, data in paths for dcm in _process_dicom(path, data)]


def _process_patients(root: str):
    return [dcm for patient in os.listdir(os.path.join(root, 'IMAGES')) for dcm in _process_patient(root, patient)]


def _sorted_categories(dicoms: list[_dicom]):
    sizes = {}
    for dcm in dicoms:
        sizes[dcm.category] = sizes.get(dcm.category, 0) + 1
    return sorted(sizes.keys(), key=lambda category: -sizes[category])


def _divide_dicoms(dicoms: list[_dicom], categories: list[str]) -> list[list[_dicom]]:
    split = [[] for _ in categories]
    for dcm in dicoms:
        split[categories.index(dcm.category)].append(dcm)
    return split


def _save_image(dcm: _dicom, category_index: int, category_name: str, dst_dir: str, index: int):
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
    main('/home/student/sdsHD/sd21c015/DataBaseMammography/OMI_SAMPLE', '/home/student/gmic/omisample_data')
