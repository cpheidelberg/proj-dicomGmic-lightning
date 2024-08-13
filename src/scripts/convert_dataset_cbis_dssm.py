import pydicom
import pandas as pd, numpy as np
import os, sys, dataclasses

import pydicom.errors

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

import src.cropping.crop_mammogram as cropping
import src.optimal_centers.get_optimal_centers as centers
import src.data.loading as loading


@dataclasses.dataclass(frozen=True)
class DICOM:
    path: str
    image: np.ndarray
    view: str
    horizontal_flip: str
    category: str
    center: tuple[int, int]


def _parse_folder_name(folder: str):
    label, rest = folder.split('-', maxsplit=1)
    parts = rest.split('_')
    image_laterality = 'R' if parts[3] == 'RIGHT' else 'L'
    view_position = 'MLO' if parts[4].startswith('MLO') else 'CC'

    view = f'{image_laterality}-{view_position}'
    horizontal_flip = 'NO'
    category = 'Suspicious Calcification' if label == 'Calc' else 'Mass'
    return view, horizontal_flip, category


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


def _process_dicom(path: str, prefix: str) -> list[DICOM]:
    try:
        with pydicom.read_file(path) as file:
            image = file.pixel_array
            folder = path.removeprefix(prefix).lstrip('/').split('/')[0]
            view, horizontal_flip, category = _parse_folder_name(folder)
            center = _get_center(image, view, horizontal_flip)
            return DICOM(path, image, view, horizontal_flip, category, center)
    except BaseException as err:
        print('Error: ', path, err)
        return []


def _get_paths(root: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.dcm'):
                paths.append(os.path.join(dirpath, filename))
    return paths


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
    dicoms = [dcm for path in _get_paths(src_dir) for dcm in _process_dicom(path, src_dir)]

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
    main('/home/ubuntu/sdsHD/sd18a006/DataBaseMammography/CBIS-DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/CBIS-DDSM', '/home/ubuntu/gmic/cbis_dssm_data')
