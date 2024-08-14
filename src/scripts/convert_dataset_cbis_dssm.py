import pydicom
import pandas as pd, numpy as np
import os, sys, dataclasses

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
    category: str


def _parse_folder_name(folder: str):
    label, rest = folder.split('-', maxsplit=1)
    parts = rest.split('_')
    image_laterality = 'R' if parts[3] == 'RIGHT' else 'L'
    view_position = 'MLO' if parts[4].startswith('MLO') else 'CC'

    view = f'{image_laterality}-{view_position}'
    category = 'Suspicious Calcification' if label == 'Calc' else 'Mass'
    return view, category


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


def _read_dicom(view: str, category: str, path: str) -> DICOM:
    with pydicom.read_file(path) as file:
        image = file.pixel_array.astype('int32')
        center = _get_center(image, view, horizontal_flip='NO')
        image = loading.flip_and_crop(image, view, 'NO', center)
        return DICOM(path, image, category)


def _subdirs(root: str):
    return (e for e in os.scandir(root) if e.is_dir())


def _subfiles(root: str):
    return (e for e in os.scandir(root) if e.is_file())


def _get_paths(root: str):
    for folder in _subdirs(root):
        view, category = _parse_folder_name(folder.name)
        for sub1 in _subdirs(folder.path):
            for sub2 in _subdirs(sub1.path):
                for file in _subfiles(sub2.path):
                    yield view, category, file.path


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

    loading.write_image(os.path.join(dst_dir, dst_name), dcm.image)
    return [dst_name, dcm_name, category_name]


def main(src_dir: str, dst_dir: str):
    paths = list(_get_paths(src_dir))
    dicoms = []
    for i, (view, category, path) in enumerate(paths):
        try:
            dicoms.append(_read_dicom(view, category, path))
            print(f'Reading DICOMs: {100*(i+1)//len(paths)} %')
        except:
            pass

    categories = _sorted_categories(dicoms)
    buckets = _divide_dicoms(dicoms, categories)

    mapp = pd.DataFrame({'png': [], 'dicom': [], 'label': []}, dtype=str)

    counter = 0
    for category_index, category_name in enumerate(categories):
        os.makedirs(os.path.join(dst_dir, str(category_index)), exist_ok=True)

        for i, dcm in enumerate(buckets[category_index]):
            mapp.loc[len(mapp), :] = _save_image(dcm, category_index, category_name, dst_dir, i)
            counter += 1
            print(f'Saving PNGs: {100*counter//len(buckets)} %')

    mapp.to_csv(os.path.join(dst_dir, 'mapping.csv'))
    print('Done!')


if __name__ == '__main__':
    main('/home/ubuntu/sdsHD/sd18a006/DataBaseMammography/CBIS-DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/CBIS-DDSM', '/home/ubuntu/gmic/cbis_dssm_data')
