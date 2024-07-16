import pydicom
import pandas as pd
import os, sys

import pydicom.errors

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

import src.cropping.crop_mammogram as cropping
import src.optimal_centers.get_optimal_centers as centers
import src.data.loading as loading


class _dicom:
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


    def __init__(self, path: str):
        with pydicom.read_file(path) as file:
            self.path = path
            self.image = file.pixel_array

            image_laterality = 'R' if 'RIGHT' in path else 'L'
            view_position = 'MLO' if 'MLO' in path else 'CC'
            self.view = f'{image_laterality}-{view_position}'

            self.horizontal_flip = 'NO'
            self.category = 'Suspicious Calcification' if 'Calc' in path else 'Mass'
            self.center = self.get_center()


def _get_paths(root: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.dcm'):
                paths.append(os.path.join(dirpath, filename))
    return paths


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
    dicoms = [_dicom(path) for path in _get_paths(src_dir)]

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
