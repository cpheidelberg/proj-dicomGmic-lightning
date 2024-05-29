import numpy as np
import pandas as pd
import h5py as h5
import os
import sys
import ast

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.data import loading


def _format_image(tab: pd.DataFrame, image_dir: str, i: int):
    path = f"{os.path.join(image_dir, tab.loc[i, 'image'])}.png"
    view = tab.loc[i, 'view']
    category = tab.loc[i, 'finding_categories'][0]
    center = tab.loc[i, 'best_center'][view][0]
    return {'path': path, 'view': view, 'category': category, 'center': center}

def _format_images(tab: pd.DataFrame, image_dir: str):
    return [_format_image(tab, image_dir, i) for i in range(len(tab))]

def _read_csv_info(image_dir: str, dict_path: str):
    tab = pd.read_csv(dict_path, converters={'best_center': ast.literal_eval, 'finding_categories': ast.literal_eval})

    images = _format_images(tab, image_dir)
    images.sort(lambda image: image['category'])

    category_names = sorted({image['category'] for image in images})
    category_sizes = [sum(image['category'] == category for image in images) for category in category_names]

    return category_names, category_sizes, images

def _encode_image(categories: list[str], category: str):
    enc = np.zeros(len(categories), dtype=np.float32)
    enc[categories.index(category)] = 1.0
    return enc

def _encode_images(categories: list[str], images: list[dict]):
    return [_encode_image(categories, img['category']) for img in images]

def _load_image(data: dict):
    array = loading.load_image(data['path'], data['view'], horizontal_flip='NO')
    array = loading.process_image(array, data['view'], data['center'])
    array = np.expand_dims(array, 0).copy()
    return array


def main():
    image_dir = '/home/ubuntu/data/output/cropped_images'
    dict_path = '/home/ubuntu/code/medken/removedTop5.csv'
    result_path = '/home/ubuntu/data_2/input.h5'

    categories, sizes, images = _read_csv_info(image_dir, dict_path)
    enc = _encode_images(categories, images)

    output = h5.File(result_path, mode='w', libver='latest')
    output.swmr_mode = True

    output.create_dataset('category_size', data=np.array(sizes, dtype=np.int32))
    output.create_dataset('category_name', data=np.array(categories, dtype='object'))
    output.create_dataset('image_encoding', data=np.array(enc, dtype=np.float32))

    first_image = _load_image(images[0])
    image_ds = output.create_dataset('image_data', (len(images), *first_image.shape), dtype=first_image.dtype)
    for i in range(len(images)):
        image_ds[i] = _load_image(images[i])
        print(f'{round(i / len(images) * 100)}% \t{i + 1}/{len(images)}')

    output.close()

if __name__ == '__main__':
    main()
