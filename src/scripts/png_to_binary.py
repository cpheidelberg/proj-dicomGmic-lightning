import numpy as np
import pandas as pd
import torch
import os
import sys
import ast

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = '/'.join(current_dir.split('/')[:-2])
sys.path.append(parent_dir)

from src.data import loading


def _format_image(tab: pd.DataFrame, image_dir: str, i: int):
    path = f"{os.path.join(image_dir, tab.loc[i, 'image'])}.png"
    view = tab.loc[i, 'view']
    category = tab.loc[i, 'finding_categories'][0]
    center = tab.loc[i, 'best_center'][view][0]
    return {'path': path, 'view': view, 'category': category, 'center': center}

def _format_images(tab: pd.DataFrame, image_dir: str):
    return [_format_image(tab, image_dir, i) for i in range(len(tab)) if len(tab.loc[i, 'finding_categories']) == 1]

def _read_csv_info(image_dir: str, dict_path: str, top_c: int):
    tab = pd.read_csv(dict_path, converters={'best_center': ast.literal_eval, 'finding_categories': ast.literal_eval})

    images = _format_images(tab, image_dir)

    category_set = {image['category'] for image in images}
    category_dict = {category: sum(image['category'] == category for image in images) for category in category_set}
    category_names = np.array(sorted(category_set, key=lambda name: -category_dict[name])[:top_c], dtype='object')
    category_sizes = np.array([category_dict[name] for name in category_names], dtype=np.int32)

    images.sort(key=lambda image: -category_dict[image['category']])
    images = images[:sum(category_sizes)]

    return category_names, category_sizes, images

def _encode_image(categories: np.ndarray[str], category: str):
    enc = np.zeros(len(categories), dtype=np.float32)
    enc[np.argmax(categories == category)] = 1.0
    return enc

def _encode_images(categories: list[str], images: list[dict]):
    return np.array([_encode_image(categories, img['category']) for img in images], dtype=np.float32)

def _load_image(data: dict):
    array = loading.load_image(data['path'], data['view'], horizontal_flip='NO')
    array = loading.process_image(array, data['view'], data['center'])
    array = np.expand_dims(array, 0).copy()
    return torch.Tensor(array)


def main():
    image_dir = '/home/ubuntu/data/output/cropped_images'
    dict_path = '/home/ubuntu/data/output/dictionary.csv'

    category_names, category_sizes, images = _read_csv_info(image_dir, dict_path, top_c=6)
    labels = _encode_images(category_names, images)

    os.makedirs('/home/ubuntu/data_2/input/images', exist_ok=True)
    np.save('/home/ubuntu/data_2/input/category_names.np', category_names)
    np.save('/home/ubuntu/data_2/input/category_sizes.np', category_sizes)
    np.save('/home/ubuntu/data_2/input/labels.np', labels)

    for i in range(len(images)):
        image = _load_image(images[i])
        torch.save(image, f'/home/ubuntu/data_2/input/images/{i}.torch')
        print(f'{round(i / len(images) * 100)}% \t{i + 1}/{len(images)}')

if __name__ == '__main__':
    main()
