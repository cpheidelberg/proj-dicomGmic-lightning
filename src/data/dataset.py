import numpy as np
import pandas as pd
import os, ast, h5py, time, sys, random, dataclasses
from tqdm import tqdm
import multiprocessing
import albumentations, albumentations.pytorch

import torch
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)
from src.data import loading


@dataclasses.dataclass(frozen=True)
class ClassificationImage:
    path: str
    view: str
    category: str
    center: tuple[int, int]

    @staticmethod
    def load(image_dir: str, line: pd.Series):
        path = f"{os.path.join(image_dir, line['image'])}.png"
        view = line['view']
        category = line['finding_categories'][0] if len(line['finding_categories']) == 1 else path
        center = line['best_center'][view][0]
        return ClassificationImage(path, view, category, center)

    def image(self):
        img = loading.read_image(self.path)
        return loading.process_image(img, self.view, 'NO', self.center)

    def encoding(self, categories: list[str]):
        enc = np.zeros(len(categories), dtype=np.float32)
        enc[categories.index(self.category)] = 1.0
        return enc


class ClassificationImages(Dataset):
    def __init__(self, image_dir: str, dict_path: str, top_c: int):
        tab = pd.read_csv(dict_path, converters={'best_center': ast.literal_eval, 'finding_categories': ast.literal_eval})

        images = [ClassificationImage.load(image_dir, line) for line in tab.iloc]

        category_set = {image.category for image in images}
        category_dict = {category: sum(image.category == category for image in images) for category in category_set}

        self.category_names = sorted(category_set, key=lambda name: -category_dict[name])[:top_c]
        self.category_sizes = [category_dict[name] for name in self.category_names]

        self.images = [image for image in images if image.category in self.category_names]
        self.class_weights = [len(self.images) / (len(self.category_sizes) * size) for size in self.category_sizes]

        
        self.transform = albumentations.Compose([
            albumentations.RandomResizedCrop(size=(2944, 1920), p=0.3),
            albumentations.RandomBrightnessContrast(p=0.3),
            albumentations.pytorch.ToTensorV2()
        ])

    def undersample(self):
        smallest = min(self.category_sizes)
        images = []
        for category in self.category_names:
            images += random.sample([image for image in self.images if image.category == category], smallest)

        self.images = images
        self.class_weights = [1.0] * len(self.class_weights)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i: int):
        x = self.transform(image=self.images[i].image())['image']
        y = self.images[i].encoding(self.category_names)
        return x, y


def convert_vindr_mammo_dataset_to_our_storage_format():
    parent_path = '/home/student/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
    image_dir = os.path.join(parent_path, 'cropped_images')
    dict_path = os.path.join(parent_path, 'dictionary.csv')
    top_c = 6

    dest_path = '/home/student/gmic/vindr-mammo'
    labels_path = os.path.join(dest_path, 'labels.txt')

    tab = pd.read_csv(dict_path, converters={'best_center': ast.literal_eval, 'finding_categories': ast.literal_eval})
    
    # Delete images which don't have exactly one category
    for i in range(len(tab)):
        if len(tab.loc[i, 'finding_categories']) != 1:
            tab.drop(i, inplace=True)
        else:
            tab.loc[i, 'finding_categories'] = tab.loc[i, 'finding_categories'][0]

    tab = tab.reset_index()
    categories = list(tab['finding_categories'].value_counts()[:top_c].keys())

    for index, category in enumerate(categories):
        os.makedirs(os.path.join(dest_path, str(index)), exist_ok=True)

        for line in tab[tab['finding_categories'] == category].iloc:
            image = loading.read_image(
                path=f"{os.path.join(image_dir, line['image'])}.png",
                dtype=None
            )
            image = loading.flip_and_crop(
                image=image,
                view=line['view'],
                horizontal_flip=line['horizontal_flip'],
                best_center=line['best_center'][line['view']][0]
            )
            loading.write_image(
                path=f"{os.path.join(dest_path, str(index), line['image'])}.png",
                image=image
            )

    with open(labels_path, 'w') as labels_file:
        labels_file.write('\n'.join(categories))


if __name__ == "__main__":
    convert_vindr_mammo_dataset_to_our_storage_format()
