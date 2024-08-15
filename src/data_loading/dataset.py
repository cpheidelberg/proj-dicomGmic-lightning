import os, h5py, time, sys, random
import multiprocessing
import numpy as np
import pandas as pd
import torch

import albumentations as alb, albumentations.pytorch as alp

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.data_loading import loading


def _geometric_mean(a: int, b: int, ratio: float) -> int:
    return round(a ** ratio * b ** (1 - ratio))


_aug = alb.Compose([alb.RandomResizedCrop((2944, 1920), p=0.3), alb.RandomBrightnessContrast(p=0.3), alp.ToTensorV2()])


class ClassificationImages(Dataset):
    def __init__(self, data_dirs: list[str], undersampling_rate: float, augmentation_rate: float, binary: bool, augment: bool):
        tables = []
        for data_dir in data_dirs:
            data_dir = data_dir.removesuffix('/')

            mapping = pd.read_csv(f'{data_dir}/mapping.csv')
            mapping['png'] = f'{data_dir}/' + mapping['png']
            tables.append(mapping)

        table = pd.concat(tables)
        labels = list(table['label'].value_counts().keys())

        if binary:
            self.images = [list(table[table['label'] == 'No Finding']['png']), list(table[table['label'] != 'No Finding']['png'])]
            self.labels = ['No Finding', 'Suspicious']
        else:
            self.images = [list(table[table['label'] == label]['png']) for label in labels]
            self.labels = labels

        c_max = _geometric_mean(len(self.images[1]), len(self.images[0]), undersampling_rate)
        self.images[0] = random.sample(self.images[0], c_max)

        self.sizes = [_geometric_mean(c_max, len(images), augmentation_rate) for images in self.images]
        self.offsets = [sum(self.sizes[:i]) for i in range(len(self.images) + 1)]

        self.augment = augment


    def class_weights(self):
        avg = self.offsets[-1] / len(self.sizes)
        return [avg / size for size in self.sizes]


    def __len__(self):
        return self.offsets[-1]


    def __getitem__(self, index: int):
        label = next(i for i in range(len(self.images)) if self.offsets[i] <= index < self.offsets[i + 1])
        position = (index - self.offsets[label]) * len(self.images[label]) // self.sizes[label]

        x = loading.read_image_standardized(self.images[label][position])
        if self.augment:
            x = _aug(image=x)['image']
        else:
            x = np.expand_dims(x, 0)

        y = np.zeros(len(self.images), dtype=np.float32)
        y[label] = 1.0
        return x, y


class H5Dataset(Dataset):
    def __init__(self, h5_filepath, relevant_labels=None):
        self.valid_indices = []
        self.h5_filepath = h5_filepath
        self.relevant_labels = relevant_labels

        self.h5_file = h5py.File(self.h5_filepath, "r")
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']

        if relevant_labels:
            self.encoding = {'No Finding': 0, 'Mass': 1, 'Asymmetry': 2, 'Focal Asymmetry': 3, 'Suspicious Calcification': 4, 'Architectural Distortion': 5}
            self.relevant_label_indices = [self.encoding[label] for label in relevant_labels]
            self.filter_data()

        print(self.getLabelCount())

    def getLabelCount(self):
        labelCount = {k: 0 for k in range(len(self.encoding))}
        for l in self.labels:
            label_index = l.argmax(axis=0)
            if label_index in self.relevant_label_indices:
                labelCount[label_index] += 1
        return labelCount

    def filter_data(self):
        for i, label in enumerate(self.labels):
            label_index = label.argmax(axis=0)
            if label_index in self.relevant_label_indices:
                self.valid_indices.append(i)

    def __len__(self):
        if self.relevant_labels:
            return len(self.valid_indices)
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        if self.relevant_labels:
            idx = self.valid_indices[idx]
        image = torch.from_numpy(self.images[idx].astype('float32'))
        label = torch.from_numpy(self.labels[idx].astype('float32'))

        if self.relevant_labels:
            label = label[self.relevant_label_indices]
        return image, label

    def close(self):
        self.h5_file.close()


def create_chunked_h5(data):
    num_workers = multiprocessing.cpu_count() - 2
    dataloader = DataLoader(data, shuffle=True, num_workers=num_workers)

    num_images = len(data) // 4
    image_shape = (1, 2944, 1920) 
    label_shape = (6,)

    image_chunk_size = (1, 1, 2944, 1920)
    label_chunk_size = (1, 6)

    start_time = time.time()
    with h5py.File('balanced_top6/dataset.h5', 'w') as f:
        # Create datasets for images and labels with appropriate chunk sizes
        dset_images = f.create_dataset('images', shape=(num_images,) + image_shape, dtype='float32', chunks=image_chunk_size)
        dset_labels = f.create_dataset('labels', shape=(num_images,) + label_shape, dtype='float32', chunks=label_chunk_size)

        for i, (image, label) in enumerate(tqdm(dataloader, total=num_images)):
            if i == num_images:
                break
            image_np = image.numpy()
            label_np = label.numpy()

            # Store in HDF5 dataset
            dset_images[i] = image_np
            dset_labels[i] = label_np

    print("Data has been successfully saved to 'balanced_top6/dataset.h5'")

    print(f"Gesamtzeit: {time.time() - start_time}")



if __name__ == "__main__":
    h5Path = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_top6/dataset.h5"
    data = H5Dataset(h5_filepath=h5Path, relevant_labels=["No Finding", "Mass", "Suspicious Calcification"])
    # create_chunked_h5(data)

    print(data[0])
