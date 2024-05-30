import numpy as np
import pandas as pd
import os, ast, pickle, h5py, time, sys, random
from tqdm import tqdm
import multiprocessing

import torch
from torch.utils.data import Dataset, DataLoader

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
    return [_format_image(tab, image_dir, i) for i in range(len(tab)) if len(tab.loc[i, 'finding_categories']) == 1]

def _read_csv_info(image_dir: str, dict_path: str, top_c: int):
    tab = pd.read_csv(dict_path, converters={'best_center': ast.literal_eval, 'finding_categories': ast.literal_eval})

    images = _format_images(tab, image_dir)

    category_set = {image['category'] for image in images}
    category_dict = {category: sum(image['category'] == category for image in images) for category in category_set}
    category_names = sorted(category_set, key=lambda name: -category_dict[name])[:top_c]
    category_sizes = [category_dict[name] for name in category_names]

    images = filter((lambda image: image['category'] in category_names), images)
    images = sorted(images, key=lambda image: -category_dict[image['category']])

    return category_names, category_sizes, images

def _load_image(data: dict):
    array = loading.load_image(data['path'], data['view'], horizontal_flip='NO')
    array = loading.process_image(array, data['view'], data['center'])
    array = np.expand_dims(array, 0).copy()
    return torch.Tensor(array)

def _encode_image(categories: list[str], category: str):
    enc = np.zeros(len(categories), dtype=np.float32)
    enc[categories.index(category)] = 1.0
    return enc


class ClassificationImages(Dataset):
    def __init__(self, image_dir: str, dict_path: str, top_c: int):
        self.category_names, self.category_sizes, self.images = _read_csv_info(image_dir, dict_path, top_c)

    def __len__(self):
        return len(self.category_names) * self.category_sizes[0]

    def __getitem__(self, index: int):
        return self._nth_image(self._convert_index(index))

    def _convert_index(self, index: int):
        category, scaled = divmod(index, self.category_sizes[0])
        scaled = scaled * self.category_sizes[category] // self.category_sizes[0]
        return sum(self.category_sizes[:category]) + scaled

    def _nth_image(self, index: int):
        image = _load_image(self.images[index])
        label = _encode_image(self.category_names, self.images[index]['category'])
        return image, label

    def num_classes(self):
        return len(self.category_names)


def store_as_hdf5(src: ClassificationImages, dst: str):
    with h5py.File(dst, mode='w', libver='latest') as f:
        f.swmr_mode = True
        f.create_dataset('category_names', data=np.array(src.category_names))
        f.create_dataset('category_sizes', data=np.array(src.category_sizes))

        image0, label0 = src[0]
        image0 = image0.numpy()
        images = f.create_dataset('images', dtype=image0.dtype, shape=(len(src), *image0.shape), chunks=(1, *image0.shape))
        labels = f.create_dataset('labels', dtype=label0.dtype, shape=(len(src), *label0.shape), chunks=(1, *label0.shape))

        for i in range(len(src)):
            image, label = src[i]
            images[i] = image.numpy()
            labels[i] = label
            print(f'{i}/{len(src)}')
        
        print('Done!')

class ClassificationImagesHDF5(Dataset):
    def __init__(self, data_path: str):
        file = h5py.File(data_path, libver='latest', swmr=True)
        self.images = file['images']
        self.labels = list(file['labels'])
        self.category_names = list(file['category_names'])
        self.category_sizes = list(file['category_sizes'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return torch.Tensor(self.images[index]), self.labels[index]

    def num_classes(self):
        return len(self.category_names)


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


def main(): 

    image_path_train = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_cropped_top5/'
    image_path_test = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images/'
    label_file = "sample_data/annotations/finding_annotations.csv"
    data_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/data.pkl'

    h5Path = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_top6/dataset.h5"

    # data = ClassificationImages(imageFolder=[image_path_train, image_path_test], top_c=6)
    # data = ClassificationFromLabels(imageFolder=[image_path_train, image_path_test], dictPath=data_path, labelPath=label_file, top_c=3)
    data = H5Dataset(h5_filepath=h5Path, relevant_labels=["No Finding", "Mass", "Suspicious Calcification"])
    # create_chunked_h5(data)

    print(data[0])


if __name__ == "__main__":
    main()