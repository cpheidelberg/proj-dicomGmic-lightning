import os, h5py, time, sys, random
import multiprocessing
import numpy as np
import pandas as pd
import torch
import scipy.spatial

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = '/'.join(current_dir.split('/')[:-2])
sys.path.append(parent_dir)

from src.data_loading import loading, augmentations


def _geometric_mean(a: int, b: int, ratio: float) -> int:
    return round(a ** ratio * b ** (1 - ratio))


class ClassificationImages(Dataset):
    def __init__(self, data_dirs: list[str], undersampling_rate: float, augmentation_rate: float, binary: bool, augment: bool):
        """
        Create a Dataset of images stored in data directories. Apply
        undersampling and/or augmentation if requested.
        - data_dirs: Directories containing the classification images as PNGs.
            Each directory contains a file mapping.csv which lists all images
            in the directory. The CSV file has three columns: png (file path
            of the PNG, relative to the directory), dicom (the original DICOM)
            and label (the string representing the class).
        - undersampling_rate: How much to undersample the largest class.
            Its value can be from 0.0 up to 1.0. The largest class will be
            undersampled to the size of A' = (A ** (1 - U)) * (B ** U), where
            A is the size of the largest class, B is the size of the second
            largest class and U is the undersampling rate.
        - augmentation_rate: How much to oversample the smaller classes by augmentation.
            If augmentation_rate > 0.0, all classes except for the largest one
            will be resized to C' = (C ** (1 - G)) * (A' ** G), where C is
            the original size of the given class, A' is the size of the largest
            class after undersampling and G is the augmentation rate.
            Dataset will return the same image multiple times with different
            random augmentations to simulate a larger class.
        - binary: Whether to merge all suspicious classes into one.
            If binary=True, all classes which are not "No Finding" are merged
            together to one class called "Suspicious"
        - augment: Whether to enable augmentation.
            If augment=False, augmentation is disabled. In such a case, setting
            augmentation_rate other than 0.0 has no effect other than making
            the program slower. If augment=True, images are randomly augmented
            each time they are loaded.
        """

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

            if len(self.images[0]) < len(self.images[1]):
                self.images.reverse()
                self.labels.reverse()
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
        x = augmentations.augment_image(x, augment=self.augment)

        y = np.zeros(len(self.images), dtype=np.float32)
        y[label] = 1.0
        return x, y



class SMOTE:
    def __init__(self, points: list[np.ndarray]):
        self._points = points
        self._tree = scipy.spatial.KDTree(points)
        self._rng = np.random.default_rng()

    def _nearest_neighbours(self, point: np.ndarray, k: int) -> np.ndarray:
        _, indices = self._tree.query(point, k=k+1)
        return self._points[indices[1:]]

    def _sample_with_replacement(self, items: np.ndarray, size: int):
        return items[self._rng.integers(0, len(items), size)]

    def _random_linear_combination(self, u: np.ndarray, v: np.ndarray):
        return u + (v - u) * self._rng.random(len(v))

    def generate(self, n: int, k: int):
        for point in self._points:
            nearest = self._nearest_neighbours(point, k)
            selected = self._sample_with_replacement(nearest, n)
            for neighbour in selected:
                yield self._random_linear_combination(point, neighbour)



class FeatureVectors(Dataset):
    def __init__(self, smote_rate: float):
        self._original = []
        self._synthetic = []
        self._smote_rate = smote_rate
        self._smote = None


    def synthesise(self):
        if self._smote is None:
            self._smote = [SMOTE(np.array(original, dtype=np.float32)) for original in self._original]

        self._synthetic.clear()
        largest = max(len(original) for original in self._original)

        for original, smote in zip(self._original, self._smote):
            target = _geometric_mean(largest, len(original), self._smote_rate)

            if len(original) >= target:
                self._synthetic.append(original)
            else:
                self._synthetic.append(list(smote.generate(target // len(original), k=5)))


    def class_weights(self):
        sizes = [len(vectors) for vectors in self._synthetic]
        avg = sum(sizes) / len(sizes)
        return [avg / size for size in sizes]


    def _to_vector(self, global_vec: np.ndarray, h_crops: np.ndarray):
        return np.concatenate((global_vec.flatten(), h_crops.flatten()), dtype=np.float32)


    def _from_vector(self, vector: np.ndarray):
        global_vec = vector[:256].reshape((256,)).astype(np.float32, copy=False)
        h_crops = vector[256:].reshape((-1, 512)).astype(np.float32, copy=False)
        return global_vec, h_crops


    def add(self, labels, global_vec, h_crops):
        for label, gv, hc in zip(labels, global_vec, h_crops):
            if label >= len(self._original):
                self._original += [[] for _ in range(label - len(self._original) + 1)]

            self._original[label].append(self._to_vector(gv, hc))


    def __len__(self):
        return sum(len(vectors) for vectors in self._synthetic)


    def __getitem__(self, i: int):
        label = 0
        while i >= len(self._synthetic[label]):
            i -= len(self._synthetic[label])
            label += 1

        x = self._from_vector(self._synthetic[label][i])
        y = np.zeros(len(self._synthetic), dtype=np.float32)
        y[label] = 1
        return x, y



class H5Dataset(Dataset):
    def __init__(self, h5_filepath, relevant_labels=None):
        self.valid_indices = []
        self.h5_filepath = h5_filepath
        self.relevant_labels = relevant_labels

        self.h5_file = h5py.File(self.h5_filepath, 'r')
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

    print('Data has been successfully saved to balanced_top6/dataset.h5')
    print('Gesamtzeit:', time.time() - start_time)


if __name__ == '__main__':
    h5Path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_top6/dataset.h5'
    data = H5Dataset(h5Path, ['No Finding', 'Mass', 'Suspicious Calcification'])
    # create_chunked_h5(data)

    print(data[0])
