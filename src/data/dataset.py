import numpy as np
import pandas as pd
import os, ast, sys, random
import albumentations as alb, albumentations.pytorch as alp
from torch.utils.data import Dataset
import scipy.spatial

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split('/')[:-2])
sys.path.append(parent_dir)
from src.data import loading


def _geometric_mean(a: int, b: int, ratio: float) -> int:
    return round(a ** ratio * b ** (1 - ratio))


_aug = alb.Compose([alb.RandomResizedCrop((2944, 1920), p=0.3), alb.RandomBrightnessContrast(p=0.3), alp.ToTensorV2()])


class ClassificationImages(Dataset):
    def __init__(self, data_dir: str, undersampling_rate: float, augmentation_rate: float):
        with open(os.path.join(data_dir, 'labels.txt')) as labels_file:
            self.labels = [label.strip() for label in labels_file]

        self.images = [[entry.path for entry in os.scandir(os.path.join(data_dir, f'{i}'))] for i in range(len(self.labels))]

        c_max = _geometric_mean(len(self.images[1]), len(self.images[0]), undersampling_rate)
        self.images[0] = random.sample(self.images[0], c_max)

        self.sizes = [_geometric_mean(c_max, len(images), augmentation_rate) for images in self.images]
        self.offsets = [sum(self.sizes[:i]) for i in range(len(self.labels) + 1)]


    def class_weights(self):
        avg = self.offsets[-1] / len(self.sizes)
        return [avg / size for size in self.sizes]


    def __len__(self):
        return self.offsets[-1]


    def __getitem__(self, index: int):
        label = next(i for i in range(len(self.labels)) if self.offsets[i] <= index < self.offsets[i + 1])
        position = (index - self.offsets[label]) * len(self.images[label]) // self.sizes[label]

        x = loading.read_image_standardized(self.images[label][position])
        x = _aug(image=x)['image']

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


    def _to_vector(global_vec: np.ndarray, h_crops: np.ndarray):
        return np.concatenate((global_vec.flatten(), h_crops.flatten()), dtype=np.float32)


    def _from_vector(vector: np.ndarray):
        global_vec = vector[:256].reshape((256,))
        h_crops = vector[256:].reshape((-1, 512))
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



def convert_vindr_mammo_dataset_to_our_storage_format():
    parent_path = '/home/student/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
    image_dir = os.path.join(parent_path, 'cropped_images')
    dict_path = os.path.join(parent_path, 'dictionary.csv')
    top_c = 6

    tab = pd.read_csv(dict_path, converters={'best_center': ast.literal_eval, 'finding_categories': ast.literal_eval})
    
    # Delete images which don't have exactly one category
    for i in range(len(tab)):
        if len(tab.loc[i, 'finding_categories']) != 1:
            tab.drop(i, inplace=True)
        else:
            tab.loc[i, 'finding_categories'] = tab.loc[i, 'finding_categories'][0]

    tab.reset_index(inplace=True)
    categories = list(tab['finding_categories'].value_counts()[:top_c].keys())

    dest_path = '/home/student/gmic/vindrmammo_data'
    for index, category in enumerate(categories):
        os.makedirs(os.path.join(dest_path, str(index)), exist_ok=True)

        for line in tab[tab['finding_categories'] == category].iloc:
            image = loading.read_image(f"{os.path.join(image_dir, line['image'])}.png", dtype=None)
            image = loading.flip_and_crop(image, line['view'], line['horizontal_flip'], line['best_center'][line['view']][0])
            loading.write_image(f"{os.path.join(dest_path, str(index), line['image'])}.png", image)

    with open(os.path.join(dest_path, 'labels.txt'), 'w') as labels_file:
        labels_file.write('\n'.join(categories))


if __name__ == "__main__":
    convert_vindr_mammo_dataset_to_our_storage_format()
