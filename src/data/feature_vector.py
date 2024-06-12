import torch.utils.data
import numpy as np
import scipy.spatial


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


def _to_vector(global_vec: np.ndarray, h_crops: np.ndarray):
    return np.concatenate((global_vec.flatten(), h_crops.flatten()), dtype=np.float32)


def _from_vector(vector: np.ndarray):
    global_vec = vector[:256].reshape((256,))
    h_crops = vector[256:].reshape((-1, 512))
    return global_vec, h_crops


class Storage(torch.utils.data.Dataset):
    def __init__(self, num_classes: int):
        self._num_classes = num_classes
        self._original = [[] for _ in range(num_classes)]
        self._synthetic = [[] for _ in range(num_classes)]


    def synthesise(self):
        self._original = np.array(self._original, dtype=np.float32, copy=False)
        self._synthetic[0] = self._original[0]

        for label in range(1, self._num_classes):
            ratio = len(self._original[0]) // len(self._original[label])
            self._synthetic[label] = list(SMOTE(self._original[label]).generate(n=min(20, ratio), k=5))


    def class_weights(self):
        sizes = [len(vectors) for vectors in self._synthetic]
        avg = sum(sizes) / len(sizes)
        return [avg / size for size in sizes]


    def add(self, labels, global_vec, h_crops):
        for label, gv, hc in zip(labels, global_vec, h_crops):
            self._original[label].append(_to_vector(gv, hc))


    def __len__(self):
        return sum(len(vectors) for vectors in self._synthetic)


    def __getitem__(self, i: int):
        label = 0
        while i >= len(self._synthetic[label]):
            i -= len(self._synthetic[label])
            label += 1

        x = _from_vector(self._synthetic[label][i])
        y = np.zeros(self._num_classes, dtype=np.float32)
        y[label] = 1
        return x, y
