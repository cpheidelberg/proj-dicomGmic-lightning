import numpy as np
import scipy.spatial


class SMOTE:
    def __init__(self, points: np.ndarray):
        self._shape = points.shape[1:]
        self._points = points.reshape((points.shape[0], np.prod(self._shape)))
        self._tree = scipy.spatial.KDTree(self._points)
        self._rng = np.random.default_rng()

    def _nearest_neighbours(self, point: np.ndarray, k: int) -> np.ndarray:
        _, indices = self._tree.query(point, k=k+1)
        return self._points[indices[1:]]

    def _sample_with_replacement(self, items: np.ndarray, size: int):
        return items[self._rng.integers(0, len(items), size)]

    def _random_linear_combination(self, u: np.ndarray, v: np.ndarray):
        return u + (v - u) * self._rng.random(len(v))

    def generate(self, n: int, k: int):
        point_num, attr_num = self._points.shape
        result = np.zeros([point_num * n, attr_num])

        i = 0
        for point in self._points:
            nearest = self._nearest_neighbours(point, k)
            selected = self._sample_with_replacement(nearest, n)
            for neighbour in selected:
                result[i] = self._random_linear_combination(point, neighbour).reshape(self._shape)
                i += 1

        return result
