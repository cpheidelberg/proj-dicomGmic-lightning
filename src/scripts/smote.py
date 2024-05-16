import numpy as np
import scipy.spatial


class NearestNeighbours:
    def __init__(self, points: np.ndarray):
        self._points = points
        self._tree = scipy.spatial.KDTree(points)
    
    def find(self, point, k) -> np.ndarray:
        _, indices = self._tree.query(point, k=k+1)
        return self._points[indices[1:]]


_rng = np.random.default_rng()

def _sample_with_replacement(items: np.ndarray, size: int):
    return items[_rng.integers(0, len(items), size)]

def _random_linear_combination(u: np.ndarray, v: np.ndarray):
    return u + (v - u) * _rng.random(len(v))


def smote(points: np.ndarray, mul: int, k: int) -> np.ndarray:
    point_count, attr_count = points.shape

    synthetic = np.zeros([point_count * mul, attr_count])
    nn = NearestNeighbours(points)

    i = 0
    for pt in points:
        selected = _sample_with_replacement(nn.find(pt, k), mul)
        for nb in selected:
            synthetic[i] = _random_linear_combination(pt, nb)
            i += 1

    return synthetic
