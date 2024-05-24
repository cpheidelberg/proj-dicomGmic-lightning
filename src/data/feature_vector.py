import pickle
import sqlite3
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


class Storage(torch.utils.data.Dataset):
    def __init__(self, path: str, no_finding: int, class_num: int):
        self._no_finding = no_finding
        self._class_num = class_num
        self._global_vec_shape = (1, )
        self._h_crops_shape = (1, )

        self._con = sqlite3.connect(path)
        self._cur = self._con.cursor()
        self._cur.execute('DROP TABLE IF EXISTS original')
        self._cur.execute('DROP TABLE IF EXISTS synthetic')

        self._cur.execute('CREATE TABLE original  (label, vector)')
        self._cur.execute('CREATE TABLE synthetic (label, vector)')
        self._con.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._cur.close()
        self._con.close()

    def _get_original_vectors(self, label: int):
        vectors = []
        for vector, in self._cur.execute('SELECT vector FROM original WHERE label = ?', (label,)):
            vectors.append(pickle.loads(vector))
        return np.array(vectors)

    def _synthesise_vectors(self, label: int, n: int, k: int):
        vectors = self._get_original_vectors(label)
        if len(vectors) > k + 1:
            for vec in SMOTE(vectors).generate(n, k):
                self._cur.execute('INSERT INTO synthetic VALUES (?, ?)', (label, pickle.dumps(vec)))

    def _marshall(self, global_vec: np.ndarray, h_crops: np.ndarray):
        self._global_vec_shape = global_vec.shape
        self._h_crops_shape = h_crops.shape
        print(f'{global_vec.dtype=}; {h_crops.dtype=}')
        return pickle.dumps(np.concatenate((global_vec.flatten(), h_crops.flatten())))

    def _unmarshall(self, vector: bytes):
        vector = pickle.loads(vector)
        split_index = np.prod(self._global_vec_shape)
        global_vec = vector[:split_index].reshape(self._global_vec_shape)
        h_crops = vector[split_index:].reshape(self._h_crops_shape)
        return global_vec, h_crops

    def reset(self):
        self._cur.execute('DELETE FROM synthetic')

        for label in range(self._class_num):
            if label != self._no_finding:
                self._synthesise_vectors(label, n=2, k=5)

        self._cur.execute('DELETE FROM original')
        self._con.commit()

    def add(self, labels, global_vec, h_crops):
        for label, gv, hc in zip(labels, global_vec, h_crops):
            if label != self._no_finding:
                self._cur.execute('INSERT INTO original (label, vector) VALUES (?, ?)', (label, self._marshall(gv, hc)))
        self._con.commit()

    def __len__(self):
        return self._cur.execute('SELECT COUNT(*) FROM synthetic').fetchone()[0]

    def __getitem__(self, index):
        label, vector = self._cur.execute('SELECT label, vector FROM synthetic ORDER BY rowid LIMIT 1 OFFSET ?', (index,)).fetchone()

        x = self._unmarshall(vector)
        y = np.zeros(self._class_num)
        y[label] = 1
        return x, y
