import pickle
import sqlite3 as sql
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


def _to_bytes(array: np.ndarray) -> bytes:
    return pickle.dumps(array.tolist())

def _to_array(array: bytes) -> np.ndarray:
    return np.array(pickle.loads(array), dtype=np.float32)


class Storage(torch.utils.data.Dataset):
    def __init__(self, path: str, category_sizes: list[int]):
        self._category_sizes = category_sizes

        self._path = path
        with sql.connect(self._path) as con:
            con.execute('DROP TABLE IF EXISTS original')
            con.execute('DROP TABLE IF EXISTS synthetic')

            con.execute('CREATE TABLE original  (label, vector)')
            con.execute('CREATE TABLE synthetic (label, vector)')

    def _get_original_vectors(self, cur: sql.Cursor, label: int):
        vectors = []
        for vector, in cur.execute('SELECT vector FROM original WHERE label = ?', (label,)):
            vectors.append(_to_array(vector))
        return np.array(vectors)

    def _synthesise_vectors(self, cur: sql.Cursor, label: int, n: int, k: int):
        vectors = self._get_original_vectors(cur, label)
        if len(vectors) > k + 1:
            for vec in SMOTE(vectors).generate(n, k):
                cur.execute('INSERT INTO synthetic VALUES (?, ?)', (label, _to_bytes(vec)))

    def _marshall(self, global_vec: np.ndarray, h_crops: np.ndarray):
        return _to_bytes(np.concatenate((global_vec.flatten(), h_crops.flatten())))

    def _unmarshall(self, vector: bytes):
        vector = _to_array(vector)
        global_vec = vector[:256].reshape((256,))
        h_crops = vector[256:].reshape((6, 512))
        return global_vec, h_crops

    def reset(self):
        with sql.connect(self._path) as con:
            con.execute('DELETE FROM synthetic')

            largest = max(self._category_sizes)
            for category, size in enumerate(self._category_sizes):
                self._synthesise_vectors(con.cursor(), category, largest // size, k=5)

            con.execute('DELETE FROM original')

    def add(self, labels, global_vec, h_crops):
        with sql.connect(self._path) as con:
            cur = con.cursor()
            for label, gv, hc in zip(labels, global_vec, h_crops):
                cur.execute('INSERT INTO original (label, vector) VALUES (?, ?)', (label, self._marshall(gv, hc)))

    def __len__(self):
        with sql.connect(self._path) as con:
            return con.execute('SELECT COUNT(*) FROM synthetic').fetchone()[0]

    def __getitem__(self, index):
        with sql.connect(self._path) as con:
            label, vector = con.execute('SELECT label, vector FROM synthetic ORDER BY rowid LIMIT 1 OFFSET ?', (index,)).fetchone()

            x = self._unmarshall(vector)
            y = np.zeros(len(self._category_sizes), dtype='float32')
            y[label] = 1
            return x, y
