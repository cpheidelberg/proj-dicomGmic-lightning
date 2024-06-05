import io
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


def _np_dumps(array: np.ndarray) -> bytes:
    with io.BytesIO() as buffer:
        np.save(buffer, array)
        return buffer.getvalue()


def _np_loads(array: bytes) -> np.ndarray:
    return np.load(io.BytesIO(array))


class Storage(torch.utils.data.Dataset):
    def __init__(self, path: str, num_classes: int):
        self._num_classes = num_classes
        self._path = path

        with sql.connect(self._path) as con:
            con.execute('CREATE TABLE IF NOT EXISTS original  (label, vector)')
            con.execute('CREATE TABLE IF NOT EXISTS synthetic (label, vector)')


    def _select_original(self, cur: sql.Cursor, label: int):
        vectors = []
        for vector, in cur.execute('SELECT vector FROM original WHERE label = ? ORDER BY RANDOM()', (label,)):
            vectors.append(_np_loads(vector))
        return np.array(vectors)


    def _insert_synthetic(self, cur: sql.Cursor, label: int, synthetic):
        for vector in synthetic:
            cur.execute('INSERT INTO synthetic VALUES (?, ?)', (label, _np_dumps(vector)))


    def _synthesise_vectors(self, cur: sql.Cursor, label: int, n: int, k: int):
        if n == 1:
            cur.execute('INSERT INTO synthetic SELECT * FROM original WHERE label = ?', (label, ))
        else:
            self._insert_synthetic(cur, label, SMOTE(self._select_original(cur, label)).generate(n, k))


    def _marshall(self, global_vec: np.ndarray, h_crops: np.ndarray):
        return _np_dumps(np.concatenate((global_vec.flatten(), h_crops.flatten())))


    def _unmarshall(self, vector: bytes):
        vector = _np_loads(vector)
        global_vec = vector[:256].reshape((256,))
        h_crops = vector[256:].reshape((self._num_classes, 512))
        return global_vec, h_crops


    def synthesise(self):
        with sql.connect(self._path) as con:
            cur = con.cursor()
            cur.execute('DELETE FROM synthetic')

            sizes = [size for size, in con.execute('SELECT COUNT(*) FROM synthetic GROUP BY label ORDER BY label')]
            largest = max(sizes)

            for label, size in enumerate(sizes):
                self._synthesise_vectors(cur, label, n=min(20, largest // size), k=5)


    def clear(self):
        with sql.connect(self._path) as con:
            con.execute('DELETE FROM original')
            con.execute('DELETE FROM synthetic')


    def class_weights(self):
        with sql.connect(self._path) as con:
            sizes = [size for size, in con.execute('SELECT COUNT(*) FROM synthetic GROUP BY label ORDER BY label')]
            avg = sum(sizes) / len(sizes)
            return [avg / size for size in sizes]


    def add(self, labels, global_vec, h_crops):
        with sql.connect(self._path) as con:
            cur = con.cursor()
            for label, gv, hc in zip(labels, global_vec, h_crops):
                cur.execute('INSERT INTO original (label, vector) VALUES (?, ?)', (label, self._marshall(gv, hc)))


    def __len__(self):
        with sql.connect(self._path) as con:
            return con.execute('SELECT COUNT(*) FROM synthetic').fetchone()[0]


    def __getitem__(self, i: int):
        with sql.connect(self._path) as con:
            label, vector = con.execute('SELECT label, vector FROM synthetic ORDER BY rowid LIMIT 1 OFFSET ?', (i,)).fetchone()

            x = self._unmarshall(vector)
            y = np.zeros(self._num_classes, dtype='float32')
            y[label] = 1
            return x, y
