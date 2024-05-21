import numpy as np
import io
import shutil
import base64
import lzma

class FeatureVectorStorage:
    def __init__(self, path: str, no_finding: int):
        self._path = path
        self._no_finding = no_finding
        self._labels = set()

    def clear(self):
        for label in self._labels:
            shutil.rmtree(f'{self._path}.{label}', ignore_errors=True)

    def add(self, label: int, vector: np.ndarray):
        if label != self._no_finding:
            self._labels.add(label)
            with io.BytesIO() as buf, lzma.LZMAFile(f'{self._path}.{label}', mode='a') as file:
                np.save(buf, vector)
                file.write(base64.b85encode(buf.getvalue()) + b'\n')

    def add_many(self, labels: np.ndarray, vectors: np.ndarray):
        for label, vector in zip(labels, vectors):
            self.add(label, vector)

    def read(self, label: int):
        try:
            with lzma.LZMAFile(f'{self._path}.{label}') as file:
                return (np.load(io.BytesIO(base64.b85decode(ln))) for ln in file if ln)
        except OSError:
            return []
