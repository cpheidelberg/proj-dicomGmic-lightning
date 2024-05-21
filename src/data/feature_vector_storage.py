import numpy as np
import io
import shutil
import base64

class FeatureVectorStorage:
    def __init__(self, path: str, no_finding: int):
        self._path = path
        self._no_finding = no_finding

    def clear(self):
        shutil.rmtree(self._path, ignore_errors=True)

    def add(self, label: int, vector: np.ndarray):
        if label != self._no_finding:
            with io.BytesIO() as buf, open(self._path, mode='a') as file:
                np.save(buf, np.append(vector, label))
                file.write(base64.b85encode(buf.getvalue()).decode() + '\n')

    def add_many(self, labels: np.ndarray, vectors: np.ndarray):
        for label, vector in zip(labels, vectors):
            self.add(label, vector)

    def read(self):
        def parse(line: str) -> tuple[int, np.ndarray]:
            array = np.load(io.BytesIO(base64.b85decode(line)))
            return round(array[-1]), array[:-1]

        with open(self._path) as file:
            return (parse(line) for line in file if line)
