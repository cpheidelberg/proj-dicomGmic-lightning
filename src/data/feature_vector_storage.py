import numpy as np
import io
import os
import base64

class FeatureVectorStorage:
    def __init__(self, path: str, no_finding: int):
        os.remove(path)
        self._path = path
        self._no_finding = no_finding

    def add(self, label: int, vector: np.ndarray):
        if label != self._no_finding:
            with io.BytesIO() as buf, open(self._path, mode='a') as file:
                np.save(buf, vector)
                file.write(f'{label} {base64.b64encode(buf.getvalue()).decode()}\n')

    def add_many(self, labels: np.ndarray, vectors: np.ndarray):
        for label, vector in zip(labels, vectors):
            self.add(label, vector)

    def read(self):
        def parse(line: str) -> tuple[int, np.ndarray]:
            label, vector = line.split(' ', maxsplit=1)
            return int(label), np.load(io.BytesIO(base64.b64decode(vector)))

        with open(self._path) as file:
            return (parse(line) for line in file if line)
