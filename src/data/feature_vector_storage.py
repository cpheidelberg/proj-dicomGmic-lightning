import numpy as np
import pandas as pd
import os

class FeatureVectorStorage:
    def __init__(self, path: str, no_finding: int):
        self._path = path
        self._no_finding = no_finding

    def clear(self):
        try:
            os.remove(self._path)
        except:
            pass

    def add(self, label: int, global_vec: np.ndarray, h_crops: np.ndarray):
        if label != self._no_finding:
            tab = pd.DataFrame({'label': [label], 'global_vec': [global_vec], 'h_crops': [h_crops]})
            tab.to_csv(self._path, mode='a', header=not os.path.exists(self._path))

    def add_many(self, labels: np.ndarray, vectors: np.ndarray):
        for label, vector in zip(labels, vectors):
            self.add(label, vector)

    def read(self, label: int):
        try:
            return pd.read_csv(self._path)
        except:
            return []
