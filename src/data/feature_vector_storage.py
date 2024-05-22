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

    def add(self, labels: np.ndarray, global_vec: np.ndarray, h_crops: np.ndarray):
        tab = pd.DataFrame({'label': labels, 'global_vec': list(global_vec), 'h_crops': list(h_crops)})
        tab.to_csv(self._path, mode='a', header=not os.path.exists(self._path))

    def read(self):
        try:
            tab = pd.read_csv(self._path)
            return list(tab.loc[:, 'label']), list(tab.loc[:, 'global_vec']), list(tab.loc[:, 'h_crops'])
        except:
            return []
