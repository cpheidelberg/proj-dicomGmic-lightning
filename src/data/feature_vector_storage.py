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
        tab = pd.DataFrame({'label': labels.tolist(), 'global_vec': global_vec.tolist(), 'h_crops': h_crops.tolist()})
        tab.to_csv(self._path, mode='a', header=not os.path.exists(self._path))

    def read(self):
        try:
            tab = pd.read_csv(self._path)
            labels = np.array(list(tab.loc[:, 'label']))
            global_vec = np.array(list(tab.loc[:, 'global_vec']))
            h_crops = np.array(list(tab.loc[:, 'h_crops']))
            return labels, global_vec, h_crops
        except:
            return []
