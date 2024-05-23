import pickle
import sqlite3
import torch.utils.data as torchdata

class Storage(torchdata.Dataset):
    def __init__(self, path: str, no_finding: int):
        self._no_finding = no_finding

        self._con = sqlite3.connect(path)
        self._cur = self._con.cursor()
        self.reset()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._cur.close()
        self._con.close()

    def reset(self):
        self._cur.execute('DROP TABLE IF EXISTS feature_vector')
        self._cur.execute('CREATE TABLE feature_vector (i, label, global_vec, h_crops)')
        self._con.commit()
        self._i = 0

    def add(self, labels, global_vec, h_crops):
        for label, gv, hc in zip(labels, global_vec, h_crops):
            if label != self._no_finding:
                self._cur.execute('''
                INSERT INTO feature_vector (i, label, global_vec, h_crops)
                VALUES (?, ?, ?, ?)
                ''', (self._i, label, pickle.dumps(gv), pickle.dumps(hc)))
                self._i += 1

        self._con.commit()

    def __len__(self):
        return self._i

    def __getitem__(self, index):
        return self._cur.execute('SELECT label, global_vec, h_crops FROM feature_vector WHERE i = ?', (index,))
