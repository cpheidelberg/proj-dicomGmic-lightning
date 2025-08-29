from torch.utils.data import Dataset

class PathReturningDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.paths = base_dataset.paths  # assumes your original dataset stores `self.paths`

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        path = self.paths[idx]
        return image, label, path