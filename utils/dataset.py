import torch
import numpy as np

class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = np.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
