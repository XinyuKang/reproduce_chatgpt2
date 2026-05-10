import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

class GPTDataset(Dataset):
    def __init__(self, path, block_size):
        self.data = np.memmap(
            path,
            dtype=np.uint16,
            mode="r",
        )
        self.block_size = block_size
        self.total_len = (len(self.data) + self.block_size - 1) // self.block_size
        
        
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx == self.total_len - 1:
            x = self.data[idx * self.block_size:-1]
            y = self.data[idx * self.block_size+1:]
            return torch.from_numpy(x), torch.from_numpy(y)

        x = self.data[idx * self.block_size:(idx+1)*self.block_size-1]
        y = self.data[idx * self.block_size+1:(idx+1)*self.block_size]
        return torch.tensor(x), torch.tensor(y)