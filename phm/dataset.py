
import os
import glob
import ntpath
from torch.utils.data import Dataset

class RepetitiveDatasetWrapper(Dataset):
    def __init__(self, ds : Dataset, iteration : int  = 1) -> None:
        super().__init__()
        if iteration < 1:
            raise ValueError('iteration cannot be lower than one!')
        self.iteration = iteration
        self.dataset_ = ds
        self.actual_size = len(self.dataset_)

    @property
    def wrapped_dataset(self):
        return self.dataset_

    def __len__(self):
        return len(self.dataset_) * self.iteration
    
    def __getitem__(self, idx):
        actual_idx = idx + 1
        iter = actual_idx % self.actual_size
        sample_index = (actual_idx // actual_idx) + 1
        return self.dataset_[sample_index]