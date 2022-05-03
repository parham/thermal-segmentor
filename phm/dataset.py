
""" 
    @name dataset.py   
    @info   dataset.py provides customizable dataset
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import os
import numpy as np

from typing import Dict
from torch.utils.data import Dataset
from gimp_labeling_converter import generate_cmap, gimp_helper


class FileRepeaterDataset(Dataset):

    __xcf_filext = '.xcf'

    def __init__(self, file : str, 
        category : Dict, iteration : int = 1,
        transform = None,
        target_transform = None
    ) -> None:
        super().__init__()

        self.iteration = iteration
        if not os.path.isfile(file) or \
            not file.endswith(self.__xcf_filext):
            raise ValueError(f'{file} is invalid!')

        res = generate_cmap(file=file, helper=gimp_helper, category=category)
        self.image = np.asarray(res['original'])
        self.target = np.asarray(res['target'])
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.iteration
    
    def __getitem__(self, idx):
        img = self.image
        target = self.target
        if self.transform is not None:
            img = self.transform(self.image)
        if self.target_transform is not None:
            target = self.target_transform(self.target)
        return self.image, self.target

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