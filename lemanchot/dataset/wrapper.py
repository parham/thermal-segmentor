
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

from torch.utils.data import Dataset

class WrapperDataset(Dataset):
    def __init__(self,
        dataset : Dataset
    ) -> None:
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class BothTransformWrapperDataset(WrapperDataset):
    def __init__(self, 
        dataset: Dataset,
        both_transform = None
    ) -> None:
        super().__init__(dataset)
        self.both_transform = both_transform
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        record = data
        if self.both_transform is not None:
            input, target = self.both_transform(data[0], data[1])
            reitem = data[2:] if len(data) > 2 else []
            record = [ input, target, *reitem ]
        return record