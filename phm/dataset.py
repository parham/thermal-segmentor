
import os
import glob
import ntpath
from torch.utils.data import Dataset

__original_dir__ = 'original'
__fileformat__ = '.png'

def read_dataset_struct(root_dir : str):
    if not os.path.isdir(root_dir):
        raise ValueError('root_dir is not a directory!')
    # List of subdirectories representing class labels
    clss = os.listdir(root_dir)
    class_lookup = {}
    for c in clss:
        cdir = os.path.join(root_dir, c)
        if os.path.isdir(cdir):
            class_lookup[c] = cdir
    # List of unique files 
    orig_dir = os.path.join(root_dir, __original_dir__)
    files = [ntpath.basename(x) for x in glob.glob(os.path.join(orig_dir,'*' + __fileformat__))]
    
    return class_lookup, files


class phmMaskDataset(Dataset):

    def __init__(self, root_dir : str, transform = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        class_lookup, files = read_dataset_struct(root_dir)
        self.class_lookup = class_lookup
        self.files = files
        self.orig_dir = os.path.join(self.root_dir, __original_dir__)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= len(self.files):
            return None
        for c, cpath in self.class_lookup.items():
            
if __name__ == '__main__':
    testdir = '/home/phm/Documents/temp'
    class_lookup, files = read_dataset_struct(testdir)
    