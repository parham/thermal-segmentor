
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @email      parham.nooralishahi.1@ulaval.ca
"""

import os
import glob
import ntpath
import numpy as np

from typing import Dict
from torch.utils.data import Dataset

from gimp_labeling_converter import generate_cmap, gimp_helper, phmError

from lemanchot.processing import classmap_2_multilayer_numpy

class XCFDataset(Dataset):
    def __init__(self, 
        root_dir : str, 
        category : Dict[str, int],
        transform = None,
        target_transform = None,
        both_transformation = None,
        multilayer_target = False,
        class_list = None
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        if not os.path.isdir(self.root_dir):
            raise phmError('Directory path is invalid!')
        if category is None or len(category.keys()) == 0:
            raise phmError('Category should be given!')
        self.category = category
        self.transform = transform
        self.target_transform = target_transform
        self.both_transforms = both_transformation
        self.files = glob.glob(os.path.join(root_dir,'*.xcf'))
        self.multilayer_target = multilayer_target
        self.class_list = class_list
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        filename = os.path.splitext(os.path.basename(file))[0]
        res = generate_cmap(file=file, helper=gimp_helper, category=self.category)
        img = np.asarray(res['original'])
        target = res['target']
        
        if self.multilayer_target:
            target = classmap_2_multilayer_numpy(target, self.class_list)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.both_transforms is not None:
            img, target = self.both_transforms(img, target)

        return img, target, filename