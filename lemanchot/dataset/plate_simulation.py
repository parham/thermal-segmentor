
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import glob
import os
import numpy as np

from PIL import Image

from torch.utils.data import Dataset

class PlateSimulationDataset(Dataset):
    def __init__(self,
        root_dir : str,
        transforms = None,
        target_transforms = None             
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.root_dir = root_dir
        # Check if the root directory exist!
        if not os.path.isdir(root_dir):
            raise ValueError('The directory "%s" does not exist.' % root_dir)
        label_file = os.path.join(root_dir, 'label.png')
        if not os.path.isfile(label_file):
            raise ValueError('The label image does not exist!')
        self.label = Image.open(label_file)
        self.label = np.asarray(self.label)
        if self.target_transforms is not None:
            self.label = self.target_transforms(self.label)
        # Loading label
        self.file_list = glob.glob(os.path.join(self.root_dir, '*.png'))
        if len(self.file_list) == 0:
            raise ValueError('No mat file does not exist.')
        # Extract list of files
        self.file_list = glob.glob(os.path.join(self.root_dir, 'Image', '*.png'))
        if len(self.file_list) == 0:
            raise ValueError('No image file does exist.')
    
    def __len__(self):
        """the count of image files in the root directory.

        Returns:
            int: number of mat files.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """Getting the data with the given index

        Args:
            idx (int): index of the required file

        Raises:
            ValueError: if the given input tag does not exist in the given mat file
            ValueError: if the given target tag does not exist in the given mat file

        Returns:
            Tuple: input, target, abd filename
        """
        fs = self.file_list[idx]
        filename = os.path.splitext(os.path.basename(fs))[0]
        img = np.asarray(Image.open(fs))
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return (img, self.label, filename)