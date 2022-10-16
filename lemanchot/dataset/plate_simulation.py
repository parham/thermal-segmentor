
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

from lemanchot.processing import classmap_2_multilayer_numpy

class PlateSimulationDataset(Dataset):
    def __init__(self,
        root_dir : str,
        transforms = None,
        target_transforms = None,
        both_transformation = None,
        zero_background = False,
        background_class = 255,
        multilayer_target = False,
        class_list = None
    ) -> None:
        """_summary_

        Args:
            root_dir (str): The directory containing the images.
            transforms (_type_, optional): The transformation applying on the input. Defaults to None.
            target_transforms (_type_, optional): The transformation applying on the target. Defaults to None.
            both_transformation (_type_, optional): The transformation applying on both input and target. Defaults to None.

        Raises:
            ValueError: the root directory does not exist!
            ValueError: the label image does not exist!
            ValueError: no file exist in the dataset directory.
            ValueError: _description_
        """
        super().__init__()
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.both_transforms = both_transformation
        self.root_dir = root_dir
        self.zero_background = zero_background
        self.background_class = background_class
        self.multilayer_target = multilayer_target
        self.class_list = class_list
        # Check if the root directory exist!
        if not os.path.isdir(root_dir):
            raise ValueError('The directory "%s" does not exist.' % root_dir)
        label_file = os.path.join(root_dir, 'label.png')
        if not os.path.isfile(label_file):
            raise ValueError('The label image does not exist!')
        self.label = np.asarray(Image.open(label_file))
        if len(self.label.shape) > 2 and self.label.shape[-1] > 1:
            self.label = self.label[:,:,0]
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
        
        target = self.label
        if self.zero_background:
            tmp = np.uint8(img.mean(axis=2))
            tmp = np.where(target != self.background_class, tmp, 0.)
            img = np.dstack((tmp,tmp,tmp))
        
        if self.multilayer_target:
            target = classmap_2_multilayer_numpy(target, self.class_list)
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        
        if self.both_transforms is not None:
            img, target = self.both_transforms(img, target)
        
        return (img, target, filename)