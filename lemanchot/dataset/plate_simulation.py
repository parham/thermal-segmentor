
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
        target_transforms = None,
        both_transformation = None,
        zero_background = False,
        background_class = 255
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
        # Check if the root directory exist!
        if not os.path.isdir(root_dir):
            raise ValueError('The directory "%s" does not exist.' % root_dir)
        label_file = os.path.join(root_dir, 'label.png')
        if not os.path.isfile(label_file):
            raise ValueError('The label image does not exist!')
        self.label = Image.open(label_file)
        self.label = np.asarray(self.label)
        if len(self.label.shape) > 2 and self.label.shape[-1] > 1:
            self.label = self.label[:,:,0]
        if self.target_transforms is not None:
            self.label = self.target_transforms(self.label)
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
        
        target = self.label
        if self.zero_background:
            tmp = np.uint8(np.mean(img, axis=2))
            tmp = np.where(target != 0, tmp, 0.)
            img = np.dstack((tmp,tmp,tmp))
        
        if self.both_transforms is not None:
            trg = target.clone()
            if trg.shape == 2:
                target = target.unsqueeze(0)
            target = self.both_transforms(target)
            if trg.shape == 2:
                target = target.squeeze(0)
            img = self.both_transforms(img)
            
        return (img, target, filename)