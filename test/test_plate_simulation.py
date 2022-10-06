
import os
import sys
import unittest
import torch
import numpy as np

from PIL import Image

from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.dataset import PlateSimulationDataset

class TestDataset(unittest.TestCase):
    
    def test_PlateSimulationDataset(self):
        dataset = PlateSimulationDataset(
            root_dir = '/data/thermal-segmentation/Plate_Simulation/Curve-CFRP',
            zero_background=True,
            background_class=0
        )
        batch_size = 1
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for img, target, fname in data_loader:
            print(f'Reading {fname}')
            Image.fromarray(np.uint8(img.cpu().detach().numpy().squeeze(0) * 255)).save('/data/a.png')
            

if __name__ == '__main__':
    unittest.main()