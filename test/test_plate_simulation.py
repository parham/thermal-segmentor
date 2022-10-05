
import os
import sys
import unittest
import torch
import numpy as np

from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.dataset import PlateSimulationDataset

class TestDataset(unittest.TestCase):
    
    def test_PlateSimulationDataset(self):
        dataset = PlateSimulationDataset(
            root_dir = '/data/thermal-segmentation/Plate_Simulation/Curve-CFRP'
        )
        batch_size = 2
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for img, target, fname in data_loader:
            print(f'Reading {fname}')

if __name__ == '__main__':
    unittest.main()