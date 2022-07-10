
""" 
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from crfseg import CRF

class Phm2022Module (nn.Module):
    def __init__(self, num_dim, num_channels: int = 100, num_convs: int = 6):
        super(Phm2022Module, self).__init__()

        self.num_convs = num_convs
        self.num_channel = num_channels

        layers = []
        layers.append(('conv1', nn.Conv2d(
            num_dim, self.num_channel,
            kernel_size=3, stride=1, padding=1)))
        layers.append(('relu1', nn.ReLU()))
        layers.append(('bnorm1', nn.BatchNorm2d(self.num_channel)))
        # Feature space including multiple convolutional layers
        for i in range(self.num_convs-1):
            layers.append((f'conv2{i+1}', nn.Conv2d(
                self.num_channel, self.num_channel,
                kernel_size=3, stride=1, padding=1)))
            layers.append((f'relu2{i+1}', nn.ReLU()))
            layers.append((f'bnorm2{i+1}', nn.BatchNorm2d(self.num_channel)))
        # The reference normalization for extracting class labels
        layers.append(('conv3', nn.Conv2d(self.num_channel, 
            self.num_channel, kernel_size=1, 
            stride=1, padding=0)))
        layers.append(('relu3', nn.ReLU()))
        layers.append(('bnorm3', nn.BatchNorm2d(self.num_channel)))
        # CRF layer to smoothen the prediction
        layers.append('crf', CRF(n_spatial_dims=2))

        self._model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self._model(x)
        return x