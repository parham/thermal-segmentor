
""" 
    @name kanezaki2018.py   
    @info   kanezaki2018.py provides the model for Kanezaki2018 unsupervised segmentation
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import torch.nn as nn
import torch.nn.functional as F

class Kanezaki2018Module(nn.Module):
    """ Implementation of the model presented in:
    @name           Unsupervised Image Segmentation by Backpropagation
    @journal        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    @year           2018
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation
    @citation       Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.
    """

    def __init__(self, num_dim, num_channels: int = 100, num_convs: int = 3):
        super(Kanezaki2018Module, self).__init__()

        self.num_convs = num_convs
        self.num_channel = num_channels

        self.conv1 = nn.Conv2d(
            num_dim, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(num_convs-1):
            self.conv2.append(nn.Conv2d(
                num_channels, num_channels,
                kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(num_channels))
        self.conv3 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(self.num_convs - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
