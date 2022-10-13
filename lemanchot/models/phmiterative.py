
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lemanchot.models.core import BaseModule, model_register

@model_register('phm_iterative')
class phmIterativeModule (BaseModule):

    # def __init__(self, name : str, config) -> None:
    #     super().__init__(
    #         name='wonjik2020',
    #         config=config
    #     )

    #     # First convolutional layer
    #     self.conv1 = nn.Conv2d(
    #         self.num_dim, 
    #         self.num_channels,
    #         kernel_size=3, 
    #         stride=1, 
    #         padding=1
    #     )
    #     self.dp1 = nn.Dropout2d(self.zero_probability)
    #     self.bn1 = nn.BatchNorm2d(self.num_channels)
    #     # Feature space including multiple convolutional layers
    #     self.conv2 = nn.ModuleList()
    #     self.bn2 = nn.ModuleList()
    #     self.dp2 = nn.ModuleList()
    #     # The number of convolutional layers are determined based on "nCov" parameters.
    #     for i in range(self.num_convs-1):
    #         tmpConv = nn.Conv2d(
    #             self.num_channels, 
    #             self.num_channels,
    #             kernel_size=3, 
    #             stride=1, 
    #             padding=1
    #         )
    #         tmpBatch = nn.Dropout2d(self.zero_probability)
    #         tmpBatch2 = nn.BatchNorm2d(self.num_channels)
    #         self.conv2.append(tmpConv)
    #         self.bn2.append(tmpBatch)
    #         self.dp2.append(tmpBatch2)
    #     # The reference normalization for extracting class labels
    #     self.conv3 = nn.Conv2d(
    #         self.num_channels, 
    #         self.num_channels, 
    #         kernel_size=1, 
    #         stride=1, 
    #         padding=0
    #     )
    #     self.bn3 = nn.BatchNorm2d(self.num_channels)
    #     self.dp3 = nn.Dropout2d(self.zero_probability)

    def __init__(self, name : str, config) -> None:
        super().__init__(
            name='wonjik2020',
            config=config
        )

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            self.num_dim, 
            self.num_channels,
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.dp1 = nn.Dropout2d(self.zero_probability)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        # Feature layer
        self.conv11 = nn.Conv2d(
            self.num_channels + self.num_dim, 
            self.num_channels,
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.dp11 = nn.Dropout2d(self.zero_probability)
        self.bn11 = nn.BatchNorm2d(self.num_channels)
        # Feature space including multiple convolutional layers
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.dp2 = nn.ModuleList()
        # The number of convolutional layers are determined based on "nCov" parameters.
        for i in range(self.num_convs-1):
            tmpConv = nn.Conv2d(
                self.num_channels * 2, 
                self.num_channels,
                kernel_size=3, 
                stride=1, 
                padding=1
            )
            tmpBatch = nn.Dropout2d(self.zero_probability)
            tmpBatch2 = nn.BatchNorm2d(self.num_channels)
            self.conv2.append(tmpConv)
            self.bn2.append(tmpBatch)
            self.dp2.append(tmpBatch2)
        # The reference normalization for extracting class labels
        self.conv3 = nn.Conv2d(
            self.num_channels * 2, 
            self.num_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.dp3 = nn.Dropout2d(self.zero_probability)

    def forward(self, out):
        x = self.conv1(out)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.bn1(x)
        x = torch.cat([x, out], 1)
        for i in range(self.num_convs-1):
            out = x
            x = self.conv2[i](out)
            x = F.relu(x)
            x = self.dp2[i](x)
            x = self.bn2[i](x)
            x = torch.cat([x, out], 1)
        x = self.conv3(x)
        x = self.dp3(x)
        x = self.bn3(x)
        return x
